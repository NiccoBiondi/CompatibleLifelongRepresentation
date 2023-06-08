import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data.sampler import BatchSampler

import numpy as np
from PIL import Image
from collections import defaultdict as dd

from continuum.datasets import CIFAR10


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class ImagesDataset(Dataset):
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = None if transform is None else transforms.Compose(transform)

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            x = Image.open(self.data[index]).convert('RGB')
        else:
            if self.transform: 
                x = Image.fromarray(self.data[index].astype(np.uint8))
            else:
                x = self.data[index]

        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, n_classes, n_samples, seen_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.seen_classes = seen_classes
        self.n_batches = self.n_samples // self.batch_size # drop last
        self.index_dic = dd(list)
        self.indices = []
        self.seen_indices = []
        for index, y in enumerate(self.dataset._y):
            if y not in self.seen_classes:
                self.indices.append(index)
            else:
                self.seen_indices.append(index)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            batch.extend(np.random.choice(self.seen_indices, size=self.batch_size//2, replace=False))
            batch.extend(np.random.choice(self.indices, size=self.batch_size//2, replace=False))
            yield batch

    def __len__(self):
        return self.n_batches


def create_pairs(data_path, num_pos_pairs=3000, num_neg_pairs=3000):

    dataset = CIFAR10(data_path=data_path, train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                    ])
    
    data = np.array(dataset.dataset.data)
    targets = np.asarray(dataset.dataset.targets) 
    
    imgs = []
    labels = []
    c_p = 0
    c_n = 0
    
    while len(labels) < num_neg_pairs+num_neg_pairs:
        id0, id1 = np.random.choice(np.arange(0,len(targets)),2)
        if targets[id0] == targets[id1] and c_p < num_pos_pairs:
            labels.append(True)
            c_p += 1
        elif targets[id0] != targets[id1] and c_n < num_neg_pairs:
            labels.append(False)
            c_n += 1
        else:
            continue
        if isinstance(data[id0], str):
            img0 = Image.open(data[id0]).convert('RGB')
        else:
            img0 = data[id0]
        if isinstance(data[id1], str):
            img1 = Image.open(data[id1]).convert('RGB')
        else: 
            img1 = data[id1]
        img0 = transform(img0)
        img1 = transform(img1)
        imgs.append(torch.unsqueeze(img0,0))
        imgs.append(torch.unsqueeze(img1,0))
        print(f"{c_p+c_n}/{num_neg_pairs+num_pos_pairs} pairs", end="\r")
    
    print(f"{len(labels)}/{num_neg_pairs+num_pos_pairs} pairs")
    data = torch.cat(imgs).detach().numpy()
    targets = np.asarray(labels)

    query_set = ImagesDataset(data[0::2], targets)
    gallery_set = ImagesDataset(data[1::2], targets)

    return query_set, gallery_set
   

def update_criterion_weight(args, num_old_classes, num_new_classes):
    args.criterion_weight = args.criterion_weight_base * np.sqrt(num_new_classes / num_old_classes)
    return args.criterion_weight

  
def extract_features(args, net, loader):
    features = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs[0].cuda(args.device)
            f, _ = net(inputs)
            f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
            else:
                features = f
    
    return features.detach().cpu().numpy()


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_epoch(n_epochs=None, loss=None, acc=None, epoch=None, task=None, time=None, classification=False):
    acc_str = f"Task {task + 1}" if task is not None else f""
    acc_str += f" Epoch [{epoch + 1}]/[{n_epochs}]" if epoch is not None else f""
    acc_str += f"\t Training Loss: {loss:.4f}" if loss is not None else f""
    acc_str += f"\t Training Accuracy: {acc:.2f}" if acc is not None else f""
    acc_str += f"\t Time: {time:.2f}" if time is not None else f""
    if classification:
        acc_str = acc_str.replace("Training", "Classification")   
    print(acc_str)
