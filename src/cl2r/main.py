import argparse
import yaml
import os
import os.path as osp
import numpy as np

from continuum import ClassIncremental
from continuum.datasets import CIFAR100
from continuum import rehearsal

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cl2r.params import ExperimentParams
from cl2r.utils import create_pairs, update_criterion_weight, BalancedBatchSampler
from cl2r.model import ResNet32Cifar
from cl2r.train import train, classification
from cl2r.eval import validation, evaluate


def main():
    # load params from the config file from yaml to dataclass
    parser = argparse.ArgumentParser(description='CL2R: Compatible Lifelong Learning Represenations')
    parser.add_argument("--config_path",
                        help="path of the experiment yaml",
                        default=os.path.join(os.getcwd(), "config.yml"),
                        type=str)
    params = parser.parse_args()
    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    args = ExperimentParams()
    for k, v in loaded_params.items():
        args.__setattr__(k, v)
    args.yaml_name = os.path.basename(params.config_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current args:\n{vars(args)}")
    
    # dataset
    data_path = osp.join(args.root_folder, "data")
    if not osp.exists(data_path):
        os.makedirs(data_path)
    if not osp.exists(osp.join(args.root_folder, "checkpoints")):
        os.makedirs(osp.join(args.root_folder, "checkpoints"))
        
    print(f"Loading Training Dataset")
    train_transform = [transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                       ]
    dataset_train = CIFAR100(data_path=data_path, train=True, download=True)
    # create task-sets for lifelong learning
    scenario_train = ClassIncremental(dataset_train,
                                      initial_increment=args.start,
                                      increment=args.increment,
                                      transformations=train_transform)

    args.num_classes = scenario_train.nb_classes
    args.nb_tasks = scenario_train.nb_tasks

    val_transform = [transforms.ToTensor(),
                     transforms.Normalize((0.5071, 0.4867, 0.4408),
                                          (0.2675, 0.2565, 0.2761))
                    ]
    dataset_val = CIFAR100(data_path=data_path, train=False, download=True)
    # create task-sets for lifelong learning
    scenario_val = ClassIncremental(dataset_val,
                                    initial_increment=args.start,
                                    increment=args.increment,
                                    transformations=val_transform)
    
    # create episodic memory dataset
    memory = rehearsal.RehearsalMemory(memory_size=args.num_classes * args.rehearsal,
                                       herding_method="random",
                                       fixed_memory=True,
                                       nb_total_classes=args.num_classes
                                    )

    print(f"Creating Pairs Dataset")
    query_set, gallery_set = create_pairs(data_path=data_path)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)

    print(f"Starting Training")
    for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
        print(f"Task {task_id+1} Classes in task: {train_task_set.get_classes()}")

        rp = ckpt_path if (task_id > 0) else None
        net = ResNet32Cifar(resume_path=rp, 
                            starting_classes=100, 
                            feat_size=99, 
                            device=args.device)

        if task_id > 0:
            previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device)
            previous_net.eval() 
        else:
            previous_net = None
        
        print(f"Created model {'and old model' if task_id > 0 else ''}")

        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-04)
        scheduler_lr = MultiStepLR(optimizer, milestones=args.stages, gamma=0.1)
        criterion_cls = nn.CrossEntropyLoss().cuda(args.device)

        if task_id > 0:
            args.criterion_weight = update_criterion_weight(args, args.seen_classes.shape[0], train_task_set.nb_classes)
            mem_x, mem_y, mem_t = memory.get()
            train_task_set.add_samples(mem_x, mem_y, mem_t)
            batchsampler = BalancedBatchSampler(train_task_set, n_classes=train_task_set.nb_classes, 
                                                batch_size=args.batch_size, n_samples=len(train_task_set._x), 
                                                seen_classes=args.seen_classes)
            train_loader = DataLoader(train_task_set, batch_sampler=batchsampler, num_workers=args.num_workers) 
        else:
            train_loader = DataLoader(train_task_set, batch_size=args.batch_size, shuffle=True, 
                                      drop_last=True, num_workers=args.num_workers) 

        val_dataset = scenario_val[:task_id + 1]
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                sampler=None, drop_last=False, num_workers=args.num_workers)
            
        best_acc = 0
        print(f"Starting Epoch Loop at task {task_id + 1}/{scenario_train.nb_tasks}")
        for epoch in range(args.epochs):
            train(args, net, train_loader, optimizer, epoch, criterion_cls, previous_net, task_id)
            # acc_val = validation(args, net, query_loader, gallery_loader, task_id, selftest=(task_id == 0))
            # if task_id > 0:
            #     acc_val = validation(args, query_loader, gallery_loader, task_id)
            # else:
            acc_val = classification(args, net, val_loader, criterion_cls)
            scheduler_lr.step()
            
            if acc_val > best_acc:
                best_acc = acc_val
                print("Saving model")
                ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{task_id}.pt"))
                torch.save(net.state_dict(), ckpt_path)
        
        memory.add(*scenario_train[task_id].get_raw_samples(), z=None)  
        args.seen_classes = torch.tensor(list(memory.seen_classes), device=args.device)
        
    print(f"Starting Evaluation")
    query_set, gallery_set = create_pairs(data_path=data_path)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)
    evaluate(args, query_loader, gallery_loader)


if __name__ == '__main__':
    main()