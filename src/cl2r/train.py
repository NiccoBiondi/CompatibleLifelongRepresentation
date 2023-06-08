import torch
import torch.nn.functional as F

import time

from cl2r.utils import AverageMeter, log_epoch, l2_norm


def train(args, net, train_loader, optimizer, epoch, criterion_cls, previous_net, task_id):
    
    start = time.time()
    
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for inputs, targets, t in train_loader:
        
        inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
        feature, output = net(inputs)
        loss = criterion_cls(output, targets)
        
        if previous_net is not None:
            with torch.no_grad():
                feature_old, logits_old = previous_net(inputs)

            feat_old = feature_old[:args.batch_size//2] # only on memory samples
            feat_new = feature[:args.batch_size//2]     # only on memory samples

            norm_feature_old, norm_feature_new = l2_norm(feat_old), l2_norm(feat_new)
            loss_fd = EmbeddingsSimilarity(norm_feature_new, norm_feature_old)
            loss = loss + args.criterion_weight * loss_fd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(output, targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, time=end-start)


def EmbeddingsSimilarity(feature_a, feature_b):
    return F.cosine_embedding_loss(
        feature_a, feature_b,
        torch.ones(feature_a.shape[0]).to(feature_a.device)
    )


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def classification(args, net, loader, criterion_cls):
    net.eval()
    classification_loss_meter = AverageMeter()
    classification_acc_meter = AverageMeter()
    with torch.no_grad():
        for inputs, targets, t in loader:

            inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
            feature, output = net(inputs)
            loss = criterion_cls(output, targets)

            classification_loss_meter.update(loss.item(), inputs.size(0))
            classification_acc = accuracy(output, targets, topk=(1,))
            classification_acc_meter.update(classification_acc[0].item(), inputs.size(0))

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, classification=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc