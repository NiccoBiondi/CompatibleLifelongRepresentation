import numpy as np
import os.path as osp
from sklearn.model_selection import KFold

from cl2r.utils import extract_features
from cl2r.model import ResNet32Cifar
from cl2r.metrics import average_compatibility, backward_compatibility, forward_compatibility


def evaluate(args, query_loader, gallery_loader):

    compatibility_matrix = np.zeros((args.nb_tasks, args.nb_tasks))
    targets = query_loader.dataset.targets

    for task_id in range(args.nb_tasks):
        ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt")) 
        net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device)
        net.eval() 
        query_feat = extract_features(args, net, query_loader)

        for i in range(task_id+1):
            ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{i}.pt")) 
            previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device)
            previous_net.eval() 
        
            gallery_feat = extract_features(args, previous_net, gallery_loader)
            acc = verification(query_feat, gallery_feat, targets)
            compatibility_matrix[task_id][i] = acc

            if i != task_id:
                acc_str = f'Cross-test accuracy between model at task {task_id+1} and {i+1}:'
            else:
                acc_str = f'Self-test of model at task {i+1}:'
            print(f'{acc_str} {acc*100:.2f}')

    # compatibility metrics
    ac = average_compatibility(matrix=compatibility_matrix)
    bc = backward_compatibility(matrix=compatibility_matrix)
    fc = forward_compatibility(matrix=compatibility_matrix)

    print(f"Avg. Comp. {ac:.2f}")
    print(f"Backw. Comp. {bc:.3f}")
    print(f"Forw. Comp. {fc:.3f}")

    print(f"Compatibility Matrix:\n{compatibility_matrix}")
    np.save(osp.join(f"./{args.checkpoint_path}/compatibility_matrix.npy"), compatibility_matrix)


def validation(args, net, query_loader, gallery_loader, task_id, selftest=False):
    targets = query_loader.dataset.targets

    net.eval() 
    query_feat = extract_features(args, net, query_loader)

    if selftest:
        previous_net = net
    else:
        ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{task_id-1}.pt")) 
        previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                        starting_classes=100, 
                                        feat_size=99, 
                                        device=args.device)
        previous_net.eval() 
    gallery_feat = extract_features(args, previous_net, gallery_loader)
    acc = verification(query_feat, gallery_feat, targets)
    print(f"{'Self' if selftest else 'Cross'} Compatibility Accuracy: {acc*100:.2f}")
    return acc


"""From [insightface](https://github.com/deepinsight/insightface)"""
def verification(query_feature, gallery_feature, targets):
    thresholds = np.arange(0, 4, 0.001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, query_feature, gallery_feature, targets)
    return accuracy.mean()


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                actual_issame[test_set])

        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

