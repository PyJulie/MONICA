import numpy as np
import datetime
import os
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
def calculate_metrics_single(labels, outputs, correct, total):

    prediction = torch.argmax(outputs, 1)
    res = prediction == labels
    for label_idx in range(len(labels)):
        label_single = labels[label_idx]
        correct[label_single] += res[label_idx].item()
        total[label_single] += 1
    return correct, total


def calculate_metrics(configs, all_outputs, all_labels):
    all_outputs = torch.cat(all_outputs, dim=0).numpy()  # 合并输出
    all_labels = torch.cat(all_labels, dim=0).numpy()    # 合并标签

    accs, group_accs = calculate_accs(configs, all_outputs, all_labels)
    aucs, group_aucs = calculate_AUROC(configs, all_outputs, all_labels)
    auprcs, group_auprcs = calculate_AUPRC(configs, all_outputs, all_labels)
    f1s, group_f1s = calculate_F1(configs, all_outputs, all_labels)
    valid_results = {
            'class_acc': accs,
            'accuracy': group_accs,
            'class_auc': aucs,
            'aucs': group_aucs,
            'class_auprc': auprcs,
            'auprcs': group_auprcs,
            'class_f1': f1s,
            'f1s': group_f1s,
        }
    return valid_results


def calculate_AUROC(configs, all_outputs, all_labels):
    num_classes = configs.general.num_classes
    auc_scores = []

    # 计算每个类别的AUC
    for i in range(num_classes):
        true_binary_labels = (all_labels == i).astype(int)  # 当前类别的二分类标签
        auc = roc_auc_score(true_binary_labels, all_outputs[:, i]) if np.any(true_binary_labels) else 0.0
        auc_scores.append(auc)

    auc_scores = np.array(auc_scores)

    # 根据类别范围计算分组AUC
    head, medium = configs.datasets.head, configs.datasets.medium
    many_auc = np.average(auc_scores[:head]) if head > 0 else 0.0
    medium_auc = np.average(auc_scores[head:medium]) if medium > head else 0.0
    tail_auc = np.average(auc_scores[medium:]) if medium < num_classes else 0.0
    avg_auc = np.average([many_auc, medium_auc, tail_auc])

    return auc_scores, [many_auc, medium_auc, tail_auc, avg_auc]


def calculate_accs(configs, all_outputs, all_labels):
    num_classes = configs.general.num_classes
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    # 获取预测标签
    predicted_labels = np.argmax(all_outputs, axis=1)
    true_labels = all_labels

    # 统计每个类别的正确和总数
    for true, pred in zip(true_labels, predicted_labels):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    # 计算每个类别的Accuracy
    accs = [
        (class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    ]

    accs = np.array(accs)

    # 根据类别范围计算分组Accuracy
    head, medium = configs.datasets.head, configs.datasets.medium
    many_acc = np.average(accs[:head]) if head > 0 else 0.0
    medium_acc = np.average(accs[head:medium]) if medium > head else 0.0
    tail_acc = np.average(accs[medium:]) if medium < num_classes else 0.0
    avg_acc = np.average([many_acc, medium_acc, tail_acc])

    return accs, [many_acc, medium_acc, tail_acc, avg_acc]

def calculate_AUPRC(configs, all_outputs, all_labels):
    num_classes = configs.general.num_classes
    auprc_scores = []

    # 计算每个类别的AUPRC
    for i in range(num_classes):
        true_binary_labels = (all_labels == i).astype(int)  # 当前类别的二分类标签
        auprc = average_precision_score(true_binary_labels, all_outputs[:, i]) if np.any(true_binary_labels) else 0.0
        auprc_scores.append(auprc)

    auprc_scores = np.array(auprc_scores)

    # 根据类别范围计算分组AUPRC
    head, medium = configs.datasets.head, configs.datasets.medium
    many_auprc = np.average(auprc_scores[:head]) if head > 0 else 0.0
    medium_auprc = np.average(auprc_scores[head:medium]) if medium > head else 0.0
    tail_auprc = np.average(auprc_scores[medium:]) if medium < num_classes else 0.0
    avg_auprc = np.average([many_auprc, medium_auprc, tail_auprc])

    return auprc_scores, [many_auprc, medium_auprc, tail_auprc, avg_auprc]

def calculate_F1(configs, all_outputs, all_labels):
    num_classes = configs.general.num_classes
    f1_scores = []

    # 获取预测类别
    predicted_labels = np.argmax(all_outputs, axis=1)
    true_labels = all_labels

    # 计算每个类别的F1 Score
    for i in range(num_classes):
        true_binary_labels = (true_labels == i).astype(int)  # 当前类别的二分类标签
        pred_binary_labels = (predicted_labels == i).astype(int)  # 当前类别的预测标签
        f1 = f1_score(true_binary_labels, pred_binary_labels) if np.any(true_binary_labels) else 0.0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # 根据类别范围计算分组F1 Score
    head, medium = configs.datasets.head, configs.datasets.medium
    many_f1 = np.average(f1_scores[:head]) if head > 0 else 0.0
    medium_f1 = np.average(f1_scores[head:medium]) if medium > head else 0.0
    tail_f1 = np.average(f1_scores[medium:]) if medium < num_classes else 0.0
    avg_f1 = np.average([many_f1, medium_f1, tail_f1])

    return f1_scores, [many_f1, medium_f1, tail_f1, avg_f1]

def log_results_train(configs, results):
    outputs_dir = 'outputs/%s/%s/' %(configs.general.dataset_name, configs.general.save_name)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    with open(os.path.join(outputs_dir, 'logs.txt'), 'a+') as f:
        epoch_status_info = 'epoch: %s %s\n' %(results['epoch'], results['status'])
        loss_info = 'loss: %s\n' %results['loss']
        class_acc_info = 'class acc: '
        for _num_class in range(configs.general.num_classes):
            class_acc_info += '%.2f ' %results['class_acc'][_num_class]
        class_acc_info += '\n'
        group_acc_info = 'group acc: '
        for _acc in results['accuracy']:
            group_acc_info += '%.2f ' %_acc
        group_acc_info += '\n'

        f.writelines(epoch_status_info)
        f.writelines(loss_info)
        f.writelines(class_acc_info)
        f.writelines(group_acc_info)

def log_results(configs, results):
    outputs_dir = f'outputs/{configs.general.dataset_name}/{configs.general.save_name}/'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    with open(os.path.join(outputs_dir, 'logs.txt'), 'a+') as f:
        epoch_status_info = f'epoch: {results["epoch"]} {results["status"]}\n'
        loss_info = f'loss: {results["loss"]}\n'

        class_acc_info = 'class acc: ' + ' '.join(f'{acc:.2f}' for acc in results['class_acc']) + '\n'
        group_acc_info = 'group acc: ' + ' '.join(f'{acc:.2f}' for acc in results['accuracy']) + '\n'

        class_auc_info = 'class auc: ' + ' '.join(f'{100*auc:.2f}' for auc in results['class_auc']) + '\n'
        group_auc_info = 'group auc: ' + ' '.join(f'{100*auc:.2f}' for auc in results['aucs']) + '\n'

        class_auprc_info = 'class auprc: ' + ' '.join(f'{100*auprc:.2f}' for auprc in results['class_auprc']) + '\n'
        group_auprc_info = 'group auprc: ' + ' '.join(f'{100*auprc:.2f}' for auprc in results['auprcs']) + '\n'

        class_f1_info = 'class f1: ' + ' '.join(f'{100*f1:.2f}' for f1 in results['class_f1']) + '\n'
        group_f1_info = 'group f1: ' + ' '.join(f'{100*f1:.2f}' for f1 in results['f1s']) + '\n'

        f.writelines(epoch_status_info)
        f.writelines(loss_info)
        f.writelines(class_acc_info)
        f.writelines(group_acc_info)
        f.writelines(class_auc_info)
        f.writelines(group_auc_info)
        f.writelines(class_auprc_info)
        f.writelines(group_auprc_info)
        f.writelines(class_f1_info)
        f.writelines(group_f1_info)


