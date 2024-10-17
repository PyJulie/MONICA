import numpy as np
import datetime
import os
import torch
def calculate_metrics_single(labels, outputs, correct, total):

    prediction = torch.argmax(outputs, 1)
    res = prediction == labels
    for label_idx in range(len(labels)):
        label_single = labels[label_idx]
        correct[label_single] += res[label_idx].item()
        total[label_single] += 1
    return correct, total

def calculate_metrics(configs, correct, total):
    accs = []
    for acc_idx in range(configs.general.num_classes):
        try:
            acc = correct[acc_idx]/total[acc_idx]
        except:
            acc = 0
        finally:
            accs.append(round(acc,4))
    accs = np.array(accs)
    many_acc = np.average(accs[:configs.datasets.head])
    medium_acc = np.average(accs[configs.datasets.head:configs.datasets.medium])
    tail_acc = np.average(accs[configs.datasets.medium:])
    avg_acc = np.average([many_acc, medium_acc, tail_acc])
    return accs, [many_acc, medium_acc, tail_acc, avg_acc]

def log_results(configs, results, epoch):
    outputs_dir = 'outputs/%s/%s/' %(configs.general.dataset_name, configs.general.save_name)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    with open(os.path.join(outputs_dir, 'logs.txt'), 'a+') as f:
        epoch_status_info = 'epoch: %s %s\n' %(results['epoch'], results['status'])
        loss_info = 'loss: %s\n' %results['loss']
        class_acc_info = 'class acc: '
        for _num_class in range(configs.general.num_classes):
            class_acc_info += '%s ' %results['class_acc'][_num_class]
        class_acc_info += '\n'
        group_acc_info = 'group acc: '
        for _acc in results['accuracy']:
            group_acc_info += '%s ' %_acc
        group_acc_info += '\n'

        f.writelines(epoch_status_info)
        f.writelines(loss_info)
        f.writelines(class_acc_info)
        f.writelines(group_acc_info)




    # savers[epoch] = results
    # np.save(os.path.join(outputs_dir, 'saver.npy'), savers)
    # np.save(os.path.join(outputs_dir, 'configs.npy'), configs)