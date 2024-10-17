import numpy as np
import torch
import torch.nn as nn
def create_RWLoss_1(cls_num_list):
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    train_loss = nn.CrossEntropyLoss(weight=per_cls_weights)
    return train_loss

def create_RWLoss_2(cls_num_list):
    num = sum(cls_num_list)
    prob = [i/num for i in cls_num_list]
    prob = torch.FloatTensor(prob)
    # normalization
    max_prob = prob.max().item()
    prob = prob / max_prob
    # class reweight
    weight = - prob.log() + 1
    train_loss = nn.CrossEntropyLoss(weight=weight.cuda())
    return train_loss