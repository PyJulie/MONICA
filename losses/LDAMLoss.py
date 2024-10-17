import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, configs, cls_num_list, max_m=0.5, weight=True, s=30, rule='DRW'):
        super(LDAMLoss, self).__init__()
        print('cls_num_list:', cls_num_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.cls_num_list = cls_num_list
        assert s > 0
        self.s = s
        self.rule = rule
        self.configs = configs
        self.weight = weight
        

    def _get_weight(self, epoch):
        if self.rule == 'DRW':
            idx = epoch // int(self.configs.general.train_epochs*4/5)
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            effective_num = 1.0 - np.power(0, self.cls_num_list)
            per_cls_weights = (1.0) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            return per_cls_weights

    def forward(self, x, target, epoch):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        if self.weight:
            weight = self._get_weight(epoch)
        else:
            weight = None

        return F.cross_entropy(self.s*output, target, weight=weight)