

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from losses.PriorCELoss import *

class LADELoss(nn.Module):
    def __init__(self, configs, cls_num_list, remine_lambda=0.1):
        super().__init__()

        self.prior = None
        self.num_classes = configs.general.num_classes
        self.img_num_per_cls = torch.Tensor(cls_num_list)
        self.balanced_prior = torch.tensor(1. / self.num_classes).float()
        self.remine_lambda = remine_lambda
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float()))
        if configs.cuda.use_cuda:
            self.img_num_per_cls = self.img_num_per_cls.cuda()
            self.balanced_prior = self.balanced_prior.cuda()
            self.cls_weight = self.cls_weight.cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

def create_loss(configs, cls_num_list, remine_lambda=0.1):
    print("Loading LADELoss.")
    return LADELoss(
        configs=configs, 
        cls_num_list=cls_num_list,
        remine_lambda=remine_lambda,
    )

class UnifiedLoss(nn.Module):
    def __init__(self, configs, cls_num_list, remine_lambda=0.1):
        super().__init__()
        self.priorce_loss = create_priorce_loss(configs, cls_num_list)
        self.lade_loss = create_loss(configs, cls_num_list)
    def forward(outputs, labels):
        loss_priorce = self.priorce_loss(outputs, labels)
        loss_lade = self.lade_loss(outputs, labels)
        loss = loss_priorce + loss_lade
        return loss
