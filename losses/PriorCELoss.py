import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, configs, cls_num_list):
        super().__init__()
        self.img_num_per_cls = torch.Tensor(cls_num_list)
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        if configs.cuda.use_gpu:
            self.prior = self.prior.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = configs.general.num_classes

    def forward(self, x, y):
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


def create_priorce_loss(configs, cls_num_list):
    print('Loading PriorCELoss Loss.')
    return PriorCELoss(
        configs = configs,
        cls_num_list=cls_num_list
    )