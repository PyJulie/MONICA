import torch.nn as nn
import torch
import numpy as np
from losses.Focal import FocalLoss
from losses.BalancedSoftmax import create_balanced_softmax_loss
from losses.CBLoss import CBLoss
from losses.LDAMLoss import LDAMLoss
from losses.MixUp import *
from losses.SEQLLoss import *
from losses.PriorCELoss import *
from losses.LADELoss import *
from losses.RWLoss import *
from losses.MocoV2 import *
from losses.LabelAwareSmoothing import *
from losses.RangeLoss import *
from losses.SADELoss import DiverseExpertLoss
from losses.VSLoss import VSLoss
from losses.GRWLoss import GRWCrossEntropyLoss
from losses.GCLLoss import GCLLoss
from losses.CenterLoss import *
def get_loss_functions(configs, cls_num_list):
    # Define a dictionary mapping methods to their corresponding loss function
    method_to_loss = {
        'ERM': lambda: nn.CrossEntropyLoss(),
        'RW': lambda: create_RWLoss_1(cls_num_list),
        'WeightedSoftmax': lambda: create_RWLoss_2(cls_num_list),
        'Focal': lambda: FocalLoss(),
        'BalancedSoftmax': lambda: create_balanced_softmax_loss(cls_num_list),
        'CBLoss': lambda: CBLoss(cls_num_list, configs.general.num_classes, 'softmax'),
        'CBLoss_Focal': lambda: CBLoss(cls_num_list, configs.general.num_classes, 'focal'),
        'LDAM': lambda: LDAMLoss(configs, cls_num_list),
        'SAM': lambda: LDAMLoss(configs, cls_num_list),
        'SEQLLoss': lambda: create_seql_loss(cls_num_list),
        'PriorCELoss': lambda: create_priorce_loss(configs, cls_num_list),
        'LADELoss': lambda: UnifiedLoss(configs, cls_num_list),
        'MixUp': lambda: nn.CrossEntropyLoss(),
        'MiSLAS': lambda: LabelAwareSmoothing(configs, cls_num_list),
        'RangeLoss': lambda: create_range_loss(configs, cls_num_list),
        'RSG': lambda: LDAMLoss(configs, configs.datasets.ori_cls_num_list),
        'SADE': lambda: DiverseExpertLoss(cls_num_list),
        'mocov2': lambda: MocoV2Loss,
        'DisAlign': lambda: GRWCrossEntropyLoss(num_samples_list=cls_num_list, num_classes=configs.general.num_classes),
        'VS': lambda: VSLoss(cls_num_list),
        'CenterLoss': lambda: get_center_loss(configs),  # Handle separately in helper
        'GCL': lambda: GCLLoss(cls_num_list=cls_num_list, m=0., s=30, noise_mul=0.5, weight=None)
    }

    # Fallback loss if method not found
    train_loss = method_to_loss.get(configs.general.method, lambda: nn.CrossEntropyLoss())()

    # Handle additional CenterLoss case
    def get_center_loss(configs):
        if configs.general.loss_type == 'Origin':
            center_loss = CenterLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        elif configs.general.loss_type == 'Cos':
            center_loss = CenterCosLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        elif configs.general.loss_type == 'Triplet':
            center_loss = CenterTripletLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        ce_loss = nn.CrossEntropyLoss()
        return [ce_loss, center_loss]

    # Validation loss (always CrossEntropy)
    val_loss = nn.CrossEntropyLoss()

    return {'train': train_loss, 'val': val_loss}


def calculate_loss(configs, outputs, labels, loss_func, status='val', **kwargs):
    if status != 'train':
        # Validation loss is straightforward
        return loss_func(outputs, labels)

    # Dictionary mapping methods to their specific loss calculation
    method_to_loss = {
        'MixUp': lambda: mixup_criterion(loss_func, outputs, labels[0], labels[1], labels[2]),
        'GCL': lambda: mixup_criterion(loss_func, outputs, labels[0], labels[1], labels[2]),
        'BBN': lambda: (
            configs.general.l * loss_func(outputs, labels[0])
            + (1 - configs.general.l) * loss_func(outputs, labels[1])
        ),
        'SADE': lambda: loss_func(outputs, labels, kwargs),
        'LDAM': lambda: loss_func(outputs, labels, kwargs['epoch']),
        'SAM': lambda: loss_func(outputs, labels, kwargs['epoch']),
        'RSG': lambda: loss_func(outputs, labels, kwargs['epoch']),
        'CenterLoss': lambda: calculate_center_loss(configs, outputs, labels, loss_func, kwargs)
    }

    # Default to standard loss calculation
    calculate = method_to_loss.get(configs.general.method, lambda: loss_func(outputs, labels))
    return calculate()


def calculate_center_loss(configs, outputs, labels, loss_func, kwargs):
    """
    Handles the special case of CenterLoss computation.
    """
    triplet = configs.general.loss_type == 'Triplet'
    epoch = kwargs['epoch']
    features, targets_a, targets_b, lam = labels
    ce_loss, center_loss = loss_func

    # Compute individual losses
    ce_loss_value = mixup_criterion(ce_loss, outputs, targets_a, targets_b, lam)
    center_loss_value = mixup_center_criterion(center_loss, features, outputs, targets_a, targets_b, lam, triplet)
    center_weight = get_center_weight(epoch, configs.general.train_epochs)

    return ce_loss_value + center_loss_value * center_weight
