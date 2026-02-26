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
from losses.ConMixLoss import ConMixLoss
def get_loss_functions(configs, cls_num_list):
    losses = {}
    if configs.general.method == 'ERM':
        train_loss = nn.CrossEntropyLoss()
    elif configs.general.method == 'RW':
        train_loss = create_RWLoss_1(cls_num_list)
    elif configs.general.method == 'WeightedSoftmax':
        train_loss = create_RWLoss_2(cls_num_list)
    elif configs.general.method == 'Focal':
        train_loss = FocalLoss()
    elif configs.general.method == 'BalancedSoftmax':
        train_loss = create_balanced_softmax_loss(cls_num_list)
    elif configs.general.method == 'CBLoss':
        train_loss = CBLoss(cls_num_list, configs.general.num_classes, 'softmax')
    elif configs.general.method == 'CBLoss_Focal':
        train_loss = CBLoss(cls_num_list, configs.general.num_classes, 'focal')
    elif configs.general.method == 'LDAM':
        train_loss = LDAMLoss(configs, cls_num_list)
    elif configs.general.method == 'SAM':
        train_loss = LDAMLoss(configs, cls_num_list)
    elif configs.general.method == 'SEQLLoss':
        train_loss = create_seql_loss(cls_num_list)
    elif configs.general.method == 'PriorCELoss':
        train_loss = create_priorce_loss(configs, cls_num_list)
    elif configs.general.method == 'LADELoss':
        train_loss = UnifiedLoss(configs, cls_num_list)
    elif configs.general.method == 'MixUp':
        train_loss = nn.CrossEntropyLoss()
    elif configs.general.method == 'MiSLAS':
        train_loss = LabelAwareSmoothing(configs, cls_num_list)
    elif configs.general.method == 'RangeLoss':
        train_loss = create_range_loss(configs, cls_num_list)
    elif configs.general.method == 'RSG':
        train_loss = LDAMLoss(configs, configs.datasets.ori_cls_num_list)
    elif configs.general.method == 'SADE':
        train_loss = DiverseExpertLoss(cls_num_list)
    elif configs.general.method == 'mocov2':
        train_loss = MocoV2Loss
    elif configs.general.method == 'DisAlign':
        train_loss = GRWCrossEntropyLoss(num_samples_list=cls_num_list, num_classes=configs.general.num_classes)
    elif configs.general.method == 'VSLoss':
        train_loss = VSLoss(cls_num_list)
    elif configs.general.method == 'CenterLoss':
        if configs.general.loss_type == 'Origin':
            center_loss = CenterLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        elif configs.general.loss_type == 'Cos':
            center_loss = CenterCosLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        elif configs.general.loss_type == 'Triplet':
            center_loss = CenterTripletLoss(num_classes=configs.general.num_classes, feat_dim=configs.model.feat_dim)
        ce_loss = nn.CrossEntropyLoss()
        train_loss = [ce_loss, center_loss]
    elif 'GCL' in configs.general.method:
        print('GCL Loss.')
        train_loss = GCLLoss(cls_num_list=cls_num_list, m=0., s=30, noise_mul =0.5, weight=None)
    elif configs.general.method == 'ConMix':
        temperature = configs.general.temperature if hasattr(configs.general, 'temperature') else 0.5
        train_loss = ConMixLoss(temperature=temperature)
    else:
        train_loss = nn.CrossEntropyLoss()
    val_loss = nn.CrossEntropyLoss()
    losses['train'] = train_loss
    losses['val'] = val_loss
    return losses

def calculate_loss(configs, outputs, labels, loss_func, status='val', **kwargs):
    if status == 'train':
        if configs.general.method == 'MixUp' or 'GCL' in configs.general.method:
            targets_a, targets_b, lam = labels[0], labels[1], labels[2]
            loss = mixup_criterion(loss_func, outputs, targets_a, targets_b, lam)
        elif configs.general.method == 'BBN':
            targets_a, targets_b = labels[0], labels[1]
            loss = configs.general.l * loss_func(outputs, targets_a) + (1 - configs.general.l) * loss_func(outputs, targets_b)
        elif configs.general.method == 'SADE':
            extra_info = kwargs
            loss = loss_func(outputs, labels, extra_info)
        elif configs.general.method == 'LDAM' or configs.general.method == 'SAM' or configs.general.method == 'RSG':
            epoch = kwargs['epoch']
            loss = loss_func(outputs, labels, epoch)
        elif configs.general.method == 'CenterLoss':
            if configs.general.loss_type == 'Triplet':
                triplet = True
            else:
                triplet = False
            epoch = kwargs['epoch']
            features, targets_a, targets_b, lam = labels
            ce_loss, center_loss = loss_func
            ce_loss_value = mixup_criterion(ce_loss, outputs, targets_a, targets_b, lam)
            center_loss_value = mixup_center_criterion(center_loss, features, outputs, targets_a, targets_b, lam, triplet)
            center_weight = get_center_weight(epoch, configs.general.train_epochs)
            loss = ce_loss_value + center_loss_value*center_weight

        elif configs.general.method == 'ConMix':
            feature1, feature2, true_labels = labels
            loss = loss_func(feature1, feature2, true_labels)
        else:
            loss = loss_func(outputs, labels)
    else:
        loss = loss_func(outputs, labels)
    
        
    return loss