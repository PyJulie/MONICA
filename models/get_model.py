import timm
import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
import copy

from models.TNormClassifier import DotProduct_Classifier
from models.CausalNormClassifier import Causal_Norm_Classifier
from models.resnet import resnext50_32x4d
from models import Expert_ResNet
from models.DisAlign import *
from models.GCLLayers import NormedLinear
def get_SSL_model(configs):
    if configs.general.method == 'mocov2':
        q_encoder = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes)


        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(q_encoder.fc.in_features, 100)),
            ('added_relu1', nn.ReLU()),
            ('fc2', nn.Linear(100, 50)),
            ('added_relu2', nn.ReLU()),
            ('fc3', nn.Linear(50, 25))
        ]))

        # replace classifier 
        # and this classifier make representation have 25 dimention 
        q_encoder.fc = classifier

        # define encoder for key by coping q_encoder
        k_encoder = copy.deepcopy(q_encoder)

        # move encoders to device
        q_encoder = q_encoder.cuda()
        k_encoder = k_encoder.cuda()
        return [q_encoder, k_encoder]
    else:
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes)
        model = model.cuda()
        return model

def get_model(configs):
    if configs.model.if_resume:
        print('loading model from %s...' %configs.model.resume_path)
        model = torch.load(configs.model.resume_path)
        if configs.model.type == 'mocov2':
            num_ftrs = 2048
            model.fc = nn.Linear(num_ftrs, configs.general.num_classes) 
    elif configs.general.method == 'RSG':
        configs.datasets.ori_cls_num_list = configs.datasets.cls_num_list.copy()
        cls_num_list = configs.datasets.cls_num_list
        print('cls num list:')
        print(cls_num_list)

        if 'vit' in configs.model.model_name:
            print('RSG does not support VIT and will apply ResNet instead.')
        head_lists = []
        Inf = 0
        for i in range(configs.datasets.head):
            head_lists.append(cls_num_list.index(max(cls_num_list)))
            cls_num_list[cls_num_list.index(max(cls_num_list))]=Inf
        model = resnext50_32x4d(num_classes=configs.general.num_classes, head_lists=head_lists, phase_train=True, epoch_thresh=160)
    elif configs.general.method == 'SADE':
        if 'vit' in configs.model.model_name:
            print('SADE does not support VIT and will apply ResNet instead.')
        model = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=configs.general.num_classes, 
                             reduce_dimension=True, layer3_output_dim=None, 
                             layer4_output_dim=None, use_norm=True, num_experts=3, returns_feat=True)
    else:
        if configs.model.model_name == 'vit_base_patch16_224':
            pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
            pretrained_cfg['file'] = '/mnt/sda/julie/datasets/checkpoints/vit_base_patch16_224/model.safetensors'
            model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
        else:
            model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes)

        if 'GCL' in configs.general.method:
            num_ftrs = model.fc.in_features
            model.fc = NormedLinear(num_ftrs, configs.general.num_classes)

    if configs.if_freeze_encoder:
        print('freezing encoders...')
        if configs.model_name.startswith('resnet'):
            print('freezing ResNet encoders...')
            for name, param in model.named_parameters():
                if 'layer' in name: 
                    param.requires_grad = False
            if configs.general.method == 'DisAlign':
                num_ftrs = model.fc.in_features
                model.fc = DisAlignLinear(in_features=num_ftrs, out_features=configs.general.num_classes).cuda()
            elif 'GCL' in configs.general.method:
                num_ftrs = model.fc.in_features
                model.fc = NormedLinear(num_ftrs, configs.general.num_classes)
            else:
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, configs.general.num_classes) 
    if configs.general.method == 'BBN':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(2*num_ftrs, configs.general.num_classes) 
    if configs.general.method == 'LWS' or configs.general.method == 'MiSLAS':
        tnormclassifier = DotProduct_Classifier(configs)
        tnormclassifier.fc.weight = model.fc.weight
        model.fc = tnormclassifier
    if configs.general.method == 'De-Confound':
        causalnormclassifier = Causal_Norm_Classifier(configs)
        model.fc = causalnormclassifier
    
    
    if configs.cuda.use_gpu:
        if configs.cuda.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
    return model