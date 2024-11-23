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
        if 'resnet' in configs.model.model_name:
            num_ftrs = model.fc.in_features
            model.classifier = model.fc
        else:
            num_ftrs = model.head.in_features
            model.classifier = model.head
        if configs.model.type == 'mocov2':
            num_ftrs = 2048
            model.classifier = nn.Linear(num_ftrs, configs.general.num_classes) 
    elif configs.general.method == 'RSG':
        configs.datasets.ori_cls_num_list = configs.datasets.cls_num_list.copy()
        cls_num_list = configs.datasets.cls_num_list
        print('cls num list:')
        print(cls_num_list)

        if 'resnet' not in configs.model.model_name:
            print('RSG does not support %s and will apply ResNet instead.' %configs.model.model_name)
        head_lists = []
        Inf = 0
        for i in range(configs.datasets.head):
            head_lists.append(cls_num_list.index(max(cls_num_list)))
            cls_num_list[cls_num_list.index(max(cls_num_list))]=Inf
        model = resnext50_32x4d(num_classes=configs.general.num_classes, head_lists=head_lists, phase_train=True, epoch_thresh=160)
        model.classifier = model.fc
    elif configs.general.method == 'SADE':
        if 'resnet' not in configs.model.model_name:
            print('SADE does not support %s and will apply ResNet instead.' %configs.model.model_name)
        model = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=configs.general.num_classes, 
                             reduce_dimension=True, layer3_output_dim=None, 
                             layer4_output_dim=None, use_norm=True, num_experts=3, returns_feat=True)
    elif configs.model.model_name.startswith("vit"):
        if configs.model.model_name == 'vit_base_patch16_224':
            pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
            pretrained_cfg['file'] = '/mnt/sdb/julie/datasets/checkpoints/vit_base_patch16_224/model.safetensors'
            model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
        elif configs.model.model_name == 'vit_swin_base_patch4_window7_224':
            _model_name = 'swin_base_patch4_window7_224'
            pretrained_cfg = timm.create_model(model_name=_model_name).default_cfg
            pretrained_cfg['file'] = '/mnt/sdb/julie/datasets/checkpoints/swin_base_patch4_window7_224/model.safetensors'
            model = timm.create_model(model_name=_model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
        model.classifier = model.head
        num_ftrs = model.head.in_features
    elif configs.model.model_name.startswith('resnet'):
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes)
        model.classifier = model.fc
        num_ftrs = model.fc.in_features
    elif configs.model.model_name.startswith('convnext'):
        pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
        pretrained_cfg['file'] = '/home/julie/.cache/torch/hub/checkpoints/convnext_base_22k_1k_224.pth'
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
        model.classifier = model.head
        num_ftrs = model.head.in_features
    else:
        raise ValueError("Unsupported model name!")

    if 'GCL' in configs.general.method:
        model.classifier.fc = NormedLinear(num_ftrs, configs.general.num_classes)

    if configs.if_freeze_encoder:
        print('freezing encoders...')
        if configs.model.model_name.startswith('resnet'):
            print('freezing ResNet encoders...')
            for name, param in model.named_parameters():
                if 'layer' in name: 
                    param.requires_grad = False
            if configs.general.method == 'DisAlign':
                model.classifier = DisAlignLinear(in_features=num_ftrs, out_features=configs.general.num_classes).cuda()
            elif 'GCL' in configs.general.method:
                model.classifier = NormedLinear(num_ftrs, configs.general.num_classes)
            else:
                model.classifier = nn.Linear(num_ftrs, configs.general.num_classes) 
        elif configs.model_name.startswith('vit'):
            print("Freezing ViT encoders...")
            for name, param in model.named_parameters():
                if 'patch_embed' in name or 'blocks' in name:
                    param.requires_grad = False
            if configs.general.method == 'DisAlign':
                model.classifier = DisAlignLinear(in_features=num_ftrs, out_features=configs.general.num_classes).cuda()
            elif 'GCL' in configs.general.method:
                model.classifier = NormedLinear(num_ftrs, configs.general.num_classes)
            else:
                model.classifier = nn.Linear(num_ftrs, configs.general.num_classes).cuda()

    if configs.general.method == 'BBN':
        if 'swin' in configs.model.model_name:
            from timm.layers import ClassifierHead
            model.head = ClassifierHead(
                model.num_features*2,
                configs.general.num_classes,
                input_fmt='NHWC',
        )
            model.classifier = model.head
        else:
            model.classifier = nn.Linear(2*num_ftrs, configs.general.num_classes) 
    if configs.general.method == 'LWS' or configs.general.method == 'MiSLAS':
        tnormclassifier = DotProduct_Classifier(configs, flatten=True)
        if 'swin' in configs.model.model_name or 'convnext' in configs.model.model_name:
            tnormclassifier.fc.weight = model.classifier.fc.weight
        else:
            tnormclassifier.fc.weight = model.classifier.weight
        model.classifier = tnormclassifier
    if configs.general.method == 'De-Confound':
        causalnormclassifier = Causal_Norm_Classifier(configs)
        model.classifier = causalnormclassifier

    if configs.model.model_name.startswith('resnet') and configs.general.method != 'SADE':
        model.fc = model.classifier
    elif configs.model.model_name.startswith('vit'):
        model.head = model.classifier
    
    
    if configs.cuda.use_gpu:
        if configs.cuda.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
    return model