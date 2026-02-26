import timm
import torch.nn as nn
import torch

from models.TNormClassifier import DotProduct_Classifier
from models.BBN import get_BBN_model
from models.CausalNormClassifier import Causal_Norm_Classifier
from models.resnet import get_RSG_model
from models.Expert_ResNet import get_SADE_model
from models.DisAlign import *
from models.GCLLayers import NormedLinear


def get_model(configs):
    if configs.model.model_name.startswith("vit"):
        pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
        pretrained_cfg['file'] = '/mnt/sda/julie/datasets/checkpoints/vit_base_patch16_224/model.safetensors'
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
    elif configs.model.model_name.startswith("swin"):
        pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
        pretrained_cfg['file'] = '/mnt/sda/julie/datasets/checkpoints/swin_base_patch4_window7_224/model.safetensors'
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
    elif configs.model.model_name.startswith('resnet'):
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes)
    elif configs.model.model_name.startswith('convnext'):
        pretrained_cfg = timm.create_model(model_name=configs.model.model_name).default_cfg
        pretrained_cfg['file'] = '/mnt/sda/julie/datasets/checkpoints/convnext_base_22k_1k_224.pth'
        model = timm.create_model(model_name=configs.model.model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
    else:
        raise ValueError("Unsupported model name!")
    
    if 'resnet' in configs.model.model_name:
        num_ftrs = model.fc.in_features
        model.classifier = model.fc
    elif 'vit' in configs.model.model_name:
        num_ftrs = model.head.in_features
        model.classifier = model.head
    elif 'convnext' in configs.model.model_name or 'swin' in configs.model.model_name:
        num_ftrs = model.head.in_features
        model.classifier = model.head.fc

    elif configs.model.model_name == 'mocov2':
        num_ftrs = model.fc.in_features
        model.classifier = nn.Linear(num_ftrs, configs.general.num_classes) 

    if configs.model.if_resume:
        print('loading model from %s...' %configs.model.resume_path)
        model = torch.load(configs.model.resume_path)

    if configs.model.if_freeze_encoder:
        print('freezing encoders...')
        if configs.model.model_name.startswith('resnet'):
            head = 'fc.'
        else:
            head = 'head'
        for name, param in model.named_parameters():
            if head not in name: 
                param.requires_grad = False


        if configs.general.method == 'DisAlign':
            model.classifier = DisAlignLinear(in_features=num_ftrs, out_features=configs.general.num_classes).cuda()
        elif 'GCL' in configs.general.method:
            model.classifier = NormedLinear(num_ftrs, configs.general.num_classes)
        elif configs.general.method == 'LWS' or configs.general.method == 'MiSLAS':
            print('DotProduct_Classifier')
            tnormclassifier = DotProduct_Classifier(configs, flatten=True)
            tnormclassifier.fc.weight = model.classifier.weight
            model.classifier = tnormclassifier
        else:
            model.classifier = nn.Linear(num_ftrs, configs.general.num_classes).cuda()
        
    if configs.general.method == 'RSG':
        model = get_RSG_model(configs)
        return model
    elif configs.general.method == 'SADE':
        model = get_SADE_model(configs)
        return model
    elif configs.general.method == 'BBN':
        model = get_BBN_model(configs, model)
        return model
    


    if 'GCL' in configs.general.method:
        model.classifier = NormedLinear(num_ftrs, configs.general.num_classes)
    elif configs.general.method == 'De-Confound':
        causalnormclassifier = Causal_Norm_Classifier(configs, num_ftrs)
        model.classifier = causalnormclassifier

    if configs.model.model_name.startswith('resnet'):
        model.fc = model.classifier
    elif configs.model.model_name.startswith('vit'):
        model.head = model.classifier
    elif 'convnext' in configs.model.model_name or 'swin' in configs.model.model_name:
        model.head.fc = model.classifier
    
    
    if configs.cuda.use_gpu:
        if configs.cuda.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()

    return model