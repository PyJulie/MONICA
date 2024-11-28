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
    def load_pretrained_model(configs):
        print(f'Loading model from {configs.model.resume_path}...')
        model = torch.load(configs.model.resume_path)
        num_ftrs = model.fc.in_features if 'resnet' in configs.model.model_name else model.head.in_features
        model.classifier = model.fc if 'resnet' in configs.model.model_name else model.head
        if configs.model.type == 'mocov2':
            num_ftrs = 2048
            model.classifier = nn.Linear(num_ftrs, configs.general.num_classes)
        return model, num_ftrs

    def create_vit_model(configs):
        vit_config = {
            'vit_base_patch16_224': '/mnt/sdb/julie/datasets/checkpoints/vit_base_patch16_224/model.safetensors',
            'vit_swin_base_patch4_window7_224': '/mnt/sdb/julie/datasets/checkpoints/swin_base_patch4_window7_224/model.safetensors',
        }
        model_name = configs.model.model_name if configs.model.model_name in vit_config else 'swin_base_patch4_window7_224'
        pretrained_cfg = timm.create_model(model_name=model_name).default_cfg
        pretrained_cfg['file'] = vit_config.get(model_name)
        model = timm.create_model(model_name=model_name, pretrained=configs.model.pretrained, num_classes=configs.general.num_classes, pretrained_cfg=pretrained_cfg)
        num_ftrs = model.head.in_features
        model.classifier = model.head
        return model, num_ftrs

    def create_timm_model(model_name, pretrained, num_classes):
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        num_ftrs = model.fc.in_features if model_name.startswith('resnet') else model.head.in_features
        model.classifier = model.fc if model_name.startswith('resnet') else model.head
        return model, num_ftrs

    def freeze_encoder(model, model_name, num_ftrs, method):
        print('Freezing encoder layers...')
        for name, param in model.named_parameters():
            if ('layer' in name and model_name.startswith('resnet')) or ('patch_embed' in name or 'blocks' in name and model_name.startswith('vit')):
                param.requires_grad = False

        if method == 'DisAlign':
            model.classifier = DisAlignLinear(in_features=num_ftrs, out_features=configs.general.num_classes).cuda()
        elif 'GCL' in method:
            model.classifier = NormedLinear(num_ftrs, configs.general.num_classes)
        else:
            model.classifier = nn.Linear(num_ftrs, configs.general.num_classes).cuda()
        return model

    if configs.model.if_resume:
        model, num_ftrs = load_pretrained_model(configs)
    elif configs.general.method == 'RSG':
        print('Using RSG method...')
        configs.datasets.ori_cls_num_list = configs.datasets.cls_num_list.copy()
        cls_num_list = configs.datasets.cls_num_list
        head_lists = [cls_num_list.index(max(cls_num_list)) for _ in range(configs.datasets.head)]
        for head in head_lists:
            cls_num_list[head] = float('-inf')
        model = resnext50_32x4d(num_classes=configs.general.num_classes, head_lists=head_lists, phase_train=True, epoch_thresh=160)
        model.classifier = model.fc
    elif configs.general.method == 'SADE':
        print('Using SADE method...')
        model = Expert_ResNet.ResNet(Expert_ResNet.Bottleneck, [3, 4, 6, 3], dropout=None, num_classes=configs.general.num_classes,
                                     reduce_dimension=True, layer3_output_dim=None, layer4_output_dim=None, use_norm=True,
                                     num_experts=3, returns_feat=True)
    elif configs.model.model_name.startswith("vit"):
        model, num_ftrs = create_vit_model(configs)
    elif configs.model.model_name.startswith(('resnet', 'convnext')):
        model, num_ftrs = create_timm_model(configs.model.model_name, configs.model.pretrained, configs.general.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {configs.model.model_name}")

    if configs.if_freeze_encoder:
        model = freeze_encoder(model, configs.model.model_name, num_ftrs, configs.general.method)

    if configs.general.method in ['BBN', 'LWS', 'MiSLAS']:
        model.classifier = DotProduct_Classifier(configs, flatten=True)
    elif configs.general.method == 'De-Confound':
        model.classifier = Causal_Norm_Classifier(configs)

    if configs.cuda.use_gpu:
        model = nn.DataParallel(model) if configs.cuda.multi_gpu else model
        model = model.cuda()

    return model
