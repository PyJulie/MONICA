from timm.layers import ClassifierHead
import torch.nn as nn
def get_BBN_model(configs, model):
    if 'swin' in configs.model.model_name or 'convnext' in configs.model.model_name:
        model.head = ClassifierHead(
            model.num_features*2,
            configs.general.num_classes,
            input_fmt='NHWC')
        model.classifier = model.head.fc
    else:
        if configs.model.model_name.startswith('vit'):
            model.head = nn.Linear(model.num_features*2, configs.general.num_classes) 
            model.classifier = model.head
        else:
            model.fc = nn.Linear(model.num_features*2, configs.general.num_classes) 
            model.classifier = model.fc
    if configs.cuda.use_gpu:
        if configs.cuda.multi_gpu:
            model = nn.DataParallel(model)
        model = model.cuda()
    return model