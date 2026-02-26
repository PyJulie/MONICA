import timm
import torch.nn as nn
from collections import OrderedDict
import copy
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