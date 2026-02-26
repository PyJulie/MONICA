import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class DotProduct_Classifier(nn.Module):
    
    def __init__(self, configs, feat_dim=2048, flatten=False, *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.num_classes = configs.general.num_classes
        self.fc = nn.Linear(feat_dim, self.num_classes)
        self.scales = Parameter(torch.ones(self.num_classes))
        self.flatten = flatten
        if configs.general.method == 'LWS':
            for param_name, param in self.fc.named_parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.fc(x)
        x *= self.scales
        return nn.Flatten(1)(x) if self.flatten == True else x