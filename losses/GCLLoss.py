import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)  
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()

class GCLLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
           
                                         
    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
             
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))  
        
        #cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list   
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list         
        output = torch.where(index, cosine-self.m, cosine)                    
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight)     