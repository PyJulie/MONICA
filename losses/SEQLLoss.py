import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 
 
 

class SEQL(nn.Module):
    def __init__(self, c_num_list, gamma=0.9, lambda_n=0.00043):
        super(SEQL, self).__init__()
        self.gamma = gamma
        self.lambda_n = lambda_n 
        dist = c_num_list
        
        num = sum(dist)
        prob = [i/num for i in dist]
        prob = torch.FloatTensor(prob)     
        self.prob = prob
        class_weight = torch.zeros(len(dist)).cuda()
        for i in range(len(dist)):
            class_weight[i] = 1 if self.prob[i] > lambda_n else 0 
        self.class_weight=class_weight

    def replace_masked_values(self, tensor, mask, replace_with):
        assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
        one_minus_mask = 1 - mask
        values_to_add = replace_with * one_minus_mask
        return tensor * mask + values_to_add


    def forward(self, input, target):
        N, C = input.shape
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        over_prob = (torch.rand(input.shape).cuda() > self.gamma).float()
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        input = self.replace_masked_values(input, weights, -1e7)
        loss = F.cross_entropy(input, target)
        return loss
    
def create_seql_loss(c_num_list):
    print('Loading SEQL Loss.')
    return SEQL(c_num_list)