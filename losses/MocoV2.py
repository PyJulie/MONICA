import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *


def MocoV2Loss(q,k,queue,t=0.05):
    # t: temperature
    N = q.shape[0] # batch_size
    C = q.shape[1] # channel
    # bmm: batch matrix multiplication
    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N,1),t))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C),torch.t(queue)),t)),dim=1)
    # denominator is sum over pos and neg
    denominator = pos + neg
    return torch.mean(-torch.log(torch.div(pos,denominator)))