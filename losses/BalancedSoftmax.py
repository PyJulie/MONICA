import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, sample_per_class):
        super(BalancedSoftmax, self).__init__()

        freq = torch.tensor(sample_per_class)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(input, label, self.sample_per_class, reduction)


def balanced_softmax_loss(logits, labels, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_balanced_softmax_loss(sample_per_class):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(sample_per_class)