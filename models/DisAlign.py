import torch
from torch.functional import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _ntuple

class NormalizedLinear(torch.nn.Module):
    """
    A advanced Linear layer which supports weight normalization or cosine normalization.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        feat_norm=True,
        scale_mode='learn',
        scale_init=1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.scale_mode = scale_mode
        self.scale_init = scale_init

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if self.scale_mode == 'constant':
            self.scale = scale_init
        elif self.scale_mode == 'learn':
            self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
        else:
            raise NotImplementedError

class DisAlignLinear(torch.nn.Linear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        logit_before = F.linear(input, self.weight, self.bias)
        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
            confidence * self.logit_bias
        return logit_after


class DisAlignNormalizedLinear(NormalizedLinear):
    """
    A wrapper for nn.Linear with support of DisAlign method.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, **args) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **args)
        self.confidence_layer = torch.nn.Linear(in_features, 1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = torch.nn.Parameter(torch.zeros(1, out_features))
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input: Tensor):
        if self.feat_norm:
            input = torch.nn.functional.normalize(input, dim=1)

        output = input.mm(torch.nn.functional.normalize(self.weight, dim=1).t())
        logit_before = self.scale * output

        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (1 + confidence * self.logit_scale) * logit_before + \
            confidence * self.logit_bias
        return logit_after