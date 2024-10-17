import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
class LabelAwareSmoothing(nn.Module):
    def __init__(self, configs, cls_num_list, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        self.smooth_head = 0.3
        self.smooth_tail = 0.0

        if shape == 'concave':
            self.smooth = self.smooth_tail + (self.smooth_head - self.smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = self.smooth_tail + (self.smooth_head - self.smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = self.smooth_head + (self.smooth_head - self.smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = self.smooth_tail + (self.smooth_head - self.smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if configs.cuda.use_gpu:
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()