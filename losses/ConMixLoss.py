import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_negative_mask(batch_size, device):
    mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    for i in range(batch_size):
        mask[i, batch_size + i] = False
        mask[batch_size + i, i] = False
    return mask


def nt_xent(x, features2, t=0.5):
    """NT-Xent contrastive loss between two sets of L2-normalized features."""
    batch_size = x.shape[0]
    out_1 = F.normalize(x, dim=-1)
    out_2 = F.normalize(features2, dim=-1)
    out = torch.cat([out_1, out_2], dim=0)                          # (2B, D)
    sim = torch.mm(out, out.t()) / t                                # (2B, 2B)
    neg = torch.exp(sim)
    mask = _get_negative_mask(batch_size, x.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)          # (2B, 2B-2)
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)          # (B,)
    pos = torch.cat([pos, pos], dim=0)                              # (2B,)
    loss = -torch.log(pos / (pos + neg.sum(dim=-1)))
    return loss.mean()


class ConMixLoss(nn.Module):
    """
    监督版 ConMix：Contrastive Mixup at Representation Level（ICLR 2025）。

    原文使用随机伪标签对特征分组混合，此处改为使用真实类别标签，
    使同类样本的表示在对比学习中被聚合，有利于长尾场景下少数类的表示学习。

    用法：
        loss_fn = ConMixLoss(temperature=0.5)
        loss = loss_fn(feature1, feature2, labels)

    Args:
        temperature: NT-Xent 温度系数，默认 0.5
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature1, feature2, labels):
        """
        Args:
            feature1: 第一个视角的特征 (B, D)，无需提前归一化
            feature2: 第二个视角的特征 (B, D)，无需提前归一化
            labels:   真实类别标签 (B,)
        Returns:
            标量 loss
        """
        n_samples = labels.shape[0]
        num_classes = int(labels.max().item()) + 1

        # 按类别构建混合权重矩阵 (num_classes, B)
        weight = torch.zeros(num_classes, n_samples, device=feature1.device)
        weight[labels, torch.arange(n_samples, device=feature1.device)] = 1.0
        # 去掉当前 batch 中未出现的类（避免全零行）
        weight = weight[weight.sum(dim=1) != 0]
        # L1 归一化：同类样本取均值
        weight = F.normalize(weight, p=1, dim=1)

        # 混合后的类级特征 (K, D)，K ≤ num_classes
        mix_f1 = torch.mm(weight, feature1)
        mix_f2 = torch.mm(weight, feature2)

        return nt_xent(mix_f1, mix_f2, t=self.temperature)
