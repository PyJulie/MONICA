import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterCosLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterCosLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def forward(self, x, labels):
        center = self.centers[labels]
        norm_c = self.l2_norm(center)
        norm_x = self.l2_norm(x)
        similarity = (norm_c * norm_x).sum(dim=-1)
        dist = 1.0 - similarity
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterTripletLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterTripletLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x, preds, labels):
        # use most likely categories as negative samples
        preds = preds.softmax(-1)
        batch_size = x.shape[0]
        idxs = torch.arange(batch_size).to(x.device)
        preds[idxs, labels] = -1
        adv_labels = preds.max(-1)[1]

        anchor = x                           # num_batch, num_dim
        positive = self.centers[labels]      # num_batch, num_dim
        negative = self.centers[adv_labels]  # num_batch, num_dim

        output = self.triplet_loss(anchor, positive, negative)
        return output
    
def mixup_center_criterion(loss_func, feat, pred, y_a, y_b, lam, triplet):
    if triplet:
        return lam * loss_func(feat, pred, y_a) + (1 - lam) * loss_func(feat, pred, y_b)
    else:
        return lam * loss_func(feat, y_a) + (1 - lam) * loss_func(feat, y_b)


def get_center_weight(epoch, total_epochs):
    center_weights = [0.0, 0.001, 0.005]
    center_milestones = [0, int(total_epochs*0.6), int(total_epochs*0.8)]
    center_weight = center_weights[0]
    for i, ms in enumerate(center_milestones):
        if epoch >= ms:
            center_weight = center_weights[i]
    return center_weight