from collections import Counter
import torch
import tqdm
import numpy as np

import torch

# WARNING: 
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.


def adjust_rho(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    rho_steps = [0.05, 0.1, 0.5, 0.5]
    if epoch <= 5:
        rho = rho_steps[0]
    elif epoch > 75:
        rho = rho_steps[3]
    elif epoch > 60:
        rho = rho_steps[2]

    else:
        rho = rho_steps[1]
    for param_group in optimizer.param_groups:
        param_group['rho'] = rho


def get_cls_num_list(image_dataset):
    c = Counter(image_dataset.img_label)
    c = sorted(c.items(), key=lambda x: x[0])  # 按类别 id 升序
    cls_num_list = []
    for i in c:
        cls_num_list.append(i[1])
    return cls_num_list

def dotproduct_similarity(A, B):
    AB = torch.mm(A, B.t())

    return AB

def forward(weights, feat):
    feat = torch.Tensor(feat)
    logits = dotproduct_similarity(feat, weights)
    
    return logits

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

def get_knncentroids(dataloaders, model):

    print('===> Calculating KNN centroids.')
    torch.cuda.empty_cache()
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    feats_all, labels_all = [], []
    # Calculate initial centroids only on training data.
    with torch.set_grad_enabled(False):
        for data in tqdm.tqdm(dataloaders['train']):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # Calculate Features of each training data
            feature_x = model.forward_features(inputs)
            feature_x = model.forward_head(feature_x, pre_logits=True)
            feats_all.append(feature_x.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    
    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    featmean = feats.mean(axis=0)
    def get_centroids(feats_, labels_):
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[labels_==i], axis=0))
        return np.stack(centroids)
    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels)

    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels)
    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)
    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,   
            'cl2ncs': cl2n_centers}