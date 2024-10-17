from __future__ import absolute_import

import torch
from torch import nn


class RangeLoss(nn.Module):
    """
        Range_loss = alpha * intra_class_loss + beta * inter_class_loss
        intra_class_loss is the harmonic mean value of the top_k largest distances beturn intra_class_pairs
        inter_class_loss is the shortest distance between different class centers
    """
    def __init__(self, configs, cls_num_list, k=2, margin=0.1, alpha=0.5, beta=0.5):
        super(RangeLoss, self).__init__()
        self.use_gpu = configs.cuda.use_gpu
        self.cls_num_list = cls_num_list
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta


    def _pairwise_distance(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
         Return: 
            pairwise distance matrix with shape(batch_size, batch_size)
        """
        n = features.size(0)
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, features, features.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _compute_top_k(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
         Return: 
            top_k largest distances
        """
	
        dist_array = self._pairwise_distance(features)
        dist_array = dist_array.view(1, -1)
        top_k = dist_array.sort()[0][0, -self.k * 2::2]     # Because there are 2 same value of same feature pair in the dist_array

        return top_k

    def _compute_min_dist(self, center_features):
        """
         Args:
            center_features: center matrix (before softmax) with shape (center_number, center_dim)
         Return: 
            minimum center distance
        """
        n = center_features.size(0)
        dist_array2 = self._pairwise_distance(center_features)
        min_inter_class_dist2 = dist_array2.view(1, -1).sort()[0][0][n]  # exclude self compare, the first one is the min_inter_class_dist
        return min_inter_class_dist2

    def _calculate_centers(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            center_features: center matrix (before softmax) with shape (center_number, center_dim)
        """
        if self.use_gpu:
            unique_labels = targets.cpu().unique().cuda()
        else:
            unique_labels = targets.unique()
        center_features = torch.zeros(unique_labels.size(0), features.size(1))
        if self.use_gpu:
            center_features = center_features.cuda()

        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
        return center_features

    def _inter_class_loss(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            margin: inter class ringe loss margin
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            inter_class_loss
        """
        center_features = self._calculate_centers(features, targets)
        min_inter_class_center_distance = self._compute_min_dist(center_features)
        # print('min_inter_class_center_dist:', min_inter_class_center_distance)
        return torch.relu(self.margin - min_inter_class_center_distance)

    def _intra_class_loss(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
         Return: 
            intra_class_loss
        """
        if self.use_gpu:
            unique_labels = targets.cpu().unique().cuda()
        else:
            unique_labels = targets.unique()

        intra_distance = torch.zeros(unique_labels.size(0))
        if self.use_gpu:
            intra_distance = intra_distance.cuda()

        for i in range(unique_labels.size(0)):
            label = unique_labels[i]
            same_class_distances = 1.0 / self._compute_top_k(features[targets == label])
            intra_distance[i] = self.k / torch.sum(same_class_distances)
        # print('intra_distace:', intra_distance)
        return torch.sum(intra_distance)

    def _range_loss(self, features, targets):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             range_loss
        """
        inter_class_loss = self._inter_class_loss(features, targets)
        intra_class_loss = self._intra_class_loss(features, targets)
        range_loss = self.alpha * intra_class_loss + self.beta * inter_class_loss
        return inter_class_loss, intra_class_loss, range_loss

    def forward(self, features, targets):
        """
        Args:
            features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
            targets: ground truth labels with shape (batch_size)
            ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
            ids_per_batch: num of different ids per batch
            imgs_per_id: num of images per id
        Return:
             range_loss
        """
        assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
        if self.use_gpu:
            features = features.cuda()
            targets = targets.cuda()

        inter_class_loss, intra_class_loss, range_loss = self._range_loss(features, targets)
        #return inter_class_loss, intra_class_loss, range_loss
        return range_loss

 

def create_range_loss(configs, cls_num_list):
    return RangeLoss(configs, cls_num_list)