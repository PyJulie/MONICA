import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def get_sampler(configs, datasets, **extra_info):
    if configs.datasets.sampler == 'RS':
        target = datasets['train'].img_label
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle = False
    elif configs.datasets.sampler == 'RSG':
       sampler = RSGDatasetSampler(dataset=datasets['train'])
       shuffle = False
    elif configs.datasets.sampler == 'GCL':
       sampler = BalancedDatasetSampler(dataset=datasets['train'])
       shuffle = False
    elif configs.datasets.sampler == 'FlexRS':
       if extra_info:
        print('use valid acc.')
        valid_accuracy = extra_info['valid_accuracy']
       else:
          print('No valid acc.')
          valid_accuracy = None
       sampler = DifficultyDatasetSampler(dataset=datasets['train'], valid_accuracy=valid_accuracy)
       shuffle = False
    else:
        sampler = None
        shuffle = True
    return sampler, shuffle

class RSGDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, label_count=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        if label_count == None:
         label_to_count = [0] * len(np.unique(dataset.img_label))
         for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        else:
         label_to_count = label_count
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        if label_count == None:
         weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        else:

         weights = [per_cls_weights[self._get_label_inaturalist(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label_inaturalist(self, dataset, idx):
        return dataset[idx][1]
    
    def _get_label(self, dataset, idx):
        return dataset.img_label[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.img_label))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        
        per_cls_weights = 1 / np.array(label_to_count)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        self.per_cls_weights = per_cls_weights
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.img_label[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
    
class DifficultyDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, valid_accuracy=None, indices=None, num_samples=None, pow=2):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 

        per_cls_weights = 1 / np.array(valid_accuracy)
        per_cls_weights = per_cls_weights / per_cls_weights.sum()
        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                for idx in self.indices]

        self.per_cls_weights = per_cls_weights
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.img_label[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples