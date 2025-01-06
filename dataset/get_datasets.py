from .dataloader import customData
from torchvision import transforms
from sampler import get_sampler
import torch
from randaugment.randaugment import RandAugment
import numpy as np
def get_dataloaders(datasets, configs):
    # datasets = get_datasets(configs)
    dataloaders = {}
    sampler, shuffle = get_sampler.get_sampler(configs, datasets)
    seed = configs.general.seed
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=configs.datasets.batch_size, shuffle=shuffle, num_workers=configs.general.num_workers, sampler = sampler, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=configs.datasets.batch_size, shuffle=False, num_workers=configs.general.num_workers, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=configs.datasets.batch_size, shuffle=False, num_workers=configs.general.num_workers, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    if configs.general.method == 'BBN':
        configs.datasets.sampler = 'RS'
        sampler, shuffle = get_sampler.get_sampler(configs, datasets)
        dataloaders['RS_train'] =  torch.utils.data.DataLoader(datasets['train'], batch_size=configs.datasets.batch_size, shuffle=shuffle, num_workers=configs.general.num_workers, sampler = sampler, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    return dataloaders
        

def get_datasets(configs):
    data_transforms = get_transforms(configs)
    datasets = {}
    if configs.general.method == 'mocov2':
        twice = True
    else:
        twice = False
    if configs.general.dataset_name == 'nct':
        loader = 'tif'
    else:
        loader = 'others'
    datasets['train'] = customData(img_path=configs.datasets.img_path,
                                np_path = configs.datasets.train.np_path,
                                dict_path=configs.datasets.train.dict_path,
                                loader=loader,
                                data_transforms=data_transforms,
                                dataset='train',
                                twice = twice)
    datasets['val'] = customData(img_path=configs.datasets.img_path,
                                np_path = configs.datasets.val.np_path,
                                dict_path=configs.datasets.val.dict_path,
                                loader=loader,
                                data_transforms=data_transforms,
                                dataset='val')
    datasets['test'] = customData(img_path=configs.datasets.img_path,
                                np_path = configs.datasets.test.np_path,
                                dict_path=configs.datasets.test.dict_path,
                                loader=loader,
                                data_transforms=data_transforms,
                                dataset='test')
    return datasets
        


def get_transforms(configs):
    img_size = configs.general.img_size
    data_transforms = {}
    if configs.datasets.transforms.train == 'strong':
        data_transforms['train'] = transforms.Compose([
        transforms.Resize((img_size+40, img_size+40)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    elif configs.datasets.transforms.train == 'weak':
        data_transforms['train'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    elif configs.datasets.transforms.train == 'randaugment':
        data_transforms['train'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        
    if configs.datasets.transforms.val_test == 'crop':
        data_transforms['val'] = transforms.Compose([
        transforms.Resize((img_size+40, img_size+40)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        data_transforms['test'] = transforms.Compose([
        transforms.Resize((img_size+40, img_size+40)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    else:
        data_transforms['val'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        data_transforms['test'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        
    return data_transforms