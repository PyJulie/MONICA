import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
from torch.utils.data import Dataset
import numpy as np
import time
import os
from PIL import Image
import tifffile
# import matplotlib.pyplot as plt



def default_loader(img_path, path):
    try:
        img = Image.open(img_path+path)
        return img.convert('RGB')
    except Exception as e:
        print(e)
        print("Cannot read image: {}".format(path))
        
def default_loader_jpeg(img_path, path):
    try:
        img = Image.open(img_path+path+'.jpeg')
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def tiff_loader(img_path, path):
    try:
        img = tifffile.imread(img_path+path)
        img = Image.fromarray(img)
        return img
    except:
        print("Cannot read image: {}".format(path))

class customData(Dataset):
    def __init__(self, img_path, np_path, dict_path, dataset = '', data_transforms=None, loader = 'others', twice = False):
        self.np_dict = np.load(dict_path,allow_pickle=True).item()
        self.np_path = np.load(np_path,allow_pickle=True)
        self.img_name = []
        self.img_label = []
        for _ in self.np_path:
            self.img_name.append(_)
            self.img_label.append(self.np_dict[_])
        self.data_transforms = data_transforms
        self.dataset = dataset
        if loader == 'jpeg':
            self.loader = default_loader_jpeg
        elif loader == 'tif':
            self.loader = tiff_loader
        else:
            self.loader = default_loader
        self.path = img_path
        self.twice = twice

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(self.path, img_name)

        if self.data_transforms is not None:
            try:
                if not self.twice:
                    img = self.data_transforms[self.dataset](img)
                else:
                    img1 = self.data_transforms[self.dataset](img)
                    img2 = self.data_transforms[self.dataset](img)
                    img = [img1, img2]
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label