# MONICA
MONICA: Benchmarking on Long-tailed Medical Image Classification

## Introduction
We build a unified codebase called Medical OpeN-source Long-taIled ClassifiCAtion (MONICA), which implements over 30 methods developed in long-tailed Learning and evaluated on
12 long-tailed medical datasets covering 6 medical domains.
![alt text](fig.png)

## Installation
First, clone the repo and cd into the directory:
```shell
git clone this repo.
cd MONICA
```
Then create a conda env and install the dependencies:
```shell
conda create -n MONICA
conda activate MONICA
conda env create -f MONICA.yml
```

## 1. Prepare Datasets

### Data Download
| Domain           | Dataset         | Link                                                                                   | License        |
|------------------|-----------------|----------------------------------------------------------------------------------------|----------------|
| Dermatology      | ISIC2019        | https://challenge.isic-archive.com/data/#2019                                          | CC-BY-NC       |
| Dermatology      | DermaMNIST      | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Ophthalmology    | ODIR            | https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k            | not specified  |
| Ophthalmology    | RFMiD           | https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid | CC BY-NC 4.0   |
| Radiology        | OragnA/C/SMNIST | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Radiology        | CheXpert        | https://stanfordmlgroup.github.io/competitions/chexpert/                               | [Stanford University Dataset Research Use Agreement](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) |
| Pathology        | PathMNIST       | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Pathology        | WILDS-Camelyon17 (In Progress)      | https://wilds.stanford.edu/datasets/                                                                  | CC0 1.0   |
| Hematology       | BloodMNIST      | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Histology        | TissueMNIST     | https://medmnist.com/                                                                  | CC BY-NC 4.0   |
| Gastroenterology | KVASIR          | https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset                             | ODbL 1.0       |

Please follow the same license as the original datasets.
### Image Preprocessing for Non-Image Datasets
If you download from the correct links, most of the evaluated datasets are conducted in image format.
We need to process the MedMNIST data for the unified training. Please find ./utils/process_medmnist.ipynb for reference.
```python
for idx in range(train_images.shape[0]):
  img = train_images[idx]
  label = train_labels[idx][0]
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  save_name = 'train_%s_%s.jpg' %(idx, label)
  cv2.imwrite('%s/%s' %(target_dir+dataset, save_name), img_rgb)
```
To match the pre-split set in `./numpy/medmnist`, all the images in MedMNIST will be stored in the format **<em>{split}\_{image_idx}\_{label}.jpg</em>**.


## 2. The Structure of Pre-Split Numpy Files

For the unified benchmark, we have pre-split the `train/val/test` sets, which are stored in `numpy files`.

There are four numpy files for each dataset/setting.

Take the ISIC datasets as an example, we have the three split files which start with the split `'train/val/test'` followed by the imbalance ratio `100/200/500`, e.g, `train_100.npy`.

We also have one dictionary file, e.g, `dic.npy`.

In `./dataset/dataloader.py`, you can find how we access the image_name and its label:

```python
self.np_dict = np.load(dict_path,allow_pickle=True).item()
self.np_path = np.load(np_path,allow_pickle=True)
self.img_name = []
self.img_label = []
for _ in self.np_path:
  self.img_name.append(_)
  self.img_label.append(self.np_dict[_])
```

### Customized Datasets
You can conduct your datasets in the following format:
```python
import numpy as np
import random
data_name = ['image_%s'%i for i in range(1000) ]
label = np.random.randint(0,10,1000)
dic = {data_name[i]:label[i] for i in range(1000)}
random.shuffle(data_name)
train = data_name[:700]
val = data_name[700:800]
test = data_name[800:1000]
np.save('train',train)
np.save('val',val)
np.save('test',test)
np.save('dic',dic)
```
And replace the `img_path` and `np_path` with your numpy files in the config files.

## 3. Start Training

### Config Files

Take the isic_GCL_2nd as the example (only keep some important hyperparameters).
```yaml
general: # Define some general parameters.
  img_size: 224
  seed: 1
  num_classes: 8
  dataset_name: 'isic'
  method: 'GCL_2nd'

model:
  if_resume: True # If this is set as True, the model will load for the resume_path.
  resume_path: './outputs/isic/100_GCL_224_resnet50_True_256_1_50/best.pt' # If if_resume is set as False, this will not work.
  if_freeze_encoder: True # If if_resume is set as False, this will not work.
  model_name: resnet50
  pretrained: True # If load the default pretrained weights provided by timm (from huggingface).


datasets:
  sampler: GCL # Sampler strategy.
  img_path: '/mnt/sda/datasets/isic2019/train/' # The image will be loaded as `img_path + name stored in np_path'. So if you can leave this blank if the full path is stored in 'np_path'.
  train:
    np_path: './numpy/isic/train_100.npy'
    dict_path: './numpy/isic/dic.npy' # Make sure this is consistency to the keys stored in the dic files.
  val:
    np_path: './numpy/isic/val_100.npy'
    dict_path: './numpy/isic/dic.npy'
  test:
    np_path: './numpy/isic/test_100.npy'
    dict_path: './numpy/isic/dic.npy'
  transforms:
    train: 'strong'
    val_test: 'crop'

```
### Support Methods
| Methods                | Paper                                                                                      | Link (TBD) | Offical Codes (TBD)  |
|------------------------|--------------------------------------------------------------------------------------------|------------|----------------------|
| ERM (Crossentropy)     | NA                                                                                         |            |                      |
| Re-sampling            | NA                                                                                         |            |                      |
| Re-weighting           | NA                                                                                         |            |                      |
| MixUp                  | mixup: Beyond empirical risk minimization                                                  |            |                      |
| Focal Loss             | Focal loss for dense object detection                                                      |            |                      |
| Classifier Re-training | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| T-Norm                 | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| LWS                    | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| KNN                    | Decoupling representation and classifier for long-tailed recognition                       |            |                      |
| CBLoss                 | Class-balanced loss based on effective number of samples                                   |            |                      |
| CBLoss_Focal           | Class-balanced loss based on effective number of samples                                   |            |                      |
| LADELoss               | Disentangling label distribution for long-tailed visual recognition                        |            |                      |
| LDAM                   | Learning imbalanced datasets with label-distribution-aware margin loss                     |            |                      |
| Logits Adjust Loss     | Long-tail learning via logit adjustment                                                    |            |                      |
| Logits Adjust Posthoc  | Long-tail learning via logit adjustment                                                    |            |                      |
| PriorCELoss            | Disentangling label distribution for long-tailed visual recognition                        |            |                      |
| RangeLoss              | Range Loss for Deep Face Recognition with Long-Tailed Training Data                        |            |                      |
| SEQLLoss               | Equalization loss for long-tailed object recognition                                       |            |                      |
| VSLoss                 | Label-imbalanced and group-sensitive classification under overparameterization             |            |                      |
| WeightedSoftmax        | Deep Long-Tailed Learning: A Survey                                                        |            |                      |
| BalancedSoftmax        | Balanced meta-softmax for long-tailed visual recognition                                   |            |                      |
| De-Confound            | Long-tailed classification by keeping the good and removing the bad momentum causal effect |            |                      |
| DisAlign               | Distribution alignment: A unified framework for long-tail visual recognition               |            |                      |
| GCL first stage        | Long-tailed visual recognition via gaussian clouded logit adjustment                       |            |                      |
| GCL second stage       | Long-tailed visual recognition via gaussian clouded logit adjustment                       |            |                      |
| MiSLAS                 | Improving calibration for long-tailed recognition                                          |            |                      |
| RSG                    | Rsg: A simple but effective module for learning imbalanced datasets                        |            |                      |
| SADE                   | Long-tailed recognition by routing diverse distribution-aware experts                      |            |                      |
| SAM                    | Sharpness-aware minimization for efficiently improving generalization                      |            |                      |
| BBN                    | Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition  |            |                      |

| Methods                | Paper                                                                                      | Link (TBD) | Offical Codes (TBD)  |
|------------------------|--------------------------------------------------------------------------------------------|------------|----------------------|
| BYOL                   | Bootstrap your own latent-a new approach to self-supervised learning                       |            |                      |
| MOCOv2                 | Improved baselines with momentum contrastive learning                                      |            |                      |
| MAE (RetFound)         | RETFound: a foundation model for generalizable disease detection from retinal image        |            |                      |
| CAEv2 (PanDerm)        | A General-Purpose Multimodal Foundation Model for Dermatology                              |            |                      |
| DINOv2 (TBD)           | Dinov2: Learning robust visual features without supervision                                |            |                      |

### Support Backbones
| Backbones        | Paper                                                                                |
|------------------|--------------------------------------------------------------------------------------|
| ResNet           | Deep residual learning for image recognition                                         |
| ViT              | An image is worth 16x16 words: Transformers for image recognition at scale           |
| Swin Transformer | Swin transformer: Hierarchical vision transformer using shifted windows              |
| ConvNext         | A convnet for the 2020s                                                              |
| RetFound         | RETFound: a foundation model for generalizable disease detection from retinal images |
| PanDerm          | A General-Purpose Multimodal Foundation Model for Dermatology                        |

### Script Training
For non-MedMNIST Training, please use command like:
```bash
python main.py --config ./configs/isic/100/isic_ERM.yml
```

For MedMNIST Training, please find the `./train.sh` script for reference.


