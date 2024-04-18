from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from argumentation import *
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from argumentation import *
from data import *
import matplotlib.pyplot as plt
from simclr_model import *
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from classify_model import *
from feature_map import *
from pytorch_lightning.accelerators import MPSAccelerator

writer = SummaryWriter("logs")

class label_Data(Dataset):
    def __init__(self, train_root_dir, label_dir, transform, label):
        self.train_root_dir = train_root_dir
        self.label_dir = label_dir
        self.image_dir_path = os.path.join(self.train_root_dir, self.label_dir)
        self.image_list = os.listdir(self.image_dir_path)
        self.image_list.sort()
        self.transform = transform
        self.label = label

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.train_root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path).convert("RGB")  # Convert to RGB
        if self.transform:
            img = self.transform(img)
        return img , self.label

    def __len__(self):
        return len(self.image_list)

    
#只要是unlabel的資料就將label設為-1
class unlabel_Data(Dataset):

    def __init__(self, train_root_dir, label_dir, transform):
        self.train_root_dir = train_root_dir
        self.label_dir = label_dir
        self.image_dir_path = os.path.join(self.train_root_dir, self.label_dir)
        self.image_list = os.listdir(self.image_dir_path)
        self.image_list.sort()
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.train_root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path).convert("RGB")  # Convert to RGB
        if self.transform:
            img = self.transform(img)
        label = -1
        return img , label

    def __len__(self):
        return len(self.image_list)
  
#L0:0 L2:1 ,,, T12_L1:11
train_root_dir = 'data/train'
valid_root_dir = 'data/valid'
test_root_dir = 'data/test'
L0_label_dir = 'L0'
L1_label_dir = 'L1'
L1_L2_label_dir = 'L1_L2'
L2_label_dir = 'L2'
L2_L3_label_dir = 'L2_L3'
L3_label_dir = 'L3'
L3_L4_label_dir = 'L3_L4'
L4_label_dir = 'L4'
L4_L5_label_dir = 'L4_L5'
L5_label_dir = 'L5'
T12_label_dir = 'T12'
T12_L1_label_dir = 'T12_L1'
unlabel = -1

#train contrast
contrast_L0_dataset = label_Data(valid_root_dir,L0_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=0)
contrast_L1_dataset = label_Data(valid_root_dir,L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=1)
contrast_L1_L2_dataset = label_Data(valid_root_dir,L1_L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=2)
contrast_L2_dataset = label_Data(valid_root_dir,L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=3)
contrast_L2_L3_dataset = label_Data(valid_root_dir,L2_L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=4)
contrast_L3_dataset = label_Data(valid_root_dir,L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=5)
contrast_L3_L4_dataset = label_Data(valid_root_dir,L3_L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=6)
contrast_L4_dataset = label_Data(valid_root_dir,L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=7)
contrast_L4_L5_dataset = label_Data(valid_root_dir,L4_L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=8)
contrast_L5_dataset = label_Data(valid_root_dir,L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=9)
contrast_T12_dataset = label_Data(valid_root_dir,T12_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=10)
contrast_T12_L1_dataset = label_Data(valid_root_dir,T12_L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2),label=11)

contrast_other_dataset = contrast_L0_dataset + contrast_L1_dataset + contrast_L1_L2_dataset + contrast_L2_dataset + contrast_L2_L3_dataset + contrast_L3_L4_dataset + contrast_L4_dataset + contrast_L4_L5_dataset + contrast_L5_dataset + contrast_T12_dataset + contrast_T12_L1_dataset
contrast_all_dataset = contrast_L3_dataset + contrast_other_dataset

#unlabeled_data
unlabeled_L0_dataset = unlabel_Data(train_root_dir,L0_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L1_dataset = unlabel_Data(train_root_dir,L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L1_L2_dataset = unlabel_Data(train_root_dir,L1_L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L2_dataset = unlabel_Data(train_root_dir,L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L2_L3_dataset = unlabel_Data(train_root_dir,L2_L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L3_dataset = unlabel_Data(train_root_dir,L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L3_L4_dataset = unlabel_Data(train_root_dir,L3_L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L4_dataset = unlabel_Data(train_root_dir,L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L4_L5_dataset = unlabel_Data(train_root_dir,L4_L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_L5_dataset = unlabel_Data(train_root_dir,L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_T12_dataset = unlabel_Data(train_root_dir,T12_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
unlabeled_T12_L1_dataset = unlabel_Data(train_root_dir,T12_L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))

unlabeled_other_dataset = unlabeled_L0_dataset + unlabeled_L1_dataset + unlabeled_L1_L2_dataset + unlabeled_L2_dataset + unlabeled_L2_L3_dataset + unlabeled_L3_L4_dataset + unlabeled_L4_dataset + unlabeled_L4_L5_dataset + unlabeled_L5_dataset + unlabeled_T12_dataset + unlabeled_T12_L1_dataset
unlabeled_all_dataset = unlabeled_L3_dataset + unlabeled_other_dataset

#test data
img_transforms = transforms.Compose([transforms.ToTensor()])

test_L0_dataset =  label_Data(test_root_dir,L0_label_dir,transform=img_transforms,label=0)
test_L1_dataset = label_Data(test_root_dir,L1_label_dir,transform=img_transforms,label=1)
test_L1_L2_dataset = label_Data(test_root_dir,L1_L2_label_dir,transform=img_transforms,label=2)
test_L2_dataset = label_Data(test_root_dir,L2_label_dir,transform=img_transforms,label=3)
test_L2_L3_dataset = label_Data(test_root_dir,L2_L3_label_dir,transform=img_transforms,label=4)
test_L3_dataset = label_Data(test_root_dir,L3_label_dir,transform=img_transforms,label=5)
test_L3_L4_dataset = label_Data(test_root_dir,L3_L4_label_dir,transform=img_transforms,label=6)
tset_L4_dataset = label_Data(test_root_dir,L4_label_dir,transform=img_transforms,label=7)
test_L4_L5_dataset = label_Data(test_root_dir,L4_L5_label_dir,transform=img_transforms,label=8)
test_L5_dataset = label_Data(test_root_dir,L5_label_dir,transform=img_transforms,label=9)
test_T12_dataset = label_Data(test_root_dir,T12_label_dir,transform=img_transforms,label=10)
test_T12_L1_dataset = label_Data(test_root_dir,T12_L1_label_dir,transform=img_transforms,label=11)

test_all_dataset =  test_L0_dataset + test_L1_dataset + test_L1_L2_dataset + test_L2_dataset + test_L2_L3_dataset + test_L3_dataset + test_L3_L4_dataset + tset_L4_dataset + test_L4_L5_dataset + test_L5_dataset + test_T12_dataset + test_T12_L1_dataset

#train calssifier
train_L0_dataset =  label_Data(train_root_dir,L0_label_dir,transform=img_transforms,label=0)
train_L1_dataset = label_Data(train_root_dir,L1_label_dir,transform=img_transforms,label=1)
train_L1_L2_dataset = label_Data(train_root_dir,L1_L2_label_dir,transform=img_transforms,label=2)
train_L2_dataset = label_Data(train_root_dir,L2_label_dir,transform=img_transforms,label=3)
train_L2_L3_dataset = label_Data(train_root_dir,L2_L3_label_dir,transform=img_transforms,label=4)
train_L3_dataset = label_Data(train_root_dir,L3_label_dir,transform=img_transforms,label=5)
train_L3_L4_dataset = label_Data(train_root_dir,L3_L4_label_dir,transform=img_transforms,label=6)
train_L4_dataset = label_Data(train_root_dir,L4_label_dir,transform=img_transforms,label=7)
train_L4_L5_dataset = label_Data(train_root_dir,L4_L5_label_dir,transform=img_transforms,label=8)
train_L5_dataset = label_Data(train_root_dir,L5_label_dir,transform=img_transforms,label=9)
train_T12_dataset = label_Data(train_root_dir,T12_label_dir,transform=img_transforms,label=10)
train_T12_L1_dataset = label_Data(train_root_dir,T12_L1_label_dir,transform=img_transforms,label=11)

train_all_dataset = train_L0_dataset + train_L1_dataset + train_L1_L2_dataset + train_L2_dataset + train_L2_L3_dataset + train_L3_dataset + train_L3_L4_dataset + train_L4_dataset + train_L4_L5_dataset + train_L5_dataset + train_T12_dataset + train_T12_L1_dataset


