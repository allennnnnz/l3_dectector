from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from argumentation import *
import matplotlib.pyplot as plt

writer = SummaryWriter("logs")

class label_Data(Dataset):

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
        label = self.label_dir
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_list)

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
        label = -1
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_list)
    
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
contrast_L0_dataset = label_Data(valid_root_dir,L0_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L1_dataset = label_Data(valid_root_dir,L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L1_L2_dataset = label_Data(valid_root_dir,L1_L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L2_dataset = label_Data(valid_root_dir,L2_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L2_L3_dataset = label_Data(valid_root_dir,L2_L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L3_dataset = label_Data(valid_root_dir,L3_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L3_L4_dataset = label_Data(valid_root_dir,L3_L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L4_dataset = label_Data(valid_root_dir,L4_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L4_L5_dataset = label_Data(valid_root_dir,L4_L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_L5_dataset = label_Data(valid_root_dir,L5_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_T12_dataset = label_Data(valid_root_dir,T12_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))
contrast_T12_L1_dataset = label_Data(valid_root_dir,T12_L1_label_dir,transform=ContrastiveTransformations(contrast_transforms,n_views=2))

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

img_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

test_L0_dataset =  label_Data(test_root_dir,L0_label_dir,transform=img_transforms)
test_L1_dataset = label_Data(test_root_dir,L1_label_dir,transform=img_transforms)
test_L1_L2_dataset = label_Data(test_root_dir,L1_L2_label_dir,transform=img_transforms)
test_L2_dataset = label_Data(test_root_dir,L2_label_dir,transform=img_transforms)
test_L2_L3_dataset = label_Data(test_root_dir,L2_L3_label_dir,transform=img_transforms)
test_L3_dataset = label_Data(test_root_dir,L3_label_dir,transform=img_transforms)
test_L3_L4_dataset = label_Data(test_root_dir,L3_L4_label_dir,transform=img_transforms)
tset_L4_dataset = label_Data(test_root_dir,L4_label_dir,transform=img_transforms)
test_L4_L5_dataset = label_Data(test_root_dir,L4_L5_label_dir,transform=img_transforms)
test_L5_dataset = label_Data(test_root_dir,L5_label_dir,transform=img_transforms)
test_T12_dataset = label_Data(test_root_dir,T12_label_dir,transform=img_transforms)
test_T12_L1_dataset = label_Data(test_root_dir,T12_L1_label_dir,transform=img_transforms)

test_all_dataset = test_L0_dataset + test_L1_dataset + test_L1_L2_dataset + test_L2_dataset + test_L2_L3_dataset + test_L3_dataset + test_L3_L4_dataset + tset_L4_dataset + test_L4_L5_dataset + test_L5_dataset + test_T12_dataset + test_T12_L1_dataset


print(contrast_all_dataset[0])