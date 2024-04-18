from copy import deepcopy
import torch
from torch import torch_version
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from data import *
import os
device = torch.device('mps')

##將原本訓練好的simclr model複製一份並將projection head 去除改成 linear ，利用模型訓練好的的參數去獲得其特徵圖
@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model)
    network.fc = nn.Identity()  # Removing  g(.) projection head
    network.eval()
    network.to(device)
    
    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=os.cpu_count(), shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)
    
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]
    
    return torch.utils.data.TensorDataset(feats, labels)
