import numpy as np # linear algebra 
import pandas as pd # Data processing 
import os # for os operations 
import torch 
import torchvision
from torchvision import datasets, models # datasets and models for torch vision 
from torchvision.transforms import functional as FT # functional transformers 
from torchvision import transforms as T # transformers 
from torch import nn, optim # nn modules and optimizers 
from torch.nn import functional as F # functional nn operations 
from torch.utils.data import DataLoader, sampler, random_split, Dataset # data loading utilities 
import copy # for deep copying objects 
import math # for mathematical functions 
from PIL import Image 
import cv2
import albumentations as A # data augmented lib
from pycocotools.coco import COCO # COCO dataset API
import matplotlib.pyplot as plt # for plotting 
from albumentations.pytorch import ToTensorV2 # convert images for pytorch tensors 
import sys # for system specification functions 
from torch_snippets import show
import selectivesearch 
from pathlib import Path

from itertools import chain

# Add normalization and device setup for image preprocessing
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()

class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs, label2target):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
        self.label2target = label2target
    def __len__(self): return len(self.fpaths)
    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[...,::-1]
        H, W, _ = image.shape
        sh = np.array([W,H,W,H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois)*sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y,x:X] for (x,y,X,Y) in bbs]  # bounding box image crops
        return image, crops, bbs, labels, deltas, gtbbs, fpath
    def collate_fn(self, batch):
        '''Performing actions on a batch of images'''
        input, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            crops = [preprocess_image(crop/255.)[None] for crop in crops]
            input.extend(crops)
            labels.extend([self.label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, labels, deltas