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


class RCNN(nn.Module):
    def __init__(self, backbone, label2target):
        super().__init__()
        feature_dim = 25088
        self.backbone = backbone
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(
              nn.Linear(feature_dim, 512),
              nn.ReLU(),
              nn.Linear(512, 4),
              nn.Tanh(),
            )
        self.cel = nn.CrossEntropyLoss() # loss for classification
        self.sl1 = nn.L1Loss() # loss for regression
    def forward(self, input):
        feat = self.backbone(input)  # both classification and regression takes 'feat' as input
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
    def calc_loss(self, probs, _deltas, labels, deltas):
        # probs is basically the predicted class
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != 1) #removing the label 1, which is background
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            # every ix is detected as background
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss
        
