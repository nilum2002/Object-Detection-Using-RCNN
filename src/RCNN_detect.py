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



IMG_ROOT = r"D:\01. projects\ObjectDetectionRCNN\DataSet\images\images"
DF_RAW = pd.read_csv(r"D:\01. projects\ObjectDetectionRCNN\DataSet\df.csv")
print(DF_RAW.head())


class OpenImages(Dataset):
    
    def __init__(self, df, image_folder = IMG_ROOT):
        self.df = df 
        self.root = image_folder
        # Get unique image Id's 
        self.unique_images = df["ImageID"].unique()
    # return the num of unique images 
    def __len__(self): return len(self.unique_images)
    # Get an Image and its bounding boxes, classes and path by index
    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = rf"{self.root}\{image_id}.jpg"
        print(image_path)
        # Read image and convert from BGR to RGB 
        image = cv2.imread(image_path,1)[...,::-1]
        h, w, _ = image.shape 
        df = self.df.copy()
        # filter data from current image Id 
        df = df[df["ImageID"]==image_id]
        # Extract Bounding Boxes and scale them to image dimensions 
        boxes = df["XMin,YMin,XMax,YMax".split(",")].values
        boxes = (boxes*np.array([w,h,w,h])).astype(np.uint16).tolist()
        # extract the class labels 
        classes = df["LabelName"].values.tolist()
        return image, boxes, classes, image_path
    


# create the instance of the dataset 
ds = OpenImages(df=DF_RAW)

# Display an example image with bounding boxes 
im, bbs, clss, _ = ds[6] # select 7 unique images from the data set
show(im, bbs=bbs, texts=clss, sz=10)
print(bbs)



