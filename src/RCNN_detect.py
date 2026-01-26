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

# # Display an example image with bounding boxes 
# im, bbs, clss, _ = ds[6] # select 7 unique images from the data set
# show(im, bbs=bbs, texts=clss, sz=10)
# print(bbs)

# # Display an example image with bounding boxes 
# im, bbs, clss, _ = ds[15] # select 16 unique images from the data set
# show(im, bbs=bbs, texts=clss, sz=10)
# print(bbs)


# Function to extract the candidate bounding boxes using selective search
def extract_candidates(img):
    # perform the selective search algorithm search regions 
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2]) # the product of hight and width of the image 
    candidates = []
    for r in regions:
        # skip duplicate regions 
        if r['rect'] in candidates: continue
        # skip the areas that are too small 
        if r['size'] < (0.05*img_area): continue
        # skip the areas that are too large 
        if r['size'] > (1*img_area): continue
        x, y, w, h  = r['rect']
        # add the regions rectangles 
        candidates.append(list(r['rect']))
    return candidates

# select the intersection over union - IOU
def extract_iou(boxA, boxB, epsilon=1e-5):
    # determine the coordinates of the intersection rectangle 
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    # calculate the area of the intersection rectangle 
    width = x2-x1 # change in x 
    height = y2-y1 # change in y
    # if the intersection is empty, return 0
    if width<0 and height<0:
        return 0
    # area of the intersection 
    area_overlap = width*height
    # area of the both boundary boxes 
    area_a = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    area_b = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    # total area of the union
    area_comb = area_a+area_b-area_overlap
    # IOU 
    iou = area_overlap/(area_comb+epsilon)
    return iou

# Ex of df[15]
# im, bbs, clss, _ = ds[12]
# # candidates 
# candidates = extract_candidates(im)
# # convert to [x, y, w, h]
# candidates_xyxy = [[x, y, x+w, y+h] for x, y, w, h in candidates]
# print(np.shape(candidates_xyxy))
# print(type(candidates_xyxy))
# # show the images with the candidate bounding boxes 
# show(im, bbs=candidates_xyxy)

(im, bbs, clss, fpath) = ds[15]
H, W, _ = im.shape
# extract candidates bounding boxes 
candidates = extract_candidates(im)
# convert to format 
candidates = np.array([[x, y, x+w, y+h ] for x, y, w, h in candidates])

# initialize lists to store IoUs, ROIs, classes, deltas, and best IoUs
ious, rois, clss, deltas, best_ious = [], [], [], [], []
temp_best_bbs = []
# calculate IoU between each candidate and each ground truth boounding box 
ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T


# iterate through each candidate bounding box 

for jx, candidate in enumerate(candidates):
    cx, cy, cX, cY = candidate
    candidate_ious = ious[jx]
    best_ious_at = np.argmax(candidate_ious)
    best_iou = candidate_ious[best_ious_at]
    best_ious.append(best_iou)
    best_bb = _x, _y, _X, _Y = bbs[best_ious_at]
    temp_best_bbs.append(best_bb)
    if best_iou > 0.3: clss.append(clss[best_ious_at])
    else: clss.append("Background")
    # delta 
    delta = np.array([_x-cx, _y-cy, _X-cY, _Y-cY])/np.array([W, H, W, H])
    # append delta to list 
    deltas.append(delta)
    rois.append(candidate/np.array([W, H,W,H]))
    
    

# Find the index of the candidate with the overall best IoU
overall_best_iou_idx = np.argmax(best_ious)
print("Best IoU:", best_ious[overall_best_iou_idx])

# Get the candidate bounding box with the best IoU
best_candidate = candidates[overall_best_iou_idx]
# Get the ground truth bounding box corresponding to the overall best IoU
best_bbs = temp_best_bbs[overall_best_iou_idx]

candidates = extract_candidates(im)
show(im, bbs = [best_bbs, best_candidate], confs= [0,0.5], texts = ['Bbox', 'Best candidate Bbox'])