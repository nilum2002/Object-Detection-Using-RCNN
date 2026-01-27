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
import RCNN_DataSet
import RCNN
from torch.utils.data import TensorDataset, DataLoader
from torch_snippets import Report


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
# show(im, bbs = [best_bbs, best_candidate], confs= [0,0.5], texts = ['Bbox', 'Best candidate Bbox'])


#####################################################################################################
#                             Process All images and Extract Data                                   #
#####################################################################################################                      

candidates = extract_candidates(im)
# show(im, bbs = [best_bbs, best_candidate], confs= [0,0.5], texts = ['Bbox', 'Best candidate Bbox'])

# initialize lists to store file paths, ground truth bounding boxes, classes, deltas, ROIs and IoUs
FPATHS , GTBBS, CLSS , DELTAS, ROIS, IOUS = [], [], [], [], [], []
N = 500 # number of images to process 
for ix, (im, bbs, lables, fpath) in enumerate(ds):
    # break after N images 
    if (ix==N):
        break
    H, W, _ = im.shape
    # extract candidate bounding boxes 
    candidates = extract_candidates(im)
    # convert to format 
    candidates = np.array([[x, y, x+w, y+h ] for x, y, w, h in candidates])
    # list for current image 
    ious, rois, clss, deltas = [], [], [], []
    # calculate IoU between each candiatae 
    ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
    
    # process the candidate for current image 
    for jx, candidate in enumerate(candidates):
        cx, cy, cX, cY = candidate
        # get the iou values for current candidates 
        candidate_ious = ious[jx]
        # find the index of the best IoU for the current candidate 
        best_iou_at = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_ious_at]
        
        best_bb = _x, _y, _X, _Y = bbs[best_ious_at]
        
        if best_iou > 0.3: clss.append(lables[best_ious_at])
        else: clss.append("Background")
        # delta 
        delta = np.array([_x-cx, _y-cy, _X-cY, _Y-cY])/np.array([W, H, W, H])
        # append delta to list 
        deltas.append(delta)
        rois.append(candidate/np.array([W, H,W,H]))
    FPATHS.append(fpath)
    IOUS.append(ious)
    ROIS.append(rois)
    CLSS.append(clss)
    DELTAS.append(deltas)
    GTBBS.append(bbs)


FPATHS = [f'{IMG_ROOT}/{Path(f).stem}.jpg' for f in FPATHS] 
FPATHS, GTBBS, CLSS, DELTAS, ROIS = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]
print(len(FPATHS))

targets = pd.DataFrame(list(chain.from_iterable(CLSS)), columns=['label'])
label2target = {l:t for t,l in enumerate(targets['label'].unique())}
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['Background']



print("The label to target values dictionary formed is:" ,label2target)


# normalizing with the mean, std used while training the model
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(img):
    """
    Convert to pytorch tensor 
    changes the channel order to (C, H, W) C - # of channels H - Height W - width 
    """
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()
def decode(_y):
    """
    the model output (usually logits or probabilities) 
    returns the predicted class index by finding the maximum value along the last dimension.
    """
    _, preds = _y.max(-1)
    return preds





n_train = 9*len(FPATHS)//10  
print(n_train)
train_ds = RCNN_DataSet.RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train], label2target)
test_ds = RCNN_DataSet.RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:], label2target)
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)

"""
The following strategy is adopted for R-CNN network architecture

    Define a VGG backbone.
    Fetch the features post passing the normalized crop through a pretrained model.
    Attach a linear layer with sigmoid activation to the VGG backbone to predict the class corresponding to the region proposal.
    Attach an additional linear layer to predict the four bounding box offsets.
    Define the loss calculations for each of the two outputs (one to predict class and the other to predict the four bounding box offsets).
    Train the model that predicts both the class of region proposal and the four bounding box offsets
"""
# For feature extraction 
vgg_backbone = models.vgg16(pretrained=True)
vgg_backbone.classifier = nn.Sequential()
for param in vgg_backbone.parameters():
    param.requires_grad = False #not to do a re-train
vgg_backbone.eval().to(device)


def train_batch(inputs, model, optimizer, criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)  # as model outputs we will be getting classes and delta (bbox offsets)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()


@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss,_deltas = model(input)
        loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
        _, _clss = _clss.max(-1)  # more like a softmax np argmax
        accs = clss == _clss
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()



rcnn = RCNN.RCNN(vgg_backbone, label2target).to(device)
criterion = rcnn.calc_loss
optimizer = optim.SGD(rcnn.parameters(), lr=1e-3)
n_epochs = 1
log = Report(n_epochs) #records the metrics as report, can be used to plot later


#####################################################################################################
#                             Training LOOP                                                         #
#####################################################################################################

# loc_loss: loss on classification
# regr_loss: loss on regression

for epoch in range(n_epochs):

    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, optimizer, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, trn_regr_loss=regr_loss, trn_acc=accs.mean(), end='\r')
        
    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        _clss, _deltas, loss, \
        loc_loss, regr_loss, accs = validate_batch(inputs, rcnn, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, val_regr_loss=regr_loss, val_acc=accs.mean(), end='\r')
        
log.plot_epochs('trn_loss,val_loss'.split(','))
log.plot_epochs('trn_acc,val_acc'.split(','))


# ...existing code...

def test_single_image(image_path, model, label2target, device):
    # Load and preprocess the image
    image = cv2.imread(image_path, 1)[..., ::-1]
    H, W, _ = image.shape
    candidates = extract_candidates(image)
    candidates_xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in candidates])

    crops = []
    valid_bboxes = []
    for (x1, y1, x2, y2) in candidates_xyxy:
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crop = cv2.resize(crop, (224, 224))
        crop = preprocess_image(crop / 255.)[None]
        crops.append(crop)
        valid_bboxes.append((x1, y1, x2, y2))
    if not crops:
        print("No valid crops found.")
        return

    inputs = torch.cat(crops).to(device)
    model.eval()
    with torch.no_grad():
        cls_scores, bbox_preds = model(inputs)
        preds = decode(cls_scores).cpu().numpy()

    # Map predictions to labels
    inv_label_map = {v: k for k, v in label2target.items()}
    pred_labels = [inv_label_map.get(p, "Unknown") for p in preds]

    # Print results
    for i, (bbox, label) in enumerate(zip(valid_bboxes, pred_labels)):
        print(f"ROI {i}: BBox={bbox}, Predicted Label={label}")

# Example usage:
test_image_path = r"D:\01. projects\ObjectDetectionRCNN\DataSet\images\images\0a00eb17a14585f7.jpg"
test_single_image(test_image_path, rcnn, label2target, device)