import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import pickle
import random

from unet_model import UNET
from tqdm import tqdm

from train_unet import MRIDataset


def calculate_dice_score(preds, targets, smooth=1e-6):
    """
    preds: tensor of shape (B, 1, H, W) with values 0 or 1
    targets: tensor of shape (B, 1, H, W) with values 0 or 1
    """
    # Flatten the tensors to 1D vectors for easy calculation
    # view(-1) flattens the tensor regardless of batch size or dimensions
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    # Intersection is where both are 1 (1 * 1 = 1)
    intersection = (preds_flat * targets_flat).sum()
    
    # Dice Formula: 2 * Intersection / (Sum of elements in both)
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    return dice


if __name__=="__main__":
    unet_model=UNET(3,1)
    print(unet_model)
    unet_model.load_state_dict(torch.load('/home/kuka/Harsh Stuff/Unet-Mri/saves/unet_model_epoch_99.pth', map_location=torch.device('cpu')))

    device="cuda" if torch.cuda.is_available() else "cpu"
    unet_model.to(device)

    root_dir="/home/kuka/Harsh Stuff/MRI Dataset Kaggle/archive/kaggle_3m"
    dirs=os.listdir(root_dir)
    
    total_data_x=[]
    total_data_y=[]

    for i in dirs:
        if i[0]!='T':
            continue

        imgs=os.listdir(root_dir+"/"+i)
        for j in range(1,(len(imgs)//2)+1):
            img_x=cv2.imread(f"{root_dir}/{i}/{i}_{j}.tif")
            img_x=cv2.cvtColor(img_x,cv2.COLOR_BGR2RGB)
            img_y=cv2.imread(f"{root_dir}/{i}/{i}_{j}_mask.tif")
            img_y=cv2.cvtColor(img_y,cv2.COLOR_BGR2GRAY)

            total_data_x.append(img_x)
            total_data_y.append(img_y)


    filename = '/home/kuka/Harsh Stuff/Unet-Mri/test_indices.pkl'
    test_indx = []

    with open(filename, 'rb') as file:
        # 'rb' means read in binary mode
        test_indx = pickle.load(file)
    
    test_x=[total_data_x[i] for i in test_indx]
    test_y=[total_data_y[i] for i in test_indx]

    transform=transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset=MRIDataset(test_x,test_y,transform=transform)

    BATCH_SIZE=1

    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    total_outs=[]
    total_masks=[]
    unet_model.eval()
    dice_scores = []

    # No need to track gradients for validation
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            mask = mask.to(device) # Ensure mask is on GPU
            
            # 1. Forward Pass
            outputs = unet_model(img)
            
            # 2. Convert logits to binary mask (0 or 1)
            # Apply sigmoid to squash to [0,1], then round to nearest integer
            preds = torch.round(torch.sigmoid(outputs))

            save_image(preds.squeeze(),f"/home/kuka/Harsh Stuff/Unet-Mri/preds/pred_img{batch_idx}.png")
            save_image(mask.squeeze(),f"/home/kuka/Harsh Stuff/Unet-Mri/true_preds/true_img{batch_idx}.png")
            save_image(img.squeeze(),f"/home/kuka/Harsh Stuff/Unet-Mri/real_imgs/real_img{batch_idx}.png")
            
            # 3. Calculate Dice for this batch and move to CPU to save GPU memory
            score = calculate_dice_score(preds, mask)
            dice_scores.append(score.item())

    # 4. Average the scores
    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice Score on Test Set: {avg_dice:.4f}")

