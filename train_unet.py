import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import random
import pickle

from unet_model import UNET
from tqdm import tqdm


""" Class to load the MRI Dataset """
class MRIDataset(Dataset):
    def __init__(self,data,masks,transform=None):
        self.data=data
        self.masks=masks
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample=self.data[idx]
        mask=self.masks[idx]

        if self.transform:
            sample=self.transform(sample)
            mask=self.transform(mask)

        return sample, mask

if __name__=="__main__":
    root_dir="path/" # Path to root directory
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

    # Train test val split (80% -> 10% -> 10%)
    indx_lst=list(range(len(total_data_x)))
    random.shuffle(indx_lst)
    train_indx=indx_lst[0:int(0.8*len(indx_lst))]
    val_indx=indx_lst[int(0.8*len(indx_lst)):int(0.9*len(indx_lst))]
    test_indx=indx_lst[int(0.9*len(indx_lst)):len(indx_lst)]


    # Saving the test indices for testing script
    filename = 'test_indices.pkl'

    with open(filename, 'wb') as file:
        # 'wb' means write in binary mode
        pickle.dump(test_indx, file)

    train_x=[total_data_x[i] for i in train_indx]
    train_y=[total_data_y[i] for i in train_indx]

    test_x=[total_data_x[i] for i in test_indx]
    test_y=[total_data_y[i] for i in test_indx]

    val_x=[total_data_x[i] for i in val_indx]
    val_y=[total_data_y[i] for i in val_indx]

    # Transform to convert to tensor
    transform=transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset=MRIDataset(train_x,train_y,transform=transform)
    test_dataset=MRIDataset(test_x,test_y,transform=transform)
    val_dataset=MRIDataset(val_x,val_y,transform=transform)

    BATCH_SIZE=8

    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    unet_model=UNET(3,1)
    print(unet_model)

    device="cuda" if torch.cuda.is_available() else "cpu"
    unet_model.to(device)

    NUM_EPOCHS=100
    optimizer=optim.AdamW(unet_model.parameters(),lr=0.0001)
    criterion=nn.BCEWithLogitsLoss()

    for epoch in range(0,NUM_EPOCHS):
        unet_model.train()
        epoch_loss=0
        step=0
        
        # Train
        for batch_idx,(img,mask) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Training')):
            
            img=img.to(device)
            mask=mask.to(device)

            optimizer.zero_grad()

            outputs=unet_model(img)

            loss=criterion(outputs,mask)

            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

            step+=1

        print(f"Train Epoch: {epoch}, Loss: {epoch_loss/len(train_loader)}")

        # Validate
        unet_model.eval()
        val_loss=0

        for batch_idx,(img,mask) in enumerate(tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Validation')):
            img=img.to(device)
            mask=mask.to(device)

            outputs=unet_model(img)

            loss_v=criterion(outputs,mask)

            val_loss+=loss_v.item()

        print(f"Val Loss: {val_loss}")

        if((epoch+1)%10==0):
            torch.save(unet_model.state_dict(), f'saves/unet_model_epoch_{epoch+1}.pth')
