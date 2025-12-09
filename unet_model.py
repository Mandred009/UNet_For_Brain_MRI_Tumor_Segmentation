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

class DoubleConv(nn.Module):
    def __init__(self,inp_channels,out_channels):
        super().__init__()

        self.doubleconv=nn.Sequential(
            nn.Conv2d(inp_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.doubleconv(x)
    

class UpConv(nn.Module):
    def __init__(self,inp_channels,out_channels):
        super().__init__()

        self.up_s=nn.ConvTranspose2d(inp_channels,inp_channels//2,kernel_size=2,stride=2)
        self.double_conv=DoubleConv(inp_channels,out_channels)

    def forward(self,x1,x2):
        x1=self.up_s(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x=torch.cat([x2,x1],dim=1)
        return self.double_conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.inp = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(512, 1024)
        )
        
        # Decoder (Upsampling path)
        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder - save outputs for skip connections
        x1 = self.inp(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels (bottleneck)
        
        # Decoder - use skip connections
        x = self.up1(x5, x4)  # Combine 1024 -> 512 with skip from x4
        x = self.up2(x, x3)   # Combine 512 -> 256 with skip from x3
        x = self.up3(x, x2)   # Combine 256 -> 128 with skip from x2
        x = self.up4(x, x1)   # Combine 128 -> 64 with skip from x1
        
        # Output
        logits = self.out_conv(x)

        return logits