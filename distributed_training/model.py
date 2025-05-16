import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

class SoftmaxRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入是一个通道，28*28的大小，所以conv1的输入通道数是1
        # 用32个卷积核来卷，所以输出是32通道
        # padding和stride都是1，所以输出是28*28
        # 故conv1是从（1，28，28）到（32，28，28）特征图大小不变
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=1) # 在图像边缘补一圈零，使得卷积后图像尺寸不发生变化
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 变为（32，28，28）下文省略Batchsize
        x = self.pool(x) # 变为（32，14，14）
        x = F.relu(self.conv2(x)) # 变为（64，14，14）总参数量32*64*3*3+64 = 18496
        x = self.pool(x) # 变为（64，7，7）
        x = x.view(-1, 64 * 7 * 7) # 将形状为 [batch_size, 64, 7, 7] 的 tensor 展平成 [batch_size, 64*7*7=3136]
        x = self.fc1(x) # 变为（128）参数量64*7*7*128+128 = 401536 = 0.4M
        x = F.relu(x)
        x = self.fc2(x)
        return x