import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    绘制图像列表，若传入的 Tensor 图像为归一化后的数据（例如 [-1,1]），则转换为 [0,1] 范围
    """
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 判断 img 是否为tensor，如果是则进行反归一化
        if torch.is_tensor(img):
            img_np = img.numpy()
            # 假设归一化参数是 (mean=0.5, std=0.5)，则反归一化公式如下：
            img_np = (img_np * 0.5) + 0.5  
            ax.imshow(img_np, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    return fig, axes