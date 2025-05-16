# cnn_ddp_torchrun.py

import argparse
import os
import torch
import torch.distributed as dist  # 分布式通信模块 (Distributed)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler  # 分布式采样器
from torchvision import transforms, datasets
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式模型包装器
import matplotlib.pyplot as plt

from model import SoftmaxRegressionModel  
from utilis.tools import show_images  

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed CNN training with torchrun')
    parser.add_argument('--lr', type=float, default=0.001)  
    parser.add_argument('--batch_size', type=int, default=128)  
    parser.add_argument('--epochs', type=int, default=50)  
    parser.add_argument('--save_path', type=str, default='cnn_model_ddp.pth')  # 模型保存路径
    return parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)) 
])

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 🧩 模块：分布式训练初始化，返回当前进程的 GPU 编号
def setup_distributed():
    dist.init_process_group(backend='nccl')  # 初始化默认进程组，使用 NCCL 后端（高性能 GPU 通信）
    local_rank = int(os.environ["LOCAL_RANK"])  # 获取当前进程的 GPU 编号（torchrun 自动设置）
    torch.cuda.set_device(local_rank)  # 设置该进程使用的 GPU
    return local_rank  

# 🧩 模块：销毁进程组
def cleanup_distributed():
    dist.destroy_process_group()  # 清除所有进程间的通信资源


def train(args):
    local_rank = setup_distributed()  # 设置当前进程 GPU、初始化通信组
    device = torch.device(f"cuda:{local_rank}")  # 告诉 PyTorch 当前进程使用哪个 GPU
    train_dataset = datasets.FashionMNIST(root='/mnt/bn/occupancy3d/workspace/mzj/DL/09/data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dataset)  # 每个进程将读取不同的数据（不重复）
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler) # 创建 DataLoader，使用分布式采样器

    model = SoftmaxRegressionModel().to(device)
    model = DDP(model, device_ids=[local_rank])  # 使用 DDP 包装模型，实现多进程之间的梯度同步


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 只在主进程（rank 0）上显示训练图像
    if dist.get_rank() == 0:  # rank 是当前进程编号，rank = 0 是主进程
        X, y = next(iter(train_loader))
        X_show, y_show = X[:16], y[:16]
        fig, _ = show_images(X_show.reshape(16, 28, 28), 4, 4, titles=get_fashion_mnist_labels(y_show))
        fig.savefig("train_images.png")
        print("Train images saved to train_images.png")

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)  # 每轮训练都要设置 epoch，确保不同 epoch 的数据划分不同
        running_loss = 0.0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

            preds = model(features_batch)  

            loss = criterion(preds, labels_batch)  
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()       

            running_loss += loss.item()

        # 仅主进程打印日志
        if dist.get_rank() == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # 仅主进程保存模型
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), args.save_path)  # model.module 是 DDP 包装的原始模型
        print(f"Model saved to {args.save_path}")

    cleanup_distributed()  # 释放通信资源
    print("Distributed training completed.")

# 🧩 模块：推理函数（非分布式）
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoftmaxRegressionModel().to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))  
    model.eval()

    test_dataset = datasets.FashionMNIST(root='/mnt/bn/occupancy3d/workspace/mzj/DL/09/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    test_data, test_target = next(iter(test_loader))
    test_data, test_target = test_data.to(device), test_target.to(device)

    with torch.no_grad():
        test_output = model(test_data)
        pred_labels = test_output.argmax(dim=1)

    # 可视化前 16 张图像的预测结果
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(test_data[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"True: {classes[test_target[i]]}\nPred: {classes[pred_labels[i]]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("inference_results.png")
    print("Inference results saved to inference_results.png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    train(args)          # 开始训练（torchrun 会自动在每张 GPU 启动一个进程）

    # 仅在主进程（rank 0）上执行推理
    if int(os.environ.get("RANK", 0)) == 0: # dictionary.get(key, default_value)
        inference(args)