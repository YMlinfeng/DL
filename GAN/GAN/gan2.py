# written by mzj
"""
分布式 GAN 训练代码（基于 torchrun）  
请使用如下命令启动训练，例如：
    torchrun --nproc_per_node=NUM_GPUS gan_ddp_torchrun.py [args]
"""

import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed GAN training with torchrun")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8192, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads for data loading")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    parser.add_argument("--data_root", type=str, default="/mnt/bn/occupancy3d/workspace/mzj/DL/data/mnist", help="root directory for MNIST dataset")
    parser.add_argument("--save_path_G", type=str, default="generator_ddp.pth", help="path to save the generator model")
    parser.add_argument("--save_path_D", type=str, default="discriminator_ddp.pth", help="path to save the discriminator model")
    parser.add_argument("--sample_dir", type=str, default="images", help="directory to save generated sample images")
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        """
        Generator 初始化  
        :param latent_dim: 噪声向量维度  
        :param img_shape: 图像尺寸，形如 (channels, height, width)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        """
        Discriminator 初始化  
        :param img_shape: 图像尺寸，形如 (channels, height, width)
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def train(args):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()  # 当前进程编号
    local_rank = rank % torch.cuda.device_count()  # 为每个进程分配 GPU
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Process initialized on device: {device}, PID: {os.getpid()}")

    # 构造数据集与 DataLoader（使用 DistributedSampler 确保每个进程读取不同数据）
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.n_cpu)

    # 定义图像尺寸，并构造 Generator 与 Discriminator 模型
    img_shape = (args.channels, args.img_size, args.img_size)
    generator = Generator(args.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # 使用 DDP 包装模型，实现多进程梯度同步
    generator = DDP(generator, device_ids=[local_rank])
    discriminator = DDP(discriminator, device_ids=[local_rank])

    # 定义损失函数及优化器
    adversarial_loss = nn.BCELoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    total_batches = len(dataloader)
    for epoch in range(args.n_epochs):
        generator.train()
        discriminator.train()
        sampler.set_epoch(epoch)  # 每个 epoch 改变采样顺序
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)

            # 构造真实与虚假标签
            valid = torch.ones((batch_size, 1), device=device, dtype=torch.float)
            fake = torch.zeros((batch_size, 1), device=device, dtype=torch.float)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # 随机噪声作为生成器输入
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)
            # 生成器损失：希望判别器将生成图像判定为真实（valid label）
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if rank == 0:
                print(f"[Epoch {epoch+1}/{args.n_epochs}] [Batch {i+1}/{total_batches}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

            batches_done = epoch * total_batches + i
            # 周期性保存生成的样例图片（只在主进程保存，防止重复写入）
            if batches_done % args.sample_interval == 0 and rank == 0:
                sample_path = os.path.join(args.sample_dir, f"{batches_done}.png")
                save_image(gen_imgs.data[:25], sample_path, nrow=5, normalize=True)
                print(f"Saved sample image to {sample_path}")

    # 保存模型（仅在主进程执行）
    if rank == 0:
        torch.save(generator.module.state_dict(), args.save_path_G)
        torch.save(discriminator.module.state_dict(), args.save_path_D)
        print(f"Generator model saved to {args.save_path_G}")
        print(f"Discriminator model saved to {args.save_path_D}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("Distributed training completed.")


def inference(args):
    """
    推理函数：使用保存的 Generator 模型从噪声生成一组图片，并展示（或保存）结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (args.channels, args.img_size, args.img_size)
    generator = Generator(args.latent_dim, img_shape).to(device)
    generator.load_state_dict(torch.load(args.save_path_G, map_location=device))
    generator.eval()

    # 使用固定随机噪声生成一批图像（例如 25 张）
    z = torch.randn(25, args.latent_dim, device=device)
    with torch.no_grad():
        gen_imgs = generator(z)

    sample_path = "inference_results.png"
    save_image(gen_imgs.data, sample_path, nrow=5, normalize=True)
    print(f"Inference results saved to {sample_path}")
    
    # 可选的：使用 matplotlib 显示其中一张图片
    img = gen_imgs[0].cpu().numpy().reshape(args.img_size, args.img_size)
    plt.imshow(img, cmap='gray')
    plt.title("Sample Generated Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
    # 仅在主进程（rank 0）上执行推理
    if int(os.environ.get("RANK", 0)) == 0:
        inference(args)