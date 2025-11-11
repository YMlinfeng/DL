
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):                   # 训练入口函数，接收包含超参的字典
    device = torch.device(modelConfig["device"])        # 选择 CPU / GPU

    # ----------------------------- 数据集部分 ----------------------------- #
    dataset = CIFAR10(                                   # 初始化 CIFAR-10 训练集
        root='./CIFAR10', train=True, download=True,     # 数据目录 & 自动下载
        transform=transforms.Compose([                   # 数据增强流水线
            transforms.RandomHorizontalFlip(),           # 随机水平翻转
            transforms.ToTensor(),                       # 转为张量并归一化到 [0,1]
            transforms.Normalize((0.5, 0.5, 0.5),        # 再标准化到 [-1,1]
                                 (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(                             # 构建批量数据加载器
        dataset,
        batch_size=modelConfig["batch_size"],            # 每批样本数
        shuffle=True,                                    # 打乱数据
        num_workers=4,                                   # 4 个进程并行加载
        drop_last=True,                                  # 最后不足一批的样本舍弃
        pin_memory=True)                                 # 加速 GPU 迁移

    # ----------------------------- 模型部分 ----------------------------- #
    net_model = UNet(                                    # 构造 U-Net 网络
        T=modelConfig["T"],                              # 扩散步数
        ch=modelConfig["channel"],                       # 初始通道数
        ch_mult=modelConfig["channel_mult"],             # 每层通道倍率
        attn=modelConfig["attn"],                        # 是否在特定分辨率加注意力
        num_res_blocks=modelConfig["num_res_blocks"],    # 每层残差块数量
        dropout=modelConfig["dropout"]                   # dropout 比例
    ).to(device)                                         # 移动到设备

    if modelConfig["training_load_weight"] is not None:  # 若指定预训练权重
        net_model.load_state_dict(                       # 加载参数
            torch.load(
                os.path.join(modelConfig["save_weight_dir"],
                             modelConfig["training_load_weight"]),
                map_location=device))

    optimizer = torch.optim.AdamW(                       # AdamW 优化器
        net_model.parameters(),
        lr=modelConfig["lr"],                            # 初始学习率
        weight_decay=1e-4)                               # 权重衰减

    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(   # 余弦退火
        optimizer=optimizer,
        T_max=modelConfig["epoch"],                      # 完整周期 = 总 epoch
        eta_min=0,                                       # 最低 LR
        last_epoch=-1)                                   # 从头开始

    warmUpScheduler = GradualWarmupScheduler(            # 先线性 warm-up
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],            # warm-up 结束时 LR 倍数
        warm_epoch=modelConfig["epoch"] // 10,           # warm-up 轮数
        after_scheduler=cosineScheduler)                 # 之后接余弦调度

    trainer = GaussianDiffusionTrainer(                  # 封装扩散损失计算
        net_model,
        modelConfig["beta_1"],                           # β₁
        modelConfig["beta_T"],                           # β_T
        modelConfig["T"]                                 # 步数
    ).to(device)

    # ----------------------------- 训练循环 ----------------------------- #
    for e in range(modelConfig["epoch"]):                # 遍历所有 epoch
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:  # 进度条
            for images, _ in tqdmDataLoader:             # 取出图像（标签忽略）
                optimizer.zero_grad()                    # 梯度清零
                x_0 = images.to(device)                  # 输入移动到设备
                loss = trainer(x_0).sum() / 1000.        # 计算扩散损失并缩放
                loss.backward()                          # 反向传播
                torch.nn.utils.clip_grad_norm_(          # 梯度裁剪防爆 #type:ignore
                    net_model.parameters(),
                    modelConfig["grad_clip"])
                optimizer.step()                         # 参数更新
                tqdmDataLoader.set_postfix(              # 更新进度条信息
                    ordered_dict={
                        "epoch": e,
                        "loss": loss.item(),
                        "img shape": tuple(x_0.shape),
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })
        warmUpScheduler.step()                           # 每个 epoch 调整 LR
        torch.save(                                      # 保存权重快照
            net_model.state_dict(),
            os.path.join(modelConfig["save_weight_dir"],
                         f'ckpt_{e}_.pt'))


def eval(modelConfig: Dict):                            # 推断 / 采样函数
    with torch.no_grad():                               # 关闭梯度以节省显存
        device = torch.device(modelConfig["device"])     # 设备选择
        model = UNet(                                   # 与训练同配置的 U-Net
            T=modelConfig["T"],
            ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            attn=modelConfig["attn"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=0.).to(device)                      # 推断时关掉 dropout

        ckpt = torch.load(                              # 加载指定权重
            os.path.join(modelConfig["save_weight_dir"],
                         modelConfig["test_load_weight"]),
            map_location=device)
        model.load_state_dict(ckpt)                     # 填充参数
        print("model load weight done.")                # 打印提示
        model.eval()                                    # 设置 eval 模式

        sampler = GaussianDiffusionSampler(             # 推断阶段的采样器
            model,
            modelConfig["beta_1"],
            modelConfig["beta_T"],
            modelConfig["T"]
        ).to(device)

        # ----------------------------- 生成采样 ----------------------------- #
        noisyImage = torch.randn(                       # 从标准正态采噪声
            size=[modelConfig["batch_size"], 3, 32, 32],
            device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)  # 转到 [0,1] 可视域
        save_image(                                     # 保存噪声可视图
            saveNoisy,
            os.path.join(modelConfig["sampled_dir"],
                         modelConfig["sampledNoisyImgName"]),
            nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)               # 通过扩散逆过程采样
        sampledImgs = sampledImgs * 0.5 + 0.5           # 反归一化到 [0,1]
        save_image(                                     # 保存最终生成图像
            sampledImgs,
            os.path.join(modelConfig["sampled_dir"],
                         modelConfig["sampledImgName"]),
            nrow=modelConfig["nrow"])