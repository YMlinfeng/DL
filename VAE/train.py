# 简易VAE训练框架 written by mzj
import argparse
import os
import time
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_without_note import VAE 
from tqdm import tqdm
from torchvision.transforms import ToPILImage


def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE model')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='ckpt/vae_model.pth', help='Path to save model')
    parser.add_argument('--load_path', type=str, default='/mnt/bn/occupancy3d/workspace/mzj/DL/VAE/ckpt/ckpt_epoch_20.pth', help='Path to save model')
    # 通过 nargs=2 接受2个整数，并在 __init__ 中转换成 tuple
    parser.add_argument('--img_shape', type=int, nargs=2, default=[128, 128], # 例如：--img_shape 128 128
                        help='Image shape as two integers representing height and width')
    parser.add_argument('--kl_weight', type=float, default=0.00025, help='KL divergence weight')
    parser.add_argument('--ckpt_interval', type=int, default=1, help='Save a checkpoint every N epochs')
    return parser.parse_args()

class CelebADataset(Dataset):
    def __init__(self, root, img_shape, **kwargs) -> None:
        super().__init__()
        self.root = root
        self.img_shape = tuple(img_shape)  # 转换为 tuple
        self.filenames = sorted(os.listdir(root))
    
    def __len__(self):
        return len(self.filenames) # 一共?条数据
    
    def __getitem__(self, idx:int):
        path = os.path.join(self.root, self.filenames[idx])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)) #可注释
        ])
        return pipeline(img)

# transform = transforms.Compose([
#     transforms.ToTensor(), # 将图像数据从PIL类型变换成32位浮点数格式，同时会将像素值从 [0, 255] 范围缩放到 [0.0, 1.0]
#     transforms.Normalize((0.5,), (0.5,)) # 归一化，x减去均值（去中心化）再除以标准差（缩放），原本 [0.0, 1.0] 的像素值被转换为 [-1.0, 1.0] 的形式
# ])
# train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform) # 是 MNIST 的替代品，包含 10 类服装图像（如 T 恤、鞋子等），每张是 28x28 的灰度图像
# test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

def get_dataloader(args, root, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    dataset = CelebADataset(root=root, img_shape=args.img_shape)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=32)


def check_traindata(args):
    """
    检验数据集和预处理，将16张图片拼成一张大图保存，验证数据处理流程是否正确
    """
    # 数据检查时使用较小的 batch_size 保证能拼成 4x4 的网格
    dataloader = get_dataloader(args, root='/mnt/bn/occupancy3d/workspace/dataset/img_align_celeba', batch_size=16)
    img = next(iter(dataloader))
    N, C, H, W = img.shape
    assert N == 16, f"Expected batch size of 16 but got {N}"
    # 重新排列成 4x4 网格
    img = torch.permute(img, (1, 0, 2, 3))  # 从 [N, C, H, W] 转为 [C, N, H, W]
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    os.makedirs('work_dirs', exist_ok=True)
    img.save('work_dirs/tmpwithoutnorm.jpg')
    print("Data check completed. Sample image saved to work_dirs/tmpwithoutnorm.jpg.")


def loss_fn(y, y_hat, mean, logvar, kl_weight):
    """
    计算 VAE 的损失：重构损失（均方误差损失）加上 KL 散度项
    """
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), dim=1), dim=0)
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子，提高复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    train_loader = get_dataloader(args, root='/mnt/bn/occupancy3d/workspace/dataset/img_align_celeba')
    print(f"Train dataset size: {len(train_loader.dataset)}")

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    begin_time = time.time()
    total_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            preds, mean, logvar = model(x)
            loss = loss_fn(x, preds, mean, logvar, args.kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 若 loss.item() 为 batch 内平均损失，乘以当前 batch 大小以便后续计算全数据集的平均损失
            running_loss += loss.item() * x.size(0)
            progress_bar.set_postfix({
                "Batch": f"{batch_idx+1}/{total_batches}",
                "Loss": f"{loss.item():.6f}"
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        training_time = time.time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.6f}, Time: {minute}m {second}s")

        # 每隔 args.ckpt_interval 个epoch保存一次 checkpoint
        if (epoch + 1) % args.ckpt_interval == 0:
            ckpt_path = f'ckpt/ckpt_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")
    
    # # 保存模型
    # torch.save(model.state_dict(), args.save_path)
    # print(f"Model saved to {args.save_path}")

def visualize(test):
    N, C, H, W = test.shape
    assert N == 16, f"Expected batch size of 16 but got {N}"
    # 重新排列成 4x4 网格
    img = torch.permute(img, (1, 0, 2, 3))  # 从 [N, C, H, W] 转为 [C, N, H, W]
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    os.makedirs('output', exist_ok=True)
    img.save(f'output/test_output.jpg')

def generate1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()
    
    with torch.no_grad():
        test_output_sample = model.sample(device)

    test_output_sample = test_output_sample[0].cpu()
    img = ToPILImage()(test_output_sample)
    os.makedirs('output', exist_ok=True)
    img.save('output/test_output_sample.jpg')

    print("Generation1 done!")

def generate2(args):
    '''
    效果肯定不如1，因为不是从隐空间中采样的
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()
    
    # 使用 torch.randn 生成随机噪声测试
    test_data_rand = torch.randn((1, 3, 128, 128)).to(device)
    with torch.no_grad():
        test_output_rand, _, _ = model(test_data_rand)

    test_output_rand = test_output_rand[0].cpu()
    img = ToPILImage()(test_output_rand)
    img.save('output/test_output_rand.jpg')

    print("Generation2 done!")

def reconstruct(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()

    # 从测试集随机选择一张图片
    test_loader = get_dataloader(args, root='/mnt/bn/occupancy3d/workspace/dataset/img_align_celeba', batch_size=1)
    test_data = next(iter(test_loader))
    with torch.no_grad():
        test_output, _, _ = model(test_data.to(device))

    test_output_reconstructed = test_output[0].cpu()
    # 保存原始图片和重建图片
    img = ToPILImage()(test_output_reconstructed)
    img.save('output/test_output_reconstructed.jpg')
    print("Reconstruction done!")

if __name__ == "__main__":
    args = parse_args()
    # 若需要检查数据预处理流程，可取消下一行注释
    # check_traindata(args)
    # train(args)
    generate1(args)
    generate2(args)
    reconstruct(args)