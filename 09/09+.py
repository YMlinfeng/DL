# Softmax回归——CNN分类器标准版 written by mzj

import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a linear regression model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='cnn_model.pth', help='Path to save model')
    return parser.parse_args()

class SyntheticDataset(Dataset):
    def __init__(self, w, b, n):
        self.features = torch.normal(0, 1, (n, 10))
        self.labels = (torch.matmul(self.features, w) + b).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features) # 一共2000条数据
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

transform = transforms.Compose([
    transforms.ToTensor(), # 将图像数据从PIL类型变换成32位浮点数格式，同时会将像素值从 [0, 255] 范围缩放到 [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,)) # 归一化，x减去均值（去中心化）再除以标准差（缩放），原本 [0.0, 1.0] 的像素值被转换为 [-1.0, 1.0] 的形式
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform) # 是 MNIST 的替代品，包含 10 类服装图像（如 T 恤、鞋子等），每张是 28x28 的灰度图像
test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 设置随机种子，提高复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 可视化部分：展示batch中的前16张图像（先保存后展示）
    X, y = next(iter(train_loader))
    X_show, y_show = X[:16], y[:16]
    fig_train, _ = show_images(X_show.reshape(16, 28, 28), 4, 4, titles=get_fashion_mnist_labels(y_show))
    train_img_path = os.path.join(os.getcwd(), "train_images.png")
    fig_train.savefig(train_img_path)
    print(f"Train images saved to {train_img_path}")
    plt.show()

    model = SoftmaxRegressionModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train() # 实例化一个 nn.Module（例如 model = MyModel()）后，模型默认处于训练模式（即 train 模式）
        running_loss = 0.0
        for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            preds = model(features_batch)
            loss = loss_fn(preds, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 使用之前计算出的梯度（param.grad），更新模型参数
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SoftmaxRegressionModel().to(device)
    print(model.load_state_dict(torch.load(args.save_path))) 
    model.eval()
    
    test_data, test_target = next(iter(test_loader)) # iter(test_loader) 将 DataLoader 转换成一个迭代器。next(...) 则取出这个迭代器的第一个元素。
    test_data, test_target = test_data.to(device), test_target.to(device)

    with torch.no_grad(): # 返回一个一维 tensor，长度为 batch_size，每个元素是对应样本预测的类别索引
        test_output = model(test_data)
        pred_labels = test_output.argmax(dim=1) # 输出张量通常形状为 [batch_size, num_classes]，所以 dim=1 是正确选择
    
    # 可视化测试结果：展示前 16 张图像的预测标签与真实标签
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(test_data[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"gd: {classes[test_target[i]]}\npred: {classes[pred_labels[i]]}", fontsize=8)
        plt.axis('off')
    plt.suptitle("inference results")
    plt.tight_layout()
    
    inference_img_path = os.path.join(os.getcwd(), "inference_results.png")
    plt.savefig(inference_img_path)
    print(f"Inference results saved to {inference_img_path}")
    plt.show()

    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
    inference(args)

