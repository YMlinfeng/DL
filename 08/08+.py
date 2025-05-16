# 线性回归标准版 written by mzj
'''
TODO list:
1. 没有用cuda来训练


'''
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a linear regression model')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='linear_model.pth', help='Path to save model')
    return parser.parse_args()

class SyntheticDataset(Dataset):
    def __init__(self, w, b, n):
        self.features = torch.normal(0, 1, (n, 10))
        self.labels = (torch.matmul(self.features, w) + b).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features) # 一共2000条数据
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1), # 默认kaiming初始化，配合ReLU
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        # with torch.no_grad():
        #     self.net[0].weight.normal_(0, 1) # PyTorch 推荐直接对参数操作时加 no_grad() 上下文
        #     self.net[0].bias.fill_(0) #in-place操作
        with torch.no_grad():
            self.linear.weight.normal_(0, 1)
            self.linear.bias.fill_(0)

    def forward(self, x):
        return self.linear(x)
        # return self.net(x)
        
def train(args):
    true_w = torch.Tensor([1,2,3,4,5,6,7,8,9,10])
    true_b = torch.Tensor([1])
    dataset = SyntheticDataset(true_w, true_b, 2000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LinearRegressionModel(10)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    unlearned_w = model.linear.weight.data.clone().view(-1) 
    unlearned_b = model.linear.bias.data.item()
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (features_batch, labels_batch) in enumerate(dataloader):
            preds = model(features_batch)
            loss = loss_fn(preds, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 使用之前计算出的梯度（param.grad），更新模型参数
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")
        
    # 打印参数误差
    learned_w = model.linear.weight.data.view(-1) # (10, ) # view() 的前提是张量在内存中是 连续的，否则会报错。可以用 .contiguous() 来变成连续的张量（如果必要）：
    learned_b = model.linear.bias.data.item()
    print(f"unlearned weights: {unlearned_w}")
    print(f"unlearned bias: {unlearned_b}")
    print(f"Learned weights: {learned_w}") #(1,10)
    print(f"Learned bias: {learned_b}")
    print(f"Weight error: {true_w - learned_w}")
    print(f"Bias error: {true_b - learned_b}")

    # 保存模型
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

def inference(args):
    model = LinearRegressionModel(10)
    model.eval()
    with torch.no_grad():
        a = torch.Tensor([[10,9,8,7,6,5,4,3,2,1]])
        before = model(a)
        print(model.load_state_dict(torch.load(args.save_path)))
        after = model(a) # 正确的输出如下
    after_true = 0.0
    for i in range(1, 11):
        after_true += i * (11 - i)

    print(f"before, after, after_true: {before}, {after}, {after_true}", sep="\n")
    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
    inference(args)

