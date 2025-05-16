# 线性回归的从零开始实现
# :label:`sec_linear_scratch`
#
# 本节旨在通过代码实现线性回归的关键流程：
#  1. 数据流水线（生成合成数据集、迭代小批量数据）
#  2. 定义模型（线性回归模型）
#  3. 定义损失函数（均方损失）
#  4. 定义小批量随机梯度下降优化器（SGD）
#  5. 训练模型，并输出训练过程中的损失及参数估计误差
#
# 尽管现代深度学习框架可以自动化这些工作，但手写实现有助于理解底层原理。

# ----------------------
# 配置环境并导入包
# ----------------------

import random            # 用于生成随机数及打乱样本顺序
import torch             # PyTorch张量及自动求导
from d2l import torch as d2l  # d2l工具包（简化数据处理、画图等操作）

# ----------------------
# 生成合成数据集
# ----------------------
# 为了简单起见，这里构造一个带有噪声的线性模型数据集。
# 假设真实模型参数为：w = [2, -3.4], b = 4.2。我们生成数据集满足：
#    y = Xw + b + noise
# 其中 X 的每一行为一个二维样本（从标准正态分布采样），
# noise 为均值为 0，标准差为 0.01 的噪声。

def synthetic_data(w, b, num_examples):  #@save
    """
    生成合成数据集
    输入:
        w: 权重张量，形状(d,)
        b: 偏置（标量）
        num_examples: 样本数
    输出:
        X: 特征矩阵，形状 (num_examples, d)
        y: 标签向量，形状 (num_examples, 1)存疑
    """
    # 从标准正态分布中采样特征，其形状为 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算真实的 y = Xw + b
    # 在 PyTorch 中，torch.matmul 对于两个张量相乘时有一些特殊规则：如果其中一个输入是二维（即矩阵），而另一个输入是1维（即向量），则将这个1维张量视为向量；乘积操作会把矩阵的每一行与向量做点积，结果得到一个1维张量。 因此，当 X 的形状为 [4, 2]，而 true_w 是 [2] 时，程序执行的就是矩阵-向量乘法，每一行与该向量做内积，结果的形状为 [4]
    y = torch.matmul(X, w) + b
    # 加入噪声，噪声符合均值0、标准差0.01的正态分布
    y += torch.normal(0, 0.005, y.shape)
    return X, y.reshape((-1, 1))

# 定义真实参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成包含1000个样本的合成数据集
features, labels = synthetic_data(true_w, true_b, 1000)

# 打印第一条数据以查看
print('features:', features[0])
print('label:', labels[0])

# ----------------------
# 可视化数据集
# ----------------------
# 通过散点图查看特征与标签之间的线性关系，
# 这里仅选择第二个特征(features[:, 1])与标签进行可视化
import matplotlib.pyplot as plt

# 使用matplotlib绘制散点图
plt.figure(figsize=(6, 4))  # 设置图像大小
plt.scatter(features[:, 1].detach().numpy(),  # 第二个特征
            labels.detach().numpy(),         # 标签
            s=1, c='blue')                   # 点大小为1，颜色蓝色
plt.xlabel('Feature 2')
plt.ylabel('Label')
plt.title('Feature 2 vs. Label')  # 添加标题
plt.grid(True)                    # 添加网格线

# 保存图像到本地
plt.savefig('feature2_vs_label.png', dpi=300)  # 保存为PNG文件，分辨率300dpi

# 显示图像
print('显示图像')
plt.show()

# ----------------------
# 定义数据迭代器
# ----------------------
# 在训练过程中，我们需要以小批量方式加载数据
# 一个简单的data_iter函数能够打乱数据并对数据集进行切分

def data_iter(batch_size, features, labels):
    '''
    小批量数据迭代器
    输入：
        batch_size: 每个批量的样本数
        features: 特征矩阵
        labels: 标签向量
    输出：
        每次返回一个 (X, y) 小批量，其中 X 的形状为 (batch_size, d)，
        y 的形状为 (batch_size, 1)
    说明：
        1. 将所有样本的索引打乱，保证随机性。
        2. 按照批量大小切分样本
    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机打乱样本序号
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 测试data_iter，读取第一个批量并打印出来
batch_size = 10
for X_batch, y_batch in data_iter(batch_size, features, labels):
    print('一个小批量的特征：\n', X_batch)
    print('一个小批量的标签：\n', y_batch)
    break  # 只打印第一个批量

# ----------------------
# 初始化模型参数
# ----------------------
# 在开始训练之前，我们必须初始化模型参数w和b。
# 这里权重w从均值为0、标准差为0.01的正态分布中采样；偏置b初始化为0。
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ----------------------
# 定义线性回归模型
# ----------------------
# 线性回归模型的输出为：y_hat = Xw + b
def linreg(X, w, b):  #@save
    """
    线性回归模型
    输入:
        X: 特征矩阵
        w: 权重张量
        b: 偏置标量
    输出:
        模型预测值
    """
    # 计算矩阵乘法，再加上偏置。注意b是标量，会通过广播机制加到每一行上
    return torch.matmul(X, w) + b

# ----------------------
# 定义损失函数
# ----------------------
# 这里使用均方损失函数：loss = 1/2 *(y_hat - y)^2
def squared_loss(y_hat, y):  #@save
    """
    均方损失函数
    输入:
        y_hat: 模型预测值（形状为(batch_size, 1)）
        y: 实际标签（可能需要reshape为与y_hat相同形状）
    输出:
        每个样本的损失值（不求和）
    """
    # 注意：这里将y通过reshape变为与y_hat相同形状
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# ----------------------
# 定义优化算法：小批量随机梯度下降 (SGD)
# ----------------------
# 在每一步中，我们使用一个小批量计算损失的梯度，然后按照梯度下降的方式更新参数。
# 由于我们计算的损失是整个批量上所有样本的和，所以除以批量大小以规范化步长。
def sgd(params, lr, batch_size):  #@save
    """
    小批量随机梯度下降
    输入:
        params: 要更新的参数列表
        lr: 学习率
        batch_size: 批量大小（用于梯度标准化）
    """
    # 在此上下文管理器中，关闭自动求导
    with torch.no_grad():
        for param in params:
            # 执行原地操作：参数 -= (学习率 * 梯度 / 批量大小)
            param -= lr * param.grad / batch_size
            # 更新完之后，将梯度清零
            param.grad.zero_()

# ----------------------
# 训练模型
# ----------------------
# 现在我们已经准备好了所有构成要素，
# 接下来利用小批量随机梯度下降来更新模型参数。
#
# 流程:
#   1. 对整个数据集进行多次迭代（epoch）。
#   2. 每个epoch中，遍历所有的小批量数据：
#         - 对每个小批量计算预测值
#         - 计算对应的损失（此处损失为批量样本之和）
#         - 反向传播计算梯度
#         - 使用SGD更新参数
#   3. 每个epoch结束时，计算全数据集的平均损失并打印

lr = 0.03          # 学习率
num_epochs = 3     # 总迭代周期数
net = linreg       # 模型
loss = squared_loss  # 损失函数

for epoch in range(num_epochs):
    # 遍历整个数据集，每次取一个小批量
    for X_batch, y_batch in data_iter(batch_size, features, labels):
        # 模型对小批量的预测输出
        l = loss(net(X_batch, w, b), y_batch)
        # l 的形状为 (batch_size, 1)，将所有损失加和，
        # 并以此计算参数的梯度
        l.sum().backward()
        # 使用SGD更新参数；注意w和b都是全局变量，
        # 原地更新会影响它们的内存里面的值。
        sgd([w, b], lr, batch_size)
    # 每个epoch结束后计算全数据集上的平均损失，并打印
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ----------------------
# 评估参数估计效果
# ----------------------
# 因为我们生成数据时使用了真实参数true_w和true_b，
# 训练完成后可以对比估计误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# ----------------------
# 小结
# ----------------------
# 1. 本节从零开始实现了线性回归模型，包括数据生成、数据加载、模型定义、
#    损失函数和小批量随机梯度下降优化器。
# 2. 我们使用PyTorch的张量和自动求导来计算梯度，手动更新模型参数。
# 3. 训练过程结束后，模型参数与真实参数非常接近，说明模型已经学到了数据的规律。
#

