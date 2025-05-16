import torch

# 1、 入门
x = torch.arange(12)

X = torch.arange(16).reshape(4, 4)

Y = torch.arange(16).reshape(2, 4, -1)

a = torch.zeros(((((2,3,4)))))
b = torch.ones(2,3,4)

print(a.size() == b.size()) # True

c = torch.randn(3, 4) # 每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样

torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) # 最外层的列表对应于轴0，内层的列表对应于轴1

# 2、运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

x = torch.tensor([1, 2, 4, 8]).reshape(1, 4)
y = torch.tensor([2, 2, 2, 2]).reshape(4, 1)
print(x, y, x + y)

zz = torch.exp(x + y)

before = id(Y)
Y = Y + X
id(Y) == before # 之前的Y已经被析构掉了
Z = torch.zeros_like(Y) #???
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))


# 3\
A = X.numpy()
B = torch.tensor(A)
# type(A), type(B)
a = torch.tensor([3.5])
# a, a.item(), float(a), int(a)

# 4\
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
X1 = torch.cat((X, Y), dim=0), X2 = torch.cat((X, Y), dim=1)
print(X == Y) # 只会对比两个矩阵都有元素存在的位置

print(X)


