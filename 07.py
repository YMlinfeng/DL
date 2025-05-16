# 自动微分演示：涵盖标量/非标量求导、梯度累积、detach、控制流等知识点

import torch
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. 基础示例：标量函数的求导
# y = 2 * x^T x 的导数应为 dy/dx = 4x
# -------------------------------

x = torch.arange(4.0)
x.requires_grad_(True)  # 开启对x的自动求导
print("x:", x)

y = 2 * torch.dot(x, x)  # 标量函数
print("y = 2 * x^T x:", y.item())

y.backward()  # 自动求导
print("dy/dx:", x.grad)
print("验证 dy/dx 是否等于 4x:", x.grad == 4 * x)

# -------------------------------
# 2. 梯度累积与清零
# -------------------------------
x.grad.zero_()  #! 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
y = x.sum()
y.backward()
print("dy/dx for y = x.sum():", x.grad)

# -------------------------------
# 3. 非标量函数的反向传播
# y = x^2 是向量，不能直接 backward()，需要先 sum() 得标量
# -------------------------------
x.grad.zero_()
y = x * x  # 非标量输出
y.sum().backward()  # 等价于 y.backward(torch.ones_like(x))和 y.backward(torch.tensor([1.0, 1.0, 1.0]))
# y.sum() 是一个标量，表示 y[0] + y[1] + y[2]。
# y.backward(torch.tensor([1.0, 1.0, 1.0])) 的含义是：对 y 中每个元素乘以 1，然后求总导数。
# 从链式法则的角度看，这两种方式传递给 x 的梯度都是一样的。
print("dy/dx for y = x^2:", x.grad)

# -------------------------------
# 4. detach: 分离计算图
# y = x^2, u = y.detach(), z = u * x
# 不希望z的梯度影响y
# -------------------------------
x.grad.zero_()
y = x * x
u = y.detach()  # u 不再与计算图绑定
z = u * x
z.sum().backward()
print("dz/dx when z = detach(y) * x:", x.grad)
print("是否等于 u:", x.grad == u)

# 如果我们对 y 再做一次 backward，则梯度应为 2x
x.grad.zero_()
y.sum().backward()
print("dy/dx for y = x^2:", x.grad)
print("是否等于 2x:", x.grad == 2 * x)

# -------------------------------
# 5. 控制流中的梯度计算
# -------------------------------

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
print("\na =", a.item())
d = f(a)
d.backward()
print("f(a) =", d.item())
print("df/da =", a.grad.item())
print("验证 df/da == d/a:", torch.allclose(a.grad, d / a)) # 验证某个数学等式是否成立
print("验证 df/da == d/a:", a.grad == d / a)

# -------------------------------
# 6. 控制流扩展：向量输入
#!以下内容属于进阶内容，这里没有详细研究
# -------------------------------
a = torch.randn(size=(3, 1), requires_grad=True)
print("\n向量 a:\n", a)

d = f(a)
# 注意：如果 d 不是标量，不能直接 backward，需要 sum()
d.sum().backward()
print("df/da:\n", a.grad)

# -------------------------------
# 7. 绘图：y = sin(x)，dy/dx = cos(x)
# -------------------------------

x = torch.linspace(0, 3 * np.pi, 128, requires_grad=True)
y = torch.sin(x)
y.sum().backward()

plt.figure(figsize=(8, 4))
plt.plot(x.detach(), y.detach(), label='y = sin(x)')
plt.plot(x.detach(), x.grad, label='dy/dx = cos(x)')
plt.legend()
plt.title("Autograd: sin(x) and its gradient")
plt.grid(True)
plt.show()

# -------------------------------
# 8. 二阶导数（高阶导数示例）
# -------------------------------
x = torch.tensor([2.0], requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]  # dy/dx = 3x^2
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]               # d2y/dx2 = 6x
print("\n函数 y = x^3")
print("一阶导 dy/dx:", dy_dx.item())
print("二阶导 d2y/dx2:", d2y_dx2.item())