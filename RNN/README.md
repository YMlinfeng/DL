注：
该部分仅完结RNN1部分，torch构建的RNN尚未完善。


---


1. Download [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

2. Modify the directory in `read_imdb`.

3. Run `main.py` to train and test the language model. You can:

- Use `rnn1` or `rnn2`
- Switch the dataset by modifying `is_vocab` parameter of `get_dataloader_and_max_length`
- Tune the hyperparameters

to do more experiments.


---


你记得得非常对！GRU 和传统的 RNN 确实结构和公式都不同。下面我们来对比一下：

---

## 🧠 一、传统 RNN 的结构和公式

### ✅ 核心思想：

RNN（Recurrent Neural Network）是一种处理序列数据的神经网络。它通过将前一个时间步的隐藏状态传递到下一个时间步，使得模型能够“记住”先前的信息。

### 🧾 RNN 的数学公式：

对于每一个时间步 `t`，RNN 的更新规则如下：

\[
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\]

- \( h_t \)：当前时间步的隐藏状态
- \( h_{t-1} \)：前一个时间步的隐藏状态
- \( x_t \)：当前时间步的输入
- \( W_{hh}, W_{xh} \)：权重矩阵
- \( b_h \)：偏置项
- \( \tanh \)：激活函数

### ❗ 问题：

传统 RNN 存在两个主要问题：

1. **梯度消失或爆炸**：在时间步数较长的情况下，梯度在反向传播中会迅速变得非常小或非常大。
2. **长期依赖难以建模**：RNN 很难记住远距离的信息。

---

## 🔁 二、GRU（Gated Recurrent Unit）的结构和公式

### ✅ 引入门控机制：

GRU 通过引入 **门控机制（Gates）**，控制信息的遗忘和保留，从而有效缓解 RNN 的问题。

### 🧾 GRU 的数学公式：

\[
\begin{align*}
z_t &= \sigma(W_z x_t + U_z h_{t-1}) &\text{（更新门）} \\
r_t &= \sigma(W_r x_t + U_r h_{t-1}) &\text{（重置门）} \\
\tilde{h}_t &= \tanh(W x_t + U (r_t \odot h_{t-1})) &\text{（候选状态）} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t &\text{（最终输出）}
\end{align*}
\]

- \( z_t \)：更新门，决定保留多少过去的信息
- \( r_t \)：重置门，决定忘记多少过去的信息
- \( \tilde{h}_t \)：候选隐藏状态
- \( h_t \)：当前隐藏状态（最终输出）
- \( \sigma \)：sigmoid 函数
- \( \odot \)：元素乘（逐元素相乘）

### 🔍 门控机制的作用：

- **更新门 \( z_t \)**：
  - 如果 \( z_t \) 接近 1，模型更倾向于使用新的候选状态。
  - 如果 \( z_t \) 接近 0，模型保留之前的状态。
- **重置门 \( r_t \)**：
  - 控制如何将过去的信息整合进当前候选状态，如果 \( r_t \) 小，则重置过去信息。

---

## 🆚 三、RNN vs GRU：对比总结

| 特征 | 传统 RNN | GRU |
|------|----------|-----|
| 是否有门控 | ❌ 没有 | ✅ 有（更新门 + 重置门） |
| 能否捕捉长期依赖 | ❌ 较差 | ✅ 较强 |
| 是否存在梯度消失问题 | ❌ 容易出现 | ✅ 得到缓解 |
| 计算效率 | ✅ 更快（结构简单） | ⚠️ 稍慢（结构复杂） |
| 表达能力 | ⚠️ 有限 | ✅ 更强 |
| 参数量 | 少 | 多（多个门） |
| 适用场景 | 简单序列任务 | 复杂序列任务（如语言建模） |

---

## 🧠 四、GRU 和 LSTM 的关系

GRU 是 LSTM（Long Short-Term Memory）的简化版本：

| 结构 | 门控数量 | 是否有记忆单元 |
|------|----------|----------------|
| LSTM | 3 个（输入门、遗忘门、输出门） | ✅ 有 |
| GRU  | 2 个（更新门、重置门） | ❌ 没有（隐藏状态即记忆） |

---

## ✅ 总结一句话：

> **RNN 是最基础的序列模型，GRU 是 RNN 的升级版，具有更强的长期记忆能力，使用门控机制来控制信息流动。**
