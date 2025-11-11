# ===============================================================
# 1. 依赖项与工具函数
# ===============================================================
import math                                     # 数学库：用于 log / exp / sin / cos 等
import torch                                    # PyTorch 主入口
from torch import nn                            # nn 模块：提供网络层基类
from torch.nn import init                       # 参数初始化工具
from torch.nn import functional as F            # 函数式 API：interpolate、softmax 等

# ===============================================================
# 2. 实用层或函数
# ===============================================================
class Swish(nn.Module):                         # Swish 激活：相比 ReLU 更平滑
    def forward(self, x):                       # x: 任意张量
        return x * torch.sigmoid(x)             # y = x · σ(x)，同形状返回

# ---------------------------------------------------------------
# TimeEmbedding : 将离散的时间步 (0~T-1) 编码为高维连续向量
# 扩散模型中，时间步代表“噪声注入程度”，需要与图像特征对齐后再输入 UNet
# ---------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, T: int, d_model: int, dim: int):
        assert d_model % 2 == 0                       # 正余弦对必须成对出现
        super().__init__()

        # 生成 [0, d_model/2) 的指数衰减频率             (动机：不同频率编码不同尺度的信息)
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)                         # 频率 = 10000^{-2i/d_model}

        pos = torch.arange(T).float()                 # 离散时间步 [0,1,...,T-1]
        emb = pos[:, None] * emb[None, :]             # 外积 → [T, d_model/2]

        # 正弦和余弦双通道                               (动机：使任意时间步线性可插值)
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # [T, d_model/2, 2]
        emb = emb.view(T, d_model)                    # 展平 → [T, d_model]

        # timembedding：查表 + 两层 MLP 做“可学习投影”
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),        # 输入 t:[B] → [B, d_model] (固定权重)
            nn.Linear(d_model, dim),                  # 投影到更高维 (理由：提供更多容量)
            Swish(),                                  # 非线性
            nn.Linear(dim, dim),                      # 再投影一次，让网络自主调整
        )
        self.initialize()                             # 调用权重初始化

    def initialize(self):
        for module in self.modules():                 # 遍历子模块
            if isinstance(module, nn.Linear):         # 仅对线性层初始化
                init.xavier_uniform_(module.weight)   # Xavier：保持前后方差一致
                init.zeros_(module.bias)

    def forward(self, t: torch.LongTensor):           # t: [B] int64
        emb = self.timembedding(t)                    # 输出 [B, dim]
        return emb

# ---------------------------------------------------------------
# DownSample : 特征图在空间维度缩小一半，通道保持不变
# 使用 stride=2 卷积(比 maxpool 学习性更强)
# ---------------------------------------------------------------
class DownSample(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=2, padding=1)    # [B,C,H,W]→[B,C,H/2,W/2]
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):                       # temb 未用，仅保持签名一致
        x = self.main(x)
        return x                                      # 返回缩小后的特征

# ---------------------------------------------------------------
# UpSample : 特征图在空间维度放大 2 倍
# 先最近邻插值(避免棋盘效应)→再卷积细化
# ---------------------------------------------------------------
class UpSample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):                       # x:[B,C,H,W]
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # [B,C,2H,2W]
        x = self.main(x)                              # 卷积融合插值造成的空洞
        return x

# ---------------------------------------------------------------
# AttnBlock : 分辨率内的自注意力 (像素/patch 互相关注)
# 允许网络在同一尺度捕获长距离相关性，弥补卷积局限
# ---------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)        # GN 不依赖 batch 大小
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)         # 1×1 卷积生 Q
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)         # 生 K
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)         # 生 V
        self.proj   = nn.Conv2d(in_ch, in_ch, 1)         # 输出投影
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        # 输出投影权重缩小 gain，避免残差过大
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):                                # x:[B,C,H,W]
        B, C, H, W = x.shape
        h = self.group_norm(x)                           # GN 归一化
        q = self.proj_q(h)                               # [B,C,H,W]
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).reshape(B, H*W, C)     # 展平成序列
        k = k.reshape(B, C, H*W)                         # 匹配 bmm 维度
        w = torch.bmm(q, k) * C**-0.5                   # 点积注意力 / sqrt(C)
        w = F.softmax(w, dim=-1)                         # attention 权重

        v = v.permute(0, 2, 3, 1).reshape(B, H*W, C)     # [B,H*W,C]
        h = torch.bmm(w, v)                              # 加权求和
        h = h.reshape(B, H, W, C).permute(0, 3, 1, 2)    # 复原 4D
        h = self.proj(h)                                 # 输出线性
        return x + h                                     # 残差：稳定训练

# ---------------------------------------------------------------
# ResBlock : 时间条件化的残差块 (含可选注意力)
# ---------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        # ---- 第一层卷积前的归一化+激活 ----
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch), Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )

        # ---- 时间嵌入投影 ----
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch)                     # 将 temb 融合到通道维
        )

        # ---- 第二层卷积 + dropout ----
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), Swish(),
            nn.Dropout(dropout),                       # 减少过拟合
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

        # ---- shortcut 通道匹配 ----
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

        # ---- 可选注意力 ----
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self.initialize()

    def initialize(self):
        for m in self.modules():                         # 遍历所有子模块
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        # 第二个 3×3 卷积权重缩放，避免方差爆炸
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):                          # x:[B,C,H,W] temb:[B,tdim]
        h = self.block1(x)                               # 归一化+卷积
        h += self.temb_proj(temb)[:, :, None, None]       # 时间信息广播到 H/W
        h = self.block2(h)                               # 再卷积

        h = h + self.shortcut(x)                         # 残差
        h = self.attn(h)                                 # 如配置则加自注意力
        return h                                         # [B,out_ch,H,W]

# ===============================================================
# 3. U-Net 主干：多尺度下采样-上采样结构
# ===============================================================
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        # --- 检查：要加注意力的层索引不得越界 ---
        assert all(i < len(ch_mult) for i in attn), 'attn index out of bound'

        tdim = ch * 4                                    # 时间嵌入维度(经验倍数)
        self.time_embedding = TimeEmbedding(T, ch, tdim) # 时序编码模块

        # -------- Stem --------
        self.head = nn.Conv2d(3, ch, kernel_size=3, padding=1)  # RGB→ch
        self.downblocks = nn.ModuleList()               # 下采样阶段层组
        chs = [ch]                                      # 记录每次通道，用于 skip
        now_ch = ch

        # -------- Down path (Encoder) --------
        for i, mult in enumerate(ch_mult):              # 遍历每个分辨率层级
            out_ch = ch * mult
            for _ in range(num_res_blocks):             # 堆叠多个 ResBlock
                self.downblocks.append(
                    ResBlock(now_ch, out_ch, tdim, dropout,
                             attn=(i in attn)))         # 部分层加注意力
                now_ch = out_ch
                chs.append(now_ch)                      # 记录输出通道

            if i != len(ch_mult) - 1:                   # 最后一级不再下采样
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)                      # DownSample 也要记录

        # -------- Middle (Bottleneck) --------
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),   # 加注意力
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        # -------- Up path (Decoder) --------
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):        # 对称遍历
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):        # 比 encoder 多 1，原因：额外处理 skip 拼接后增大的通道数
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, # 拼接后通道 = encoder_out + now_ch
                             out_ch=out_ch,
                             tdim=tdim,
                             dropout=dropout,
                             attn=(i in attn)))
                now_ch = out_ch
            if i != 0:                                 # 第一层恢复原分辨率后不再上采样
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0                           # 所有 skip 已消费完

        # -------- Tail --------
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, padding=1)         # 输出回 3 通道 (RGB 残差或噪声预测)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    # -----------------------------------------------------------
    # forward : x 为图像，t 为时间步
    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.LongTensor):
        # ---- 0. 时间步编码 ----
        temb = self.time_embedding(t)                   # [B, tdim]

        # ---- 1. Stem ----
        h = self.head(x)                                # [B,ch,H,W] (H=W=32 对 CIFAR)
        hs = [h]                                        # 用栈记录跳连

        # ---- 2. Encoder ----
        for layer in self.downblocks:
            h = layer(h, temb)                          # 传入 temb 以做条件化
            hs.append(h)                                # 每一步输出入栈

        # ---- 3. Bottleneck ----
        for layer in self.middleblocks:
            h = layer(h, temb)

        # ---- 4. Decoder ----
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):             # 只有 ResBlock 需要拼接 skip
                h = torch.cat([h, hs.pop()], dim=1)     # 通道拼接 skip
            h = layer(h, temb)                          # 继续前向

        h = self.tail(h)                                # 最终 3 通道输出

        assert len(hs) == 0                             # skip 栈应清空，逻辑自洽
        return h                                        # [B,3,H,W] 与输入同分辨率

# ===============================================================
# 4. 快速自检 (仅在直接运行本文件时执行)
# ===============================================================
if __name__ == '__main__':
    batch_size = 8
    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2],  # 网络配置
                 attn=[1], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)              # 假设 CIFAR10 尺度
    t = torch.randint(0, 1000, (batch_size, ))          # 随机时间步
    y = model(x, t)                                     # 前向
    print('output shape:', y.shape)                     # 预期 [8,3,32,32]