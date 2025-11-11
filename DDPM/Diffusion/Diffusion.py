# ===============================================================
# 1. 依赖项
# ===============================================================
import torch                           # 深度学习主库
import torch.nn as nn                  # 神经网络模块
import torch.nn.functional as F        # 常用函数式 API：mse_loss / pad 等
import numpy as np                     # 仅作类型补全（本文件未显式用到）

# ===============================================================
# 2. 工具函数
# ===============================================================
def extract(v: torch.Tensor,
            t: torch.LongTensor,
            x_shape: torch.Size) -> torch.Tensor:
    """
    从一维向量 v（长度=T）中抽取与时间步 t 对应的元素，
    再 reshape 成 [B,1,1,1,...]，方便与任意维度的 x 做广播相乘。

    参数说明
    ----------
    v        : [T]               —— 需要索引的系数表
    t        : [B] (int64)       —— 每个样本的随机时间步
    x_shape  : 形如 [B,C,H,W]    —— 用来确定要加多少个 1 维度

    返回值
    ----------
    out      : [B,1,1,1...] 与 x 形状可广播
    """
    device = t.device                                 # 设备保持一致
    # torch.gather 在 dim=0 维上按索引 t 取值，得到 [B]
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # view 成 [B,1,1,...]，长度=len(x_shape)-1 使得可与 x 对齐
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# ===============================================================
# 3. 训练阶段：q(x_t|x_0) 前向扩散 + 拟合噪声  ε_θ
# ===============================================================
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 beta_1: float,
                 beta_T: float,
                 T: int):
        super().__init__()

        self.model = model          # 预测噪声 ε_θ(x_t, t) 的 UNet
        self.T = T                  # 总扩散步数

        # ---- 3.1 预生成 β_t 系数表（线性调度） -------------------
        self.register_buffer(       # register_buffer => 保存到 state_dict，但不作为参数优化
            'betas', torch.linspace(beta_1, beta_T, T).double())  # [T]

        # 计算 α_t = 1 - β_t ； ᾱ_t = ∏_{i≤t} α_i
        alphas = 1. - self.betas                            # [T]
        alphas_bar = torch.cumprod(alphas, dim=0)           # [T]

        # ---- 3.2 预存 q(x_t|x_0) 的闭式系数 ----------------------
        # x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε
        self.register_buffer('sqrt_alphas_bar',
                             torch.sqrt(alphas_bar))        # √ᾱ_t
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             torch.sqrt(1. - alphas_bar))    # √(1-ᾱ_t)

    # -----------------------------------------------------------
    # forward : 实现论文 Algorithm 1 的无偏噪声重建损失
    # -----------------------------------------------------------
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        输入
        ------
        x_0 : [B,C,H,W]，取值范围 [-1,1] 的真实图像

        输出
        -----
        loss : [B,C,H,W] 与噪声逐像素 MSE（可再对 batch 求均值）
        """
        # 1) 随机采样时间步 t ~ Uniform{0,...,T-1}
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)

        # 2) 生成与 x_0 同形状的高斯噪声 ε ~ N(0,I)
        noise = torch.randn_like(x_0)

        # 3) 用闭式公式得到 x_t
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +     # √ᾱ_t·x₀
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise  # √(1-ᾱ_t)·ε
        )

        # 4) 让模型预测 ε_θ(x_t, t)，与真实 ε 做 MSE (DDPM 损失)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss  # 形状同 x_0，调用者通常再对[B,C,H,W]做 mean

# ===============================================================
# 4. 采样阶段：p_θ(x_{t-1}|x_t) 反向去噪
# ===============================================================
class GaussianDiffusionSampler(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 beta_1: float,
                 beta_T: float,
                 T: int):
        super().__init__()

        self.model = model
        self.T = T

        # ---- 4.1 同样预生成 β/α/ᾱ 表 --------------------------------
        self.register_buffer('betas',
                             torch.linspace(beta_1, beta_T, T).double())  # [T]
        alphas = 1. - self.betas                      # [T]
        alphas_bar = torch.cumprod(alphas, dim=0)     # [T]
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # ᾱ_{t-1}

        # ---- 4.2 预计算 Algorithm 2 中涉及的常数 --------------------
        # 见 DDPM (Eq.12)：μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))        # 1/√α_t
        self.register_buffer('coeff2',
                             self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # 预测均值公式中的第二项系数：β_t / √(1-ᾱ_t)

        # 后验方差 σ²_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t) （Eq.7）
        self.register_buffer('posterior_var',
                             self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # -----------------------------------------------------------
    # 4.3 由 ε_θ 反推 μ_θ(x_t,t)  （Eq.12）
    # -----------------------------------------------------------
    def predict_xt_prev_mean_from_eps(self,
                                      x_t: torch.Tensor,
                                      t: torch.LongTensor,
                                      eps: torch.Tensor) -> torch.Tensor:
        """
        x_t, eps : [B,C,H,W] 形状相同
        返回      : μ_θ(x_{t-1}|x_t)  的预测均值
        """
        assert x_t.shape == eps.shape

        return (
            extract(self.coeff1, t, x_t.shape) * x_t -    # 1/√α_t · x_t
            extract(self.coeff2, t, x_t.shape) * eps      # β_t/√(1-ᾱ_t) · ε_θ
        )

    # -----------------------------------------------------------
    # 4.4 给定 x_t 计算 p_θ(x_{t-1}|x_t) 的参数 (均值+方差)
    # -----------------------------------------------------------
    def p_mean_variance(self,
                        x_t: torch.Tensor,
                        t: torch.LongTensor):
        """
        返回
        ------
        xt_prev_mean : [B,C,H,W] 预测均值 μ_θ
        var          : [B,1,1,1]   后验方差 σ²_t      （对每个样本可广播）
        """
        # 采样时仅需方差，不用 log_variance；KL 评估时会用
        var_table = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var_table, t, x_t.shape)            # [B,1,1,1]

        # 1) 模型预测当前噪声 ε_θ
        eps = self.model(x_t, t)                          # [B,C,H,W]
        # 2) 由 ε_θ 反推出 μ_θ(x_{t-1}|x_t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)

        return xt_prev_mean, var

    # -----------------------------------------------------------
    # forward : DDPM Algorithm 2 —— 逐步去噪生成样本
    # -----------------------------------------------------------
    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        """
        输入
        ------
        x_T : [B,C,H,W] 纯高斯噪声 (t=T-1)

        输出
        -----
        x_0 : [B,C,H,W] 生成的样本（裁剪到 [-1,1]）
        """
        x_t = x_T                                         # 初始设为纯噪声
        for time_step in reversed(range(self.T)):         # t = T-1 ... 0
            print(time_step)                              # 调试：打印当前 t

            # 构造全 batch 同一个时间整数张量 [B]
            t = x_t.new_full((x_T.shape[0],), time_step, dtype=torch.long)

            # 4.4 计算 p_θ 的均值/方差
            mean, var = self.p_mean_variance(x_t=x_t, t=t)

            # 4.5 重参数化：x_{t-1} = μ + σ·z （若 t>0）; t=0 时无噪声
            if time_step > 0:
                noise = torch.randn_like(x_t)             # z ~ N(0,I)
            else:
                noise = 0.                                # 最后一步直接取均值
            x_t = mean + torch.sqrt(var) * noise          # 更新 x_t → x_{t-1}

            # 防御性检查：出现 NaN 立即报错
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        x_0 = x_t                                         # 最后得到复原图像
        return torch.clip(x_0, -1, 1)                     # 限幅到合法范围