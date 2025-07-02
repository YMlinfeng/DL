import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
import os

class SpatioTemporalAttention(nn.Module):
    """时空注意力模块"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x):
        """x: [batch, (time*space), dim]"""
        b, n, d = x.shape
        h = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class STDiTBlock(nn.Module):
    """STDiT基础块"""
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatioTemporalAttention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class STDiT3(nn.Module):
    """STDiT3模型实现"""
    def __init__(self, 
                 dim=512,
                 depth=12,
                 heads=8,
                 dim_head=64,
                 mlp_dim=1024,
                 in_channels=3,
                 patch_size=16,
                 temporal_patch=2):
        super().__init__()
        
        # 时空patch嵌入
        self.patch_embed = nn.Conv3d(in_channels, dim, 
                                    kernel_size=(temporal_patch, patch_size, patch_size),
                                    stride=(temporal_patch, patch_size, patch_size))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, dim))
        
        # Transformer层
        self.blocks = nn.ModuleList([
            STDiTBlock(dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_channels * temporal_patch * patch_size * patch_size)
        
    def forward(self, x):
        """x: [batch, channels, time, height, width]"""
        b, c, t, h, w = x.shape
        
        # 时空patch嵌入
        x = self.patch_embed(x)  # [b, dim, t', h', w']
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Transformer
        for block in self.blocks:
            x = block(x)
            
        # 输出重建
        x = self.norm(x)
        x = self.head(x)
        x = rearrange(x, 'b (t h w) (c p1 p2 p3) -> b c (t p1) (h p2) (w p3)',
                      p1=self.patch_embed.kernel_size[0],
                      p2=self.patch_embed.kernel_size[1],
                      p3=self.patch_embed.kernel_size[2])
        return x

# 使用示例
if __name__ == "__main__":
    model = STDiT3(dim=512, depth=12, heads=8).cuda()
    input_tensor = torch.randn(2, 3, 8, 64, 64).cuda()  # [batch, channels, time, h, w]
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")