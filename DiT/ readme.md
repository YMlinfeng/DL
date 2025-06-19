# DiT - Diffusion Transformer

**Paper**: [DiT: Self-supervised Pretraining for Diffusion Models](https://arxiv.org/abs/2303.16203)  
**Conference**: ICCV 2023  
**Author**: Saining Xie et al.

本项目是 DiT（Diffusion Transformer）的 PyTorch 实现，复现并简化了论文中的主要结构和训练流程。

## 📌 参考实现

本实现参考了以下优秀项目：

- [lucidrains/imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [facebookresearch/DiT](https://github.com/facebookresearch/DiT)

## 🧠 主要特性

- 使用 Transformer 架构构建扩散模型
- 支持多分辨率图像训练
- 支持微调和条件生成
- 兼容 HuggingFace Datasets 和 PyTorch Lightning

## 🚀 快速开始

### 训练模型

```bash
python main.py --config configs/dit_small.yaml
```

### 生成样本

```bash
python scripts/sample.py --checkpoint path_to_checkpoint.ckpt
```
