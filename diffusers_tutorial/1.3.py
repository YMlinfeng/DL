#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diffusers 教程（精简整合版）
作者：林风
仓库：https://github.com/YMlinfeng/DL

示例涵盖三大部分：

1. 使用 StableDiffusionPipeline 进行文本生成图像
2. 更换调度器（以 EulerDiscreteScheduler 为例）
3. 拆解模型 / 调度器，自定义 DDPM 去噪流程（以公开猫猫模型为例）

所有超参数均使用固定数值，力求示例最小可复现且逻辑清晰。
如需改动，只需修改脚本顶部的「用户可配置区域」。

运行要求：
- Python ≥ 3.8
- diffusers >= 0.25.0
- transformers >= 4.40.0
- accelerate >= 0.29.0
- torch >= 2.0
- 有兼容 CUDA 的 GPU（强烈推荐，否则速度会非常慢）
-----------------------------------------------------------------
"""

# ======================== 用户可配置区域（全部写死） ============================
MODEL_ID            = "sd-legacy/stable-diffusion-v1-5"  # 文生图模型
PROMPT              = "An image of a squirrel in Picasso style"  # 文本提示
NEGATIVE_PROMPT     = None                               # 负向提示，可设 None
NUM_INFERENCE_STEPS = 30                                 # 去噪步数
GUIDANCE_SCALE      = 7.5                                # classifier-free guidance 规模
OUTPUT_IMAGE        = "image_of_squirrel_painting.png"   # 输出文件名

# 本地模型路径（若走离线流程则提前 git clone）
LOCAL_MODEL_PATH    = "./stable-diffusion-v1-5"          # 若不存在则自动切换为线上
USE_LOCAL_MODEL     = False                              # True -> 强制走本地权重

# 自定义 DDPM（猫猫）相关固定参数
CAT_MODEL_REPO      = "google/ddpm-cat-256"
CAT_TIMESTEPS       = 1000      # 与调度器保持一致
CAT_SAMPLE_STEPS    = 100       # 每 N 步保存一次可视化
CAT_OUTPUT_FINAL    = "cat_ddpm_final.png"

# ==============================================================================
import os
import time
import torch
from diffusers import (
    DiffusionPipeline, #type:ignore
    StableDiffusionPipeline, #type:ignore
    EulerDiscreteScheduler,#type:ignore
    UNet2DModel,#type:ignore
    DDPMScheduler,#type:ignore
)
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


def is_cuda_available() -> bool:
    """检测当前机器是否支持 CUDA。"""
    return torch.cuda.is_available()


# ----------------------------------------------------------------------
# 1. StableDiffusionPipeline：文本生成图像
# ----------------------------------------------------------------------
def run_stable_diffusion() -> None:
    """使用 StableDiffusionPipeline 进行文本到图像生成并保存。"""
    print("\n===== 1. 文生图示例：Stable Diffusion v1-5 =====")

    # 决定使用本地还是线上的 checkpoint
    model_source = LOCAL_MODEL_PATH if USE_LOCAL_MODEL and os.path.exists(LOCAL_MODEL_PATH) else MODEL_ID
    # model_source = MODEL_ID
    print(f"正在加载模型：{model_source}")

    # torch_dtype=torch.float16 可显著减少显存占用，运算更快
    pipe: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
    )

    # 将整个管道搬到 GPU，能大幅提速（14 亿参数若只跑 CPU 会极慢）
    device = "cuda" if is_cuda_available() else "cpu"
    pipe = pipe.to(device)

    # 生成图像。参数释义：
    # prompt：              文本提示
    # negative_prompt：     负向提示，可抑制不希望出现的元素
    # num_inference_steps： 去噪步数，越大越清晰但越慢
    # guidance_scale：      CFG 规模，越大越贴合 prompt，但过大易失真
    with torch.autocast(device):
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )

    image: Image.Image = result.images[0]
    image.save(OUTPUT_IMAGE)
    print(f"✅ 图像已保存至 {OUTPUT_IMAGE}")


# ----------------------------------------------------------------------
# 2. 更换调度器：EulerDiscreteScheduler
# ----------------------------------------------------------------------
def run_with_euler_scheduler() -> None:
    """演示如何一行代码替换调度器为 EulerDiscreteScheduler。"""
    print("\n===== 2. 替换调度器：EulerDiscreteScheduler =====")

    pipe: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda" if is_cuda_available() else "cpu")

    # 替换调度器：直接用现有 scheduler 的配置复刻一个 EulerDiscreteScheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    with torch.autocast(pipe.device.type):
        image = pipe(prompt=PROMPT, num_inference_steps=NUM_INFERENCE_STEPS).images[0]

    output_name = OUTPUT_IMAGE.replace(".png", "_euler.png")
    image.save(output_name)
    print(f"✅ Euler 调度器生成的图像已保存至 {output_name}")


# ----------------------------------------------------------------------
# 3. 手写 DDPM 去噪循环（猫猫）
# ----------------------------------------------------------------------
def run_custom_ddpm() -> None:
    """
    拆解模型与调度器：手动实现 DDPM 去噪流程。
    以 google/ddpm-cat-256 checkpoint 为例，展示从纯噪声逐步生成猫图。
    """
    print("\n===== 3. 自定义 DDPM 去噪（猫猫示例）=====")

    # 3.1 加载 UNet2DModel
    model: UNet2DModel = UNet2DModel.from_pretrained(CAT_MODEL_REPO)
    model = model.to("cuda" if is_cuda_available() else "cpu")
    model.eval()  # 推理模式，加速并节省显存

    # 3.2 加载 DDPMScheduler
    scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(CAT_MODEL_REPO)

    # 3.3 准备纯高斯噪声：shape = [batch, channels, height, width]
    torch.manual_seed(0)  # 固定随机种子，确保结果可复现
    sample = torch.randn( # 生成服从标准正态分布（均值为0，标准差为1）的随机数张量
        1,
        model.config.in_channels,
        model.config.sample_size,
        model.config.sample_size,
        device=model.device,
    )

    # 3.4 反向去噪循环
    for step, t in enumerate(tqdm(scheduler.timesteps, desc="DDPM Sampling")):
        # 1. 预测残差 ε_theta(x_t, t)
        with torch.no_grad():
            epsilon = model(sample, t).sample

        # 2. 根据调度器计算 x_{t-1}
        sample = scheduler.step(epsilon, t, sample).prev_sample

        # 3. 每隔 CAT_SAMPLE_STEPS 步保存一次可视化
        if (step + 1) % CAT_SAMPLE_STEPS == 0 or t == 0:
            img = postprocess_sample(sample)
            img.save(f"cat_step_{step + 1:04d}.png")

    # 3.5 保存最终结果
    final_img = postprocess_sample(sample)
    final_img.save(CAT_OUTPUT_FINAL)
    print(f"✅ DDPM 去噪结束，最终图像已保存至 {CAT_OUTPUT_FINAL}")


def postprocess_sample(sample: torch.Tensor) -> Image.Image:
    """
    将模型输出的张量 [-1, 1] 归一化到 [0,255]，并转为 PIL.Image.

    参数
    ----
    sample : torch.Tensor
        shape=[1, C, H, W]，且像素范围在 [-1, 1] 的张量

    返回
    ----
    PIL.Image.Image
    """
    sample = sample.detach().cpu()
    sample = (sample + 1.0) * 127.5  # [-1,1] -> [0,255]
    sample = sample.clamp(0, 255).permute(0, 2, 3, 1)  # NCHW -> NHWC
    array = sample.numpy().astype(np.uint8)[0]
    return Image.fromarray(array)


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    # 1. Stable Diffusion 文生图
    run_stable_diffusion()

    # 2. 替换为 Euler 调度器
    run_with_euler_scheduler()

    # 3. 自定义 DDPM 去噪（猫猫）
    run_custom_ddpm()

    print(f"\n🎉 全部示例运行完毕，总耗时 {time.time() - t0:.1f} 秒")