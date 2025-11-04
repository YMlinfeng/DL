#!/usr/bin/env python
# coding: utf-8
"""
Diffusers 教程示例（精简整合版）
作者：林风  GitHub: https://github.com/YMlinfeng/DL

本文件整合并修正了旧版教程中零散的文字与代码示例，
按如下逻辑逐步演示如何在 HuggingFace - diffusers 中
兼顾「速度」「显存」与「画质」三方面进行推理优化。

运行环境要求
------------------------------------------------------------------
1. Python ≥ 3.8
2. pip install diffusers==0.27.0  transformers accelerate  safetensors
3. 建议使用 GPU（CUDA 11.7+）; 若无 GPU，可删去 .to("cuda") 并把
   torch_dtype 改为 torch.float32（速度将大幅下降）。

本脚本包含 6 个主要步骤：
------------------------------------------------------------------
0. 全局配置与工具函数
1. 基础用法（fp32，PNDM，50 步）
2. 使用 fp16 加速
3. 换用更高效的调度器并减少推理步数
4. 显存优化（attention slicing & CPU offload）
5. 更换更好的 VAE 组件
6. 提示词工程（prompt engineering）示例

所有超参数均写死在常量区域，方便初学者“一键跑通”。
如需实验，请自行修改常量或函数入参。


可以在 [DiffusionPipeline] 中通过调用compatibles方法找到与当前模型兼容的调度器 (scheduler)。

pipeline.scheduler.compatibles
[
    diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
    diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
    diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
    diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
    diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
    diffusers.schedulers.scheduling_pndm.PNDMScheduler,
    diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_ddim.DDIMScheduler,
]
Stable Diffusion 模型默认使用的是 PNDMScheduler ，通常要大概50步推理, 
但是像 DPMSolverMultistepScheduler 这样更高效的调度器只要大概 20 或 25 步推理. 
使用 ConfigMixin.from_config() 方法加载新的调度器
Diffusers 还提供了更先进的优化方法，例如分组卸载（group-offloading）和区域编译（regional compilation）
想了解如何进一步提高性能，请查阅 推理优化文档

"""

# ----------------------------- 0. 全局配置 ----------------------------- #
import os
import time
from typing import List
import PIL
from PIL import Image

import torch
from diffusers import (
    DiffusionPipeline, #type:ignore
    DPMSolverMultistepScheduler, #type:ignore
    AutoencoderKL, #type:ignore
)
from diffusers.utils.pil_utils import make_image_grid  # 仅用于拼图，可删去

# ---- 常量（如需实验请自行修改） --------------------------------------- #
MODEL_ID: str = "sd-legacy/stable-diffusion-v1-5"  # 模型名称
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 设备
DTYPE: torch.dtype = torch.float16  # 推理精度；若无 GPU 建议用 torch.float32
BASE_PROMPT: str = "portrait photo of an old warrior chief"
ADVANCED_PROMPT_SUFFIX: str = (
    ", tribal panther make up, blue on red, side profile, looking away, serious eyes,"
    " 50mm portrait photography, hard rim lighting"
)
NEGATIVE_PROMPT: str = "low quality, blurry, ugly, poor details"
SEED: int = 0  # 随机种子，保证结果可复现
IMG_DIR: str = "outputs"  # 图片保存目录


# ------------------------- 0.1 工具函数 -------------------------------- #
def seed_everything(seed: int) -> torch.Generator:
    """
    设置随机种子并返回一个 cuda Generator。

    参数
    ----
    seed : int
        随机数种子；相同种子可复现完全一致的结果。

    返回
    ----
    torch.Generator
        绑定到 CUDA 的生成器；CPU 环境下也可用但速度慢。
    """
    g = torch.Generator(DEVICE).manual_seed(seed)
    return g


def save_grid(images: List["PIL.Image.Image"], rows: int, cols: int, name: str) -> None:
    """
    将多张 PIL 图片按网格拼接后保存到 IMG_DIR 目录下。

    参数
    ----
    images : List[PIL.Image]
        需要拼接的图片列表。
    rows, cols : int
        行数和列数（必须满足 rows * cols == len(images)）。
    name : str
        文件名（不含扩展名），结果保存为 PNG。
    """
    os.makedirs(IMG_DIR, exist_ok=True)
    grid = make_image_grid(images, rows, cols)
    path = os.path.join(IMG_DIR, f"{name}.png")
    grid.save(path)
    print(f"[INFO] 已保存 {path}")


# --------------------------- 0.2 初始化 Pipeline ------------------------ #
def build_pipeline(
    fp16: bool = True,
    enable_slicing: bool = True,
    cpu_offload: bool = False,
    use_dpm: bool = True,
) -> DiffusionPipeline:
    """
    根据给定选项构建并返回 DiffusionPipeline。

    参数
    ----
    fp16 : bool
        是否加载为 float16（半精度）；可显著减少显存、加速推理。
    enable_slicing : bool
        是否启用 attention slicing；在多图 batch 推理时可省显存。
    cpu_offload : bool
        是否启用模型 CPU offload；显存极小但会牺牲速度。
    use_dpm : bool
        是否将调度器替换为 DPMSolverMultistepScheduler（20-25 步即可）。

    返回
    ----
    DiffusionPipeline
        配置好的 pipeline 实例。
    """
    dtype = torch.float16 if fp16 else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # 放到 GPU / CPU
    pipe = pipe.to(DEVICE)

    # 替换调度器
    if use_dpm:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 显存相关优化
    if enable_slicing:
        pipe.enable_attention_slicing()
        # pipe.enable_attention_slicing(slice_size="auto") # diffusers 自动根据模型的 seq_len 和 num_heads 计算最优分片数，平衡显存和速度
    if cpu_offload:
        pipe.enable_model_cpu_offload()

    pipe.set_progress_bar_config(disable=False)  # 显示进度条
    return pipe


# ----------------------------- 1. 基础用法 ----------------------------- #
def demo_fp32_pndm_50_steps() -> None:
    """
    使用默认设置（fp32 + PNDM + 50 步）生成一张图片。
    速度最慢但最具可比性，可作为基准。
    """
    pipe = build_pipeline(fp16=False, enable_slicing=False, cpu_offload=False, use_dpm=False)

    start = time.perf_counter()
    image = pipe(
        BASE_PROMPT,
        generator=seed_everything(SEED),
        num_inference_steps=50,
    ).images[0]
    duration = time.perf_counter() - start
    save_grid([image], 1, 1, "step1_fp32_pndm_50")
    print(f"[STEP-1] 生成耗时 {duration:.2f} 秒 | 精度 fp32 | 调度器 PNDM | 50 步")


# --------------------------- 2. fp16 加速 ------------------------------- #
def demo_fp16_dpm_50_steps() -> None:
    """
    使用 fp16 + DPMSolverMultistepScheduler（仍保持 50 步）生成图片，
    对比 Step-1 体会半精度带来的提速。
    """
    pipe = build_pipeline(fp16=True, enable_slicing=False, cpu_offload=False, use_dpm=True)

    start = time.perf_counter()
    image = pipe(
        BASE_PROMPT,
        generator=seed_everything(SEED),
        num_inference_steps=50,
    ).images[0]
    duration = time.perf_counter() - start
    save_grid([image], 1, 1, "step2_fp16_dpm_50")
    print(f"[STEP-2] 生成耗时 {duration:.2f} 秒 | 精度 fp16 | 调度器 DPM | 50 步")


# --------------- 3. 换调度器 + 减少步数（20 步） ------------------------ #
def demo_dpm_20_steps() -> None:
    """
    继续使用 DPM 调度器，但将推理步数减到 20。
    """
    pipe = build_pipeline(fp16=True, enable_slicing=False, cpu_offload=False, use_dpm=True)

    start = time.perf_counter()
    image = pipe(
        BASE_PROMPT,
        generator=seed_everything(SEED),
        num_inference_steps=20,
    ).images[0]
    duration = time.perf_counter() - start
    save_grid([image], 1, 1, "step3_fp16_dpm_20")
    print(f"[STEP-3] 生成耗时 {duration:.2f} 秒 | 精度 fp16 | 调度器 DPM | 20 步")


# --------------- 4. attention slicing + batch 推理 ---------------------- #
def demo_batch_generation(batch_size: int = 8) -> None:
    """
    演示如何在开启 attention slicing 后提升 batch size。

    参数
    ----
    batch_size : int
        批量大小；如显存不足可减小。
    """
    pipe = build_pipeline(fp16=True, enable_slicing=True, cpu_offload=False, use_dpm=True)

    # 构造同一 prompt 的批量输入
    prompts = [BASE_PROMPT] * batch_size
    generators = [seed_everything(i) for i in range(batch_size)]

    start = time.perf_counter()
    images = pipe(
        prompt=prompts,
        generator=generators,
        num_inference_steps=20,
    ).images
    duration = time.perf_counter() - start
    save_grid(images, rows=2, cols=batch_size // 2, name=f"step4_batch_{batch_size}")
    print(
        f"[STEP-4] batch={batch_size} 耗时 {duration:.2f} 秒 | attention slicing 已开启"
    )


# ----------- 5. 替换更好的 VAE（提升细节与去噪质量） -------------------- #
def demo_replace_vae(batch_size: int = 8) -> None:
    """
    使用公开的 MSE-finetuned VAE 替换默认 VAE，可改善细节。
    """
    pipe = build_pipeline(fp16=True, enable_slicing=True, cpu_offload=False, use_dpm=True)

    # 加载并替换 VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16,
    ).to(DEVICE)
    pipe.vae = vae

    prompts = [BASE_PROMPT + ADVANCED_PROMPT_SUFFIX] * batch_size
    generators = [seed_everything(i) for i in range(batch_size)]

    images = pipe(
        prompt=prompts,
        generator=generators,
        num_inference_steps=20,
    ).images
    save_grid(images, 2, batch_size // 2, "step5_replace_vae")
    print("[STEP-5] 已替换 VAE 并生成高质量批量图片")


# ------------- 6. 提示词工程（高级 prompt & 多种年龄） ------------------ #
def demo_prompt_engineering() -> None:
    """
    通过改写 prompt + 不同种子，探索不同年龄段的酋长形象。
    """
    pipe = build_pipeline(fp16=True, enable_slicing=True, cpu_offload=False, use_dpm=True)

    prompts = [
        "portrait photo of the oldest warrior chief" + ADVANCED_PROMPT_SUFFIX,
        "portrait photo of an old warrior chief" + ADVANCED_PROMPT_SUFFIX,
        "portrait photo of a warrior chief" + ADVANCED_PROMPT_SUFFIX,
        "portrait photo of a young warrior chief" + ADVANCED_PROMPT_SUFFIX,
    ]
    generators = [seed_everything(i + 1) for i in range(len(prompts))]

    images = pipe(
        prompt=prompts,
        generator=generators,
        num_inference_steps=25,
        negative_prompt=[NEGATIVE_PROMPT] * len(prompts),
    ).images
    save_grid(images, 2, 2, "step6_prompt_engineering")
    print("[STEP-6] 提示词工程完成，已生成多年龄对比图")


# ------------------------------ main ----------------------------------- #
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # 可能略微提升 A100 上速度

    demo_fp32_pndm_50_steps()
    demo_fp16_dpm_50_steps()
    demo_dpm_20_steps()
    demo_batch_generation(batch_size=8)
    demo_replace_vae(batch_size=8)
    demo_prompt_engineering()

    # 输出最大显存占用
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\n[SUMMARY] 脚本执行完毕，GPU 最大显存占用 ≈ {used:.2f} GB")


        