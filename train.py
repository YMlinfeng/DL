"""
本脚本基于 Qwen3-1.7B 模型演示了医学场景下的 SFT(Fine-tuning) 流程：
1. 下载并加载 Qwen3-1.7B 权重与 tokenizer；
2. 将自定义 JSONLines 数据集转换为指令微调格式；
3. 预处理 → 训练 → 推理 → 结果记录到 SwanLab。
依赖库：torch, transformers, datasets, pandas, modelscope, swanlab(=wandb 的国货替代)。
"""

# ====================== 导包 ======================
import json                                     # 标准库；序列化 / 反序列化
import pandas as pd                             # 结构化数据处理(DataFrame)
import torch                                    # 深度学习框架
from datasets import Dataset                    # 🤗 Datasets；轻量级内存/磁盘映射数据集
from modelscope import snapshot_download, AutoTokenizer
from transformers import (                      # HuggingFace Transformers
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
import swanlab                                  # experiment tracking 平台

# ====================== 运行时全局常量 ======================
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"  # 指定 SwanLab 项目空间(等价于 wandb project)
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"  # 指令前缀
MAX_LENGTH = 2048                                       # 生成 / 截断的上下文最大长度(token 数)

# 将超参写入 SwanLab 仪表盘
swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

# =========================================================
# 1. 数据格式转换工具
# =========================================================
def dataset_jsonl_transfer(origin_path: str, new_path: str) -> None:
    """
    把原始 JSONL ➜ 指令微调 JSONL
    ----------
    origin_path: 旧文件，每行包含 {question, think, answer}
    new_path   : 新文件，每行包含 {instruction, input, output}
    """
    messages = []  # type: list[dict[str, str]]

    # --- 读取旧 JSONL --------------------------------------------------------
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)                          # line -> dict
            user_q  = data["question"]                       # str
            model_a = data["answer"]                         # str
            think   = data["think"]                          # str

            # 把“思考”包装成 <think>...</think> 标签，方便 model 学“链式思考”
            output = f"<think>{think}</think>\n{model_a}"

            message = {
                "instruction": PROMPT,  # 系统级 prompt
                "input": user_q,        # 用户提问
                "output": output,       # 带思维过程的答案
            }
            messages.append(message)

    # --- 写出新 JSONL --------------------------------------------------------
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


# =========================================================
# 2. token 级预处理函数，供 `.map` 批量调用
# =========================================================
def process_func(example: dict) -> dict:
    """
    把一条样本转换为模型可直接喂入的张量字段  
    返回值字典包含三列，Trainer 会将其转为 torch.LongTensor：  
        • input_ids       : token 序列  
        • attention_mask  : pad = 0 / 1 掩码  
        • labels          : 监督信号；-100 处的 token 不计入 loss  
    """

    # ---------- 1. 构造系统 + 用户 prompt (ChatML 格式) ----------
    # f-string 可以跨行，只要每行均以 `f"` 起始
    instruction_enc = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"          # 系统提示
        f"<|im_start|>user\n{example['input']}<|im_end|>\n" # 用户问题
        f"<|im_start|>assistant\n",                         # assistant 起始
        add_special_tokens=False                            # 已手写特殊 token
    )

    # ---------- 2. 编码答案 ----------
    response_enc = tokenizer(example["output"], add_special_tokens=False)

    # ---------- 3. 拼接 input_ids / attention_mask ----------
    # 列表 `+` 在 Python 中表示连接，而非数值加法
    input_ids = (
        instruction_enc["input_ids"] +
        response_enc["input_ids"] +
        [tokenizer.pad_token_id]          # 末尾补 1 个 pad (<|im_end|>)
    )
    attention_mask = (
        instruction_enc["attention_mask"] +
        response_enc["attention_mask"] +
        [1]                                # pad 位依旧设 1，符合 Qwen 约定
    )

    # ---------- 4. 构造 labels ----------
    # 使用 -100 忽略 prompt 部分的 loss，只训练回答
    labels = (
        [-100] * len(instruction_enc["input_ids"]) +
        response_enc["input_ids"] +
        [tokenizer.pad_token_id]
    )

    # ---------- 5. 长度裁剪 ----------
    if len(input_ids) > MAX_LENGTH:           # MAX_LENGTH = 2048
        input_ids      = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels         = labels[:MAX_LENGTH]

    # ---------- 6. 返回 ----------
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# =========================================================
# 3. 推理辅助函数：把多轮 messages 送入模型，返回 assistant 回复
# =========================================================
def predict(
        messages: list[dict[str, str]],      # [{"role": "...", "content": "..."} ...]
        model: AutoModelForCausalLM,         # 已 eval() 的因果语言模型
        tokenizer: AutoTokenizer             # 与 model 配套的分词器
    ) -> str:
    """
    单条多轮对话推理；`messages` 应符合 ChatML 角色格式  
    """

    # ---------- 1. 选择设备 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 2. 将 messages 拼回 ChatML 字符串 ----------
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,          # 仅生成字符串
        add_generation_prompt=True  # 在末尾补 <|im_start|>assistant\n
    )

    # ---------- 3. tokenizer -> tensor ----------
    model_inputs = tokenizer(
        [prompt_text],           # batch size = 1，所以放进列表
        return_tensors="pt"
    ).to(device)                 # 同时迁移 input_ids / attention_mask

    # ---------- 4. 生成 ----------
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH   # 只限制新增 token 数
    )

    # ---------- 5. 去掉 prompt ----------
    generated_ids = [
        out_ids[len(in_ids):]           # 把前缀 prompt 切掉
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # ---------- 6. 解码为字符串 ----------
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True        # 移除 <|im_end|> 等特殊标记
    )[0]                                # batch size = 1，只取第 0 条

    return response

# =========================================================
# 4. 模型与 tokenizer 加载
# =========================================================
# （1）在 ModelScope 上下载权重到本地 cache_dir
model_dir = snapshot_download(
    "Qwen/Qwen3-1.7B",
    cache_dir="/root/autodl-tmp/",
    revision="master"   # 指向分支 / commit；默认 main
)

# （2）Transformers 方式加载
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,            # Qwen 提供 python 版 tokenizer
    trust_remote_code=True     # 允许加载自定义模型文件
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",         # 多 GPU 自动分布
    torch_dtype=torch.bfloat16 # bfloat16 节省显存；需硬件支持
)
model.enable_input_require_grads()  # 开梯度检查点前的固定操作

# =========================================================
# 5. 准备训练 / 验证数据集
# =========================================================
# —— 本节目标：把磁盘上的 JSONLines 文件 → pandas.DataFrame → 🤗Dataset
#    → 再经过我们前面写好的 `process_func` 转成张量字段。
# =========================================================

# 1) 原始数据（包含 question / answer / think 字段）的文件名
train_path_raw = "train.jsonl"
val_path_raw   = "val.jsonl"

# 2) 转换后（instruction / input / output）的文件名
train_path_fmt = "train_format.jsonl"
val_path_fmt   = "val_format.jsonl"

# ---------- 若格式化文件不存在则先转换 ----------
# os.path.exists(path) : 检查 path 是否真实存在；常见替代是 Path(path).exists()
if not os.path.exists(train_path_fmt):          # 只有第一次运行才会进入 if
    dataset_jsonl_transfer(train_path_raw, train_path_fmt)

if not os.path.exists(val_path_fmt):
    dataset_jsonl_transfer(val_path_raw, val_path_fmt)

# ---------- 读文件 → DataFrame ----------
# pandas.read_json:
#   • lines=True  告诉 pandas “按行读取，每行一个 JSON 对象”
#   • 返回的是 DataFrame：二维标表结构，行索引 + 列索引
train_df = pd.read_json(train_path_fmt, lines=True)  # shape: (N, 3)
val_df   = pd.read_json(val_path_fmt,   lines=True)

# ---------- DataFrame → 🤗Dataset ----------
# Dataset.from_pandas 会：
#   1. 深拷贝一份或零拷贝映射（取决于 python 对象类型）
#   2. 把列类型保持为 Python 原生，比如 list / str / int
from datasets import Dataset    # 防止上文未导入
train_hf = Dataset.from_pandas(train_df)
eval_hf  = Dataset.from_pandas(val_df)

# ---------- map：调用我们写好的 token 预处理 ----------
# • process_func 输入是单条 dict，输出必须仍是 dict
# • remove_columns 把原来的 "instruction/input/output" 列删掉，
#   否则返回张量列时会与旧列同名冲突
train_dataset = train_hf.map(
    process_func,
    remove_columns=train_hf.column_names,  # 等价于 ["instruction","input","output"]
    desc="Tokenizing train set"
)
eval_dataset = eval_hf.map(
    process_func,
    remove_columns=eval_hf.column_names,
    desc="Tokenizing eval set"
)

# =========================================================
# 6. 训练配置 & Trainer
# =========================================================
# —— 本节使用 Transformers 内置的 Trainer 高层 API。
#    1) 先实例化 TrainingArguments：存放超参数 + I/O 路径 + 日志设置
#    2) 再实例化 Trainer：把 model / dataset / collator 等聚合到一起
#    3) 调用 .train() 进入训练循环
# =========================================================
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# ---------- 1) TrainingArguments ----------
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B",   # 保存 checkpoint 的根目录
    per_device_train_batch_size=1,                     # 单 GPU / CPU 上的 micro-batch
    per_device_eval_batch_size=1,                      # eval 同理
    gradient_accumulation_steps=4,                     # 1×4=4，等效大 batch
    #   累积梯度常用于显存不足时；原则：total_effective_batch = per_device * grad_accu * n_gpu
    evaluation_strategy="steps",                       # 何时跑 eval：'no'/'epoch'/'steps'
    eval_steps=100,                                    # 每 100 个 optimizer.step() 做一次验证
    logging_steps=10,                                  # 每 10 步写一次日志（loss/学习率）
    num_train_epochs=2,                                # 数据集完整遍历 (=epoch) 次数
    save_steps=400,                                    # 每 400 步保存一次 ckpt
    learning_rate=1e-4,                                # AdamW 初始 lr
    save_on_each_node=True,                            # 多机时，每台都各自保存
    gradient_checkpointing=True,                       # 用计算换显存；速度≈0.75×
    report_to="swanlab",                               # 日志后端：可选 'tensorboard' 'wandb' ...
    run_name="qwen3-1.7B",                             # 实验在 SwanLab 中的子目录名称
    # 其余诸如 warmup_steps、lr_scheduler_type 等可在此继续添加
)

# ---------- 2) DataCollator ----------
# DataCollatorForSeq2Seq 专为 Encoder-Decoder 设计，但这里也能用：
#   • 会动态找出 batch 内最长序列长度 → 批量 pad
#   • 同步处理 input_ids 和 labels：labels 中 pad token 变 -100
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True          # 也可传 "longest" / "max_length"
)

# ---------- 3) Trainer ----------
trainer = Trainer(
    model=model,                      # 已加载的 Qwen3-1.7B
    args=args,                        # 刚刚的超参集合
    train_dataset=train_dataset,      # 已 tokenized
    eval_dataset=eval_dataset,
    data_collator=collator,           # 控制 batch 构造逻辑
    # 还可添加 compute_metrics=xxx 自定义评估函数
)

# ---------- 4) 开始训练 ----------
# 本质流程：
#   for epoch:
#       for step, batch in enumerate(train_dataloader):
#           loss = model(**batch)
#           loss.backward()
#           if step % grad_accum == 0:
#               optimizer.step(); scheduler.step(); optimizer.zero_grad()
#           if step % logging_steps == 0: log
#           if step % eval_steps    == 0: evaluate()
#           if step % save_steps    == 0: save_ckpt()
trainer.train()   # 内部已设置 model.train() 与随机数种子

# =========================================================
# 7. 训练后简单主观评估
# =========================================================
# —— 仅抽取验证集前三条做主观检查，并把结果同步到 SwanLab
# =========================================================
# read_json + head(3) ：pandas 中常用“取前几行”调试手段
test_df = pd.read_json(val_path_fmt, lines=True).head(3)

test_text_list = []      # 用于收集 swanlab.Text 对象 (相当于 wandb.Table 的 text 列)

# DataFrame.iterrows() -> (行索引, Series)；下划线 _ 表示“我不关心该变量”
for _, row in test_df.iterrows():
    # 手动组装 role/content 列表，复用我们前面写的 predict()
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user",   "content": row["input"]},
    ]
    response = predict(messages, model, tokenizer)

    # 生成可读字符串，用三引号自然保留换行
    display_text = f"""
Question: {row['input']}

LLM: {response}
"""
    # swanlab.Text 相当于富文本记录；在仪表盘里可折叠查看
    test_text_list.append(swanlab.Text(display_text))
    print(display_text)   # 同时打印到终端，方便 CLI 里观察

# ---------- 把三条结果一次性记录到仪表盘 ----------
swanlab.log({"Prediction": test_text_list})

# 结束当前 run，刷新剩余缓存文件
swanlab.finish()
