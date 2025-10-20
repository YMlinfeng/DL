"""
本脚本功能：
1. 从 ModelScope 平台拉取数据集 `krisfu/delicate_medical_r1_data` 的 train 切分。
2. 将数据随机打乱、按 9:1 的比例拆分为训练集(train.jsonl)与验证集(val.jsonl)。
3. 最终输出拆分后各子集的样本数量。
依赖：modelscope>=1.x, json(标准库), random(标准库)
"""

# ====================== 导包 ======================
from modelscope.msdatasets import MsDataset   # ModelScope 内置的 Dataset 管理工具
import json                                    # Python 标准库，用于序列化 / 反序列化 JSON
import random                                  # Python 标准库，提供随机数相关函数

# ====================== 随机数种子 ======================
random.seed(42)                                # 固定随机种子，保证可复现性
# 在不同机器上多次运行脚本时，shuffle 之后的数据顺序将保持一致

# ====================== 加载数据集 ======================
# `load` 返回一个可迭代对象(自定义 Dataset 类)，支持切分(split)与子集(subset_name)参数
ds = MsDataset.load(
    'krisfu/delicate_medical_r1_data',          # 数据集在 ModelScope 上的命名空间:https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data/dataPeview
    subset_name='default',                      # 选取的数据子集，缺省值一般也是 'default'
    split='train'                               # 只加载官方的 train 切分
)
# `ds` 类型: ModelScope 自定义 Dataset，类似于 list/Iterable

# ====================== 数据打乱 ======================
data_list = list(ds)                           # 将 Dataset 转成普通 list，方便索引、切片等操作
random.shuffle(data_list)                      # 就地(in-place)随机打乱；等价于 Fisher–Yates 洗牌算法

# ====================== 划分训练集 / 验证集 ======================
split_idx = int(len(data_list) * 0.9)          # 9:1 的分割点，取整保证索引为整数

train_data = data_list[:split_idx]             # 列表切片: 前 90% -> 训练集
val_data   = data_list[split_idx:]             # 列表切片: 剩余 10% -> 验证集

# ====================== 写出 train.jsonl ======================
# JSON Lines(.jsonl) 格式：一行一个 JSON 对象，便于流式读取/追加
with open('train.jsonl', 'w', encoding='utf-8') as f:  # 上下文管理器，自动关闭文件
    for item in train_data:                            # item 是 dict (示例数据结构)
        json.dump(item, f, ensure_ascii=False)         # ensure_ascii=False 保留非 ASCII 字符(中文)
        f.write('\n')                                  # 每个 JSON 对象占一行

# ====================== 写出 val.jsonl ======================
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# ====================== 日志打印 ======================
print(f"The dataset has been split successfully.")     # f-string，方便插入变量
print(f"Train Set Size：{len(train_data)}")            # len(list) 返回元素个数 -> int
print(f"Val Set Size：{len(val_data)}")