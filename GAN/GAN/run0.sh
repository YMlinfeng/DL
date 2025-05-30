#!/bin/bash

# 配置环境（如果有 conda 或 virtualenv）
# source activate myenv

# # 启动分布式训练
# torchrun \
#   --nproc_per_node=2 \                            # 每台机器使用 4 张 GPU
#   --nnodes=2 \                                     # 总共有 2 台机器
#   --node_rank=0 \                                  # 当前是第 0 台机器
#   --master_addr="10.124.2.24" \                    # 主节点 IP（一般是 node_rank=0 的机器 IP）
#   --master_port=12345 \                            # 通信端口（确保未被占用）
#   09+_ddp.py \                            # 脚本名
#   --batch_size 128 \
#   --epochs 5 \
#   --lr 0.001

torchrun \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=12346 \
  gan2.py \
  # --batch_size 256 \
  # --epochs  \
  # --lr 0.001