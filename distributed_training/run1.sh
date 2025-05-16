#!/bin/bash

# 配置环境（如果有 conda 或 virtualenv）
# source activate myenv

# # 启动分布式训练
# torchrun \
#   --nproc_per_node=2 \                            # 每台机器使用 4 张 GPU
#   --nnodes=2 \                                     # 总共有 2 台机器
#   --node_rank=1 \                                  # 当前是第 1 台机器
#   --master_addr="10.124.2.24" \                    # 主节点 IP（一般是 node_rank=0 的机器 IP）
#   --master_port=12345 \                            # 通信端口（确保未被占用）
#   09+_ddp.py \                            # 脚本名
#   --batch_size 128 \
#   --epochs 5 \
#   --lr 0.001


#!/bin/sh
# 运行 torchrun 的分布式训练脚本
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="10.136.84.198" \
  --master_port=12346 \
  ddp2.py \
  --batch_size 256 \
  --epochs 15 \
  --lr 0.001