#!/bin/bash
# 启动分布式训练的启动脚本
# 本脚本适用于两张 A100 显卡环境，且训练入口文件为 train_ddp.py
#
# 修改配置：
#   NUM_GPUS      - GPU 数量（这里设置为 2）
#   NUM_NODES     - 节点总数（单节点时为 1）
#   NODE_RANK     - 当前节点排名（单节点时为 0）
#   MASTER_ADDR   - 主节点地址（通常为 localhost）
#   MASTER_PORT   - 分布式通信端口（确保该端口未被使用）
#

NUM_GPUS=2
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=12348

# 设置环境变量（虽然我们在命令行中也指定了 --master_port，但这里也设置下以防万一）
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

echo "启动分布式训练："
echo "  GPU 数量       : ${NUM_GPUS}"
echo "  节点总数       : ${NUM_NODES}"
echo "  当前节点排名   : ${NODE_RANK}"
echo "  主节点地址     : ${MASTER_ADDR}"
echo "  主节点端口     : ${MASTER_PORT}"

# 显式通过 torchrun 参数指定 master_port
torchrun --nproc_per_node=${NUM_GPUS} \
         --nnodes=${NUM_NODES} \
         --node_rank=${NODE_RANK} \
         --master_port=${MASTER_PORT} \
         main.py