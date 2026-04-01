#!/bin/bash
# AutoDL 一键运行脚本（直接执行版本）

# ==================== 配置 ====================
# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# 训练参数
N_LAYER=8
N_HEAD=8
N_EMBD=512
MAX_POS=2048
NUM_EPOCHS=50
BATCH_SIZE=4
LR=3e-4
WEIGHT_DECAY=0.01
GRAD_ACCUM=8

echo "============================================"
echo "MIDI Melody Generator - AutoDL 训练脚本"
echo "============================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "模型: n_layer=$N_LAYER, n_head=$N_HEAD, n_embd=$N_EMBD"
echo "训练: epochs=$NUM_EPOCHS, batch=$BATCH_SIZE, lr=$LR"
echo "============================================"

# ==================== 安装依赖 ====================
echo "[1/3] 安装依赖..."
pip install -q torch transformers datasets miditok mido tqdm numpy accelerate

# ==================== 检查数据 ====================
echo "[2/3] 检查数据..."
if [ ! -d "tokenized_data" ]; then
    echo "tokenized_data 不存在，开始处理数据..."
    python prepare_data.py
    if [ $? -ne 0 ]; then
        echo "数据处理失败!"
        exit 1
    fi
else
    echo "数据已存在，跳过"
fi

# ==================== 开始训练 ====================
echo "[3/3] 开始训练..."
python train_model.py

if [ $? -eq 0 ]; then
    echo "============================================"
    echo "训练完成!"
    echo "============================================"
else
    echo "训练失败!"
    exit 1
fi
