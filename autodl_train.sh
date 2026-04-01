# MIDI Melody Generator - AutoDL 一键运行脚本

# ==================== 配置 ====================
# 请根据你的 GPU 配置调整以下参数

# 模型配置
N_LAYER=8        # Transformer 层数 (4-12)
N_HEAD=8         # 注意力头数 (4-12)
N_EMBD=512       # 嵌入维度 (256-768)
MAX_POS=1024     # 最大位置嵌入

# 训练配置
NUM_EPOCHS=50    # 训练轮数 (建议 20-50)
BATCH_SIZE=4     # batch size (根据显存调整，16G显存建议4-8)
LR=3e-4          # 学习率
WEIGHT_DECAY=0.01
GRAD_ACCUM=4     # 梯度累积步数

# ==================== 环境准备 ====================
echo "=== 1. 安装依赖 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets miditok mido tqdm numpy

# ==================== 数据准备 ====================
echo "=== 2. 检查数据 ==="
if [ -d "tokenized_data" ]; then
    echo "tokenized_data 已存在，跳过数据处理"
else
    echo "开始处理数据..."
    python prepare_data.py
fi

# ==================== 开始训练 ====================
echo "=== 3. 开始训练 ==="
echo "模型配置: n_layer=$N_LAYER, n_head=$N_HEAD, n_embd=$N_EMBD"
echo "训练配置: epochs=$NUM_EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"

python train_model.py

echo "=== 训练完成 ==="
