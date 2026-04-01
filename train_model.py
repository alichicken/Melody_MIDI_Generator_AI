"""
MIDI 旋律生成模型训练脚本
- 使用 miditok REMI 编码的 Tokenizer
- 构建轻量级 GPT-2 模型
- 训练并保存到 ./model_output
"""

import os
import gc
import torch
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader

from miditok import REMI, TokenizerConfig
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    Trainer,
    TrainingArguments,
)
import numpy as np
from typing import Any, Dict, List, Union
from torch import nn


# 配置路径
OUTPUT_DIR = Path("model_output")
TOKENIZED_DATA_DIR = Path("tokenized_data")

# 超参数配置
MODEL_CONFIG = {
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "max_position_embeddings": 2048,
}

TRAINING_CONFIG = {
    "num_train_epochs": 50,
    "per_device_train_batch_size": 2,  # 减小 batch size 避免 OOM
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 8,  # 增大梯度累积以补偿小 batch
}


class MIDIDataCollator:
    """自定义 MIDI 数据收集器"""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取 input_ids
        input_ids = [f["input_ids"] for f in features]

        # 找到最大长度
        max_len = max(len(ids) for ids in input_ids)

        # padding
        batch_input_ids = []
        attention_masks = []

        for ids in input_ids:
            padding_len = max_len - len(ids)
            padded_ids = ids + [self.pad_token_id] * padding_len
            mask = [1] * len(ids) + [0] * padding_len

            batch_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(batch_input_ids, dtype=torch.long),
        }


def initialize_and_save_tokenizer():
    """初始化并保存 Tokenizer"""
    print("初始化 Tokenizer...")

    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=False,
        chord_tokens_with_root_note=True,
    )
    tokenizer = REMI(config)

    # 保存 tokenizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {vocab_size}")

    return tokenizer, vocab_size


def load_and_process_data():
    """加载并处理数据"""
    print("\n加载数据集...")

    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    print(f"数据集大小: train={len(dataset['train'])}, test={len(dataset['test'])}")

    # 截断 tokens 到 2048 (与数据集中的 max_tokens 一致)
    max_tokens = 2048
    def truncate_tokens(example):
        return {
            "input_ids": example["tokens"][:max_tokens],
            "attention_mask": [1] * min(len(example["tokens"]), max_tokens),
        }

    print(f"\n截断 tokens 到 {max_tokens} 长度...")
    train_dataset = dataset["train"].map(
        truncate_tokens,
        remove_columns=["tokens", "song_id", "original_song", "transpose"],
    )
    eval_dataset = dataset["test"].map(
        truncate_tokens,
        remove_columns=["tokens", "song_id", "original_song", "transpose"],
    )

    print(f"处理后: train={len(train_dataset)}, test={len(eval_dataset)}")
    print(f"示例 input_ids 长度: {len(train_dataset[0]['input_ids'])}")

    return train_dataset, eval_dataset


def create_model(vocab_size, max_pos_emb):
    """创建 GPT-2 模型"""
    print("\n创建模型...")

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_pos_emb,
        n_layer=MODEL_CONFIG["n_layer"],
        n_head=MODEL_CONFIG["n_head"],
        n_embd=MODEL_CONFIG["n_embd"],
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )

    model = GPT2LMHeadModel(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def train():
    """训练模型"""
    # 1. 初始化并保存 Tokenizer
    tokenizer, vocab_size = initialize_and_save_tokenizer()

    # 2. 加载并处理数据
    train_dataset, eval_dataset = load_and_process_data()

    # 3. 创建模型
    model = create_model(vocab_size, MODEL_CONFIG["max_position_embeddings"])

    # 4. 配置 DataCollator（使用自定义的 MIDIDataCollator）
    # 设置 tokenizer 的 pad_token_id
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    pad_id = tokenizer.pad_token_id
    data_collator = MIDIDataCollator(pad_token_id=pad_id)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=TRAINING_CONFIG.get("gradient_accumulation_steps", 8),
    )

    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 7. 开始训练
    print("\n开始训练...")
    trainer.train()

    # 8. 保存模型
    print("\n保存模型...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\n训练完成! 模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
