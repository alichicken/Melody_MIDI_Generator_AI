# -*- coding: utf-8 -*-
"""
MIDI 旋律生成推理脚本 - 工业级版本
- 高级和弦解析器
- 采样与防复读
- 音高/节奏约束
- MIDI 净化后处理
"""

import re
import random
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from transformers import GPT2LMHeadModel
from miditok import REMI
import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack


# 配置路径 - 使用本地训练好的模型
MODEL_DIR = Path("model_output")
OUTPUT_MIDI = Path("output.mid")
TOKENIZED_DATA_DIR = Path("tokenized_data")

# MIDI 音符范围 (C3-C7)
MIN_PITCH = 48
MAX_PITCH = 96


# ==================== 1. 高级和弦解析器 ====================

def parse_user_chords(input_str: str, tokenizer) -> List[int]:
    """
    解析用户输入的和弦字符串，返回 token 序列

    支持格式: Csus2 dm dim7 G7 Cmaj7 Am Gb7
    """
    if not input_str or not input_str.strip():
        return None

    print(f"解析和弦输入: {input_str}")

    # 标准化输入
    input_str = input_str.strip()

    # 解析每个和弦
    chord_parts = input_str.split()
    tokens = []

    # 添加起始标记
    bar_token = "Bar_None"
    if bar_token in tokenizer.vocab:
        tokens.append(tokenizer.vocab[bar_token])

    # 根音到 MIDI 音高的映射
    note_to_midi = {
        'c': 60, 'c#': 61, 'db': 61, 'd': 62, 'd#': 63, 'eb': 63,
        'e': 64, 'f': 65, 'f#': 66, 'gb': 66, 'g': 67, 'g#': 68, 'ab': 68,
        'a': 69, 'a#': 70, 'bb': 70, 'b': 71
    }

    # 和弦性质映射到相对音程
    chord_intervals = {
        'maj': [0, 4, 7],
        'maj7': [0, 4, 7, 11],
        'maj9': [0, 4, 7, 11, 14],
        'min': [0, 3, 7],
        'min7': [0, 3, 7, 10],
        'min9': [0, 3, 7, 10, 14],
        'dim': [0, 3, 6],
        'dim7': [0, 3, 6, 9],
        'aug': [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        '7': [0, 4, 7, 10],
        '9': [0, 4, 7, 10, 14],
    }

    for i, part in enumerate(chord_parts):
        part = part.strip()
        if not part:
            continue

        # 解析根音和性质
        match = re.match(r'^([a-gA-G][#b]?)(.*)$', part)
        if not match:
            continue

        root = match.group(1).upper()
        quality = match.group(2).strip() or 'maj'  # 默认大三和弦

        # 标准化根音
        root_lower = root.lower()
        if root_lower.endswith('#'):
            root_note = root_lower[:-1] + '#'
        elif root_lower.endswith('b'):
            root_note = root_lower[:-1] + 'b'
        else:
            root_note = root_lower

        # 简化性质映射
        if quality in ['m', 'min', 'minor']:
            quality = 'min'
        elif quality in ['dim', 'dim7']:
            quality = 'dim'
        elif quality in ['sus', 'sus2']:
            quality = 'sus2'
        elif quality in ['sus4']:
            quality = 'sus4'
        elif 'maj7' in quality or quality == 'maj':
            quality = 'maj'
        elif '7' in quality:
            quality = '7'
        else:
            quality = 'maj'  # 默认

        # 获取和弦音
        intervals = chord_intervals.get(quality, [0, 4, 7])
        base_note = note_to_midi.get(root_note, 60)

        # 生成 Position token (每个和弦占一个位置)
        position_token = f"Position_{i}"
        if position_token in tokenizer.vocab:
            tokens.append(tokenizer.vocab[position_token])

        # 生成和弦的根音 Pitch token
        chord_root_token = f"Pitch_{base_note}"
        if chord_root_token in tokenizer.vocab:
            tokens.append(tokenizer.vocab[chord_root_token])

        # 生成Duration
        if "Duration_1.0.0" in tokenizer.vocab:
            tokens.append(tokenizer.vocab["Duration_1.0.0"])

    print(f"和弦解析结果: {len(tokens)} tokens")
    return tokens if tokens else None


# ==================== 2. Logits Processors ====================

class PitchConstraintLogitsProcessor:
    """音高约束 - 限制 MIDI 音符范围 C3-C7 (48-96)"""

    def __init__(self, min_pitch: int = MIN_PITCH, max_pitch: int = MAX_PITCH, vocab=None):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.vocab = vocab or {}
        self.pitch_token_ids = self._get_pitch_token_ids()

    def _get_pitch_token_ids(self):
        """获取所有 Pitch token 的 id"""
        pitch_ids = []
        for name, idx in self.vocab.items():
            if name.startswith("Pitch_"):
                pitch_ids.append(idx)
        return set(pitch_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """处理 logits，限制音高范围"""
        for idx in self.pitch_token_ids:
            # 获取 token 名称
            token_name = None
            for name, tid in self.vocab.items():
                if tid == idx:
                    token_name = name
                    break

            if token_name and token_name.startswith("Pitch_"):
                try:
                    pitch = int(token_name.split("_")[1])
                    # 如果超出范围，设为 -inf
                    if pitch < self.min_pitch or pitch > self.max_pitch:
                        scores[idx] = float('-inf')
                except (ValueError, IndexError):
                    pass

        return scores


class RhythmConstraintLogitsProcessor:
    """节奏约束 - 防止过长静音"""

    def __init__(self, vocab=None, max_consecutive_time=4):
        self.vocab = vocab or {}
        self.max_consecutive_time = max_consecutive_time
        self.time_token_ids = self._get_time_token_ids()
        self.consecutive_count = 0

    def _get_time_token_ids(self):
        """获取 Time/Position token ids"""
        time_ids = []
        for name, idx in self.vocab.items():
            if name.startswith("Position_") or name.startswith("Time_"):
                time_ids.append(idx)
        return set(time_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """处理 logits，惩罚过长静音"""
        # 检查最近生成的 token
        if len(input_ids) > 0:
            last_token = input_ids[-1].item()
            if last_token in self.time_token_ids:
                self.consecutive_count += 1
            else:
                self.consecutive_count = 0

        # 如果连续生成太多 Time/Position token，惩罚它们
        if self.consecutive_count >= self.max_consecutive_time:
            for idx in self.time_token_ids:
                scores[idx] = float('-inf')
            # 重置
            self.consecutive_count = 0

        return scores


class RhythmBiasLogitsProcessor:
    """节奏偏置 - 给短时值音符增加正向偏置 (8分/16分音符)"""

    def __init__(self, vocab=None, bias: float = 2.0):
        self.vocab = vocab or {}
        self.bias = bias
        self.short_duration_ids = self._collect_short_duration_ids()
        self.position_ids = self._get_position_ids()
        print(f"RhythmBias: 找到 {len(self.short_duration_ids)} 个短时值 Duration tokens")

    def _collect_short_duration_ids(self):
        """动态收集短时值 Duration tokens (数值 < 1.0，即8分/16分音符)"""
        short_ids = []
        for name, idx in self.vocab.items():
            if name.startswith("Duration_"):
                # 格式: Duration_X.Y.Z，X 是小节数
                try:
                    beats = float(name.split("_")[1].split(".")[0])
                    if beats < 1.0:  # 短于1拍 = 8分或更短
                        short_ids.append(idx)
                except (ValueError, IndexError, KeyError):
                    pass
        return short_ids

    def _get_position_ids(self):
        """获取 Position token ids"""
        pos_ids = []
        for name, idx in self.vocab.items():
            if name.startswith("Position_"):
                pos_ids.append(idx)
        return set(pos_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        处理 logits，给短时值音符增加偏置
        防错指南：scores 形状 [batch_size, vocab_size]，使用正确的切片操作
        """
        # 检查前一个 token 是否是 Position
        if len(input_ids) > 0:
            last_token = input_ids[-1].item()
            if last_token in self.position_ids:
                # 前一个 token 是 Position，给短时值增加正向偏置
                # scores 形状: [batch_size, vocab_size]
                if self.short_duration_ids:
                    scores[:, self.short_duration_ids] += self.bias

        return scores


class GrammarEnforcerLogitsProcessor:
    """强制语法处理器 - 防止模型陷入非音符死循环，强制生成音符"""

    def __init__(self, vocab=None, force_threshold: int = 2):
        self.vocab = vocab or {}
        self.force_threshold = force_threshold
        self.pitch_ids = self._get_pitch_ids()
        self.non_note_ids = self._get_non_note_ids()
        self.consecutive_non_note = 0
        print(f"GrammarEnforcer: {len(self.pitch_ids)} pitch tokens, {len(self.non_note_ids)} control tokens")

    def _get_pitch_ids(self):
        """获取所有 Pitch token ids"""
        return [idx for name, idx in self.vocab.items() if name.startswith("Pitch_")]

    def _get_non_note_ids(self):
        """获取所有非音符 token ids (Bar, Position, Time, Velocity, Duration 等)"""
        ids = []
        for name, idx in self.vocab.items():
            if (name.startswith("Bar_") or name.startswith("Position_") or
                name.startswith("Time_") or name.startswith("Velocity_") or
                name.startswith("Duration_") or name.startswith("Tempo_") or
                name.startswith("Program_")):
                ids.append(idx)
        return ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        核心逻辑：如果连续生成多个非音符 token，强制生成音符
        """
        if len(input_ids) == 0:
            return scores

        # 检查最后一个 token 是否是非音符 token
        last_token = input_ids[-1].item()
        if last_token in self.non_note_ids:
            self.consecutive_non_note += 1
        else:
            self.consecutive_non_note = 0

        # 如果连续生成超过 threshold 个非音符 token，强制生成音符
        if self.consecutive_non_note >= self.force_threshold:
            # 方法1：禁止所有非音符 token
            for idx in self.non_note_ids:
                scores[idx] = float('-inf')

            # 方法2：给所有 pitch token 极大正向偏置
            if self.pitch_ids:
                scores[:, self.pitch_ids] += 50.0

            print(f"  [GrammarEnforcer] 连续 {self.consecutive_non_note} 个控制token，强制生成音符!")
            self.consecutive_non_note = 0  # 重置

        return scores


class RepetitionPenaltyLogitsProcessor:
    """防复读处理器"""

    def __init__(self, penalty: float = 1.2, ngram_size: int = 8):
        self.penalty = penalty
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """对重复的 n-gram 施加惩罚"""
        if input_ids.shape[0] < self.ngram_size:
            return scores

        # 获取最近的 n-gram
        last_ngram = input_ids[-self.ngram_size:].tolist()

        # 遍历所有 token
        for i in range(scores.shape[0]):
            # 检查这个 token 是否会导致重复
            for j in range(len(last_ngram)):
                if input_ids[-self.ngram_size+j:].tolist() == last_ngram[:self.ngram_size-j]:
                    # 如果匹配，施加惩罚
                    if scores[i] > 0:
                        scores[i] = scores[i] / self.penalty
                    else:
                        scores[i] = scores[i] * self.penalty
                    break

        return scores


# ==================== 3. MIDI 后处理 ====================

def clean_midi_notes(mid: MidiFile, min_duration_ticks: int = 30) -> MidiFile:
    """清理 MIDI - 单音符化 + 清理碎音"""
    print("\n执行 MIDI 净化...")

    if not mid.tracks:
        return mid

    # 只处理第一个轨道
    track = mid.tracks[0]

    # 解析音符事件
    note_events = []  # (time, note, velocity, is_note_on)
    current_time = 0

    for msg in track:
        if msg.type == 'note_on':
            current_time += msg.time
            if msg.velocity > 0:
                note_events.append((current_time, msg.note, msg.velocity, True))
            else:
                note_events.append((current_time, msg.note, 0, False))

    # 按时间分组，找出同时发生的音符
    time_to_notes = {}
    for time, note, velocity, is_on in note_events:
        if time not in time_to_notes:
            time_to_notes[time] = []
        time_to_notes[time].append((note, velocity, is_on))

    # 单音符化：每个时间点只保留音高最高的音符
    filtered_events = []
    for time in sorted(time_to_notes.keys()):
        notes = time_to_notes[time]

        # 分离 note_on 和 note_off
        note_ons = [(n, v) for n, v, is_on in notes if is_on and v > 0]
        note_offs = [(n, v) for n, v, is_on in notes if not is_on or v == 0]

        if note_ons:
            # 只保留音高最高的音符
            highest_note = max(note_ons, key=lambda x: x[0])
            filtered_events.append((time, highest_note[0], highest_note[1], True))

        if note_offs:
            # 也只保留一个 note_off
            filtered_events.append((time, note_offs[0][0], 0, False))

    # 排序
    filtered_events.sort(key=lambda x: x[0])

    # 重建轨道
    new_track = MidiTrack()
    prev_time = 0

    for time, note, velocity, is_on in filtered_events:
        delta_time = time - prev_time
        new_track.append(Message('note_on', note=note, velocity=velocity, time=delta_time))
        prev_time = time

    # 替换轨道
    mid.tracks[0] = new_track

    print(f"MIDI 净化完成")
    return mid


# ==================== 模型加载 ====================

def load_model_and_tokenizer():
    """加载模型和 Tokenizer"""
    print("加载模型和 Tokenizer...")

    # 加载 GPT-2 模型
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
    model.eval()

    # 从保存的 tokenizer.json 加载 tokenizer
    tokenizer = REMI.from_pretrained(str(MODEL_DIR))

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    return model, tokenizer


def get_training_prompt(tokenizer, sample_idx: int = None) -> List[int]:
    """从训练数据获取 prompt"""
    from datasets import load_from_disk

    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    # 如果没有指定 sample_idx，随机选择
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset["train"]) - 1)
    # 取一条训练数据的 tokens
    tokens = dataset["train"][sample_idx]["tokens"][:256]
    return tokens


# ==================== 旋律生成 ====================

def generate_melody(model, tokenizer, chord_input: str = None, num_tokens: int = 256,
                    temperature: float = 0.8, top_k: int = 30, top_p: float = 0.88):
    """生成旋律 - 使用工业级配置
    参数:
        temperature: 采样温度，越低越确定性，越高越随机 (0.1-2.0)
        top_k: top-k 采样，限制候选词数量 (1-100)
        top_p: 核采样概率，越高越保守 (0-1)
    """
    print("\n开始生成旋律...")

    # 每次生成使用不同的 prompt 和随机种子
    best_tokens = None
    best_note_count = 0

    # 尝试多次生成
    for attempt in range(10):
        # 每次使用不同的随机 prompt
        prompt = get_training_prompt(tokenizer)
        torch.manual_seed(random.randint(0, 1000000) + attempt)
        input_ids = torch.tensor([prompt], dtype=torch.long)

        # 创建 LogitsProcessors
        pitch_processor = PitchConstraintLogitsProcessor(
            min_pitch=MIN_PITCH, max_pitch=MAX_PITCH, vocab=tokenizer.vocab
        )
        grammar_enforcer = GrammarEnforcerLogitsProcessor(vocab=tokenizer.vocab, force_threshold=2)

        with torch.no_grad():
            # 流行旋律采样配置
            outputs = model.generate(
                input_ids,
                max_length=min(len(prompt) + num_tokens, 512),
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=0,
                eos_token_id=0,
            )

        generated_tokens = outputs[0].tolist()

        # 统计 Pitch token 数量
        pitch_count = 0
        for tid in generated_tokens:
            for name, idx in tokenizer.vocab.items():
                if idx == tid and name.startswith("Pitch_"):
                    pitch_count += 1
                    break

        print(f"尝试 {attempt + 1}: 生成了 {len(generated_tokens)} 个 tokens, {pitch_count} 个 Pitch tokens")

        if pitch_count > best_note_count:
            best_note_count = pitch_count
            best_tokens = generated_tokens

    # 如果模型仍然没有生成任何音符，返回原始 prompt（不注入假数据）
    if best_note_count == 0:
        print("模型未能生成任何 Pitch tokens，请检查生成策略")
        return prompt

    print(f"最佳结果: {best_note_count} 个 Pitch tokens")
    return best_tokens if best_tokens else prompt


def clean_tokens(tokens, tokenizer):
    """清理和修复不合法 token"""
    vocab_size = tokenizer.vocab_size
    valid_tokens = []

    for token in tokens:
        if 0 <= token < vocab_size:
            valid_tokens.append(token)

    print(f"清理后剩余 {len(valid_tokens)} 个 tokens")
    return valid_tokens


# ==================== MIDI 转换 ====================

def tokens_to_midi(tokenizer, tokens, output_path: Path):
    """将 tokens 转换为 MIDI 文件"""
    print("\n解码为 MIDI...")

    # 转换为 token 名称
    token_names = []
    for tid in tokens:
        for name, idx in tokenizer.vocab.items():
            if idx == tid:
                token_names.append(name)
                break
        else:
            token_names.append(f"Unknown_{tid}")

    print(f"前 30 个 token: {token_names[:30]}")

    # 构建 MIDI
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Generated Melody', time=0))
    track.append(Message('program_change', program=0, channel=0, time=0))

    time_scale = 480 * 2
    current_time = 0
    note_count = 0
    current_velocity = 80
    current_duration = time_scale // 4

    i = 0
    while i < len(token_names):
        token_name = token_names[i]

        if token_name.startswith("Pitch_"):
            try:
                pitch = int(token_name.split("_")[1])
                # 应用音高约束
                pitch = max(MIN_PITCH, min(MAX_PITCH, pitch))
                pitch = max(21, min(108, pitch))

                track.append(Message('note_on', note=pitch, velocity=current_velocity, time=current_time))
                track.append(Message('note_on', note=pitch, velocity=0, time=current_duration))
                current_time = 0
                note_count += 1

            except (ValueError, IndexError):
                pass

        elif token_name.startswith("Velocity_"):
            try:
                vel_val = int(token_name.split("_")[1])
                current_velocity = max(1, min(127, int((vel_val / 16) * 127)))
            except (ValueError, IndexError):
                pass

        elif token_name.startswith("Duration_"):
            try:
                dur_val = int(token_name.split("_")[1])
                current_duration = int(dur_val * time_scale / 4)
            except (ValueError, IndexError):
                pass

        elif token_name.startswith("Bar_"):
            current_time = 0

        i += 1

    if note_count > 0:
        mid.tracks.append(track)

        # 执行 MIDI 净化
        mid = clean_midi_notes(mid, min_duration_ticks=30)

        mid.save(str(output_path))
        print(f"MIDI 已保存到 {output_path} (包含 {note_count} 个音符)")
        return True
    else:
        print("未生成任何音符")
        return fallback_simple_midi(output_path)


def fallback_simple_midi(output_path: Path):
    """降级方案：生成简单 MIDI"""
    print("\n使用降级方案生成简单 MIDI...")

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Simple Melody', time=0))
    track.append(Message('program_change', program=0, channel=0, time=0))

    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    time_scale = 480
    current_time = 0

    for note in notes:
        track.append(Message('note_on', note=note, velocity=80, time=current_time))
        track.append(Message('note_on', note=note, velocity=0, time=time_scale))
        current_time = 0

    mid.tracks.append(track)
    mid.save(str(output_path))
    print(f"简单 MIDI 已保存到 {output_path}")
    return True


# ==================== 主函数 ====================

def main(chord_input: str = None, temperature: float = 0.8, top_k: int = 30, top_p: float = 0.88, num_tokens: int = 256):
    """主函数
    参数:
        chord_input: 和弦进行字符串
        temperature: 采样温度 (0.1-2.0)
        top_k: top-k 采样
        top_p: 核采样概率
        num_tokens: 生成 token 数量
    """
    global tokenizer

    # 1. 加载模型和 Tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2. 生成旋律
    tokens = generate_melody(model, tokenizer, chord_input, num_tokens=num_tokens,
                             temperature=temperature, top_k=top_k, top_p=top_p)

    # 3. 清理 tokens
    tokens = clean_tokens(tokens, tokenizer)

    # 4. 转换为 MIDI
    success = tokens_to_midi(tokenizer, tokens, OUTPUT_MIDI)

    if not success:
        fallback_simple_midi(OUTPUT_MIDI)

    print("\n完成!")
    return OUTPUT_MIDI


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Melody Generator")
    parser.add_argument("--chords", type=str, default=None, help="和弦进行，例如: Cmaj7 Am dm G7")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度 (0.1-2.0, 越低越确定性)")
    parser.add_argument("--top_k", type=int, default=30, help="top-k 采样 (1-100)")
    parser.add_argument("--top_p", type=float, default=0.88, help="核采样概率 (0-1)")
    parser.add_argument("--num_tokens", type=int, default=256, help="生成 token 数量")
    args = parser.parse_args()

    main(chord_input=args.chords, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, num_tokens=args.num_tokens)