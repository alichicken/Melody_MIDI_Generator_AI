# -*- coding: utf-8 -*-
"""
POP909 数据处理脚本 - Block Chord 版本 (FIXED)
- 使用 symusic 库创建 MIDI (兼容 miditok)
- 关键修复：确保同时起始的音符有相同的 duration，让 detect_chords 能正确识别和弦
"""

import os
import gc
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

from miditok import REMI, TokenizerConfig
from miditok.constants import CHORD_MAPS
from datasets import Dataset as HFDataset
from tqdm import tqdm
from symusic import Score, Track, Note

# 配置路径
DATASET_DIR = Path("dataset/POP909-Dataset-master/POP909")
OUTPUT_DIR = Path("tokenized_data")
SANITY_CHECK_PATH = Path("final_sanity_check.mid")

# MIDI 配置
TPB = 480

# 和弦名称到音高的映射 (C4 = 60)
CHORD_NOTES = {
    'maj': [0, 4, 7],
    'maj7': [0, 4, 7, 11],
    'add9': [0, 4, 7, 14],
    'min': [0, 3, 7],
    'm': [0, 3, 7],
    'min7': [0, 3, 7, 10],
    '7': [0, 4, 7, 10],
    'maj7_alt': [0, 4, 7, 11],
    'min7_alt': [0, 3, 7, 10],
    'dim': [0, 3, 6],
    'dim7': [0, 3, 6, 9],
    'aug': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'N': [0, 4, 7],
    '5': [0, 7],
}

ROOT_MAP = {
    'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11
}


def parse_root(chord_str: str) -> int:
    """解析和弦根音，返回半音偏移"""
    root = chord_str.split(':')[0].upper()
    if root in ROOT_MAP:
        return ROOT_MAP[root]
    if root[0] == '#':
        return ROOT_MAP.get(root[1], 0) + 1
    if root[0] == 'B' and len(root) > 1:
        return ROOT_MAP.get(root[1], 0) - 1
    return 0


def parse_chord(chord_str: str) -> Tuple[int, List[int]]:
    """解析和弦，返回(根音半音, 音符列表)"""
    try:
        if ':' in chord_str:
            root_part, chord_type = chord_str.split(':')
        else:
            root_part = chord_str[0]
            chord_type = chord_str[1:] if len(chord_str) > 1 else 'maj'

        root_offset = parse_root(chord_str)
        chord_type = chord_type.lower().strip()
        notes = CHORD_NOTES.get(chord_type, CHORD_NOTES['maj'])
        return root_offset, notes
    except:
        return 0, CHORD_NOTES['maj']


def parse_chord_file(chord_path: Path) -> List[Tuple[float, float, str]]:
    """解析 chord_midi.txt 文件"""
    chords = []
    if not chord_path.exists():
        return chords

    with open(chord_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                chord = parts[2]
                chords.append((start_time, end_time, chord))
    return chords


def create_block_chord_track_symusic(chords: List[Tuple[float, float, str]],
                                      transpose: int,
                                      base_note: int = 48) -> List[Note]:
    """使用 symusic.Note 创建柱式和弦音符列表

    关键修复：所有同时起始的音符必须使用相同的 duration，
    这样 detect_chords 才会认为它们是一个和弦（通过 duration 一致性检查）。
    """
    notes = []

    for start_sec, end_sec, chord_str in chords:
        root_offset, note_pattern = parse_chord(chord_str)
        duration_ticks = int((end_sec - start_sec) * TPB * 4)
        duration_ticks = max(duration_ticks, TPB // 2)  # 最小半拍

        start_tick = int(start_sec * TPB * 4)

        for note_offset in note_pattern:
            pitch = base_note + root_offset + note_offset + transpose
            pitch = max(0, min(127, pitch))
            # duration 以 tick 为单位（symusic 使用 time, duration 格式）
            note = Note(time=start_tick, duration=duration_ticks, pitch=pitch, velocity=80)
            notes.append(note)

    return notes


def transpose_score(score: Score, semitones: int) -> Score:
    """对整个 Score 进行全局移调"""
    new_score = Score()
    new_score.tpq = score.tpq
    new_score.tempos = score.tempos.copy()
    new_score.time_signatures = score.time_signatures.copy()

    for track in score.tracks:
        new_track = Track(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )
        new_track.notes = Note.from_numpy(
            np.array([n.time for n in track.notes], dtype=np.int32),
            np.array([n.duration for n in track.notes], dtype=np.int32),
            np.array([max(0, min(127, n.pitch + semitones)) for n in track.notes], dtype=np.int8),
            np.array([n.velocity for n in track.notes], dtype=np.int8),
        )
        new_track.pitch_bends = track.pitch_bends.copy()
        new_track.controls = track.controls.copy()
        new_score.tracks.append(new_track)

    return new_score


def validate_tokens(tokens: List[int], tokenizer: REMI) -> Tuple[bool, Dict]:
    """验证 token 序列是否有效"""
    if not tokens or len(tokens) < 10:
        return False, {}

    token_counts = {}
    for tid in tokens:
        token_counts[tid] = token_counts.get(tid, 0) + 1

    unique_tokens = set(tokens)
    if len(unique_tokens) < 10:
        return False, {'reason': 'unique_tokens_too_few'}

    pitch_count = sum(count for tid, count in token_counts.items()
                    if 5 <= tid <= 80)

    chord_count = 0
    chord_names_in_sample = set()
    vocab_map = {v: k for k, v in tokenizer.vocab.items()}
    for tid, count in token_counts.items():
        name = vocab_map.get(tid, '')
        if name.startswith('Chord_'):
            chord_count += count
            chord_names_in_sample.add(name)

    stats = {
        'pitch_count': pitch_count,
        'chord_count': chord_count,
        'unique_tokens': len(unique_tokens),
        'chord_names': sorted(chord_names_in_sample)
    }

    if pitch_count < 5:
        return False, {'reason': 'pitch_too_few', 'stats': stats}

    return True, stats


def print_token_names(tokens: List[int], tokenizer: REMI, count: int = 100):
    """打印 token 的实际名称"""
    names = []
    vocab_map = {v: k for k, v in tokenizer.vocab.items()}
    for tid in tokens[:count]:
        name = vocab_map.get(tid, f"ID_{tid}")
        names.append(name)
    return names


def process_all_songs():
    """处理所有歌曲"""
    print("=" * 50)
    print("开始处理 POP909 数据集 (Block Chord 修复版)...")
    print("=" * 50)

    song_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    print(f"找到 {len(song_dirs)} 首歌曲")

    # 配置 Tokenizer
    print("\n初始化 Tokenizer (use_chords=True)...")
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=False,
        chord_tokens_with_root_note=True,
    )
    tokenizer = REMI(config)
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Chord maps: {list(CHORD_MAPS.keys())}")

    # 显示和弦 token
    chord_tokens = [(name, idx) for name, idx in tokenizer.vocab.items() if 'Chord' in name]
    print(f"Chord tokens 数量: {len(chord_tokens)}")
    print(f"  示例: {sorted(chord_tokens)[:10]}")

    # 加载所有歌曲数据
    print("\n[1/4] 加载歌曲数据...")
    song_data = []

    for song_dir in tqdm(song_dirs, desc="加载中"):
        song_id = song_dir.name
        mid_path = song_dir / f"{song_id}.mid"
        chord_path = song_dir / "chord_midi.txt"

        if not mid_path.exists():
            continue

        try:
            original_score = Score(str(mid_path))
            chords = parse_chord_file(chord_path)

            if chords:
                song_data.append({
                    'song_id': song_id,
                    'score': original_score,
                    'chords': chords,
                })
        except Exception as e:
            print(f"  加载失败 {song_id}: {e}")
            continue

    print(f"成功加载 {len(song_data)} 首歌曲 (有和弦信息)")

    # 处理移调和 tokenization
    print("\n[2/4] Tokenization 和移调...")

    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    valid_count = 0
    skipped_no_chord = 0
    skipped_other = 0

    for song_info in tqdm(song_data, desc="处理中"):
        song_id = song_info['song_id']
        original_score = song_info['score']
        chords = song_info['chords']

        for transpose_step in range(12):
            try:
                # 1. 移调原 Score (包括所有轨道)
                transposed_score = transpose_score(original_score, transpose_step)

                # 2. 生成柱式和弦轨道
                chord_notes = create_block_chord_track_symusic(
                    chords, transpose_step, base_note=48
                )

                # 3. 添加和弦轨道到 Score
                chord_track = Track(
                    program=0,
                    is_drum=False,
                    name="CHORD",
                )
                chord_track.notes = Note.from_numpy(
                    np.array([n.time for n in chord_notes], dtype=np.int32),
                    np.array([n.duration for n in chord_notes], dtype=np.int32),
                    np.array([n.pitch for n in chord_notes], dtype=np.int8),
                    np.array([n.velocity for n in chord_notes], dtype=np.int8),
                )
                transposed_score.tracks.append(chord_track)

                # 4. 保存临时 MIDI（使用 symusic）
                temp_midi_path = temp_dir / f"{song_id}_t{transpose_step}.mid"
                transposed_score.dump_midi(str(temp_midi_path))

                # 5. Tokenization
                tokens = tokenizer(str(temp_midi_path))

                if not tokens or len(tokens) == 0:
                    skipped_other += 1
                    if temp_midi_path.exists():
                        temp_midi_path.unlink()
                    continue

                # Handle different tokenization outputs:
                # - use_programs=True: single TokSequence
                # - use_programs=False: list of TokSequence (one per track)
                if isinstance(tokens, list):
                    # Concatenate all track tokens into one sequence
                    combined_ids = []
                    for seq in tokens:
                        if hasattr(seq, 'ids') and seq.ids:
                            combined_ids.extend(seq.ids)
                    token_ids = combined_ids
                elif hasattr(tokens, 'ids'):
                    token_ids = tokens.ids
                else:
                    token_ids = list(tokens)

                # 6. 验证
                is_valid, stats = validate_tokens(token_ids, tokenizer)
                if not is_valid:
                    reason = stats.get('reason', 'unknown')
                    if reason == 'no_chord':
                        skipped_no_chord += 1
                    else:
                        skipped_other += 1
                    if temp_midi_path.exists():
                        temp_midi_path.unlink()
                    continue

                all_tokens.append({
                    'song_id': f"{song_id}_t{transpose_step}",
                    'original_song': song_id,
                    'transpose': transpose_step,
                    'tokens': token_ids,
                    'stats': stats
                })
                valid_count += 1

                # 删除临时文件
                if temp_midi_path.exists():
                    temp_midi_path.unlink()

            except Exception as e:
                print(f"  错误: {song_id} 移调 {transpose_step} - {e}")
                skipped_other += 1
                continue

        gc.collect()

    print(f"\n处理完成:")
    print(f"  有效样本: {valid_count}")
    print(f"  跳过 (无和弦): {skipped_no_chord}")
    print(f"  跳过 (其他): {skipped_other}")
    print(f"  总共: {len(all_tokens)} 个样本")

    # Sanity Check
    print("\n[3/4] Sanity Check...")
    if all_tokens:
        first = all_tokens[0]
        print(f"第一个样本: {first['song_id']}")
        print(f"Token 数量: {len(first['tokens'])}")
        print(f"Stats: {first['stats']}")

        # 打印 token 名称
        print(f"\n前 100 个 tokens 名称:")
        names = print_token_names(first['tokens'], tokenizer, 100)
        for i in range(0, len(names), 10):
            print(f"  {names[i:i+10]}")

        # 解码 MIDI
        try:
            mid = tokenizer.decode([first['tokens']])
            if hasattr(mid, 'dump_midi'):
                mid.dump_midi(str(SANITY_CHECK_PATH))
                print(f"\n✅ 已保存: {SANITY_CHECK_PATH}")
            elif hasattr(mid, 'save'):
                mid.save(str(SANITY_CHECK_PATH))
                print(f"\n✅ 已保存: {SANITY_CHECK_PATH}")
        except Exception as e:
            print(f"\n⚠️ 解码错误: {e}")
            import traceback
            traceback.print_exc()

    # 创建 Dataset
    print("\n[4/4] 创建 HuggingFace 数据集...")

    tokens_list = [item['tokens'] for item in all_tokens]
    song_ids = [item['song_id'] for item in all_tokens]
    original_songs = [item['original_song'] for item in all_tokens]
    transposes = [item['transpose'] for item in all_tokens]

    # 截断
    max_tokens = 2048
    tokens_list = [tokens[:max_tokens] for tokens in tokens_list]

    print(f"Token 序列长度: min={min(len(t) for t in tokens_list)}, "
          f"max={max(len(t) for t in tokens_list)}, "
          f"avg={sum(len(t) for t in tokens_list)/len(tokens_list):.0f}")

    hf_dataset = HFDataset.from_dict({
        'tokens': tokens_list,
        'song_id': song_ids,
        'original_song': original_songs,
        'transpose': transposes
    })

    print("\n划分数据集 (95% train, 5% test)...")
    dataset_dict = hf_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)

    # 保存
    print(f"\n保存到 {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if (OUTPUT_DIR / "train").exists():
        shutil.rmtree(OUTPUT_DIR / "train")
    if (OUTPUT_DIR / "test").exists():
        shutil.rmtree(OUTPUT_DIR / "test")

    dataset_dict.save_to_disk(str(OUTPUT_DIR))

    print("=" * 50)
    print("完成!")
    print(f"训练集: {len(dataset_dict['train'])} 样本")
    print(f"验证集: {len(dataset_dict['test'])} 样本")
    print(f"Sanity check: {SANITY_CHECK_PATH}")

    # 最终统计
    total_chord_tokens = sum(
        sum(1 for tid in item['tokens']
            for name, idx in tokenizer.vocab.items()
            if idx == tid and 'Chord' in name)
        for item in all_tokens
    )
    print(f"数据集中 Chord tokens 总数: {total_chord_tokens}")
    print("=" * 50)


if __name__ == "__main__":
    process_all_songs()