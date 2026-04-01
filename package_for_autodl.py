# -*- coding: utf-8 -*-
"""
AutoDL 部署打包脚本
将 MIDI 生成项目打包为可在 AutoDL 上运行的格式
"""

import os
import shutil
import zipfile
from pathlib import Path

# 项目根目录
PROJECT_DIR = Path("E:/Project/MIDI_Generator_AI")
OUTPUT_DIR = Path("E:/Project/MIDI_Generator_AI/autodl_package")

# 需要打包的文件和目录
INCLUDE_FILES = [
    "prepare_data.py",
    "train_model.py",
    "inference.py",
    "requirements.txt",
    "autodl_run.sh",
    "autodl_train.sh",
]

INCLUDE_DIRS = [
    "tokenized_data",  # 预处理后的数据（包含 Chord tokens）
]

# 排除的文件/目录
EXCLUDE = [
    "model_output",
    "output.mid",
    "__pycache__",
    ".git",
    "final_sanity_check.mid",
    "test_block.mid",
    "test_chord.mid",
    "temp_test.mid",
    "temp_test_fixed.mid",
    "temp_test_v2.mid",
    "nul",
    "dataset",           # 不打包原始数据集（太大）
    "python",
    "autodl_package",
    "temp",              # 不打包临时目录
    "*.pyc",
    "*.pyo",
]


def should_include(path: Path) -> bool:
    """判断路径是否应该包含"""
    name = path.name
    # 排除特定模式（完整匹配或前缀匹配）
    for pattern in EXCLUDE:
        if pattern.startswith("*"):
            # 通配符模式：*.pyc 匹配 .pyc 结尾
            if name.endswith(pattern[1:]):
                return False
        elif pattern.startswith("cache-"):
            # 缓存文件前缀匹配
            if name.startswith("cache-"):
                return False
        else:
            # 完整匹配（不包括 json 文件）
            if name == pattern:
                return False
    return True


def create_package():
    """创建打包文件"""
    print("=== 开始打包 ===")

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建临时打包目录
    temp_dir = OUTPUT_DIR / "midi_generator"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 复制文件
    print("复制文件...")
    for filename in INCLUDE_FILES:
        src = PROJECT_DIR / filename
        if src.exists():
            dst = temp_dir / filename
            shutil.copy2(src, dst)
            print(f"  + {filename}")
        else:
            print(f"  ! {filename} (不存在)")

    # 复制目录
    for dirname in INCLUDE_DIRS:
        src = PROJECT_DIR / dirname
        if src.exists():
            dst = temp_dir / dirname
            if src.is_dir():
                # 创建目标目录
                dst.mkdir(parents=True, exist_ok=True)
                # 复制所有内容（但排除temp目录和缓存文件）
                for item in src.iterdir():
                    if item.name == 'temp':
                        continue
                    if item.name.startswith('cache-'):
                        continue  # 排除缓存文件
                    if item.is_file():
                        # 确保复制根目录的 json 文件（如 dataset_dict.json）
                        shutil.copy2(item, dst / item.name)
                    elif item.is_dir():
                        # 对 train/test 目录，复制所有文件包括 json
                        if item.name in ['train', 'test']:
                            dst_sub = dst / item.name
                            dst_sub.mkdir(exist_ok=True)
                            for sub_item in item.iterdir():
                                if sub_item.name.startswith('cache-'):
                                    continue
                                shutil.copy2(sub_item, dst_sub / sub_item.name)
                        else:
                            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                print(f"  + {dirname}/")
            else:
                print(f"  ! {dirname}/ (不是目录)")
        else:
            print(f"  ! {dirname}/ (不存在)")

    # 创建 zip 包
    zip_path = OUTPUT_DIR / "midi_generator_package.zip"
    print(f"\n创建压缩包: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            # 过滤目录
            dirs[:] = [d for d in dirs if should_include(Path(d))]

            for file in files:
                file_path = Path(root) / file
                if should_include(file_path):
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
                    print(f"  + {arcname}")

    # 清理临时目录
    shutil.rmtree(temp_dir)

    print(f"\n=== 打包完成 ===")
    print(f"压缩包位置: {zip_path}")
    print(f"压缩包大小: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 列出包内容
    print("\n包内容预览:")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        names = zipf.namelist()
        for name in sorted(names)[:20]:
            print(f"  - {name}")
        if len(names) > 20:
            print(f"  ... 还有 {len(names) - 20} 个文件")


if __name__ == "__main__":
    create_package()
