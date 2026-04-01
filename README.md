# MIDI Melody Generator

基于 GPT-2 的轻量级 MIDI 旋律生成模型。

## 功能

- 使用 PyQt6 开发的桌面应用程序
- 基于 miditok REMI 编码的旋律生成
- 支持 Temperature、Top_K、Top_P 参数调节
- 生成的 MIDI 可直接拖拽到 DAW 软件中使用

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型文件

模型文件需要从本仓库的 **Release** 中下载：
- 下载 `model_output.zip`（包含训练好的模型和 tokenizer）
- 解压到项目根目录

### 3. 运行

```bash
# 运行 GUI
python app.py

# 或运行命令行推理
python inference.py
```

## 项目结构

```
├── app.py              # PyQt6 桌面应用
├── inference.py        # 推理脚本
├── prepare_data.py    # 数据预处理
├── train_model.py    # 模型训练
├── requirements.txt   # 依赖列表
└── model_output/     # 模型文件（需从 Release 下载）
```

## 模型说明

- 参数量：25.88M
- 训练数据：POP909 数据集
- Tokenizer：REMI 编码

## 许可证

MIT License
