"""
MIDI 旋律生成器 - PyQt6 GUI 应用
- 左侧：控制区（输入框 + 生成按钮）
- 右侧：拖拽释放区（将生成的 MIDI 拖拽到 DAW）
"""

import sys
import os
from pathlib import Path
import subprocess
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFrame, QMessageBox, QSlider, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QMimeData, QUrl, QObject, pyqtSignal, QPoint
from PyQt6.QtGui import QDrag, QPixmap, QPainter, QColor, QFont


# 配置路径
OUTPUT_MIDI = Path("output.mid")
MODEL_DIR = Path("model_output")

# 设置支持中文的字体（带回退机制）
CHINESE_FONT = "Microsoft YaHei, SimHei, PingFang SC"
TITLE_FONT = QFont("Microsoft YaHei,SimHei,PingFang SC", 18, QFont.Weight.Bold)
LABEL_FONT = QFont("Microsoft YaHei,SimHei,PingFang SC", 10)
BUTTON_FONT = QFont("Microsoft YaHei,SimHei,PingFang SC", 14, QFont.Weight.Bold)


class SignalEmitter(QObject):
    """信号发射器用于线程间通信"""
    generation_complete = pyqtSignal(bool, str)
    generation_started = pyqtSignal()


class DropZone(QFrame):
    """拖拽区域组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.is_ready = False

        # 设置样式 - 使用简单选择器避免 PyQt6 警告
        self.setStyleSheet("""
            DropZone {
                border: 3px dashed #666;
                border-radius: 15px;
                background-color: #f5f5f5;
            }
        """)

        # 布局
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Click 'Generate Melody'\nto create MIDI", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFont(QFont("Microsoft YaHei,SimHei,PingFang SC", 14, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #666;")
        layout.addWidget(self.label)

        self.setMinimumSize(300, 200)

    def set_ready(self, ready=True):
        """设置是否就绪"""
        self.is_ready = ready
        if ready:
            self.label.setText("Success!\nDrag me to your DAW!")
            self.label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.setStyleSheet("""
                DropZone {
                    border: 3px solid #4CAF50;
                    border-radius: 15px;
                    background-color: #e8f5e9;
                }
            """)
        else:
            self.label.setText("Click 'Generate Melody'\nto create MIDI")
            self.label.setStyleSheet("color: #666;")
            self.setStyleSheet("""
                DropZone {
                    border: 3px dashed #666;
                    border-radius: 15px;
                    background-color: #f5f5f5;
                }
            """)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.is_ready and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 启动拖拽"""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if not self.is_ready:
            return

        # 检查是否移动了足够距离
        if not hasattr(self, 'drag_start_pos'):
            return

        distance = (event.pos() - self.drag_start_pos).manhattanLength()
        if distance < QApplication.startDragDistance():
            return

        # 执行拖拽
        self.start_drag()

    def start_drag(self):
        """开始拖拽 MIDI 文件"""
        if not OUTPUT_MIDI.exists():
            return

        # 创建 MIME 数据
        mime_data = QMimeData()
        url = QUrl.fromLocalFile(str(OUTPUT_MIDI.absolute()))
        mime_data.setUrls([url])

        # 创建拖拽对象
        drag = QDrag(self)
        drag.setMimeData(mime_data)

        # 设置拖拽时的视觉反馈
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor(76, 175, 80, 200))
        painter = QPainter(pixmap)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setFont(QFont("Microsoft YaHei,SimHei,PingFang SC", 12, QFont.Weight.Bold))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "MIDI")
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(50, 50))

        # 执行拖拽
        drag.exec(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        event.accept()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Melody Generator")
        # 3. 窗口基础尺寸保护
        self.setMinimumSize(800, 600)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 2. 加入滚动区域 (QScrollArea)
        scroll_area = QScrollArea()
        # 1. 固定左侧面板的最大宽度
        scroll_area.setMaximumWidth(400)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        # 左侧控制面板容器
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #fff;
                border-radius: 10px;
            }
        """)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 防止标签被压扁 - 标题
        title = QLabel("MIDI Generator", self)
        title.setFont(TITLE_FONT)
        title.setStyleSheet("color: #333;")
        title.setWordWrap(True)
        title.setMinimumHeight(40)
        title.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        control_layout.addWidget(title)

        # === 参数控制区 ===
        # Temperature 滑块
        temp_label = QLabel("Temperature 温度\n高时节奏型更丰富、音域更广、变化更多", self)
        temp_label.setFont(LABEL_FONT)
        temp_label.setStyleSheet("color: #666;")
        temp_label.setWordWrap(True)
        temp_label.setMinimumHeight(40)
        temp_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        control_layout.addWidget(temp_label)

        temp_layout = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.temp_slider.setMinimum(1)
        self.temp_slider.setMaximum(20)
        self.temp_slider.setValue(8)
        self.temp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temp_slider.setTickInterval(2)
        self.temp_label = QLabel("0.8", self)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/10:.1f}"))
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        control_layout.addLayout(temp_layout)

        # Top_K 滑块
        topk_label = QLabel("Top_K 候选词数量\n少=旋律稳定重复，多=旋律变化多样", self)
        topk_label.setFont(LABEL_FONT)
        topk_label.setStyleSheet("color: #666;")
        topk_label.setWordWrap(True)
        topk_label.setMinimumHeight(40)
        topk_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        control_layout.addWidget(topk_label)

        topk_layout = QHBoxLayout()
        self.topk_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.topk_slider.setMinimum(1)
        self.topk_slider.setMaximum(100)
        self.topk_slider.setValue(30)
        self.topk_label = QLabel("30", self)
        self.topk_slider.valueChanged.connect(lambda v: self.topk_label.setText(str(v)))
        topk_layout.addWidget(self.topk_slider)
        topk_layout.addWidget(self.topk_label)
        control_layout.addLayout(topk_layout)

        # Top_P 滑块
        topp_label = QLabel("Top_P 核采样概率\n低=保守只选高概率音符，高=开放可能选低概率音符", self)
        topp_label.setFont(LABEL_FONT)
        topp_label.setStyleSheet("color: #666;")
        topp_label.setWordWrap(True)
        topp_label.setMinimumHeight(40)
        topp_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        control_layout.addWidget(topp_label)

        topp_layout = QHBoxLayout()
        self.topp_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.topp_slider.setMinimum(50)
        self.topp_slider.setMaximum(100)
        self.topp_slider.setValue(88)
        self.topp_label = QLabel("0.88", self)
        self.topp_slider.valueChanged.connect(lambda v: self.topp_label.setText(f"{v/100:.2f}"))
        topp_layout.addWidget(self.topp_slider)
        topp_layout.addWidget(self.topp_label)
        control_layout.addLayout(topp_layout)

        # 状态标签
        self.status_label = QLabel("", self)
        self.status_label.setFont(LABEL_FONT)
        self.status_label.setStyleSheet("color: #666;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumHeight(30)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        control_layout.addWidget(self.status_label)

        # 生成按钮 (放在最下面)
        self.generate_btn = QPushButton("Generate Melody", self)
        self.generate_btn.setFont(BUTTON_FONT)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        control_layout.addWidget(self.generate_btn)

        # 4. 布局弹性 - 滑块往上顶
        control_layout.addStretch()

        # 将控制面板放入滚动区域
        scroll_area.setWidget(control_panel)

        # 右侧拖拽区
        self.drop_zone = DropZone(self)

        # 添加到主布局 - 3. 右侧占满剩余空间
        main_layout.addWidget(scroll_area, stretch=0)
        main_layout.addWidget(self.drop_zone, stretch=1)

        # 设置背景和全局字体
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e0e0e0;
            }
            QLabel {
                font-family: "Microsoft YaHei", "SimHei", "PingFang SC", Arial;
                font-size: 10pt;
            }
            QPushButton {
                font-family: "Microsoft YaHei", "SimHei", "PingFang SC", Arial;
                font-size: 10pt;
            }
        """)

    def on_generate_clicked(self):
        """生成按钮点击事件"""
        # 禁用按钮
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Generating...")
        self.drop_zone.set_ready(False)

        # 在后台线程中运行生成
        thread = threading.Thread(target=self.generate_melody_thread, daemon=True)
        thread.start()

    def generate_melody_thread(self):
        """后台生成旋律"""
        try:
            # 调用 inference.py
            cmd = [sys.executable, "inference.py"]

            # 添加生成参数
            temperature = self.temp_slider.value() / 10
            top_k = self.topk_slider.value()
            top_p = self.topp_slider.value() / 100
            cmd.extend(["--temperature", str(temperature)])
            cmd.extend(["--top_k", str(top_k)])
            cmd.extend(["--top_p", str(top_p)])

            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=180
            )

            # 打印输出以便调试
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"stderr: {result.stderr}")

            # 检查输出文件
            output_path = Path("output.mid")
            if output_path.exists():
                # 延迟发送信号让 UI 更新
                self.status_label.setText("Success! MIDI generated!")
                self.drop_zone.set_ready(True)
                self.generate_btn.setEnabled(True)
            else:
                raise FileNotFoundError("Output MIDI not found")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.generate_btn.setEnabled(True)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("MIDI Melody Generator")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
