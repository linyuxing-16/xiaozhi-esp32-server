"""运行环境能力探测。

需求实现计划 §7 阶段 A：缺少 ultralytics / insightface / ffmpeg 时整体降级并明示日志。
探测结果按进程缓存，避免每个连接都重复探测。
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


@dataclass
class VideoCapabilities:
    """一次探测得到的各项能力。"""

    ffmpeg: bool = False
    cv2: bool = False
    numpy: bool = False
    onnxruntime: bool = False
    ultralytics: bool = False
    insightface: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def can_record(self) -> bool:
        """录制需要 ffmpeg 做封装。"""
        return self.ffmpeg

    @property
    def can_detect_person(self) -> bool:
        """人物检测：有 ultralytics 走 YOLO，否则走轻量 ONNX 后备检测器。"""
        return self.ultralytics or (self.onnxruntime and self.cv2)

    @property
    def can_recognize_face(self) -> bool:
        """人脸识别需要 InsightFace(ArcFace)。"""
        return self.insightface

    def as_dict(self) -> Dict[str, bool]:
        return {
            "ffmpeg": self.ffmpeg,
            "cv2": self.cv2,
            "numpy": self.numpy,
            "onnxruntime": self.onnxruntime,
            "ultralytics": self.ultralytics,
            "insightface": self.insightface,
            "can_record": self.can_record,
            "can_detect_person": self.can_detect_person,
            "can_recognize_face": self.can_recognize_face,
        }


_lock = threading.Lock()
_cached: VideoCapabilities | None = None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _ffmpeg_available() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def detect_capabilities(force: bool = False) -> VideoCapabilities:
    """探测当前机器的视频处理能力（结果缓存）。"""
    global _cached
    with _lock:
        if _cached is not None and not force:
            return _cached

        caps = VideoCapabilities()
        caps.ffmpeg = _ffmpeg_available()
        caps.cv2 = _module_available("cv2")
        caps.numpy = _module_available("numpy")
        caps.onnxruntime = _module_available("onnxruntime")
        caps.ultralytics = _module_available("ultralytics")
        caps.insightface = _module_available("insightface")

        if not caps.ffmpeg:
            caps.notes.append("ffmpeg 不可用：视频无法封装 mp4，录制功能整体降级")
        if not caps.cv2:
            caps.notes.append("OpenCV 不可用：帧解码/画框不可用，视觉分析管线降级")
        if not caps.ultralytics:
            caps.notes.append("ultralytics 未安装：人物检测回退轻量 ONNX 后备检测器")
        if not caps.insightface:
            caps.notes.append("insightface 未安装：人脸识别关闭，结构化输出中不产生 face_id")

        for note in caps.notes:
            logger.bind(tag=TAG).warning(f"[video] {note}")

        _cached = caps
        return caps


def reset_capabilities_cache() -> None:
    """仅供测试使用：清空缓存。"""
    global _cached
    with _lock:
        _cached = None
