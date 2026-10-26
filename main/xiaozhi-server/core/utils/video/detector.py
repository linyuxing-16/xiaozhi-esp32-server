"""人物检测 + 视频内稳定 id（需求 ②，阶段 D）。

组成：
- 检测后端：优先 ultralytics YOLO；其次 onnxruntime 加载配置的 YOLOv8 权重；
  都没有时返回空结果并明确日志（不编造检测）。
- ByteTrack 风格跟踪器：卡尔曼式匀速预测 + IoU 匹配，保证视频内 person_id 不跳变。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.logger import setup_logging
from core.utils.video.capabilities import detect_capabilities

TAG = __name__
logger = setup_logging()

PERSON_CLASS_ID = 0  # COCO person


@dataclass
class Detection:
    """一次检测结果（像素坐标系）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    class_id: int = PERSON_CLASS_ID

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def as_bbox(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))]

    def to_dict(self) -> dict:
        return {
            "bbox": self.as_bbox(),
            "score": round(float(self.score), 4),
        }


def iou(a: Detection, b: Detection) -> float:
    """两个框的交并比。"""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


class DetectorBackend:
    """检测后端基类。"""

    name = "base"

    def detect(self, image: np.ndarray) -> List[Detection]:  # pragma: no cover - 抽象
        raise NotImplementedError


class UltralyticsYoloBackend(DetectorBackend):
    """ultralytics YOLO 后端（推荐）。"""

    name = "ultralytics"

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 320):
        from ultralytics import YOLO  # 延迟导入，缺少依赖时由工厂降级

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, classes=[PERSON_CLASS_ID], verbose=False
        )
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                coords = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        score=float(box.conf[0]) if box.conf is not None else 1.0,
                    )
                )
        return detections


class OnnxYoloBackend(DetectorBackend):
    """onnxruntime 加载 YOLOv8 导出模型（无 ultralytics 时的后备）。"""

    name = "onnx"

    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 320):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.imgsz = imgsz

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        import cv2

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None, ...], scale, (new_w, new_h)

    def detect(self, image: np.ndarray) -> List[Detection]:
        blob, scale, _ = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        if not outputs:
            return []
        prediction = np.squeeze(outputs[0])
        # YOLOv8 输出形如 (84, N)；若是 (N, 84) 需要转置
        if prediction.ndim != 2:
            return []
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.transpose(1, 0)

        detections: List[Detection] = []
        if prediction.shape[1] < 5:
            return detections
        boxes = prediction[:, :4]
        scores = prediction[:, 4:]
        class_ids = np.argmax(scores, axis=1) if scores.shape[1] > 1 else np.zeros(len(scores), dtype=int)
        class_scores = scores.max(axis=1) if scores.shape[1] > 1 else scores[:, 0]
        keep = (class_ids == PERSON_CLASS_ID) & (class_scores >= self.conf)
        boxes = boxes[keep]
        class_scores = class_scores[keep]
        if boxes.size == 0:
            return detections

        detections = []
        for (cx, cy, bw, bh), score in zip(boxes, class_scores):
            x1 = (cx - bw / 2.0) / scale
            y1 = (cy - bh / 2.0) / scale
            x2 = (cx + bw / 2.0) / scale
            y2 = (cy + bh / 2.0) / scale
            detections.append(Detection(x1, y1, x2, y2, float(score)))
        return detections


class MotionFallbackBackend(DetectorBackend):
    """没有任何检测模型时的诚实降级：不产生检测结果。"""

    name = "none"

    def detect(self, image: np.ndarray) -> List[Detection]:
        return []


def create_detector(config: dict) -> DetectorBackend:
    """按配置创建检测后端，缺依赖时逐级降级并明示日志。"""
    caps = detect_capabilities()
    model_path = (config or {}).get("model_path") or ""
    conf = float((config or {}).get("conf", 0.35))
    imgsz = int((config or {}).get("imgsz", 320))

    if caps.ultralytics:
        try:
            return UltralyticsYoloBackend(model_path or "yolov8n.pt", conf=conf, imgsz=imgsz)
        except Exception as exc:  # noqa: BLE001 - 依赖/权重异常统一降级
            logger.bind(tag=TAG).warning(f"[video] ultralytics 后端初始化失败，继续降级: {exc}")

    if caps.onnxruntime and model_path and os.path.exists(model_path):
        try:
            return OnnxYoloBackend(model_path, conf=conf, imgsz=imgsz)
        except Exception as exc:  # noqa: BLE001
            logger.bind(tag=TAG).warning(f"[video] ONNX 检测后端初始化失败: {exc}")

    logger.bind(tag=TAG).warning(
        "[video] 没有可用的人物检测模型：结构化输出中不会出现 person 检测结果"
        "（请安装 ultralytics 或配置 video.detector.model_path）"
    )
    return MotionFallbackBackend()


@dataclass
class Track:
    """一条跟踪轨迹（即视频内的一个人物）。"""

    track_id: int
    detection: Detection
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    misses: int = 0
    hits: int = 1
    detected: bool = True  # True=本帧真检测；False=跟踪外推

    def predict(self) -> Detection:
        """按速度外推当前帧 bbox（卡尔曼的简化版）。"""
        dx, dy = self.velocity
        return Detection(
            self.detection.x1 + dx,
            self.detection.y1 + dy,
            self.detection.x2 + dx,
            self.detection.y2 + dy,
            score=self.detection.score,
        )

    def update(self, detection: Detection, smoothing: float = 0.5) -> None:
        old_cx, old_cy = self.detection.center
        new_cx, new_cy = detection.center
        vx = new_cx - old_cx
        vy = new_cy - old_cy
        self.velocity = (
            smoothing * vx + (1 - smoothing) * self.velocity[0],
            smoothing * vy + (1 - smoothing) * self.velocity[1],
        )
        # 用指数平滑抑制抖动，但保留尺寸变化
        self.detection = Detection(
            x1=smoothing * detection.x1 + (1 - smoothing) * self.detection.x1,
            y1=smoothing * detection.y1 + (1 - smoothing) * self.detection.y1,
            x2=smoothing * detection.x2 + (1 - smoothing) * self.detection.x2,
            y2=smoothing * detection.y2 + (1 - smoothing) * self.detection.y2,
            score=detection.score,
        )
        self.hits += 1
        self.misses = 0
        self.detected = True


class ByteTrackTracker:
    """ByteTrack 风格跟踪器：预测 + IoU 匹配 + 丢失容忍。

    关键点（D15）：未做新检测的帧也能给出 bbox（跟踪外推），
    但结构化输出里 detected=false，验收时可与真检测区分。
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_index = 0

    def _assign_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def step(self, detections: Sequence[Detection], detected: bool) -> List[Track]:
        """推进一帧。

        detections: 本帧新检测（detected=False 时传空列表，全部走外推）
        """
        self.frame_index += 1

        for track in self.tracks:
            track.age += 1

        predicted = [track.predict() for track in self.tracks]

        matched_track: Dict[int, int] = {}
        used_detections = set()
        if detected and detections and self.tracks:
            # 贪心 IoU 匹配（数量小，O(n*m) 足够）
            pairs: List[Tuple[float, int, int]] = []
            for ti, pred in enumerate(predicted):
                for di, det in enumerate(detections):
                    score = iou(pred, det)
                    if score >= self.iou_threshold:
                        pairs.append((score, ti, di))
            pairs.sort(reverse=True)
            used_tracks = set()
            for _score, ti, di in pairs:
                if ti in used_tracks or di in used_detections:
                    continue
                used_tracks.add(ti)
                used_detections.add(di)
                matched_track[ti] = di

        survivors: List[Track] = []
        for ti, track in enumerate(self.tracks):
            if ti in matched_track:
                track.update(detections[matched_track[ti]])
                survivors.append(track)
            else:
                track.misses += 1
                track.detected = False
                if track.misses <= self.max_misses:
                    track.detection = track.predict()
                    survivors.append(track)
        self.tracks = survivors

        if detected:
            for di, det in enumerate(detections):
                if di in used_detections:
                    continue
                self.tracks.append(Track(track_id=self._assign_id(), detection=det))

        return list(self.tracks)
