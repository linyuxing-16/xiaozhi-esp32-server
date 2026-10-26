"""人脸识别与跨视频唯一 face_id（需求 ②，决策 D4）。

口径（v4 澄清）：
- **没露出可识别人脸的不做人脸识别**：模糊 / 侧脸 / 遮挡 / 太小 / 检测置信度不足时，
  不分配 face_id，**不做模糊匹配兜底、不猜**，结构化输出里该字段直接缺失。
- 可识别的人脸用 InsightFace(ArcFace) 提特征，与 SQLite 底库做余弦匹配：
  命中 -> 复用全局 face_id；未命中 -> 新建全局 face_id 并入库。
- InsightFace 未安装时整体关闭人脸识别，face_id 一律缺失（能力探测已告警）。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.logger import setup_logging
from core.utils.video.capabilities import detect_capabilities

TAG = __name__
logger = setup_logging()

MIN_FACE_SIDE_PX = 32  # 人脸框最小边长，低于此视为"太小"，不做识别
MIN_DET_SCORE = 0.6  # 检测置信度下限
MIN_FACE_QUALITY = 0.45  # 质量分（清晰度/正脸度综合）下限
DEFAULT_SIMILARITY = 0.45  # ArcFace 余弦相似度阈值


@dataclass
class FaceObservation:
    """一个人脸检测结果。"""

    bbox: List[int]  # 相对整帧的像素框 [x1, y1, x2, y2]
    det_score: float
    embedding: Optional[np.ndarray] = None
    yaw: float = 0.0  # 左右偏转角（度），用于正脸度
    pitch: float = 0.0
    quality: float = 1.0
    landmarks: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def frontal_score(self) -> float:
        """正脸度 0-1：yaw 越接近 0 越高。"""
        return max(0.0, 1.0 - min(90.0, abs(self.yaw)) / 90.0)

    def recognizable(self) -> bool:
        """是否满足"可识别人脸"的口径。"""
        if self.embedding is None:
            return False
        if min(self.width, self.height) < MIN_FACE_SIDE_PX:
            return False
        if self.det_score < MIN_DET_SCORE:
            return False
        if self.quality < MIN_FACE_QUALITY:
            return False
        # 侧脸（|yaw| > 45°）按需求不做识别
        if abs(self.yaw) > 45.0:
            return False
        return True


class FaceDatabase:
    """SQLite 人脸底库：跨视频唯一的 face_id。"""

    def __init__(self, path: str = "data/video/faces.db", similarity: float = DEFAULT_SIMILARITY):
        self.path = path
        self.similarity = float(similarity)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return vector
        return vector / norm

    def _load_all(self) -> List[Tuple[str, np.ndarray]]:
        rows = self._conn.execute("SELECT face_id, embedding, dim FROM faces").fetchall()
        result: List[Tuple[str, np.ndarray]] = []
        for face_id, blob, dim in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size == int(dim):
                result.append((face_id, vector))
        return result

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """按余弦相似度匹配底库，返回 (face_id, score)。"""
        query = self._normalize(np.asarray(embedding, dtype=np.float32))
        best_id: Optional[str] = None
        best_score = -1.0
        with self._lock:
            for face_id, vector in self._load_all():
                score = float(np.dot(query, self._normalize(vector)))
                if score > best_score:
                    best_score = score
                    best_id = face_id
        if best_id is not None and best_score >= self.similarity:
            return best_id, best_score
        return None, best_score

    def enroll(self, embedding: np.ndarray) -> str:
        """新面孔入库，返回新分配的全局 face_id。"""
        vector = self._normalize(np.asarray(embedding, dtype=np.float32))
        face_id = f"face_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO faces (face_id, embedding, dim, created_at, last_seen_at, seen_count)"
                " VALUES (?, ?, ?, ?, ?, 1)",
                (face_id, vector.tobytes(), int(vector.size), now, now),
            )
            self._conn.commit()
        return face_id

    def touch(self, face_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE faces SET last_seen_at = ?, seen_count = seen_count + 1 WHERE face_id = ?",
                (time.time(), face_id),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class FaceRecognizer:
    """InsightFace 封装。不可用时 enabled=False，识别一律返回空。"""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.enabled = detect_capabilities().insightface
        self.det_size = int(config.get("det_size", 320))
        self.app = None
        self.database: Optional[FaceDatabase] = None
        self._cache: Dict[int, Optional[str]] = {}
        if not self.enabled:
            logger.bind(tag=TAG).warning(
                "[video] 未安装 insightface：人脸识别关闭，结构化输出不含 face_id"
            )
            return
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=config.get("model_name", "buffalo_l"),
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
            self.database = FaceDatabase(
                path=config.get("db_path", "data/video/faces.db"),
                similarity=float(config.get("similarity", DEFAULT_SIMILARITY)),
            )
        except Exception as exc:  # noqa: BLE001 - 模型下载/依赖异常统一降级
            logger.bind(tag=TAG).warning(f"[video] InsightFace 初始化失败，人脸识别关闭: {exc}")
            self.enabled = False
            self.app = None

    def detect(self, image: np.ndarray) -> List[FaceObservation]:
        """检测一帧中所有人脸及其特征。"""
        if not self.enabled or self.app is None:
            return []
        try:
            faces = self.app.get(image)
        except Exception as exc:  # noqa: BLE001
            logger.bind(tag=TAG).debug(f"[video] 人脸检测失败: {exc}")
            return []

        observations: List[FaceObservation] = []
        for face in faces:
            bbox = [int(round(v)) for v in face.bbox.tolist()]
            yaw = 0.0
            pitch = 0.0
            pose = getattr(face, "pose", None)
            if pose is not None and len(pose) >= 3:
                pitch, yaw, _roll = (float(pose[0]), float(pose[1]), float(pose[2]))
            det_score = float(getattr(face, "det_score", 1.0))
            quality = det_score * max(0.0, 1.0 - min(90.0, abs(yaw)) / 90.0 * 0.5)
            observations.append(
                FaceObservation(
                    bbox=bbox,
                    det_score=det_score,
                    embedding=np.asarray(face.normed_embedding, dtype=np.float32)
                    if getattr(face, "normed_embedding", None) is not None
                    else None,
                    yaw=yaw,
                    pitch=pitch,
                    quality=quality,
                    landmarks=getattr(face, "kps", None),
                )
            )
        return observations

    def identify(self, observation: FaceObservation) -> Optional[str]:
        """返回可识别的 face_id；不可识别或未命中且不可入库时返回 None。"""
        if not self.enabled or self.database is None:
            return None
        if not observation.recognizable():
            return None
        face_id, score = self.database.match(observation.embedding)
        if face_id is not None:
            self.database.touch(face_id)
            return face_id
        # 新面孔：自动入库并分配跨视频唯一 id
        try:
            new_id = self.database.enroll(observation.embedding)
            logger.bind(tag=TAG).info(
                f"[video] 新面孔入库 face_id={new_id}（最近邻相似度 {score:.3f} < 阈值）"
            )
            return new_id
        except Exception as exc:  # noqa: BLE001
            logger.bind(tag=TAG).warning(f"[video] 人脸入库失败: {exc}")
            return None

    def close(self) -> None:
        if self.database is not None:
            try:
                self.database.close()
            except Exception:  # noqa: BLE001
                pass
            self.database = None


def associate_faces_to_persons(
    person_boxes: Sequence[Sequence[int]], faces: Sequence[FaceObservation]
) -> Dict[int, FaceObservation]:
    """把检测到的人脸归属到人物 bbox（按人脸中心是否落在人物框内）。"""
    assignment: Dict[int, FaceObservation] = {}
    for index, box in enumerate(person_boxes):
        x1, y1, x2, y2 = box
        best: Optional[FaceObservation] = None
        best_area = 0
        for face in faces:
            cx = (face.bbox[0] + face.bbox[2]) / 2.0
            cy = (face.bbox[1] + face.bbox[3]) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if face.area > best_area:
                    best_area = face.area
                    best = face
        if best is not None:
            assignment[index] = best
    return assignment


def mouth_open_score(observation: FaceObservation) -> float:
    """嘴部开合度的近似：用人脸关键点上下唇间距 / 脸高（无关键点时返回 0）。"""
    if observation.landmarks is None:
        return 0.0
    try:
        points = np.asarray(observation.landmarks, dtype=np.float32)
        if points.shape[0] < 5:
            return 0.0
        # InsightFace 5 点：左眼、右眼、鼻尖、左嘴角、右嘴角
        mouth_left = points[3]
        mouth_right = points[4]
        mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
        if observation.height <= 0:
            return 0.0
        return max(0.0, min(1.0, mouth_width / float(observation.height)))
    except Exception:  # noqa: BLE001
        return 0.0
