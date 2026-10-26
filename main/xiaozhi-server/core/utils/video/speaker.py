"""说话人判定（需求 ②，决策 D12 方案 A：音画启发式）。

思路（不训练模型，用两条独立线索交叉打分）：
1. 声音侧：VAD 给出有声区间 + 声纹 id -> "这段时间里有 1 个人在说话"；
2. 画面侧：对区间内每个候选人物打分：
   - 人脸面积占比 0.35（越大越像正对镜头且靠近）
   - 人脸朝向 0.25（正脸度高）
   - 嘴部开合度 0.30（逐帧嘴部区域变化方差大）
   - 声纹一致性 0.10（该人物 face_id 之前与声纹 id 关联过）
3. 取最高分且超阈值者标"是"，其余标"否"；最高分不达阈值则全部标"否"（宁可漏标，不乱标）。

接口设计为可替换：定义 SpeakerDetector 抽象基类，后续可新增 AsdSpeakerDetector。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.logger import setup_logging
from core.utils.video.face import FaceObservation

TAG = __name__
logger = setup_logging()


@dataclass
class SpeakerCandidate:
    """一个候选人物在某一时刻的画面线索。"""

    person_id: int
    face_box: Optional[List[int]] = None  # 相对整帧的人脸框
    frame_width: int = 0
    frame_height: int = 0
    frontal_score: float = 0.0
    mouth_motion: float = 0.0  # 0-1，嘴部区域逐帧变化程度（方差归一化）
    face_id: Optional[str] = None


@dataclass
class SpeechSegment:
    """一个有声区间（来自 VAD / ASR）。"""

    start_ms: int
    end_ms: int
    voiceprint_id: Optional[str] = None

    def contains(self, timestamp_ms: int) -> bool:
        return self.start_ms <= timestamp_ms <= self.end_ms


@dataclass
class SpeakerScore:
    person_id: int
    score: float
    components: Dict[str, float] = field(default_factory=dict)


class SpeakerDetector(ABC):
    """说话人检测抽象接口（可替换为 TalkNet-ASD 等模型实现）。"""

    name = "base"

    @abstractmethod
    def score(
        self,
        candidates: Sequence[SpeakerCandidate],
        segment: SpeechSegment,
        voiceprint_history: Optional[Dict[str, str]] = None,
    ) -> List[SpeakerScore]:
        """对候选人物打分（同一区间）。"""

    def decide(
        self,
        scores: Sequence[SpeakerScore],
        threshold: float,
    ) -> Optional[int]:
        """取最高分且超阈值者；否则返回 None（全部标"否"）。"""
        if not scores:
            return None
        best = max(scores, key=lambda item: item.score)
        if best.score < threshold:
            return None
        return best.person_id


class HeuristicSpeakerDetector(SpeakerDetector):
    """方案 A：加权启发式。权重/阈值全部来自配置。"""

    name = "heuristic"

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.w_area = float(config.get("w_area", 0.35))
        self.w_yaw = float(config.get("w_yaw", 0.25))
        self.w_mouth = float(config.get("w_mouth", 0.30))
        self.w_voiceprint = float(config.get("w_voiceprint", 0.10))
        self.threshold = float(config.get("threshold", 0.5))

    def score(
        self,
        candidates: Sequence[SpeakerCandidate],
        segment: SpeechSegment,
        voiceprint_history: Optional[Dict[str, str]] = None,
    ) -> List[SpeakerScore]:
        results: List[SpeakerScore] = []
        frame_area = max(1, int(candidates[0].frame_width) * int(candidates[0].frame_height)) if candidates else 1
        for candidate in candidates:
            area_ratio = 0.0
            if candidate.face_box:
                face_w = max(0, candidate.face_box[2] - candidate.face_box[0])
                face_h = max(0, candidate.face_box[3] - candidate.face_box[1])
                area_ratio = (face_w * face_h) / float(frame_area)
            # 面积占比归一化：人脸占画面 5% 已算很近
            area_score = min(1.0, area_ratio / 0.05)
            yaw_score = max(0.0, min(1.0, candidate.frontal_score))
            mouth_score = max(0.0, min(1.0, candidate.mouth_motion))
            vp_score = 0.0
            if (
                voiceprint_history
                and candidate.face_id
                and segment.voiceprint_id
                and voiceprint_history.get(candidate.face_id) == segment.voiceprint_id
            ):
                vp_score = 1.0
            total = (
                self.w_area * area_score
                + self.w_yaw * yaw_score
                + self.w_mouth * mouth_score
                + self.w_voiceprint * vp_score
            )
            results.append(
                SpeakerScore(
                    person_id=candidate.person_id,
                    score=round(total, 4),
                    components={
                        "area": round(area_score, 4),
                        "yaw": round(yaw_score, 4),
                        "mouth": round(mouth_score, 4),
                        "voiceprint": round(vp_score, 4),
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results


class MouthMotionTracker:
    """嘴部区域逐帧变化量（嘴开合/说话的直接画面证据）。

    用灰度帧差在被跟踪人物的人脸下半部 ROI 上算平均绝对差，
    再用滑动窗口的均值/方差映射到 0-1。
    """

    def __init__(self, history: int = 8):
        self.history = max(2, history)
        self._prev: Dict[int, np.ndarray] = {}
        self._samples: Dict[int, List[float]] = {}

    def update(self, person_id: int, frame_gray: np.ndarray, face_box: Optional[List[int]]) -> float:
        if face_box is None:
            self._prev.pop(person_id, None)
            return 0.0
        height, width = frame_gray.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_box]
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        # 人脸下半部（嘴部区域）
        my1 = y1 + int((y2 - y1) * 0.55)
        roi = frame_gray[my1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        roi = roi.astype(np.float32)
        roi = roi.reshape(32, -1) if roi.size >= 32 else roi.reshape(-1)
        # 统一到固定长度，避免不同帧尺寸无法比较
        target = 32 * 32
        if roi.size != target:
            flat = roi.flatten()
            if flat.size < target:
                padded = np.zeros(target, dtype=np.float32)
                padded[: flat.size] = flat
                roi = padded
            else:
                roi = flat[:target]
        else:
            roi = roi.flatten()

        previous = self._prev.get(person_id)
        self._prev[person_id] = roi
        if previous is None:
            return 0.0
        diff = float(np.mean(np.abs(roi - previous)) / 255.0)
        samples = self._samples.setdefault(person_id, [])
        samples.append(diff)
        if len(samples) > self.history:
            samples.pop(0)
        if not samples:
            return 0.0
        mean_diff = float(np.mean(samples))
        # 经验映射：嘴部区域帧差 0.02 以上基本就是在动嘴
        score = min(1.0, mean_diff / 0.02)
        if len(samples) < 3:
            # 样本太少时保守打折，避免单帧抖动误判
            score *= 0.5
        return score

    def reset(self) -> None:
        self._prev.clear()
        self._samples.clear()


def build_speaker_detector(config: Optional[dict] = None) -> SpeakerDetector:
    """按配置创建说话人检测器（默认启发式，便于替换为 ASD 模型）。"""
    config = config or {}
    kind = str(config.get("speaker_detector", "heuristic")).lower()
    if kind in ("asd", "talknet", "loconet"):
        logger.bind(tag=TAG).warning(
            f"[video] 尚未实现 {kind} 检测器，回退启发式（接口已预留 SpeakerDetector）"
        )
    return HeuristicSpeakerDetector(config)
