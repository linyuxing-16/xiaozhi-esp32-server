"""三级差异化实时调度（需求 D15，阶段 D 的核心）。

四类计算代价差两个数量级，必须分开调度：

| 处理项            | 频率                    | 说明                     |
| ----------------- | ----------------------- | ------------------------ |
| YOLO 检测         | 每 detect_every_n_frames | 最贵，必须降频            |
| ByteTrack 跟踪    | 每帧                    | 只算 bbox IoU/外推，<1ms  |
| 人脸检测 + 识别   | 每 face_every_n_frames  | 人脸不变，无需高频        |
| 说话判定 + 画框   | 每帧                    | 必须逐帧，否则卡片/掉标   |

自适应：某帧处理耗时超 target_budget_ms 就临时加大检测间隔（上限 max_detect_interval），
负载回落后逐步恢复。永远不会因积压阻塞音频链路（丢帧优先丢视频帧）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


@dataclass
class SchedulerConfig:
    detect_every_n_frames: int = 3
    face_every_n_frames: int = 15
    target_budget_ms: float = 33.0
    adaptive: bool = True
    max_detect_interval: int = 6
    # 降分辨率档位（三级降级预案第 2 级）
    detector_input_size: int = 320

    @classmethod
    def from_config(cls, config: dict | None) -> "SchedulerConfig":
        config = config or {}
        return cls(
            detect_every_n_frames=max(1, int(config.get("detect_every_n_frames", 3))),
            face_every_n_frames=max(1, int(config.get("face_every_n_frames", 15))),
            target_budget_ms=float(config.get("target_budget_ms", 33)),
            adaptive=bool(config.get("adaptive", True)),
            max_detect_interval=max(1, int(config.get("max_detect_interval", 6))),
            detector_input_size=int(config.get("detector_input_size", 320)),
        )


@dataclass
class FramePlan:
    """单帧要做什么。"""

    frame_index: int
    run_detect: bool
    run_face: bool
    detect_interval: int
    face_interval: int


class FrameScheduler:
    """决定每帧跑哪些重计算，并根据耗时自适应调整。"""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self._detect_interval = self.config.detect_every_n_frames
        self._face_interval = self.config.face_every_n_frames
        self.frame_index = 0
        self.durations_ms: List[float] = []
        self.level = 1  # 降级档位：1 正常，2 降频，3 降分辨率

    def plan(self) -> FramePlan:
        index = self.frame_index
        self.frame_index += 1
        return FramePlan(
            frame_index=index,
            run_detect=index % self._detect_interval == 0,
            run_face=index % self._face_interval == 0,
            detect_interval=self._detect_interval,
            face_interval=self._face_interval,
        )

    def observe(self, duration_ms: float) -> None:
        """记录一帧耗时并做自适应调整。"""
        self.durations_ms.append(float(duration_ms))
        if len(self.durations_ms) > 120:
            self.durations_ms.pop(0)
        if not self.config.adaptive:
            return
        if duration_ms > self.config.target_budget_ms:
            # 超预算：加大检测间隔（最多 max_detect_interval），并记降级日志
            new_interval = min(self.config.max_detect_interval, self._detect_interval + 1)
            if new_interval != self._detect_interval:
                self._detect_interval = new_interval
                self.level = 2 if new_interval < self.config.max_detect_interval else 3
                logger.bind(tag=TAG).warning(
                    f"[video] 单帧耗时 {duration_ms:.1f}ms 超预算 "
                    f"{self.config.target_budget_ms:.0f}ms，检测间隔调整为每 {new_interval} 帧（降级档位 {self.level}）"
                )
        elif len(self.durations_ms) >= 30:
            recent = self.durations_ms[-30:]
            if sum(recent) / len(recent) < self.config.target_budget_ms * 0.6:
                if self._detect_interval > self.config.detect_every_n_frames:
                    self._detect_interval -= 1
                    self.level = 1 if self._detect_interval == self.config.detect_every_n_frames else 2
                    logger.bind(tag=TAG).info(
                        f"[video] 负载回落，检测间隔恢复为每 {self._detect_interval} 帧"
                    )

    def stats(self) -> dict:
        if not self.durations_ms:
            return {
                "frames": 0,
                "avg_ms": 0.0,
                "max_ms": 0.0,
                "detect_interval": self._detect_interval,
                "level": self.level,
            }
        return {
            "frames": len(self.durations_ms),
            "avg_ms": round(sum(self.durations_ms) / len(self.durations_ms), 2),
            "max_ms": round(max(self.durations_ms), 2),
            "detect_interval": self._detect_interval,
            "face_interval": self._face_interval,
            "level": self.level,
        }

    @property
    def detect_interval(self) -> int:
        return self._detect_interval

    def reset(self) -> None:
        self._detect_interval = self.config.detect_every_n_frames
        self._face_interval = self.config.face_every_n_frames
        self.frame_index = 0
        self.durations_ms.clear()
        self.level = 1
