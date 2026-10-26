"""视觉分析管线（需求 ② + D15 实时调度，阶段 D）。

输入：一段视频的 JPEG 帧序列（带时间戳）+ ASR/VAD 的句级区间（含声纹 id）。
输出：
- 标注视频（画框 + person_id + face_id + 是否说话）；
- `frames[]` 结构化 JSON：每帧每个人物都有 bbox / person_id / speaking / detected，
  face_id 只在"露出可识别人脸"时出现（缺失即不写该 key）。

实时性做法（D15）：检测降频 + 跟踪补帧 + 人脸低频 + 说话判定与画框逐帧，
超预算自适应加大检测间隔，永不积压（处理是顺序的，队列只按帧推进）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.logger import setup_logging
from core.utils.video import annotator
from core.utils.video.capabilities import detect_capabilities
from core.utils.video.detector import ByteTrackTracker, DetectorBackend, create_detector
from core.utils.video.face import FaceRecognizer, associate_faces_to_persons
from core.utils.video.frames import VideoFrame
from core.utils.video.scheduler import FrameScheduler, SchedulerConfig
from core.utils.video.speaker import (
    MouthMotionTracker,
    SpeechSegment,
    SpeakerCandidate,
    SpeakerDetector,
    build_speaker_detector,
)

TAG = __name__
logger = setup_logging()


@dataclass
class PipelineConfig:
    """视觉管线配置（对应 config.yaml 的 video.*）。"""

    fps: int = 10
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    detector: dict = field(default_factory=dict)
    face: dict = field(default_factory=dict)
    speaker: dict = field(default_factory=dict)
    annotate: bool = True
    jpeg_quality: int = 80

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "PipelineConfig":
        config = config or {}
        return cls(
            fps=int(config.get("fps", 10)),
            scheduler=SchedulerConfig.from_config(config.get("scheduler") or {}),
            detector=config.get("detector") or {},
            face=config.get("face") or {},
            speaker=config.get("speaker") or {},
            annotate=bool(config.get("annotate", True)),
            jpeg_quality=int(config.get("jpeg_quality", 80)),
        )


@dataclass
class AnalysisResult:
    """一次视觉分析的结果。"""

    ok: bool
    video_path: Optional[str] = None
    frames: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(
            {"video": self.video_path, "frames": self.frames, "stats": self.stats},
            ensure_ascii=False,
            indent=indent,
        )


class VideoPipeline:
    """把一帧帧 JPEG 变成"标注视频 + 逐帧结构化 JSON"。"""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        output_dir: str = "data/video",
        detector: Optional[DetectorBackend] = None,
        recognizer: Optional[FaceRecognizer] = None,
        speaker_detector: Optional[SpeakerDetector] = None,
    ):
        self.config = config or PipelineConfig()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.detector = detector or create_detector(self.config.detector)
        self.recognizer = recognizer if recognizer is not None else FaceRecognizer(
            self.config.face
        )
        self.speaker_detector = speaker_detector or build_speaker_detector(
            self.config.speaker
        )
        self.mouth_tracker = MouthMotionTracker()

    # ------------------------------------------------------------------ 单帧
    def _analyze_frame(
        self,
        image: np.ndarray,
        frame: VideoFrame,
        plan,
        segments: Sequence[SpeechSegment],
        voiceprint_history: Dict[str, str],
        state: dict,
    ) -> List[dict]:
        """处理一帧，返回该帧的人物结构化列表。"""
        height, width = image.shape[:2]
        started = time.monotonic()

        detections = self.detector.detect(image) if plan.run_detect else []
        tracks = state["tracker"].step(detections, detected=plan.run_detect)

        if plan.run_face and plan.run_detect:
            faces = self.recognizer.detect(image)
            boxes = [track.detection.as_bbox() for track in tracks]
            face_assignments = associate_faces_to_persons(boxes, faces)
            # 缓存按 track_id 存：只保留本帧仍存在、且确实检到人脸的轨迹，
            # 避免旧人脸结果挂在已经离开画面的人物上（口径：不猜）。
            previous_faces: Dict[int, object] = state.get("faces") or {}
            previous_ids: Dict[int, str] = state.get("face_ids") or {}
            current_ids = {track.track_id for track in tracks}
            new_faces: Dict[int, object] = {}
            new_ids: Dict[int, str] = {}
            for index, observation in face_assignments.items():
                track_id = tracks[index].track_id
                new_faces[track_id] = observation
                face_id = self.recognizer.identify(observation)
                if face_id:
                    new_ids[track_id] = face_id
                elif track_id in previous_ids:
                    # 同一张脸这次识别不出（例如侧脸）：沿用已知 face_id
                    new_ids[track_id] = previous_ids[track_id]
            state["faces"] = new_faces
            state["face_ids"] = new_ids
            # 仍在画面但本帧没检到脸的人物：沿用上一轮的人脸缓存
            for track_id in current_ids:
                if track_id not in new_faces and track_id in previous_faces:
                    new_faces[track_id] = previous_faces[track_id]
                    if track_id in previous_ids:
                        new_ids[track_id] = previous_ids[track_id]

        cached_faces: Dict[int, object] = state.get("faces") or {}
        cached_ids: Dict[int, str] = state.get("face_ids") or {}

        objects: List[dict] = []
        gray = None
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception:  # noqa: BLE001
            gray = None

        candidates: List[SpeakerCandidate] = []
        for track in tracks:
            observation = cached_faces.get(track.track_id)
            face_box = observation.bbox if observation is not None else None
            face_id = cached_ids.get(track.track_id)
            mouth_motion = 0.0
            if gray is not None:
                mouth_motion = self.mouth_tracker.update(track.track_id, gray, face_box)
            candidates.append(
                SpeakerCandidate(
                    person_id=track.track_id,
                    face_box=face_box,
                    frame_width=width,
                    frame_height=height,
                    frontal_score=observation.frontal_score if observation is not None else 0.0,
                    mouth_motion=mouth_motion,
                    face_id=face_id,
                )
            )

        segment = self._segment_for(frame.timestamp_ms, segments)
        speaking_ids = set()
        if segment is not None and candidates:
            scores = self.speaker_detector.score(
                candidates, segment, voiceprint_history=voiceprint_history
            )
            chosen = self.speaker_detector.decide(scores, self.speaker_detector.threshold)
            if chosen is not None:
                speaking_ids.add(chosen)
                # 建立 face_id -> 声纹 id 的关联，供后续轮次一致性加分
                for candidate in candidates:
                    if candidate.person_id == chosen and candidate.face_id and segment.voiceprint_id:
                        voiceprint_history.setdefault(
                            candidate.face_id, segment.voiceprint_id
                        )

        for track, candidate in zip(tracks, candidates):
            entry = {
                "timestamp_ms": int(frame.timestamp_ms),
                "person_id": int(track.track_id),
                "bbox": track.detection.as_bbox(),
                "speaking": track.track_id in speaking_ids,
                "detected": bool(track.detected and plan.run_detect),
            }
            # 关键口径：人脸不可识别时 face_id 字段缺失（不写空串、不猜）
            observation = cached_faces.get(track.track_id)
            if observation is not None and observation.recognizable():
                face_id = cached_ids.get(track.track_id)
                if face_id:
                    entry["face_id"] = face_id
                    entry["face_bbox"] = [int(v) for v in observation.bbox]
            objects.append(entry)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        state["scheduler"].observe(elapsed_ms)
        state["elapsed"].append(elapsed_ms)

        if self.config.annotate:
            annotator.draw_annotations(image, objects)
        return objects

    @staticmethod
    def _segment_for(timestamp_ms: int, segments: Sequence[SpeechSegment]) -> Optional[SpeechSegment]:
        for segment in segments:
            if segment.contains(timestamp_ms):
                return segment
        return None

    # ------------------------------------------------------------------ 整段
    def analyze(
        self,
        frames: Sequence[VideoFrame],
        segments: Optional[Sequence[SpeechSegment]] = None,
        session_id: str = "session",
        name: str = "annotated",
    ) -> AnalysisResult:
        caps = detect_capabilities()
        if not frames:
            return AnalysisResult(ok=False, error="没有视频帧")

        segments = list(segments or [])
        scheduler = FrameScheduler(self.config.scheduler)
        state = {
            "tracker": ByteTrackTracker(),
            "scheduler": scheduler,
            "elapsed": [],
            "faces": {},
            "face_ids": {},
        }
        voiceprint_history: Dict[str, str] = {}

        out_frames: List[dict] = []
        annotated: List[VideoFrame] = []
        decode_failures = 0

        for frame in frames:
            image = annotator.decode_jpeg(frame.data)
            if image is None:
                decode_failures += 1
                continue
            plan = scheduler.plan()
            objects = self._analyze_frame(
                image, frame, plan, segments, voiceprint_history, state
            )
            out_frames.append(
                {"timestamp_ms": int(frame.timestamp_ms), "persons": objects}
            )
            if self.config.annotate:
                encoded = annotator.encode_jpeg(image, self.config.jpeg_quality)
                if encoded:
                    annotated.append(
                        VideoFrame(
                            data=encoded,
                            timestamp_ms=frame.timestamp_ms,
                            width=image.shape[1],
                            height=image.shape[0],
                        )
                    )

        source = annotated if annotated else list(frames)
        video_path = None
        if caps.ffmpeg and source:
            video_path = self._mux_video(source, session_id, name, scheduler)
            if video_path is None:
                logger.bind(tag=TAG).warning("[video] 标注视频封装失败，仅返回结构化 JSON")

        elapsed = state["elapsed"] or [0.0]
        detected_frames = sum(
            1 for entry in out_frames if any(p["detected"] for p in entry["persons"])
        )
        total_persons = sum(len(entry["persons"]) for entry in out_frames)
        stats = {
            "frames": len(out_frames),
            "decode_failures": decode_failures,
            "avg_ms": round(float(np.mean(elapsed)), 2),
            "max_ms": round(float(np.max(elapsed)), 2),
            "p95_ms": round(float(np.percentile(elapsed, 95)), 2),
            "budget_ms": self.config.scheduler.target_budget_ms,
            "realtime": bool(np.mean(elapsed) <= self.config.scheduler.target_budget_ms),
            "detected_frames": detected_frames,
            "tracked_only_frames": len(out_frames) - detected_frames,
            "persons_total": total_persons,
            "scheduler": scheduler.stats(),
            "detector_backend": getattr(self.detector, "name", "unknown"),
            "face_enabled": bool(getattr(self.recognizer, "enabled", False)),
        }
        if out_frames:
            stats["tracked_only_ratio"] = round(
                (len(out_frames) - detected_frames) / float(len(out_frames)), 4
            )
        return AnalysisResult(ok=True, video_path=video_path, frames=out_frames, stats=stats)

    def _mux_video(
        self,
        frames: Sequence[VideoFrame],
        session_id: str,
        name: str,
        scheduler: FrameScheduler,
    ) -> Optional[str]:
        workdir = tempfile.mkdtemp(prefix="xzannot_")
        try:
            for index, frame in enumerate(frames):
                with open(os.path.join(workdir, f"f_{index:06d}.jpg"), "wb") as handle:
                    handle.write(frame.data)
            output_path = os.path.join(
                self.output_dir, f"{name}_{session_id}_{int(time.time())}.mp4"
            )
            args = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(max(1, self.config.fps)),
                "-i",
                os.path.join(workdir, "f_%06d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(max(1, self.config.fps)),
                output_path,
            ]
            result = subprocess.run(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
            )
            if result.returncode != 0:
                logger.bind(tag=TAG).error(
                    f"[video] ffmpeg 标注视频封装失败: "
                    f"{(result.stderr or b'').decode('utf-8', 'replace')[-800:]}"
                )
                return None
            return output_path
        except Exception as exc:  # noqa: BLE001
            logger.bind(tag=TAG).error(f"[video] 标注视频封装异常: {exc}")
            return None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def close(self) -> None:
        if self.recognizer is not None:
            self.recognizer.close()
