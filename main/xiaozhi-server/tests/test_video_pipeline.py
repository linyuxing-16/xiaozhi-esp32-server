"""视觉分析管线的单元测试（需求 ② + D15）。

覆盖：
- 跟踪补帧：降频下每一帧仍有 bbox，且可用 detected 区分真检测/外推；
- person_id 在单段视频内不跳变；
- 人脸不可识别时 face_id 字段缺失（不写空串、不猜）；
- 结构化 JSON 通过 schema 校验；
- 实时性统计可用。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.utils.video.detector import ByteTrackTracker, Detection
from core.utils.video.face import FaceObservation
from core.utils.video.frames import VideoFrame
from core.utils.video.pipeline import PipelineConfig, VideoPipeline
from core.utils.video.scheduler import FrameScheduler, SchedulerConfig
from core.utils.video.speaker import (
    HeuristicSpeakerDetector,
    SpeechSegment,
    SpeakerCandidate,
)


class FakeDetector:
    """返回固定人数与位置的假检测器（可移动）。"""

    name = "fake"

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(*box, score=0.9) for box in self.boxes]


class FakeRecognizer:
    """可配置"能否识别"的假人脸识别器。"""

    enabled = True

    def __init__(self, recognizable=True):
        self.recognizable = recognizable
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        height, width = image.shape[:2]
        return [
            FaceObservation(
                bbox=[int(width * 0.35), int(height * 0.1), int(width * 0.6), int(height * 0.45)],
                det_score=0.95 if self.recognizable else 0.3,
                embedding=np.ones(16, dtype=np.float32) if self.recognizable else None,
                yaw=5.0,
                quality=0.9 if self.recognizable else 0.1,
            )
        ]

    def identify(self, observation):
        return "face_global_1" if self.recognizable and observation.recognizable() else None

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean_output_dir():
    """测试产物写到系统临时目录，避免污染仓库 tmp/。"""
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="xzpipe_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _jpeg_frames(count=12, width=320, height=240):
    import cv2

    frames = []
    for index in range(count):
        image = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.rectangle(image, (100, 60), (180, 220), (200, 200, 200), -1)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        frames.append(
            VideoFrame(data=buffer.tobytes(), timestamp_ms=index * 100, width=width, height=height)
        )
    return frames


def test_scheduler_detects_every_n_frames():
    scheduler = FrameScheduler(SchedulerConfig(detect_every_n_frames=3, face_every_n_frames=15))
    plans = [scheduler.plan() for _ in range(7)]
    assert [p.run_detect for p in plans] == [True, False, False, True, False, False, True]
    assert plans[0].run_face is True
    assert all(not p.run_face for p in plans[1:])


def test_scheduler_adapts_when_over_budget():
    scheduler = FrameScheduler(
        SchedulerConfig(detect_every_n_frames=3, target_budget_ms=33, max_detect_interval=6)
    )
    scheduler.observe(120.0)
    assert scheduler.detect_interval == 4
    scheduler.observe(120.0)
    assert scheduler.detect_interval == 5
    stats = scheduler.stats()
    assert stats["level"] >= 2


def test_tracker_keeps_id_when_detection_skipped():
    tracker = ByteTrackTracker()
    first = tracker.step([Detection(100, 60, 180, 220, 0.9)], detected=True)
    assert len(first) == 1
    pid = first[0].track_id

    second = tracker.step([], detected=False)
    assert len(second) == 1
    assert second[0].track_id == pid
    assert second[0].detected is False
    assert second[0].detection.as_bbox() == [100, 60, 180, 220]


def test_pipeline_every_frame_has_bbox_and_detected_flag(_clean_output_dir):
    pipeline = VideoPipeline(
        PipelineConfig(fps=10, annotate=True),
        output_dir=_clean_output_dir,
        detector=FakeDetector([(100, 60, 180, 220)]),
        recognizer=FakeRecognizer(recognizable=True),
    )
    frames = _jpeg_frames(12)
    result = pipeline.analyze(frames, session_id="s1", name="unit")
    assert result.ok
    assert len(result.frames) == 12

    # ② 每一帧都要有 bbox；D15 降频下用 detected 区分真检测与外推
    for entry in result.frames:
        assert entry["persons"], "每一帧都应有人物 bbox（跟踪补帧）"
        for person in entry["persons"]:
            assert len(person["bbox"]) == 4
            assert "person_id" in person
            assert "speaking" in person
            assert "detected" in person

    detected = [p["detected"] for entry in result.frames for p in entry["persons"]]
    assert any(detected)
    assert not all(detected), "降频下应存在跟踪外推帧（detected=false）"

    # D15 判据：detected=false 帧占比 <= 2/3（检测每 3 帧一次时正好是边界，留浮点容差）
    assert result.stats["tracked_only_ratio"] <= 2.0 / 3.0 + 1e-3

    # person_id 全程不跳变
    ids = {p["person_id"] for entry in result.frames for p in entry["persons"]}
    assert len(ids) == 1

    # schema 校验
    validate_frames_schema(result.frames)
    pipeline.close()


def test_pipeline_omits_face_id_when_not_recognizable(_clean_output_dir):
    pipeline = VideoPipeline(
        PipelineConfig(fps=10, annotate=False),
        output_dir=_clean_output_dir,
        detector=FakeDetector([(100, 60, 180, 220)]),
        recognizer=FakeRecognizer(recognizable=False),
    )
    result = pipeline.analyze(_jpeg_frames(4), session_id="s2", name="unit")
    assert result.ok
    for entry in result.frames:
        for person in entry["persons"]:
            # ② 口径：没露出可识别人脸就不做人脸识别，face_id 字段缺失
            assert "face_id" not in person
    pipeline.close()


def test_pipeline_attaches_face_id_when_recognizable(_clean_output_dir):
    pipeline = VideoPipeline(
        PipelineConfig(fps=10, annotate=False),
        output_dir=_clean_output_dir,
        detector=FakeDetector([(100, 60, 180, 220)]),
        recognizer=FakeRecognizer(recognizable=True),
    )
    result = pipeline.analyze(_jpeg_frames(4), session_id="s3", name="unit")
    assert result.ok
    face_ids = {
        person.get("face_id") for entry in result.frames for person in entry["persons"]
    }
    assert face_ids == {"face_global_1"}
    pipeline.close()


def test_heuristic_speaker_prefers_frontal_moving_mouth():
    detector = HeuristicSpeakerDetector({"threshold": 0.5})
    candidates = [
        SpeakerCandidate(
            person_id=1,
            face_box=[0, 0, 160, 160],
            frame_width=320,
            frame_height=240,
            frontal_score=1.0,
            mouth_motion=0.9,
            face_id="face_a",
        ),
        SpeakerCandidate(
            person_id=2,
            face_box=[0, 0, 40, 40],
            frame_width=320,
            frame_height=240,
            frontal_score=0.2,
            mouth_motion=0.0,
            face_id="face_b",
        ),
    ]
    segment = SpeechSegment(start_ms=0, end_ms=1000, voiceprint_id="spk_1")
    scores = detector.score(candidates, segment, voiceprint_history={"face_a": "spk_1"})
    assert scores[0].person_id == 1
    assert detector.decide(scores, detector.threshold) == 1


def test_heuristic_speaker_marks_nobody_when_below_threshold():
    detector = HeuristicSpeakerDetector({"threshold": 0.99})
    candidates = [
        SpeakerCandidate(
            person_id=7,
            face_box=[0, 0, 32, 32],
            frame_width=320,
            frame_height=240,
            frontal_score=0.1,
            mouth_motion=0.0,
        )
    ]
    scores = detector.score(candidates, SpeechSegment(0, 1000))
    assert detector.decide(scores, detector.threshold) is None


def validate_frames_schema(frames):
    """② 的结构化 schema 校验（每帧每人物的必需字段）。"""
    assert isinstance(frames, list)
    for entry in frames:
        assert "timestamp_ms" in entry and isinstance(entry["timestamp_ms"], int)
        assert "persons" in entry and isinstance(entry["persons"], list)
        for person in entry["persons"]:
            assert isinstance(person["person_id"], int)
            assert isinstance(person["bbox"], list) and len(person["bbox"]) == 4
            assert all(isinstance(v, int) for v in person["bbox"])
            assert isinstance(person["speaking"], bool)
            assert isinstance(person["detected"], bool)
            if "face_id" in person:
                assert isinstance(person["face_id"], str) and person["face_id"]
                assert person["face_id"] not in ("", "unknown", "None")