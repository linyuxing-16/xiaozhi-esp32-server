"""嘴部运动追踪的轮次隔离测试。

背景：MouthMotionTracker 按 person_id(int) 记账，而 person_id 每轮视频都从 1
重新编号。管线改成进程级单例后，如果 tracker 仍挂在 pipeline 上，不同会话的
person_id=1 会共用同一份历史，导致"是否说话"判定被别的会话污染。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.utils.video import pipeline as pipeline_module
from core.utils.video.pipeline import PipelineConfig, VideoPipeline
from core.utils.video.frames import VideoFrame
from core.utils.video.speaker import MouthMotionTracker


class _StaticDetector:
    """固定返回一个人物的假检测器。"""

    name = "fake"

    def detect(self, image):
        from core.utils.video.detector import Detection

        height, width = image.shape[:2]
        return [
            Detection(
                x1=width * 0.2,
                y1=height * 0.1,
                x2=width * 0.8,
                y2=height * 0.9,
                score=0.9,
            )
        ]


class _StaticRecognizer:
    enabled = True

    def detect(self, image):
        return []

    def identify(self, observation):
        return None

    def close(self):
        pass


def _frames(count=6, base=40):
    """生成带纹理的 JPEG 帧，保证 cv2 灰度可用。"""
    import cv2

    result = []
    for index in range(count):
        image = np.full((240, 320, 3), base, dtype=np.uint8)
        cv2.rectangle(image, (100, 60), (180, 220), (200, 200, 200), -1)
        # 加一点逐帧变化的纹理，模拟嘴部运动
        cv2.circle(image, (140, 150 + index * 2), 8, (255, 255, 255), -1)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        result.append(
            VideoFrame(data=buffer.tobytes(), timestamp_ms=index * 100, width=320, height=240)
        )
    return result


@pytest.fixture()
def pipeline(tmp_path):
    return VideoPipeline(
        PipelineConfig(fps=10, annotate=False),
        output_dir=str(tmp_path),
        detector=_StaticDetector(),
        recognizer=_StaticRecognizer(),
    )


def test_pipeline_does_not_hold_mouth_tracker(pipeline):
    """管线对象上不能再挂跨轮共享的 mouth_tracker。"""
    assert not hasattr(pipeline, "mouth_tracker"), (
        "mouth_tracker 必须按轮次新建，不能挂在共享管线上"
    )


def test_mouth_tracker_recreated_per_analyze(pipeline, monkeypatch, tmp_path):
    """每次 analyze 都要新建独立的 MouthMotionTracker 实例。"""
    created = []
    original = MouthMotionTracker

    class _RecordingTracker(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(pipeline_module, "MouthMotionTracker", _RecordingTracker)

    pipeline.analyze(_frames(6), session_id="round1", name="r1")
    pipeline.analyze(_frames(6), session_id="round2", name="r2")

    assert len(created) == 2, f"两轮应各建一个 tracker，实际 {len(created)} 个"
    assert created[0] is not created[1], "两轮的 tracker 必须是不同对象"


def test_two_rounds_do_not_share_mouth_history(pipeline):
    """连续两轮之间，嘴部历史不延续，避免旧历史影响新轮次判定。"""
    first = pipeline.analyze(_frames(6, base=40), session_id="a", name="a")
    second = pipeline.analyze(_frames(6, base=40), session_id="b", name="b")
    assert first.ok and second.ok
    # 两轮输入完全相同，隔离正确时首帧都还没有历史（mouth_motion 因无前一帧而为 0）
    assert [p["speaking"] for p in first.frames[0]["persons"]] == [
        p["speaking"] for p in second.frames[0]["persons"]
    ]
