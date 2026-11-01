"""端到端验证：视觉管线单例化后的行为。

不加载真实模型（用假检测器），验证：
1. 两个会话共享同一管线实例；
2. 会话结束不销毁共享管线，第二个会话无需重建；
3. 两个会话的嘴部历史互不污染。
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from core.utils.video.frames import VideoFrame, build_video_frame
from core.utils.video.pipeline import PipelineConfig, VideoPipeline, reset_video_pipeline
from core.utils.video.session import VideoSession
from core.utils.video.store import VideoStore


class FakeDetector:
    name = "fake"

    def detect(self, image):
        from core.utils.video.detector import Detection

        h, w = image.shape[:2]
        return [Detection(w * 0.2, h * 0.1, w * 0.8, h * 0.9, 0.9)]


class FakeRecognizer:
    enabled = True

    def detect(self, image):
        return []

    def identify(self, observation):
        return None

    def close(self):
        pass


def _jpeg(seed: int) -> bytes:
    import cv2

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture(autouse=True)
def _clean():
    reset_video_pipeline()
    try:
        yield
    finally:
        reset_video_pipeline()


def test_two_sessions_share_one_pipeline(tmp_path, monkeypatch):
    """两个会话必须拿到同一个管线对象（模型只加载一次）。"""
    created = []

    class Counted(VideoPipeline):
        def __init__(self, config=None, output_dir="data/video", **_kw):
            created.append(1)
            super().__init__(
                config or PipelineConfig(fps=10, annotate=False),
                output_dir=output_dir,
                detector=FakeDetector(),
                recognizer=FakeRecognizer(),
            )

    from core.utils.video import pipeline as pm

    monkeypatch.setattr(pm, "VideoPipeline", Counted)

    store = VideoStore(retention_rounds=0)
    a = VideoSession(
        {"video": {"enable": True, "analyze": True, "output_dir": str(tmp_path)}},
        session_id="sess-A", output_dir=str(tmp_path), store=store,
    )
    b = VideoSession(
        {"video": {"enable": True, "analyze": True, "output_dir": str(tmp_path)}},
        session_id="sess-B", output_dir=str(tmp_path), store=store,
    )

    pa, pb = a._get_pipeline(), b._get_pipeline()
    assert pa is pb, "两个会话应共享同一管线"
    assert len(created) == 1, f"模型应只加载一次，实际 {len(created)} 次"

    # A 会话结束，不能影响 B
    a.close()
    assert b._get_pipeline() is pa, "A 关闭后 B 仍应拿到同一管线"
    assert len(created) == 1, "A 关闭不应触发重新加载"


def test_two_sessions_mouth_history_isolated(tmp_path):
    """两个会话并发跑，嘴部历史各自独立（不串）。"""
    pipeline = VideoPipeline(
        PipelineConfig(fps=10, annotate=False),
        output_dir=str(tmp_path),
        detector=FakeDetector(),
        recognizer=FakeRecognizer(),
    )

    def run(session_id):
        store = VideoStore(retention_rounds=0)
        session = VideoSession(
            {"video": {"enable": True, "analyze": True, "output_dir": str(tmp_path)}},
            session_id=session_id,
            output_dir=str(tmp_path),
            store=store,
            pipeline=pipeline,
        )
        session.on_user_speech_start(0)
        for i in range(6):
            session.push_frame_bytes(build_video_frame(_jpeg(i), i * 100), i * 100)
        session.on_user_speech_stop(600)
        art = session.build_round(None)
        session.close()
        return art

    result_a = run("iso-A")
    result_b = run("iso-B")

    assert result_a.has_video and result_b.has_video
    frames_a = (result_a.user_analysis or {}).get("frames", [])
    frames_b = (result_b.user_analysis or {}).get("frames", [])
    assert len(frames_a) == len(frames_b) == 6
    # 同样输入、同样轮次起点，两轮的 speaking 序列必须一致（互不污染）
    speak_a = [[p["speaking"] for p in f["persons"]] for f in frames_a]
    speak_b = [[p["speaking"] for p in f["persons"]] for f in frames_b]
    assert speak_a == speak_b
