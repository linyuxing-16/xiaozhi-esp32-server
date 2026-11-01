"""进程级视觉管线单例测试（模型只加载一次、跨会话共享）。

覆盖：
- 连续调用返回同一对象；
- 并发获取时只创建一次（双重检查锁有效）；
- reset 后能重建；
- 会话 close 不会连带关闭共享管线。
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from core.utils.video import pipeline as pipeline_module
from core.utils.video.pipeline import (
    PipelineConfig,
    VideoPipeline,
    get_video_pipeline,
    reset_video_pipeline,
)


class _FakeDetector:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return []


class _FakeRecognizer:
    enabled = True

    def __init__(self):
        self.closed = False

    def detect(self, image):
        return []

    def identify(self, observation):
        return None

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个用例前后都清空单例，避免相互污染。"""
    reset_video_pipeline()
    try:
        yield
    finally:
        reset_video_pipeline()


@pytest.fixture()
def counting_pipeline(monkeypatch):
    """把 VideoPipeline 换成计数包装，避免真的加载模型。"""
    created = []

    class _CountedPipeline(VideoPipeline):
        def __init__(self, config=None, output_dir="data/video", **kwargs):
            created.append(output_dir)
            super().__init__(
                config or PipelineConfig(fps=10, annotate=False),
                output_dir=output_dir,
                detector=_FakeDetector(),
                recognizer=_FakeRecognizer(),
            )

    monkeypatch.setattr(pipeline_module, "VideoPipeline", _CountedPipeline)
    return created


def test_get_video_pipeline_returns_same_instance(counting_pipeline, tmp_path):
    first = get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))
    second = get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))
    assert first is second
    assert len(counting_pipeline) == 1


def test_concurrent_get_creates_only_once(counting_pipeline, tmp_path):
    """8 线程并发获取，模型只能加载一次。"""
    results = []
    errors = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait()
            results.append(get_video_pipeline({"fps": 10}, output_dir=str(tmp_path)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(counting_pipeline) == 1, f"应只创建一次，实际 {len(counting_pipeline)} 次"
    assert len({id(item) for item in results}) == 1, "所有线程应拿到同一对象"


def test_reset_creates_new_instance(counting_pipeline, tmp_path):
    first = get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))
    reset_video_pipeline()
    second = get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))
    assert first is not second
    assert len(counting_pipeline) == 2


def test_shared_pipeline_close_is_not_called_by_reset_only(monkeypatch, tmp_path):
    """reset 会关闭管线（关掉人脸底库），普通取用不会。"""
    closed = []

    class _Closable(VideoPipeline):
        def __init__(self, config=None, output_dir="data/video", **_kwargs):
            super().__init__(
                config or PipelineConfig(fps=10, annotate=False),
                output_dir=output_dir,
                detector=_FakeDetector(),
                recognizer=_FakeRecognizer(),
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(pipeline_module, "VideoPipeline", _Closable)
    get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))
    assert closed == [], "取用不应关闭管线"
    reset_video_pipeline()
    assert closed == [True], "reset 应关闭管线"


def test_session_close_does_not_close_shared_pipeline(monkeypatch, tmp_path):
    """会话 close 只能清理自己的帧缓冲，不能关掉共享管线。"""
    from core.utils.video.session import VideoSession
    from core.utils.video.store import VideoStore

    closed = []

    class _Closable(VideoPipeline):
        def __init__(self, config=None, output_dir="data/video", **_kwargs):
            super().__init__(
                config or PipelineConfig(fps=10, annotate=False),
                output_dir=output_dir,
                detector=_FakeDetector(),
                recognizer=_FakeRecognizer(),
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(pipeline_module, "VideoPipeline", _Closable)
    shared = get_video_pipeline({"fps": 10}, output_dir=str(tmp_path))

    session = VideoSession(
        {"video": {"enable": True, "output_dir": str(tmp_path)}},
        session_id="sess-a",
        output_dir=str(tmp_path),
        store=VideoStore(retention_rounds=0),
    )
    assert session._get_pipeline() is shared, "会话应复用共享单例"

    session.close()
    assert closed == [], "会话 close 不应关闭共享管线"
    assert get_video_pipeline({"fps": 10}, output_dir=str(tmp_path)) is shared
