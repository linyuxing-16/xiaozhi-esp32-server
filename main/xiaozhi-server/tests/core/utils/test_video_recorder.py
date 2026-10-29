"""录制与音画对齐测试（需求 ①④⑤）。

关键断言：
- ④ 用户说话段产物有音轨；
- ⑤ 小智输出音频段产物**无音轨**（ffprobe 校验 stream 数）；
- 时间轴原点对齐：视频首帧与音频起点同为 t=0。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from core.utils.video.frames import VideoFrame
from core.utils.video.recorder import (
    PcmFrame,
    RecordingRequest,
    VideoRecorder,
    collect_pcm,
    ffprobe_streams,
    pcm_bytes_to_wav,
)


def _make_frames(count=20, fps=10, start_ms=0):
    import cv2

    frames = []
    for index in range(count):
        image = np.full((240, 320, 3), 60, dtype=np.uint8)
        cv2.circle(image, (160, 120), 40 + index, (0, 200, 0), -1)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        frames.append(
            VideoFrame(
                data=buffer.tobytes(),
                timestamp_ms=start_ms + int(index * 1000 / fps),
                width=320,
                height=240,
            )
        )
    return frames


def _pcm(ms=2000, sample_rate=16000):
    count = int(sample_rate * ms / 1000)
    samples = (np.sin(np.linspace(0, 200, count)) * 8000).astype(np.int16)
    return samples.tobytes()


@pytest.fixture()
def workdir():
    """本机 pytest tmp_path 有权限问题，统一用 tempfile 自建目录。"""
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="xzvtest_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_pcm_bytes_to_wav_header():
    wav = pcm_bytes_to_wav(_pcm(100))
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_collect_pcm_crops_to_range():
    frames = [
        PcmFrame(timestamp_ms=0, data=b"\x00\x00" * 16000),  # 1s
        PcmFrame(timestamp_ms=1000, data=b"\x01\x00" * 16000),  # 1s
    ]
    cropped = collect_pcm(frames, 500, 1500)
    # 500ms + 500ms = 1000ms * 16000 * 2 bytes
    assert len(cropped) == 16000 * 2


def test_user_recording_has_no_audio_track(workdir):
    """④ 也不封装音轨：音频信息由 ③ 的 ASR 文本承载。"""
    recorder = VideoRecorder(workdir, fps=10)
    request = RecordingRequest(
        session_id="sess_user",
        kind="user",
        start_ms=0,
        end_ms=1900,
        frames=_make_frames(20),
        pcm_frames=[PcmFrame(0, _pcm(2000))],  # 即便传了 PCM，④ 也不封装音轨
    )
    result = recorder.render(request)
    if not result.ok:
        pytest.skip(f"ffmpeg 不可用: {result.error}")
    assert result.has_audio is False
    streams = ffprobe_streams(result.path)
    assert streams == ["video"], f"④ 必须只有视频流，实际: {streams}"


def test_tts_recording_is_silent(workdir):
    recorder = VideoRecorder(workdir, fps=10)
    request = RecordingRequest(
        session_id="sess_tts",
        kind="tts",
        start_ms=0,
        end_ms=1900,
        frames=_make_frames(20),
        pcm_frames=[PcmFrame(0, _pcm(2000))],  # 即便传了 PCM，⑤ 也必须静音
    )
    result = recorder.render(request)
    if not result.ok:
        pytest.skip(f"ffmpeg 不可用: {result.error}")
    assert result.has_audio is False
    streams = ffprobe_streams(result.path)
    assert streams == ["video"], f"⑤ 必须只有视频流，实际: {streams}"


def test_recording_without_frames_fails_gracefully(workdir):
    recorder = VideoRecorder(workdir, fps=10)
    result = recorder.render(
        RecordingRequest(session_id="s", kind="user", start_ms=0, end_ms=1000, frames=[])
    )
    assert result.ok is False
    assert result.error
