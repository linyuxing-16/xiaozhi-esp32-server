"""短视频兜底：<0.5s 的片段改为每 0.1s 抽一帧发图片（image_url）。

背景：上游视频网关（minimax-m3 / CodeBuddy）对过短的 mp4 会直接返回
400（code 11133 参数被拒），实测 10fps 下 5 帧/0.5s 必失败、6 帧/0.6s 通过。
因此当片段时长低于 video.image_fallback_max_ms 时，不再发 mp4，改为抽帧发图片。

覆盖：
- image_url 内容块与 data URL 格式；
- build_content_blocks 按 attachment["type"] 分流（video / image）；
- 短视频走图片、长视频仍走 video_url（回归保护）；
- 抽帧间隔与去重；
- 图片路径已登记进 VideoStore，会话结束随 D5 一并删除；
- analyze=false 时回退原始帧仍可用；
- ⑤ 索引 < ④ 索引在"图片 + 图片"组合下依然成立。
"""
from __future__ import annotations

import base64
import os

import numpy as np
import pytest

from core.utils.dialogue import Dialogue
from core.utils.video.frames import VideoFrame, build_video_frame
from core.utils.video.multimodal import (
    build_content_blocks,
    image_data_url,
    image_url_block,
)
from core.utils.video.pipeline import PipelineConfig, VideoPipeline
from core.utils.video.round import TTS_MESSAGE_TEXT, put_round_messages
from core.utils.video.session import RoundArtifacts, VideoSession
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


def _frames(count: int, start_ms: int = 0, interval_ms: int = 100):
    return [
        VideoFrame(data=_jpeg(index), timestamp_ms=start_ms + index * interval_ms)
        for index in range(count)
    ]


@pytest.fixture()
def workdir():
    import shutil
    import tempfile

    path = tempfile.mkdtemp(prefix="xzimg_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _build_session(workdir, store, pipeline=None, **video_cfg):
    config = {"video": {"enable": True, "analyze": True, "output_dir": workdir}}
    config["video"].update(video_cfg)
    return VideoSession(
        config,
        session_id="sess-img",
        output_dir=workdir,
        store=store,
        pipeline=pipeline,
    )


def _pipeline(workdir):
    return VideoPipeline(
        PipelineConfig(fps=10, annotate=True),
        output_dir=workdir,
        detector=FakeDetector(),
        recognizer=FakeRecognizer(),
    )


# ---------------------------------------------------------------- 内容块
def test_image_url_block_is_data_url(workdir):
    path = os.path.join(workdir, "frame.jpg")
    payload = b"\xff\xd8\xff\xe0" + b"x" * 50
    with open(path, "wb") as handle:
        handle.write(payload)

    block = image_url_block(path)
    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == payload
    assert image_data_url(path) == url


def test_build_content_blocks_dispatches_image_and_video(workdir):
    image_path = os.path.join(workdir, "a.jpg")
    with open(image_path, "wb") as handle:
        handle.write(b"\xff\xd8\xff\xe0" + b"x" * 10)
    video_path = os.path.join(workdir, "b.mp4")
    with open(video_path, "wb") as handle:
        handle.write(b"\x00\x00\x00\x18ftypmp42" + b"x" * 10)

    image_blocks = build_content_blocks(
        "看图", [{"type": "image", "kind": "user", "path": image_path}]
    )
    assert [b["type"] for b in image_blocks][:2] == ["text", "image_url"]

    video_blocks = build_content_blocks(
        "看视频", [{"type": "video", "kind": "user", "path": video_path}]
    )
    assert [b["type"] for b in video_blocks][:2] == ["text", "video_url"]


# ---------------------------------------------------------------- 抽帧算法
def test_pick_image_indices_takes_first_frame_and_each_slot(workdir):
    session = _build_session(workdir, VideoStore())
    stamps = [0, 100, 200, 300, 400]
    assert session._pick_image_indices(stamps, 100) == [0, 1, 2, 3, 4]

    # 帧率比间隔高：每个 100ms 槽位只取第一帧
    dense = [0, 20, 40, 100, 120, 200]
    assert session._pick_image_indices(dense, 100) == [0, 3, 5]

    # 重复时间戳不重复抽
    dup = [0, 0, 0, 100]
    assert session._pick_image_indices(dup, 100) == [0, 3]

    assert session._pick_image_indices([], 100) == []


# ---------------------------------------------------------------- 短视频
def test_short_clip_becomes_images_not_video(workdir):
    pipeline = _pipeline(workdir)
    store = VideoStore()
    session = _build_session(workdir, store, pipeline)
    try:
        # 400ms 片段，10fps -> 5 帧
        session.on_user_speech_start(0)
        for index in range(5):
            session.push_frame_bytes(build_video_frame(_jpeg(index), index * 100), index * 100)
        session.on_user_speech_stop(400)
        artifacts = session.build_round(None)

        assert artifacts.user_images, "短视频应产出图片"
        assert artifacts.user_video is None, "短视频不应再产出 mp4"
        # 区间取闭区间：0/100/200/300/400 共 5 帧，对应 5 个 100ms 槽位
        assert len(artifacts.user_images) == 5
        for path in artifacts.user_images:
            assert os.path.exists(path)
            with open(path, "rb") as handle:
                assert handle.read(2) == b"\xff\xd8", "应是 JPEG"

        # 附件类型是 image，内容块是 image_url
        attachments = artifacts.attachments()
        assert attachments and all(a["type"] == "image" for a in attachments)
        blocks = build_content_blocks("用户说的话", attachments)
        assert [b["type"] for b in blocks].count("image_url") == len(artifacts.user_images)
        assert all(b["type"] != "video_url" for b in blocks)
    finally:
        session.close()
        pipeline.close()


def test_long_clip_still_uses_video_url(workdir):
    """回归保护：时长 >= 阈值时行为不变，仍发 mp4。"""
    pipeline = _pipeline(workdir)
    session = _build_session(workdir, VideoStore(), pipeline)
    try:
        session.on_user_speech_start(0)
        for index in range(20):
            session.push_frame_bytes(build_video_frame(_jpeg(index), index * 100), index * 100)
        session.on_user_speech_stop(1900)
        artifacts = session.build_round(None)

        assert artifacts.user_video, "长视频应仍产出 mp4"
        assert not artifacts.user_images
        blocks = build_content_blocks("用户说的话", artifacts.attachments())
        assert any(b["type"] == "video_url" for b in blocks)
        assert all(b["type"] != "image_url" for b in blocks)
    finally:
        session.close()
        pipeline.close()


def test_short_clip_with_analysis_disabled_uses_raw_frames(workdir):
    """analyze=false 时没有标注帧，回退原始帧仍应能抽帧发图。"""
    session = _build_session(workdir, VideoStore(), analyze=False)
    try:
        session.on_user_speech_start(0)
        for index in range(3):
            session.push_frame_bytes(build_video_frame(_jpeg(index), index * 100), index * 100)
        session.on_user_speech_stop(250)
        artifacts = session.build_round(None)
        assert artifacts.user_images, "关闭分析后仍应抽帧发图"
        assert artifacts.user_analysis is None
    finally:
        session.close()


def test_short_clip_images_are_registered_and_cleaned(workdir):
    """图片必须登记进 VideoStore，会话结束随 D5 一并删除。"""
    pipeline = _pipeline(workdir)
    store = VideoStore()
    session = _build_session(workdir, store, pipeline)
    try:
        session.on_user_speech_start(0)
        for index in range(4):
            session.push_frame_bytes(build_video_frame(_jpeg(index), index * 100), index * 100)
        session.on_user_speech_stop(350)
        artifacts = session.build_round(None)
        paths = list(artifacts.user_images)
        assert paths
        assert all(store.refcount(path) > 0 for path in paths)

        session.close()
        pipeline.close()
        for path in paths:
            assert not os.path.exists(path), f"会话结束应删除图片: {path}"
    finally:
        pass


def test_image_fallback_can_be_disabled(workdir):
    """image_fallback_max_ms=0 时关闭兜底，短视频仍走 mp4（保持旧行为）。"""
    pipeline = _pipeline(workdir)
    session = _build_session(workdir, VideoStore(), pipeline, image_fallback_max_ms=0)
    try:
        session.on_user_speech_start(0)
        for index in range(4):
            session.push_frame_bytes(build_video_frame(_jpeg(index), index * 100), index * 100)
        session.on_user_speech_stop(350)
        artifacts = session.build_round(None)
        assert artifacts.user_video
        assert not artifacts.user_images
    finally:
        session.close()
        pipeline.close()


# ---------------------------------------------------------------- 消息顺序
def test_tts_index_still_before_user_with_images():
    """⑤ 索引 < ④ 索引，在"图片 + 图片"组合下依然成立。"""
    dialogue = Dialogue()
    artifacts = RoundArtifacts(
        round_index=1,
        user_images=["/tmp/u0.jpg", "/tmp/u1.jpg"],
        tts_images=["/tmp/t0.jpg"],
        user_analysis={"frames": [], "stats": {}},
        tts_analysis={"frames": [], "stats": {}},
    )
    put_round_messages(dialogue, "用户说的话", artifacts)
    messages = dialogue.get_llm_dialogue()
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) == 2
    assert messages.index(user_messages[0]) < messages.index(user_messages[1])
    assert user_messages[0]["content"][0]["text"] == TTS_MESSAGE_TEXT
    assert user_messages[1]["content"][0]["text"] == "用户说的话"
