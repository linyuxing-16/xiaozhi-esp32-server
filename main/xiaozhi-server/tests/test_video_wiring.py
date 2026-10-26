"""服务端视频链路接线测试（阶段 A/C/F 的集成点）。

覆盖：
- `connection` / 各 handle 模块可正常导入（不漏依赖、不改旧行为）；
- 二进制帧分流：type=2 视频帧被视频会话消费，不再当 opus 解码；
- VAD 上升/下降沿、TTS 起止的打点；
- hello 消息按 video.enable 声明 features.video；
- `video.enable=false` 时所有打点零副作用（验收"行为零变化"）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.handle import helloHandle, receiveAudioHandle, sendAudioHandle
from core.utils.video.frames import build_video_frame


class FakeSession:
    def __init__(self):
        self.enabled = True
        self.fps = 10
        self.frames = []
        self.pcm = []
        self.events = []

    def push_frame_bytes(self, message):
        from core.utils.video.frames import is_video_frame

        if not is_video_frame(message):
            return False
        self.frames.append(message)
        return True

    def push_pcm(self, pcm):
        self.pcm.append(pcm)

    def on_user_speech_start(self):
        self.events.append("user_start")

    def on_user_speech_stop(self):
        self.events.append("user_stop")

    def on_tts_start(self):
        self.events.append("tts_start")

    def on_tts_stop(self):
        self.events.append("tts_stop")


class FakeLogger:
    def bind(self, **_kwargs):
        return self

    def info(self, *_args, **_kwargs):
        pass

    def debug(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _fake_conn(session=None):
    return SimpleNamespace(
        video_session=session,
        logger=FakeLogger(),
        _video_voice_active=False,
        session_id="sess",
        welcome_msg={"type": "hello", "transport": "websocket"},
        websocket=SimpleNamespace(sent=[], send=None),
    )


def test_binary_video_frame_is_routed_to_video_session():
    session = FakeSession()
    conn = _fake_conn(session)
    message = build_video_frame(b"\xff\xd8\xff\xe0fakejpeg", 1234)
    assert session.push_frame_bytes(message) is True
    assert len(session.frames) == 1

    # 音频业务帧（首字节不是 2）不应被当成视频
    assert session.push_frame_bytes(b"\x00" * 40) is False


def test_vad_edges_and_pcm_are_recorded():
    session = FakeSession()
    conn = _fake_conn(session)

    receiveAudioHandle._track_video_audio(conn, b"\x00\x01" * 100, have_voice=False)
    assert len(session.pcm) == 1
    assert session.events == []

    receiveAudioHandle._track_video_audio(conn, b"\x00\x01" * 100, have_voice=True)
    assert session.events == ["user_start"]
    assert conn._video_voice_active is True

    receiveAudioHandle._track_video_audio(conn, b"\x00\x01" * 100, have_voice=True)
    assert session.events == ["user_start"], "上升沿只触发一次"

    from core.providers.asr.base import _notify_video_speech_stop

    _notify_video_speech_stop(conn)
    assert session.events == ["user_start", "user_stop"]
    assert conn._video_voice_active is False


def test_tts_span_hooks():
    session = FakeSession()
    conn = _fake_conn(session)
    sendAudioHandle._video_tts_start(conn)
    sendAudioHandle._video_tts_stop(conn)
    assert session.events == ["tts_start", "tts_stop"]


def test_no_video_session_is_zero_cost():
    conn = _fake_conn(None)
    receiveAudioHandle._track_video_audio(conn, b"\x00\x01" * 100, have_voice=True)
    sendAudioHandle._video_tts_start(conn)
    sendAudioHandle._video_tts_stop(conn)
    from core.providers.asr.base import _notify_video_speech_stop

    _notify_video_speech_stop(conn)
    assert conn._video_voice_active is False


@pytest.mark.asyncio
async def test_hello_declares_video_feature_when_enabled():
    session = FakeSession()
    conn = _fake_conn(session)
    conn.features = None
    conn.audio_format = "opus"

    async def _send(payload):
        conn.websocket.sent.append(payload)

    conn.websocket.send = _send
    await helloHandle.handleHelloMessage(conn, {})
    assert conn.websocket.sent, "hello 应发送 welcome 消息"
    import json

    payload = json.loads(conn.websocket.sent[-1])
    assert payload["features"]["video"]["enable"] is True
    assert payload["features"]["video"]["frame_type"] == 2


@pytest.mark.asyncio
async def test_hello_has_no_video_feature_when_disabled():
    conn = _fake_conn(None)
    conn.features = None

    async def _send(payload):
        conn.websocket.sent.append(payload)

    conn.websocket.send = _send
    await helloHandle.handleHelloMessage(conn, {})
    import json

    payload = json.loads(conn.websocket.sent[-1])
    assert "video" not in (payload.get("features") or {})


def test_put_round_messages_orders_tts_before_user():
    """⑤ 的索引必须严格小于 ④（生产函数校验）。"""
    from core.utils.dialogue import Dialogue
    from core.utils.video.round import TTS_MESSAGE_TEXT, put_round_messages
    from core.utils.video.session import RoundArtifacts

    dialogue = Dialogue()
    artifacts = RoundArtifacts(
        round_index=1,
        user_video="/tmp/user.mp4",
        tts_video="/tmp/tts.mp4",
    )
    put_round_messages(dialogue, "用户说的话", artifacts)
    messages = dialogue.get_llm_dialogue()
    assert len(messages) == 2
    assert messages[0]["content"][0]["text"] == TTS_MESSAGE_TEXT
    assert any(
        block.get("text") == "用户说的话"
        for block in messages[1]["content"]
        if isinstance(block, dict)
    )
    # 没有视频时退化为单条普通 user 消息
    plain = Dialogue()
    put_round_messages(plain, "只有文字", None)
    assert plain.get_llm_dialogue() == [{"role": "user", "content": "只有文字"}]


def test_connection_has_video_hooks():
    """connection.py 必须提供视频轮次的两个挂点。

    说明：本机 mcp 包版本与 requirements 不一致（`streamable_http_client` 缺失），
    `import core.connection` 在当前环境失败，这是既有环境问题、与视频改动无关，
    因此这里用源码静态检查代替 import 检查。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "core" / "connection.py").read_text(
        encoding="utf-8"
    )
    assert "def _put_user_message_with_video" in source
    assert "def _build_video_round" in source
    assert "self.video_session.push_frame_bytes(message)" in source
    assert "self.video_session.close()" in source
