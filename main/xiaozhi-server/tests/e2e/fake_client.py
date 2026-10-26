"""Python 测试客户端（阶段 H，验收 1）。

模拟**客户端向服务端发送** opus 音频 + JPEG 视频帧（与修正后的验收标准 1 一致），
跑 3 个完整轮次。

两种用法：

1. 离线自检（不需要服务器）::

       python tests/e2e/fake_client.py --offline

   生成模拟帧与音频，验证帧封装、区间构造与 ①-⑥ 的数据结构。

2. 连真实服务端::

       python tests/e2e/fake_client.py --url ws://127.0.0.1:8000/xiaozhi/v1/

   需要服务端 config.yaml 里 `video.enable: true`；本机缺 opuslib_next 时纯音频
   帧会退化为静音 PCM 直发（服务端会按 opus 解码失败丢弃），视频帧仍按 type=2 发送。

退出码：0 表示全部检查通过。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import tempfile
import uuid

# 允许直接运行本文件（不需要安装包）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

VIDEO_TYPE = 2


def build_video_frame(jpeg: bytes, timestamp_ms: int) -> bytes:
    """type(1) + reserved(1) + timestamp(4 BE) + payload_size(4 BE) + JPEG。"""
    return struct.pack(">BBII", VIDEO_TYPE, 0, timestamp_ms & 0xFFFFFFFF, len(jpeg)) + jpeg


def make_jpeg(width: int = 320, height: int = 240, seed: int = 0) -> bytes:
    """生成一张测试 JPEG（有一个人形色块，便于视觉管线检出）。"""
    import cv2
    import numpy as np

    image = np.full((height, width, 3), 45, dtype=np.uint8)
    cv2.rectangle(image, (100, 60), (180, 220), (205, 205, 205), -1)
    cv2.circle(image, (140, 90), 10 + seed % 6, (110, 150, 210), -1)
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buffer.tobytes()


def make_opus_frame() -> bytes:
    """生成一帧 opus（若能编码就编码，否则给静音 PCM 占位）。"""
    try:
        import opuslib_next

        encoder = opuslib_next.Encoder(16000, 1, opuslib_next.APPLICATION_AUDIO)
        return encoder.encode(b"\x00\x00" * 960, 960)
    except Exception:  # noqa: BLE001
        return b"\x00\x00" * 960


def hello_message(device_id: str, video: bool = True) -> str:
    return json.dumps(
        {
            "type": "hello",
            "version": 3,
            "features": {"mcp": False, "aec": False, "video": video},
            "transport": "websocket",
            "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
        },
        ensure_ascii=False,
    )


class FakeClient:
    """把一轮对话（用户说话段 + 小智输出段）编码成待发送的消息序列。"""

    def __init__(self, rounds: int = 3, fps: int = 10):
        self.rounds = rounds
        self.fps = fps
        self.sent_texts: list[str] = []
        self.sent_bytes = 0
        self.video_frames_sent = 0
        self.audio_frames_sent = 0

    def round_messages(self, round_index: int, user_seconds: float = 1.5, tts_seconds: float = 1.0):
        """返回 [(kind, payload)]，kind in {text, audio, video}。"""
        messages: list[tuple[str, object]] = []
        timestamp = 0

        # 用户说话段：VAD 起 -> 止
        messages.append(("text", json.dumps({"type": "listen", "state": "start", "mode": "auto"})))
        for index in range(int(user_seconds * self.fps)):
            messages.append(("audio", make_opus_frame()))
            messages.append(("video", build_video_frame(make_jpeg(seed=index), timestamp)))
            timestamp += int(1000 / self.fps)
        messages.append(("text", json.dumps({"type": "listen", "state": "stop"})))

        # 小智输出段（服务端 TTS 时段的录像素材）
        for index in range(int(tts_seconds * self.fps)):
            messages.append(("video", build_video_frame(make_jpeg(seed=index + 50), timestamp)))
            timestamp += int(1000 / self.fps)
        return messages

    async def run(self, url: str, token: str | None = None) -> dict:
        import websockets

        device_id = "aa:bb:cc:dd:ee:ff"
        headers = {"Device-Id": device_id, "Client-Id": str(uuid.uuid4()), "Protocol-Version": "3"}
        if token:
            headers["Authorization"] = token if " " in token else f"Bearer {token}"

        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            await ws.send(hello_message(device_id))
            server_hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            self.sent_texts.append(json.dumps(server_hello, ensure_ascii=False))

            stop_reading = asyncio.Event()
            received: list[dict] = []

            async def reader():
                try:
                    while not stop_reading.is_set():
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        if isinstance(message, bytes):
                            received.append({"type": "audio", "len": len(message)})
                        else:
                            try:
                                received.append(json.loads(message))
                            except json.JSONDecodeError:
                                received.append({"raw": str(message)[:80]})
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception:  # noqa: BLE001
                    pass

            reader_task = asyncio.create_task(reader())
            for round_index in range(self.rounds):
                for kind, payload in self.round_messages(round_index):
                    if kind == "text":
                        await ws.send(payload)
                    else:
                        await ws.send(payload)
                        if kind == "video":
                            self.video_frames_sent += 1
                        else:
                            self.audio_frames_sent += 1
                        self.sent_bytes += len(payload)
                    await asyncio.sleep(0.001)
                await asyncio.sleep(1.0)

            stop_reading.set()
            reader_task.cancel()
            await ws.close()

        return {
            "server_hello": server_hello,
            "received": received,
            "video_frames_sent": self.video_frames_sent,
            "audio_frames_sent": self.audio_frames_sent,
            "bytes_sent": self.sent_bytes,
        }


def offline_check(rounds: int = 3) -> int:
    """离线自检：不连服务器，验证帧格式与区间/结构化链路。"""
    from core.utils.video.frames import parse_video_frame
    from core.utils.video.pipeline import PipelineConfig, VideoPipeline
    from core.utils.video.recorder import VideoRecorder
    from core.utils.asr_result import normalize_asr_result

    client = FakeClient(rounds=rounds)
    total_video = 0
    for round_index in range(rounds):
        messages = client.round_messages(round_index)
        video_messages = [payload for kind, payload in messages if kind == "video"]
        total_video += len(video_messages)
        parsed = [parse_video_frame(message) for message in video_messages]
        assert all(item is not None for item in parsed), "视频帧头解析失败"
        assert all(item[1].startswith(b"\xff\xd8") for item in parsed), "负载不是 JPEG"

    workdir = tempfile.mkdtemp(prefix="xzfake_")
    from core.utils.video.frames import VideoFrame

    frames = [
        VideoFrame(
            data=make_jpeg(seed=index),
            timestamp_ms=index * 100,
            width=320,
            height=240,
        )
        for index in range(15)
    ]
    recorder = VideoRecorder(workdir, fps=10)
    result = recorder.render(
        __import__("core.utils.video.recorder", fromlist=["RecordingRequest"]).RecordingRequest(
            session_id="offline", kind="tts", start_ms=0, end_ms=1400, frames=frames
        )
    )
    assert result.ok, f"录制失败: {result.error}"
    assert result.has_audio is False, "⑤ 必须无音轨"

    pipeline = VideoPipeline(PipelineConfig(fps=10, annotate=True), output_dir=workdir)
    analysis = pipeline.analyze(frames, session_id="offline", name="check")
    assert analysis.ok
    assert analysis.frames, "应有结构化帧"
    pipeline.close()

    asr = normalize_asr_result(
        {"content": "你好。今天天气不错。"}, duration_ms=2000, vad_segments=[(0, 2000)]
    )
    assert asr.sentences and asr.sentences[0].time_source == "vad"
    assert "time: [" in asr.to_yaml()

    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"[fake_client] 离线自检通过：{rounds} 轮，共 {total_video} 个视频帧消息")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="小智测试客户端（opus + JPEG 上行，3 轮）")
    parser.add_argument("--offline", action="store_true", help="离线自检，不连服务器")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/xiaozhi/v1/", help="服务端 WebSocket 地址")
    parser.add_argument("--token", default=None, help="认证 token（可选）")
    parser.add_argument("--rounds", type=int, default=3, help="轮次数")
    args = parser.parse_args()

    if args.offline:
        return offline_check(args.rounds)

    client = FakeClient(rounds=args.rounds)
    result = asyncio.run(client.run(args.url, args.token))
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
