"""视频帧数据结构、二进制协议解析与帧缓冲。

二进制帧格式（D11，type=2 独立通道）：

    +--------+---------+-------------------+----------------------+-------------+
    | type=2 | reserved| timestamp(4B, BE) | payload_size(4B, BE)| JPEG 负载   |
    +--------+---------+-------------------+----------------------+-------------+
    1 字节    1 字节    4 字节              4 字节                 payload_size

timestamp 是设备侧单调时钟（毫秒），服务端只用于相对对齐。
"""
from __future__ import annotations

import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Tuple

VIDEO_FRAME_TYPE = 2
# type(1) + reserved(1) + timestamp(4) + payload_size(4)
VIDEO_HEADER = struct.Struct(">BBII")
VIDEO_HEADER_SIZE = VIDEO_HEADER.size  # 10


@dataclass
class VideoFrame:
    """一帧视频。

    data: JPEG 字节；timestamp_ms: 采集时刻（单调毫秒，服务端收到时补齐）；
    device_ts_ms: 设备上报的原始时间戳（可能为 0）。
    """

    data: bytes
    timestamp_ms: int
    width: int = 0
    height: int = 0
    device_ts_ms: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)


def parse_video_frame(message: bytes) -> Optional[Tuple[int, bytes]]:
    """解析服务端收到的二进制视频帧，返回 (device_timestamp_ms, jpeg_bytes)。

    不是视频帧（或无负载）时返回 None。
    """
    if not message or len(message) < VIDEO_HEADER_SIZE:
        return None
    frame_type, _reserved, timestamp, payload_size = VIDEO_HEADER.unpack_from(message, 0)
    if frame_type != VIDEO_FRAME_TYPE:
        return None
    if payload_size == 0 or len(message) < VIDEO_HEADER_SIZE + payload_size:
        return None
    payload = message[VIDEO_HEADER_SIZE : VIDEO_HEADER_SIZE + payload_size]
    if not payload:
        return None
    return timestamp, payload


def is_video_frame(message: bytes) -> bool:
    """快速判断一个二进制消息是否是视频帧（按首字节分流）。"""
    return bool(message) and len(message) >= VIDEO_HEADER_SIZE and message[0] == VIDEO_FRAME_TYPE


def build_video_frame(jpeg: bytes, timestamp_ms: int) -> bytes:
    """构造二进制视频帧（测试客户端与服务端回环使用）。"""
    header = VIDEO_HEADER.pack(
        VIDEO_FRAME_TYPE, 0, timestamp_ms & 0xFFFFFFFF, len(jpeg)
    )
    return header + jpeg


def now_ms() -> int:
    """单调毫秒时间戳（用于服务端时间轴对齐）。"""
    return int(time.monotonic() * 1000)


class FrameBuffer:
    """有界帧缓冲：满了丢最旧（背压策略，保证永不积压）。"""

    def __init__(self, max_frames: int = 300):
        self.max_frames = max(1, int(max_frames))
        self._frames: Deque[VideoFrame] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self.dropped = 0

    def put(self, frame: VideoFrame) -> None:
        with self._lock:
            if len(self._frames) == self.max_frames:
                self.dropped += 1
            self._frames.append(frame)

    def extend(self, frames: Iterable[VideoFrame]) -> None:
        for frame in frames:
            self.put(frame)

    def snapshot(self) -> List[VideoFrame]:
        with self._lock:
            return list(self._frames)

    def drain(self) -> List[VideoFrame]:
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
            return frames

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


def slice_frames(
    frames: Iterable[VideoFrame], start_ms: int, end_ms: int
) -> List[VideoFrame]:
    """按时间区间裁剪帧（左闭右开）。"""
    if end_ms <= start_ms:
        return []
    return [f for f in frames if start_ms <= f.timestamp_ms < end_ms]
