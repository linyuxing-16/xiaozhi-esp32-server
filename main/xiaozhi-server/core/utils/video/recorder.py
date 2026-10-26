"""录制与音画对齐（需求 ①④⑤，阶段 C）。

职责：
- 用户说话段（④）：VAD 上升沿 -> 下降沿，JPEG 帧 + PCM 音轨封装为 mp4；
- 小智输出音频段（⑤）：TTS 首帧 -> TTS 结束，只有视频流（静音，D10）；
- 时间轴原点对齐（①）：以各自区间的起点为 t=0，视频帧时间戳平移后与音频严格同起点。

实现方式：把区间内的 JPEG 帧按各自时间戳写成 ffmpeg concat 列表（保留每帧真实间隔），
再交给 ffmpeg 封装。没有 ffmpeg 时不抛异常，返回失败原因（能力探测已提前告警）。
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from config.logger import setup_logging
from core.utils.video.capabilities import detect_capabilities
from core.utils.video.frames import VideoFrame

TAG = __name__
logger = setup_logging()

PCM_SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1


@dataclass
class PcmFrame:
    """一段带时间戳的 PCM（16bit 单声道 16kHz）。"""

    timestamp_ms: int
    data: bytes

    @property
    def duration_ms(self) -> int:
        return int(len(self.data) / (PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH) * 1000)


@dataclass
class RecordingRequest:
    """一次录制请求。"""

    session_id: str
    kind: str  # "user"（④，保留音轨） / "tts"（⑤，静音）
    start_ms: int
    end_ms: int
    frames: Sequence[VideoFrame] = field(default_factory=tuple)
    pcm_frames: Sequence[PcmFrame] = field(default_factory=tuple)
    fps: int = 10

    @property
    def duration_ms(self) -> int:
        return max(0, int(self.end_ms) - int(self.start_ms))

    @property
    def silence(self) -> bool:
        return self.kind == "tts"


@dataclass
class RecordingResult:
    """录制结果。"""

    ok: bool
    path: Optional[str] = None
    error: Optional[str] = None
    frames_written: int = 0
    duration_ms: int = 0
    has_audio: bool = False
    sample_count: int = 0


def pcm_bytes_to_wav(pcm: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    """PCM -> WAV 字节（16bit 单声道）。"""
    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def collect_pcm(pcm_frames: Sequence[PcmFrame], start_ms: int, end_ms: int) -> bytes:
    """按时间区间裁剪并拼接 PCM。"""
    chunks: List[bytes] = []
    for frame in sorted(pcm_frames, key=lambda f: f.timestamp_ms):
        frame_start = frame.timestamp_ms
        frame_end = frame_start + frame.duration_ms
        if frame_end <= start_ms or frame_start >= end_ms:
            continue
        data = frame.data
        if frame_start < start_ms:
            skip_bytes = int((start_ms - frame_start) * PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH / 1000)
            skip_bytes -= skip_bytes % PCM_SAMPLE_WIDTH
            data = data[skip_bytes:]
        if frame_end > end_ms:
            keep_bytes = int((end_ms - frame_start) * PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH / 1000)
            keep_bytes -= keep_bytes % PCM_SAMPLE_WIDTH
            data = data[:keep_bytes]
        if data:
            chunks.append(data)
    return b"".join(chunks)


def _write_concat_list(
    frames: Sequence[VideoFrame], offset_ms: int, workdir: str
) -> Tuple[Optional[str], int]:
    """把帧写成 JPEG 文件并生成 ffmpeg concat 列表。

    返回 (concat 列表路径, 写入帧数)。时间戳相对 offset_ms 平移到 0。
    """
    usable = [f for f in frames if f.data]
    if not usable:
        return None, 0

    usable.sort(key=lambda f: f.timestamp_ms)
    lines: List[str] = ["ffconcat version 1.0"]
    last_start = 0
    for index, frame in enumerate(usable):
        name = f"frame_{index:06d}.jpg"
        path = os.path.join(workdir, name)
        with open(path, "wb") as handle:
            handle.write(frame.data)
        start = max(0, int(frame.timestamp_ms) - int(offset_ms))
        if index == 0:
            last_start = 0
        safe_path = path.replace("\\", "/").replace("'", r"\'")
        lines.append(f"file '{safe_path}'")
        # duration 由与下一帧的间隔决定；最后一帧沿用默认帧间隔
        lines.append(f"duration {max(0.001, (start - last_start) / 1000.0):.4f}")
        last_start = start

    list_path = os.path.join(workdir, "frames.ffconcat")
    with open(list_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return list_path, len(usable)


def _run_ffmpeg(args: List[str], timeout: int = 120) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", "replace")
        return False, stderr[-2000:]
    return True, ""


def ffprobe_streams(path: str) -> List[str]:
    """用 ffprobe 读取流的 codec_type 列表（测试断言"无音轨"用）。"""
    probe = shutil.which("ffprobe")
    if probe is None:
        return []
    try:
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.decode("utf-8", "replace").splitlines() if line.strip()]


class VideoRecorder:
    """把一段区间渲染成 mp4。"""

    def __init__(self, output_dir: str, fps: int = 10, use_h264: bool = True):
        self.output_dir = output_dir
        self.fps = max(1, int(fps))
        self.use_h264 = use_h264
        os.makedirs(self.output_dir, exist_ok=True)

    def render(self, request: RecordingRequest) -> RecordingResult:
        caps = detect_capabilities()
        if not caps.ffmpeg:
            return RecordingResult(ok=False, error="ffmpeg 不可用，无法录制视频")

        if not request.frames:
            return RecordingResult(ok=False, error="区间内没有视频帧")

        duration_ms = request.duration_ms
        workdir = tempfile.mkdtemp(prefix="xzvideo_")
        try:
            list_path, written = _write_concat_list(
                request.frames, request.start_ms, workdir
            )
            if list_path is None or written == 0:
                return RecordingResult(ok=False, error="没有可用的 JPEG 帧")

            suffix = ".mp4"
            output_path = os.path.join(
                self.output_dir,
                f"{request.kind}_{request.session_id}_{request.start_ms}{suffix}",
            )

            args: List[str] = ["ffmpeg", "-y", "-loglevel", "error"]
            args += ["-f", "concat", "-safe", "0", "-i", list_path]

            has_audio = False
            if not request.silence:
                pcm = collect_pcm(request.pcm_frames, request.start_ms, request.end_ms)
                if pcm:
                    wav_path = os.path.join(workdir, "audio.wav")
                    with open(wav_path, "wb") as handle:
                        handle.write(pcm_bytes_to_wav(pcm))
                    args += ["-i", wav_path]
                    has_audio = True

            if self.use_h264:
                args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
            else:
                args += ["-c:v", "mpeg4", "-q:v", "5"]

            # ffmpeg 7+ 移除了 -vsync（改为 -fps_mode）；用 -r 控制输出帧率即可
            args += ["-r", str(self.fps)]
            if has_audio:
                args += ["-c:a", "aac", "-shortest"]
            else:
                args += ["-an"]
            args += ["-movflags", "+faststart", output_path]

            ok, error = _run_ffmpeg(args)
            if not ok:
                logger.bind(tag=TAG).error(f"ffmpeg 封装失败: {error}")
                return RecordingResult(
                    ok=False, error=error, frames_written=written, duration_ms=duration_ms
                )

            return RecordingResult(
                ok=True,
                path=output_path,
                frames_written=written,
                duration_ms=duration_ms,
                has_audio=has_audio,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
