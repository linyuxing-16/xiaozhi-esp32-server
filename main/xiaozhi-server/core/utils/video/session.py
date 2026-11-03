"""单连接视频会话编排（阶段 A/C/D/F/G 的接线层）。

把"设备推来的 JPEG 帧 + 服务端解出的 PCM"按需求切成两段：
- 用户说话段（需求 ④）：VAD 上升沿 -> 下降沿；保留音轨是既有行为，
  ④ 原文未规定音轨（⑤ 才明确"不录音"）；
- 小智输出音频段（需求 ⑤）：TTS 起 -> TTS 止，静音（D10）；
两段都过视觉分析（需求 ②），并在下一轮对话前组装成消息附件。

设计要点：
- 全流程同步执行，由 `ConnectionHandler.chat` 的线程池线程调用，不阻塞事件循环；
- 帧缓冲有界，满了丢最旧（背压策略，永不积压）；
- 任何一步失败都只降级（记日志），不影响音频对话主链路。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config.logger import setup_logging
from core.utils.asr_result import AsrResult, Sentence
from core.utils.video.capabilities import detect_capabilities
from core.utils.video.frames import (
    FrameBuffer,
    VideoFrame,
    build_video_frame,
    is_video_frame,
    now_ms,
    parse_video_frame,
)
from core.utils.video.pipeline import (
    PipelineConfig,
    VideoPipeline,
    get_video_pipeline,
)
from core.utils.video.recorder import PcmFrame, RecordingRequest, VideoRecorder
from core.utils.video.speaker import SpeechSegment
from core.utils.video.store import VideoStore, get_video_store

TAG = __name__
logger = setup_logging()

DEFAULT_OUTPUT_DIR = "data/video"


@dataclass
class RoundArtifacts:
    """一轮对话产出的视频与结构化结果。"""

    round_index: int
    user_video: Optional[str] = None
    user_analysis: Optional[dict] = None
    tts_video: Optional[str] = None
    tts_analysis: Optional[dict] = None
    # 短视频兜底：片段时长 < video.image_fallback_max_ms 时不发 mp4（上游网关会
    # 以 code 11133 拒绝），改为每 0.1s 抽一帧，以图片内容块发送。
    user_images: List[str] = field(default_factory=list)
    tts_images: List[str] = field(default_factory=list)
    asr: Optional[dict] = None
    notes: List[str] = field(default_factory=list)

    @property
    def has_video(self) -> bool:
        """是否带有多模态内容（视频或图片）。"""
        return bool(
            self.user_video
            or self.tts_video
            or self.user_images
            or self.tts_images
        )

    @staticmethod
    def media_attachments(
        kind: str, video: Optional[str], images: Sequence[str], analysis: Optional[dict]
    ) -> List[dict]:
        """一段（④ 或 ⑤）的附件：有图片就发图片，否则发视频。"""
        if images:
            return [
                {"type": "image", "kind": kind, "path": path, "analysis": analysis}
                for path in images
            ]
        if video:
            return [
                {"type": "video", "kind": kind, "path": video, "analysis": analysis}
            ]
        return []

    def attachments(self) -> List[dict]:
        """组装成 Message.attachments（⑤ 在前、④ 在后，与消息列表顺序一致）。"""
        result: List[dict] = []
        result.extend(
            self.media_attachments(
                "tts", self.tts_video, self.tts_images, self.tts_analysis
            )
        )
        result.extend(
            self.media_attachments(
                "user", self.user_video, self.user_images, self.user_analysis
            )
        )
        return result


class VideoSession:
    """一个 WebSocket 连接对应的视频会话。"""

    def __init__(
        self,
        config: Optional[dict] = None,
        session_id: str = "",
        output_dir: Optional[str] = None,
        store: Optional[VideoStore] = None,
        pipeline: Optional[VideoPipeline] = None,
        recorder: Optional[VideoRecorder] = None,
    ):
        video_config = dict((config or {}).get("video") or {})
        self.config = video_config
        self.enabled = bool(video_config.get("enable", False))
        self.session_id = session_id
        self.output_dir = output_dir or video_config.get("output_dir", DEFAULT_OUTPUT_DIR)
        self.fps = int(video_config.get("fps", 10))
        self.max_frames = int(video_config.get("max_buffer_frames", 600))
        self.analyze = bool(video_config.get("analyze", True))
        # 单段最长时长（毫秒）：超过后自动切段（事前约束，不做发送时体积拦截）
        self.max_segment_ms = max(1000, int(video_config.get("max_segment_ms", 60000)))
        # 短视频兜底：实测上游网关（minimax-m3）对 <0.6s 的 mp4 会回 400
        # （code 11133，参数被拒），10fps 下 5 帧/0.5s 必失败、6 帧/0.6s 通过。
        # 因此时长低于阈值时不再发 mp4，改为每 interval 毫秒抽一帧发图片。
        self.image_fallback_max_ms = max(
            0, int(video_config.get("image_fallback_max_ms", 500))
        )
        self.image_fallback_interval_ms = max(
            1, int(video_config.get("image_fallback_interval_ms", 100))
        )

        self._frames = FrameBuffer(self.max_frames)
        self._pcm: List[PcmFrame] = []
        self._lock = threading.Lock()

        self._user_span: Optional[Tuple[int, int]] = None
        self._tts_span: Optional[Tuple[int, int]] = None
        self._last_tts_span: Optional[Tuple[int, int]] = None
        self.round_index = 0

        self.store = store or get_video_store(
            int(video_config.get("retention_rounds", 0)),
            int(video_config.get("orphan_ttl_seconds", 3600)),
        )
        self._pipeline = pipeline
        self._recorder = recorder
        self._pipeline_lock = threading.Lock()

        if self.enabled:
            caps = detect_capabilities()
            logger.bind(tag=TAG).info(
                f"[video] 视频会话已启用 session={session_id} "
                f"能力={json.dumps(caps.as_dict(), ensure_ascii=False)}"
            )
            if not caps.ffmpeg:
                logger.bind(tag=TAG).warning(
                    "[video] 缺少 ffmpeg：视频只能保留结构化分析，无法封装 mp4"
                )

    # ------------------------------------------------------------ 采集入口
    def push_frame_bytes(self, message: bytes, timestamp_ms: Optional[int] = None) -> bool:
        """处理一个二进制视频帧消息。返回是否识别为视频帧。

        timestamp_ms 允许调用方注入服务端时间（测试可驱动虚拟时钟）。
        """
        if not self.enabled or not is_video_frame(message):
            return False
        parsed = parse_video_frame(message)
        if parsed is None:
            return True
        device_ts, jpeg = parsed
        self._frames.put(
            VideoFrame(
                data=jpeg,
                timestamp_ms=int(timestamp_ms if timestamp_ms is not None else now_ms()),
                device_ts_ms=int(device_ts),
            )
        )
        return True

    def push_pcm(self, pcm: bytes, timestamp_ms: Optional[int] = None) -> None:
        if not self.enabled or not pcm:
            return
        with self._lock:
            self._pcm.append(
                PcmFrame(
                    timestamp_ms=int(timestamp_ms if timestamp_ms is not None else now_ms()),
                    data=bytes(pcm),
                )
            )
            # 只保留最近 5 分钟，避免长会话内存膨胀。
            # 注意：以"最新一帧的时间"为基准而不是墙上时钟，避免时钟注入/漂移时误删。
            newest = self._pcm[-1].timestamp_ms
            cutoff = newest - 5 * 60 * 1000
            if self._pcm and self._pcm[0].timestamp_ms < cutoff:
                self._pcm = [f for f in self._pcm if f.timestamp_ms >= cutoff]

    # ------------------------------------------------------------ 区间打点
    def on_user_speech_start(self, timestamp_ms: Optional[int] = None) -> None:
        if not self.enabled:
            return
        ts = int(timestamp_ms if timestamp_ms is not None else now_ms())
        with self._lock:
            if self._user_span is None:
                self._user_span = (ts, ts)

    def on_user_speech_stop(self, timestamp_ms: Optional[int] = None) -> None:
        if not self.enabled:
            return
        ts = int(timestamp_ms if timestamp_ms is not None else now_ms())
        with self._lock:
            if self._user_span is not None:
                start, _ = self._user_span
                self._user_span = (start, ts)

    def on_tts_start(self, timestamp_ms: Optional[int] = None) -> None:
        if not self.enabled:
            return
        ts = int(timestamp_ms if timestamp_ms is not None else now_ms())
        with self._lock:
            if self._tts_span is None:
                self._tts_span = (ts, ts)

    def on_tts_stop(self, timestamp_ms: Optional[int] = None) -> None:
        if not self.enabled:
            return
        ts = int(timestamp_ms if timestamp_ms is not None else now_ms())
        with self._lock:
            if self._tts_span is not None:
                start, _ = self._tts_span
                self._tts_span = (start, ts)
                self._last_tts_span = self._tts_span
                self._tts_span = None

    # ------------------------------------------------------------ 一轮结束
    def build_round(self, asr_result: Optional[AsrResult] = None) -> RoundArtifacts:
        """把刚结束的一轮渲染成视频 + 结构化结果。"""
        self.round_index += 1
        artifacts = RoundArtifacts(round_index=self.round_index)
        if not self.enabled:
            return artifacts

        if asr_result is not None:
            artifacts.asr = asr_result.to_dict()

        with self._lock:
            user_span = self._user_span
            tts_span = self._last_tts_span
            frames = self._frames.snapshot()
            pcm_frames = list(self._pcm)
            self._user_span = None
            self._last_tts_span = None

        # 事前约束（不是发送时拦截）：单段最长 max_segment_ms，超长时只保留开头部分并记录说明
        for name, span in (("用户说话段", user_span), ("小智输出段", tts_span)):
            if span is None:
                continue
            if span[1] - span[0] > self.max_segment_ms:
                artifacts.notes.append(
                    f"{name} 时长 {span[1] - span[0]}ms 超过 max_segment_ms="
                    f"{self.max_segment_ms}，只保留开头 {self.max_segment_ms}ms"
                )
        if user_span is not None:
            user_span = (user_span[0], min(user_span[1], user_span[0] + self.max_segment_ms))
        if tts_span is not None:
            tts_span = (tts_span[0], min(tts_span[1], tts_span[0] + self.max_segment_ms))

        # 需求 ①：ASR 的句级时间相对音频起点，需要平移到视频帧的同一时间原点（用户说话段起点）
        segments: List[SpeechSegment] = []
        if asr_result is not None and user_span is not None:
            offset = int(user_span[0])
            for sentence in asr_result.sentences:
                segments.append(
                    SpeechSegment(
                        start_ms=offset + int(sentence.time[0]),
                        end_ms=offset + int(sentence.time[1]),
                        voiceprint_id=sentence.voiceprint_id,
                    )
                )

        if user_span is not None and frames:
            (
                artifacts.user_video,
                artifacts.user_images,
                artifacts.user_analysis,
            ) = self._render_and_analyze(
                "user", user_span, frames, pcm_frames, segments, artifacts
            )
            self._register_media(
                "user", artifacts.user_video, artifacts.user_images
            )
        elif user_span is not None:
            artifacts.notes.append("用户说话段没有视频帧，仅保留音频与 ASR 结果")

        if tts_span is not None and frames:
            # ⑤ 是设备输出音频的时段：画面里没有人"在说话"，因此不传句级区间（全部标否）
            (
                artifacts.tts_video,
                artifacts.tts_images,
                artifacts.tts_analysis,
            ) = self._render_and_analyze(
                "tts", tts_span, frames, pcm_frames, [], artifacts
            )
            self._register_media("tts", artifacts.tts_video, artifacts.tts_images)

        self.store.prune_rounds(self.session_id, self.round_index)
        return artifacts

    def _render_and_analyze(
        self,
        kind: str,
        span: Tuple[int, int],
        frames: Sequence[VideoFrame],
        pcm_frames: Sequence[PcmFrame],
        segments: Sequence[SpeechSegment],
        artifacts: RoundArtifacts,
    ) -> Tuple[Optional[str], List[str], Optional[dict]]:
        start_ms, end_ms = span
        window = [f for f in frames if start_ms <= f.timestamp_ms <= end_ms]
        if not window:
            # 时间窗内没有帧：退化为"区间前最近的一批帧"
            window = [f for f in frames if f.timestamp_ms <= end_ms][-int(self.fps * 5) :]
        if not window:
            return None, [], None

        # 短视频兜底判定：时长低于阈值时不发 mp4（上游网关会以 code 11133 拒绝），
        # 改为每 image_fallback_interval_ms 抽一帧发图片。
        span_ms = max(0, int(end_ms) - int(start_ms))
        image_mode = span_ms < self.image_fallback_max_ms

        analysis: Optional[dict] = None
        annotated_path: Optional[str] = None
        annotated_frames: List[VideoFrame] = []

        # 需求 ②④⑤：先跑分析拿到"经过②处理的视频"（画了 person_id / face_id /
        # 是否说话的标注视频），发给 LLM 的应当是这一版，而不是未处理的原始帧。
        if self.analyze:
            try:
                pipeline = self._get_pipeline()
                with self._pipeline_lock:
                    result = pipeline.analyze(
                        window,
                        segments=segments,
                        session_id=self.session_id,
                        name=f"{kind}_round{self.round_index}",
                        mux=not image_mode,
                    )
                if result.ok:
                    analysis = {
                        "video": result.video_path,
                        "frames": result.frames,
                        "stats": result.stats,
                    }
                    annotated_path = result.video_path
                    annotated_frames = list(result.annotated_frames or [])
                    if annotated_path:
                        self.store.register(
                            annotated_path,
                            self.session_id,
                            self.round_index,
                            "annotated",
                        )
                    # ② 的逐帧结构化 JSON 落盘，便于验收逐帧核对
                    frames_path = self._save_frames_json(analysis, kind)
                    if frames_path:
                        analysis["frames_json"] = frames_path
                else:
                    artifacts.notes.append(f"{kind} 视觉分析失败: {result.error}")
            except Exception as exc:  # noqa: BLE001
                logger.bind(tag=TAG).error(f"[video] {kind} 视觉分析异常: {exc}")
                artifacts.notes.append(f"{kind} 视觉分析异常: {exc}")

        if image_mode:
            # 短视频：不生成 mp4，直接按 0.1s 间隔抽帧存成 JPEG 发图片。
            image_paths = self._export_image_frames(kind, span_ms, window, annotated_frames)
            if image_paths:
                return None, image_paths, analysis
            artifacts.notes.append(
                f"{kind} 片段时长 {span_ms}ms 低于 {self.image_fallback_max_ms}ms，"
                f"抽帧发图片失败，退化为纯文本"
            )
            return None, [], analysis

        video_path: Optional[str] = None
        if annotated_path:
            # ④⑤ 统一使用无音轨的标注视频：音频信息已由 ③ 的 ASR 文本承载，
            # 视频本身不需要音轨（⑤ 原文也明确"不录音"）。
            video_path = annotated_path

        # 兜底：分析关闭 / 分析失败 / 封装失败时，退回原始帧录制
        if video_path is None:
            try:
                recorder = self._get_recorder()
                result = recorder.render(
                    RecordingRequest(
                        session_id=self.session_id,
                        kind=kind,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        frames=window,
                        pcm_frames=() if kind == "tts" else pcm_frames,
                        fps=self.fps,
                    )
                )
                if result.ok:
                    video_path = result.path
                else:
                    artifacts.notes.append(f"{kind} 录制失败: {result.error}")
            except Exception as exc:  # noqa: BLE001
                logger.bind(tag=TAG).error(f"[video] {kind} 录制异常: {exc}")
                artifacts.notes.append(f"{kind} 录制异常: {exc}")
        return video_path, [], analysis

    def _pick_image_indices(
        self, stamps: Sequence[int], interval_ms: int
    ) -> List[int]:
        """按固定间隔挑帧：首帧必取，之后每个间隔槽位只取最先落入的那一帧。

        stamps 必须是升序（同一槽位多帧时只保留第一帧，避免重复发同一时刻）。
        """
        if not stamps:
            return []
        interval_ms = max(1, int(interval_ms))
        picked: List[int] = [0]
        next_slot = interval_ms
        for index in range(1, len(stamps)):
            if stamps[index] >= next_slot:
                picked.append(index)
                # 跳过所有仍落在同一槽位内的帧（含重复时间戳）
                while next_slot <= stamps[index]:
                    next_slot += interval_ms
        return picked

    def _export_image_frames(
        self,
        kind: str,
        span_ms: int,
        window: Sequence[VideoFrame],
        annotated_frames: Sequence[VideoFrame],
    ) -> List[str]:
        """短视频兜底：按 image_fallback_interval_ms 抽帧落盘为 JPEG。

        优先用标注帧（与"发给模型的是 ② 处理过的那一版"口径一致），
        分析关闭/分析失败时回退到原始帧。任何一张写盘失败都只跳过该张。
        """
        source = list(annotated_frames) if annotated_frames else list(window)
        if not source:
            return []

        stamps = [int(f.timestamp_ms) for f in source]
        order = sorted(range(len(source)), key=lambda i: stamps[i])
        sorted_stamps = [stamps[i] for i in order]
        picked = self._pick_image_indices(sorted_stamps, self.image_fallback_interval_ms)

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as exc:
            logger.bind(tag=TAG).warning(f"[video] 创建图片输出目录失败: {exc}")
            return []

        base_ms = int(source[order[0]].timestamp_ms) if order else 0
        paths: List[str] = []
        for slot, index in enumerate(picked):
            frame = source[order[index]]
            if not frame.data:
                continue
            # 文件名带真实时间戳，便于按时间轴核对：偏移 + 序号
            offset_ms = max(0, int(frame.timestamp_ms) - base_ms)
            name = (
                f"{kind}_round{self.round_index}_{self.session_id}"
                f"_{base_ms}_{offset_ms:04d}.jpg"
            )
            path = os.path.join(self.output_dir, name)
            try:
                with open(path, "wb") as handle:
                    handle.write(frame.data)
            except OSError as exc:
                logger.bind(tag=TAG).warning(f"[video] 保存抽帧图片失败 {path}: {exc}")
                continue
            paths.append(path)

        logger.bind(tag=TAG).info(
            f"[video] {kind} 片段 {span_ms}ms < {self.image_fallback_max_ms}ms，"
            f"改发图片 {len(paths)} 张（间隔 {self.image_fallback_interval_ms}ms，"
            f"来源={'标注帧' if annotated_frames else '原始帧'}）"
        )
        return paths

    def _register_media(
        self, kind: str, video_path: Optional[str], image_paths: Sequence[str]
    ) -> None:
        """把本轮产物登记进 VideoStore，会话结束时由 D5 统一删除。"""
        if video_path:
            self.store.register(video_path, self.session_id, self.round_index, kind)
        for path in image_paths or ():
            self.store.register(path, self.session_id, self.round_index, "image")

    def _save_frames_json(self, analysis: dict, kind: str) -> Optional[str]:
        """把 frames[] 结构化结果落盘（内部统一 JSON）。"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(
                self.output_dir,
                f"{kind}_round{self.round_index}_{self.session_id}_frames.json",
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "round": self.round_index,
                        "kind": kind,
                        "video": analysis.get("video"),
                        "stats": analysis.get("stats"),
                        "frames": analysis.get("frames") or [],
                    },
                    handle,
                    ensure_ascii=False,
                )
            return path
        except OSError as exc:
            logger.bind(tag=TAG).warning(f"[video] 逐帧 JSON 落盘失败: {exc}")
            return None

    def _get_recorder(self) -> VideoRecorder:
        if self._recorder is None:
            self._recorder = VideoRecorder(self.output_dir, fps=self.fps)
        return self._recorder

    def _get_pipeline(self) -> VideoPipeline:
        """取视觉管线。

        - 构造时注入的管线优先（测试用）；
        - 否则复用进程级单例：模型（YOLO + InsightFace，约 12s / 620MB）只在
          进程内加载一次，所有会话共享，避免每个连接重付冷启动成本。
        """
        if self._pipeline is not None:
            return self._pipeline
        return get_video_pipeline(self.config, self.output_dir)

    # ------------------------------------------------------------ 清理
    def close(self) -> List[str]:
        """会话结束：按 D5 删除本会话全部视频。

        说明：逐帧结构化 JSON（`*_frames.json`）是分析产物、不是视频，
        会话结束后不删除（验收需要留档核对）；如需清理请用 video.output_dir 整体删除。
        """
        self._frames.clear()
        with self._lock:
            self._pcm.clear()
        # 注意：这里不关闭 _pipeline。管线已是进程级单例、被所有会话共享，
        # 关闭会连带弄坏其他活跃会话，并让下一次连接重新付 12s 模型加载成本；
        # 统一由 reset_video_pipeline() 在进程退出时关闭。
        self._pipeline = None
        return self.store.close_session(self.session_id)


def simulate_frames(jpegs: Sequence[bytes], start_ms: int = 0, interval_ms: int = 100) -> List[VideoFrame]:
    """测试辅助：把一串 JPEG 变成带时间戳的帧。"""
    return [
        VideoFrame(data=data, timestamp_ms=start_ms + index * interval_ms)
        for index, data in enumerate(jpegs)
    ]


def encode_frames(messages: Sequence[bytes]) -> List[bytes]:
    """测试辅助：内联构造二进制视频帧。"""
    return [build_video_frame(jpeg, index * 100) for index, jpeg in enumerate(messages)]
