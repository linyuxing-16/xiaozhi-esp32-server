"""多模态消息组装（需求 ④⑤，决策 D3，阶段 F）。

按 D3：视频**原样**以 `video_url` 内容块 + base64 内联交给 OpenAI 兼容网关：

    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<...>"}}

不设体积上限、不截断、不降码率（D14 已删除 20MB 保护）。
base64 按块拼接，避免同时持有"原始 + 完整 base64"两份大对象造成内存峰值翻倍。
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Sequence

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

CHUNK_SIZE = 3 * 1024 * 1024  # 3MB 原始字节 -> 约 4MB base64


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }.get(ext, "video/mp4")


def file_to_base64(path: str, chunk_size: int = CHUNK_SIZE) -> str:
    """分块读取并 base64 编码（避免 2 倍内存峰值）。

    关键：base64 以 3 字节为一组，块大小必须是 3 的倍数，否则块边界会插入
    填充符 `=` 导致拼接结果错误。这里先对齐块大小，并把余数字节留到下一轮。
    """
    chunk_size = max(3, chunk_size - (chunk_size % 3))
    parts: List[str] = []
    remainder = b""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            chunk = remainder + chunk
            usable = len(chunk) - (len(chunk) % 3)
            if usable:
                parts.append(base64.b64encode(chunk[:usable]).decode("ascii"))
            remainder = chunk[usable:]
    if remainder:
        parts.append(base64.b64encode(remainder).decode("ascii"))
    return "".join(parts)


def video_data_url(path: str, mime: Optional[str] = None) -> str:
    """生成本地视频的 data URL。"""
    mime = mime or guess_mime(path)
    return f"data:{mime};base64,{file_to_base64(path)}"


def video_url_block(path: str, mime: Optional[str] = None) -> Dict[str, Any]:
    """构造 OpenAI 兼容网关的 video_url 内容块（D3：原样内联）。"""
    return {"type": "video_url", "video_url": {"url": video_data_url(path, mime)}}


def text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text or ""}


def analysis_to_text(attachments: Sequence[dict]) -> str:
    """把 ② 的结构化结果压成一段可供 LLM 阅读的文本（不替代视频本身）。"""
    lines: List[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        kind = attachment.get("kind", "video")
        label = "用户说话" if kind == "user" else "小智说话"
        analysis = attachment.get("analysis") or {}
        stats = analysis.get("stats") or {}
        frames = analysis.get("frames") or []
        persons: Dict[int, dict] = {}
        for frame in frames:
            for person in frame.get("persons") or []:
                pid = person.get("person_id")
                if pid is None:
                    continue
                record = persons.setdefault(
                    pid, {"face_id": None, "speaking_frames": 0, "frames": 0}
                )
                record["frames"] += 1
                if person.get("speaking"):
                    record["speaking_frames"] += 1
                if person.get("face_id"):
                    record["face_id"] = person["face_id"]
        if not persons:
            lines.append(f"[{label}视频] 画面中未检出人物")
            continue
        parts = []
        for pid, record in sorted(persons.items()):
            face = record["face_id"] or "未露出可识别人脸"
            parts.append(
                f"人物{pid}（人脸:{face}，说话帧 {record['speaking_frames']}/{record['frames']}）"
            )
        lines.append(
            f"[{label}视频] 共 {len(frames)} 帧，检出 {len(persons)} 人：" + "；".join(parts)
        )
    return "\n".join(lines)


def build_content_blocks(text: str, attachments: Sequence[dict]) -> List[Dict[str, Any]]:
    """把一条消息组装成 OpenAI 兼容的 content 数组。

    顺序：文本块在前，然后按 attachments 顺序插入视频块（调用方保证 ⑤ 在 ④ 之前）。
    """
    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append(text_block(text))
    for attachment in attachments or []:
        if not isinstance(attachment, dict):
            continue
        path = attachment.get("path")
        if not path:
            continue
        if not os.path.exists(path):
            logger.bind(tag=TAG).warning(f"[video] 附件不存在，跳过: {path}")
            continue
        try:
            blocks.append(video_url_block(path))
        except OSError as exc:
            logger.bind(tag=TAG).error(f"[video] 读取视频附件失败 {path}: {exc}")
    summary = analysis_to_text(attachments or [])
    if summary:
        blocks.append(text_block(summary))
    return blocks
