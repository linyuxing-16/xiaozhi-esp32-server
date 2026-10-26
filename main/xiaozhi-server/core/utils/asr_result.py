"""ASR 结果归一化（需求 ③，决策 D13，阶段 E）。

统一输出结构（一句一个区间）：

    sentences:
      - index: 1
        text: "你好，今天天气怎么样"
        voiceprint_id: "spk_a1b2"      # 无法识别时为 null（不是空串、不是 unknown）
        time: [1200, 4300]             # 毫秒，左闭右闭
        time_source: asr               # asr（供应商句级时间戳）| vad（VAD 近似）
    time_source: ...

时间来源策略：
1. 适配器声明能力 `supports_sentence_timestamp`；
2. 为真 -> 直接采用供应商返回的句级起止时间，time_source=asr；
3. 为假或缺失 -> 用 VAD 有声区间与句子顺序对齐估算，time_source=vad；
4. 两种来源都必须满足"区间单调、不重叠、落在音频时长内"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def _to_ms(value: Any) -> Optional[int]:
    """把各种时间表达统一成毫秒整数。

    支持：int/float 毫秒、秒（<1000 且是 float 的小值按秒处理）、"12.3" 字符串。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # 小于 100 且带小数的，多半是秒；整数按毫秒
        if isinstance(value, float) and number < 100.0:
            number *= 1000.0
        return int(round(number))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _to_ms(float(text))
        except ValueError:
            return None
    return None


def normalize_voiceprint(value: Any) -> Optional[str]:
    """声纹 id 归一化：无法识别 -> None（不是空串、不是 unknown）。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in ("unknown", "none", "null", "未知", "未知说话人", "n/a"):
        return None
    return text


@dataclass
class Sentence:
    """③ 的一句话。"""

    index: int
    text: str
    voiceprint_id: Optional[str] = None
    time: Tuple[int, int] = (0, 0)
    time_source: str = "vad"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "voiceprint_id": self.voiceprint_id,
            "time": [int(self.time[0]), int(self.time[1])],
            "time_source": self.time_source,
        }


@dataclass
class AsrResult:
    """归一化后的 ASR 结果。"""

    sentences: List[Sentence] = field(default_factory=list)
    duration_ms: int = 0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "duration_ms": int(self.duration_ms),
        }

    def to_yaml(self) -> str:
        """渲染为需求 ③ 的 YAML 交付样例格式（不依赖第三方库）。"""
        lines: List[str] = ["sentences:"]
        for sentence in self.sentences:
            lines.append(f"  - index: {sentence.index}")
            lines.append(f"    text: {_yaml_quote(sentence.text)}")
            if sentence.voiceprint_id is None:
                lines.append("    voiceprint_id: null")
            else:
                lines.append(f"    voiceprint_id: {_yaml_quote(sentence.voiceprint_id)}")
            lines.append(f"    time: [{int(sentence.time[0])}, {int(sentence.time[1])}]")
            lines.append(f"    time_source: {sentence.time_source}")
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        return "".join(sentence.text for sentence in self.sentences)


def _yaml_quote(text: str) -> str:
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'


def split_sentences(text: str) -> List[str]:
    """按中英文标点切句，保留标点。"""
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def _align_vad_segments(
    texts: Sequence[str], vad_segments: Sequence[Tuple[int, int]], duration_ms: int
) -> List[Tuple[int, int]]:
    """把句子顺序对齐到 VAD 有声区间（句数多于区间时均分最后一个区间）。"""
    if not texts:
        return []
    usable = [(int(start), int(end)) for start, end in vad_segments if end > start]
    if not usable:
        # 没有任何 VAD 信息：按字符数比例均分整段音频
        total_chars = sum(max(1, len(t)) for t in texts)
        ranges: List[Tuple[int, int]] = []
        cursor = 0
        for text in texts:
            span = int(duration_ms * max(1, len(text)) / total_chars)
            ranges.append((cursor, min(duration_ms, cursor + span)))
            cursor += span
        return ranges

    if len(texts) <= len(usable):
        # 句数少于区间数：把多余区间合并到最近的句子（按顺序均摊）
        merged: List[Tuple[int, int]] = []
        step = len(usable) / float(len(texts))
        for index in range(len(texts)):
            start_index = int(round(index * step))
            end_index = int(round((index + 1) * step)) - 1
            end_index = max(start_index, min(len(usable) - 1, end_index))
            merged.append((usable[start_index][0], usable[end_index][1]))
        return merged

    # 句数多于区间数：最后一个区间内部再按字符比例切
    ranges: List[Tuple[int, int]] = []
    for index, text in enumerate(texts):
        if index < len(usable) - 1:
            ranges.append(usable[index])
        else:
            remaining = texts[index:]
            start = usable[len(usable) - 1][0]
            end = usable[len(usable) - 1][1]
            total_chars = sum(max(1, len(t)) for t in remaining)
            cursor = start
            for item in remaining:
                span = int((end - start) * max(1, len(item)) / total_chars)
                ranges.append((cursor, min(end, cursor + span)))
                cursor += span
            break
    return ranges


def _enforce_monotonic(sentences: List[Sentence], duration_ms: int) -> List[Sentence]:
    """保证区间单调、不重叠、落在音频时长内。"""
    previous_end = 0
    result: List[Sentence] = []
    for sentence in sentences:
        start, end = sentence.time
        start = max(previous_end, int(start))
        end = max(start, int(end))
        if duration_ms > 0:
            start = min(start, duration_ms)
            end = min(max(start, end), duration_ms)
        sentence.time = (start, end)
        previous_end = end
        result.append(sentence)
    return result


def normalize_asr_result(
    raw: Any,
    *,
    duration_ms: Optional[int] = None,
    vad_segments: Optional[Sequence[Tuple[int, int]]] = None,
    voiceprint_id: Optional[str] = None,
    supports_sentence_timestamp: bool = False,
) -> AsrResult:
    """把各供应商的 ASR 返回统一成 ③ 的结构。

    支持三种返回形态：
    1. 纯文本 "你好。今天天气不错。"；
    2. {"content": "...", "speaker": "..."}（FunASR 风格）；
    3. {"sentences": [{"text": ..., "start": ..., "end": ..., "voiceprint_id": ...}]}
       （带句级时间戳的供应商）。
    """
    sentences: List[Sentence] = []
    full_text = ""
    speaker = normalize_voiceprint(voiceprint_id)
    raw_sentences: List[dict] = []
    raw_duration = duration_ms

    if isinstance(raw, str):
        full_text = raw
    elif isinstance(raw, dict):
        speaker = normalize_voiceprint(raw.get("speaker")) or speaker
        if isinstance(raw.get("sentences"), list):
            raw_sentences = [item for item in raw["sentences"] if isinstance(item, dict)]
        full_text = str(raw.get("content") or raw.get("text") or "")
        if raw_duration is None:
            raw_duration = _to_ms(raw.get("duration_ms") or raw.get("duration"))
    elif raw is None:
        full_text = ""
    else:
        full_text = str(raw)

    if raw_sentences:
        for order, item in enumerate(raw_sentences, start=1):
            text = str(item.get("text") or item.get("content") or "").strip()
            if not text:
                continue
            start = _to_ms(
                item.get("start_ms", item.get("start", item.get("begin")))
            )
            end = _to_ms(item.get("end_ms", item.get("end", item.get("stop"))))
            seg_speaker = normalize_voiceprint(
                item.get("voiceprint_id") or item.get("speaker")
            ) or speaker
            if start is None or end is None:
                continue
            sentences.append(
                Sentence(
                    index=order,
                    text=text,
                    voiceprint_id=seg_speaker,
                    time=(start, end),
                    time_source="asr" if supports_sentence_timestamp else "vad",
                )
            )
        if not sentences and full_text:
            raw_sentences = []

    if not sentences:
        texts = split_sentences(full_text)
        if raw_duration is None:
            raw_duration = 0
        ranges = _align_vad_segments(texts, list(vad_segments or []), int(raw_duration or 0))
        for order, (text, span) in enumerate(zip(texts, ranges), start=1):
            sentences.append(
                Sentence(
                    index=order,
                    text=text,
                    voiceprint_id=speaker,
                    time=span,
                    time_source="vad",
                )
            )

    sentences = _enforce_monotonic(sentences, int(raw_duration or 0))
    if not raw_duration:
        raw_duration = sentences[-1].time[1] if sentences else 0

    return AsrResult(
        sentences=sentences,
        duration_ms=int(raw_duration or 0),
        text=full_text or "".join(sentence.text for sentence in sentences),
    )


def save_asr_result(
    result: AsrResult, output_dir: str, session_id: str, round_index: int
) -> Dict[str, str]:
    """落盘 ③ 的交付产物：内部 JSON + 对外 YAML。返回两个路径。"""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"asr_{session_id}_round{round_index}")
    json_path = base + ".json"
    yaml_path = base + ".yaml"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(result.to_yaml())
    return {"json": json_path, "yaml": yaml_path}
