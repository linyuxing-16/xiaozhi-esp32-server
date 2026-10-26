"""ASR 归一化测试（需求 ③，D13）。

覆盖 >= 3 种返回形态：
1. 纯文本（无句级时间戳 -> time_source=vad）；
2. FunASR 风格 dict（content + speaker）；
3. 带句级时间戳的 sentences（time_source=asr）。

另外校验：区间单调、不重叠、落在音频时长内；无法识别声纹写 null。
"""
from __future__ import annotations

import pytest

from core.utils.asr_result import (
    AsrResult,
    Sentence,
    normalize_asr_result,
    normalize_voiceprint,
    split_sentences,
)


def _assert_invariants(result: AsrResult, duration_ms: int):
    previous_end = -1
    for sentence in result.sentences:
        start, end = sentence.time
        assert 0 <= start <= end <= duration_ms
        assert start >= previous_end, "区间必须单调不重叠"
        previous_end = end


def test_plain_text_uses_vad_time_source():
    result = normalize_asr_result(
        "你好。今天天气不错。",
        duration_ms=6000,
        vad_segments=[(500, 2500), (3000, 5500)],
    )
    assert len(result.sentences) == 2
    assert all(s.time_source == "vad" for s in result.sentences)
    assert result.sentences[0].time == (500, 2500)
    _assert_invariants(result, 6000)


def test_funasr_dict_with_speaker():
    result = normalize_asr_result(
        {"content": "打开灯。", "speaker": "张三", "language": "zh"},
        duration_ms=2000,
        vad_segments=[(0, 2000)],
    )
    assert result.sentences[0].text == "打开灯。"
    assert result.sentences[0].voiceprint_id == "张三"
    assert result.sentences[0].time_source == "vad"
    _assert_invariants(result, 2000)


def test_sentence_timestamps_use_asr_source():
    raw = {
        "sentences": [
            {"text": "你好，今天天气怎么样", "start": 1200, "end": 4300, "voiceprint_id": "spk_a1b2"},
            {"text": "我想出去走走", "start": 5100, "end": 7800, "voiceprint_id": "spk_a1b2"},
        ]
    }
    result = normalize_asr_result(raw, duration_ms=8000, supports_sentence_timestamp=True)
    assert [s.time_source for s in result.sentences] == ["asr", "asr"]
    assert [s.time for s in result.sentences] == [(1200, 4300), (5100, 7800)]
    assert result.sentences[0].index == 1
    _assert_invariants(result, 8000)


def test_unknown_voiceprint_becomes_none():
    assert normalize_voiceprint("") is None
    assert normalize_voiceprint("unknown") is None
    assert normalize_voiceprint("未知说话人") is None
    assert normalize_voiceprint(None) is None
    assert normalize_voiceprint("spk_1") == "spk_1"

    result = normalize_asr_result("你好。", duration_ms=1000, voiceprint_id="unknown")
    assert result.sentences[0].voiceprint_id is None


def test_yaml_rendering_matches_spec_shape():
    result = normalize_asr_result(
        {
            "sentences": [
                {"text": "你好", "start": 1200, "end": 4300, "voiceprint_id": "spk_a1b2"},
                {"text": "我想出去走走", "start": 5100, "end": 7800},
            ]
        },
        duration_ms=8000,
        supports_sentence_timestamp=True,
    )
    yaml_text = result.to_yaml()
    assert "sentences:" in yaml_text
    assert "index: 1" in yaml_text
    assert "time: [1200, 4300]" in yaml_text
    assert "time_source: asr" in yaml_text
    assert "voiceprint_id: null" in yaml_text


def test_monotonic_enforced_when_supplier_overlaps():
    raw = {
        "sentences": [
            {"text": "a", "start": 1000, "end": 3000},
            {"text": "b", "start": 2000, "end": 4000},  # 与上一句重叠
        ]
    }
    result = normalize_asr_result(raw, duration_ms=5000, supports_sentence_timestamp=True)
    _assert_invariants(result, 5000)
    assert result.sentences[1].time[0] == 3000


def test_split_sentences_keeps_content():
    assert split_sentences("你好。今天天气不错！") == ["你好。", "今天天气不错！"]
