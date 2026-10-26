"""ASR 适配器时间戳能力探测测试（D13 的探测分支）。

模拟三种真实返回形态，验证与 `_normalize_asr_for_video` 相同的解析链路：
1. 带句级时间戳的结构化 dict；
2. FunASR 风格 timestamp 数组；
3. JSON 字符串包裹的结构化结果。
"""
from __future__ import annotations

import json

from core.providers.asr.base import parse_asr_timestamp_items
from core.utils.asr_result import normalize_asr_result


def test_structured_sentences_are_detected():
    raw = {
        "sentences": [
            {"text": "你好", "start": 1200, "end": 4300, "voiceprint_id": "spk_1"},
            {"text": "再见", "start": 5100, "end": 7800, "voiceprint_id": "spk_1"},
        ]
    }
    items = parse_asr_timestamp_items(raw)
    assert len(items) == 2

    result = normalize_asr_result(
        {"sentences": items, "speaker": "spk_1"},
        duration_ms=8000,
        supports_sentence_timestamp=True,
    )
    assert [s.time_source for s in result.sentences] == ["asr", "asr"]
    assert result.sentences[0].time == (1200, 4300)


def test_funasr_timestamp_array_is_detected():
    raw = {"text": "你好。再见。", "timestamp": [[0, 1000], [1000, 2000]]}
    items = parse_asr_timestamp_items(raw)
    assert items, "应能从 FunASR timestamp 数组提取句级时间"


def test_json_string_payload_is_detected():
    payload = {
        "sentences": [{"text": "你好", "start": 500, "end": 1500, "voiceprint_id": "spk_2"}]
    }
    items = parse_asr_timestamp_items(json.dumps(payload, ensure_ascii=False))
    assert len(items) == 1
    assert items[0]["start"] == 500


def test_plain_text_has_no_timestamps():
    assert parse_asr_timestamp_items("你好，今天天气不错。") == []
    assert parse_asr_timestamp_items(None) == []
