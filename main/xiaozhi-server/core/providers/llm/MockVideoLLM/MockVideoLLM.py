"""测试用视频 LLM 提供商（阶段 H，验收 1）。

记录收到的 messages 原文（含视频块与文本块），返回预定回复。
不发起任何网络请求，便于在无账号环境下跑完整三轮验收。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    supports_video = True

    # 进程级记录（测试断言用）
    _records: List[Dict[str, Any]] = []
    _records_lock = threading.Lock()

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.reply = config.get("reply") or "我看到了，你的画面我收到了。"
        self.record_path = config.get("record_path")

    @classmethod
    def get_records(cls) -> List[Dict[str, Any]]:
        with cls._records_lock:
            return list(cls._records)

    @classmethod
    def clear_records(cls) -> None:
        with cls._records_lock:
            cls._records.clear()

    def _record(self, session_id: str, dialogue: List[Dict[str, Any]]) -> None:
        record = {
            "session_id": session_id,
            "timestamp": time.time(),
            "dialogue": dialogue,
            # 便于断言：每条消息里视频块/文本块的数量
            "summary": [
                {
                    "role": message.get("role"),
                    "has_video": any(
                        isinstance(block, dict) and block.get("type") == "video_url"
                        for block in (message.get("content") or [])
                    )
                    if isinstance(message.get("content"), list)
                    else False,
                    "text": (
                        "".join(
                            block.get("text", "")
                            for block in message["content"]
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                        if isinstance(message.get("content"), list)
                        else str(message.get("content") or "")
                    ),
                }
                for message in dialogue
            ],
        }
        with self._records_lock:
            self._records.append(record)
        if self.record_path:
            try:
                with open(self.record_path, "w", encoding="utf-8") as handle:
                    json.dump(self._records, handle, ensure_ascii=False, indent=2)
            except OSError as exc:
                logger.bind(tag=TAG).warning(f"写入 mock 记录失败: {exc}")
        logger.bind(tag=TAG).info(
            f"MockVideoLLM 收到 {len(dialogue)} 条消息，含视频块 "
            f"{sum(item['has_video'] for item in record['summary'])} 条"
        )

    def response(self, session_id, dialogue, **kwargs):
        self._record(session_id, dialogue)
        # 逐字返回，模拟流式
        for char in self.reply:
            yield char

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        self._record(session_id, dialogue)
        for char in self.reply:
            yield char, None
