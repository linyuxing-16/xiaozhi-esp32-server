"""测试用 ASR 提供商（阶段 H）：返回带句级时间戳 + 声纹 id 的预定句子。"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional, Tuple

from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    """返回预定句子，供验收脚本核对 ③ 的归一化结构。

    supports_sentence_timestamp = True：供应商自带句级时间戳（D13 的 asr 分支）。
    """

    supports_sentence_timestamp = True

    def __init__(self, config: dict = None, delete_audio_file: bool = True):
        super().__init__()
        config = config or {}
        self.interface_type = InterfaceType.NON_STREAM
        self.delete_audio_file = delete_audio_file
        self.output_dir = config.get("output_dir", "tmp")
        self.replies = config.get("replies")
        self._cursor = 0

    def _next_reply(self) -> dict:
        if self.replies:
            reply = self.replies[self._cursor % len(self.replies)]
            self._cursor += 1
            return reply
        # 同时给 content（交给 LLM 的正文）与 sentences（③ 的归一化数据）
        return {
            "content": "你好，今天天气怎么样。我想出去走走。",
            "sentences": [
                {
                    "text": "你好，今天天气怎么样",
                    "start": 1200,
                    "end": 4300,
                    "voiceprint_id": "spk_a1b2",
                },
                {
                    "text": "我想出去走走",
                    "start": 5100,
                    "end": 7800,
                    "voiceprint_id": "spk_a1b2",
                },
            ],
            "speaker": "spk_a1b2",
        }

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, artifacts=None
    ) -> Tuple[Optional[str], Optional[str]]:
        await asyncio.sleep(0)
        payload = self._next_reply()
        text = json.dumps(payload, ensure_ascii=False)
        return text, None
