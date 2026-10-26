"""OpenAI 兼容网关的视频 LLM 适配器（需求 ④，决策 D3/D7）。

能力声明 `supports_video = True`；请求体里视频以 `video_url` 内容块 + base64 内联发送。
未配置 base_url / model_name 时降级为普通文本 LLM 并打印告警（不阻塞对话）。
"""
from __future__ import annotations

import json

import httpx
import openai

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.openai.openai import LLMProvider as OpenAILikeProvider

TAG = __name__
logger = setup_logging()


class LLMProvider(OpenAILikeProvider):
    """继承 OpenAI 兼容实现，额外声明视频能力。"""

    supports_video = True

    def __init__(self, config):
        self.video_config = {
            "base_url": config.get("video_base_url") or config.get("base_url") or config.get("url"),
            "model_name": config.get("video_model_name") or config.get("model_name"),
            "api_key": config.get("video_api_key") or config.get("api_key"),
        }
        self.video_enabled = bool(
            self.video_config["base_url"] and self.video_config["model_name"]
        )
        super().__init__(config)
        if self.video_enabled:
            self.video_client = openai.OpenAI(
                api_key=self.video_config["api_key"],
                base_url=self.video_config["base_url"],
                timeout=httpx.Timeout(300),
            )
            logger.bind(tag=TAG).info(
                f"视频 LLM 已启用: base_url={self.video_config['base_url']} "
                f"model={self.video_config['model_name']}"
            )
        else:
            logger.bind(tag=TAG).warning(
                "视频 LLM 未配置 base_url / model_name，视频消息将降级为纯文本"
            )

    def _dialogue_has_video(self, dialogue) -> bool:
        for message in dialogue:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "video_url":
                        return True
        return False

    @staticmethod
    def _strip_video(dialogue):
        """降级路径：把内容数组压成纯文本。"""
        simplified = []
        for message in dialogue:
            content = message.get("content")
            if isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                simplified.append(
                    {"role": message.get("role"), "content": "\n".join(t for t in texts if t)}
                )
            else:
                simplified.append(message)
        return simplified

    def response(self, session_id, dialogue, **kwargs):
        dialogue = self.normalize_dialogue(dialogue)
        if not self.video_enabled or not self._dialogue_has_video(dialogue):
            yield from super().response(session_id, dialogue, **kwargs)
            return

        request_params = {
            "model": self.video_config["model_name"],
            "messages": dialogue,
            "stream": True,
        }
        try:
            responses = self.video_client.chat.completions.create(**request_params)
            for chunk in responses:
                delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                content = getattr(delta, "content", "") if delta else ""
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001 - 网关拒绝时按普通错误处理并记日志
            logger.bind(tag=TAG).error(f"视频 LLM 请求失败，降级为文本: {exc}")
            yield from super().response(session_id, self._strip_video(dialogue), **kwargs)

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        dialogue = self.normalize_dialogue(dialogue)
        if not self.video_enabled or not self._dialogue_has_video(dialogue):
            yield from super().response_with_functions(
                session_id, dialogue, functions=functions, **kwargs
            )
            return
        # 视频轮次优先走纯文本回复（多数网关的视频块与 tools 不兼容）
        for token in self.response(session_id, dialogue):
            yield token, None
