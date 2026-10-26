"""把"一轮视频"组装进对话历史（需求 ④⑤ 的顺序保证）。

抽成独立纯函数，便于在不依赖 `ConnectionHandler`（其导入链依赖 mcp/opuslib）
的环境下测试"⑤ 的索引 < ④ 的索引"这条硬要求。
"""
from __future__ import annotations

from typing import Optional

from config.logger import setup_logging
from core.utils.dialogue import Dialogue, Message
from core.utils.video.session import RoundArtifacts, VideoSession

TAG = __name__
logger = setup_logging()

TTS_MESSAGE_TEXT = "[小智（设备）输出音频时段的摄像头录像，无声]"


def put_round_messages(
    dialogue: Dialogue,
    query: str,
    artifacts: Optional[RoundArtifacts],
) -> None:
    """写入本轮消息。

    需求 ⑤：⑤（设备输出音频段）的消息在消息列表中的索引必须小于 ④（用户说话段）。
    因此先 put ⑤，再 put ④。两条都是 user 角色，代表模型要看的画面上下文。
    没有视频时退化为单条普通 user 消息（与改动前行为一致）。
    """
    if artifacts is None or not artifacts.has_video:
        dialogue.put(Message(role="user", content=query))
        return

    if artifacts.tts_video:
        dialogue.put(
            Message(
                role="user",
                content=TTS_MESSAGE_TEXT,
                attachments=[
                    {
                        "type": "video",
                        "kind": "tts",
                        "path": artifacts.tts_video,
                        "analysis": artifacts.tts_analysis,
                    }
                ],
            )
        )
    if artifacts.user_video:
        dialogue.put(
            Message(
                role="user",
                content=query,
                attachments=[
                    {
                        "type": "video",
                        "kind": "user",
                        "path": artifacts.user_video,
                        "analysis": artifacts.user_analysis,
                    }
                ],
            )
        )
    else:
        dialogue.put(Message(role="user", content=query))


def build_video_round(video_session: Optional[VideoSession], asr_result) -> Optional[RoundArtifacts]:
    """渲染 + 分析本轮视频；任何异常都只降级，不影响对话主链路。"""
    if video_session is None:
        return None
    try:
        return video_session.build_round(asr_result)
    except Exception as exc:  # noqa: BLE001
        logger.bind(tag=TAG).error(f"[video] 生成本轮视频失败（不影响对话）: {exc}")
        return None
