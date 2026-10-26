"""多模态消息组装测试（需求 ④⑤，D3/D7）。

覆盖：
- Message.attachments -> get_llm_dialogue() 输出 content 数组（video_url + text）；
- base64 内联格式与 mime；
- ⑤ 索引 < ④ 索引的顺序断言；
- MockVideoLLM 收到 messages 原文并可断言。
"""
from __future__ import annotations

import base64
import os
import tempfile

import pytest

from core.utils.dialogue import Dialogue, Message
from core.utils.video.multimodal import (
    analysis_to_text,
    build_content_blocks,
    file_to_base64,
    video_data_url,
    video_url_block,
)


@pytest.fixture()
def videofile():
    import shutil

    directory = tempfile.mkdtemp(prefix="xzmm_")
    path = os.path.join(directory, "clip.mp4")
    payload = b"\x00\x00\x00\x18ftypmp42" + b"x" * 100
    with open(path, "wb") as handle:
        handle.write(payload)
    try:
        yield path, payload
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_video_url_block_is_data_url(videofile):
    path, payload = videofile
    block = video_url_block(path)
    assert block["type"] == "video_url"
    url = block["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == payload


def test_file_to_base64_chunked_matches_single_shot(videofile):
    path, payload = videofile
    assert file_to_base64(path, chunk_size=7) == base64.b64encode(payload).decode()


def test_message_attachments_become_content_array(videofile):
    path, _ = videofile
    dialogue = Dialogue()
    dialogue.put(
        Message(
            role="user",
            content="这是什么？",
            attachments=[{"type": "video", "kind": "user", "path": path}],
        )
    )
    messages = dialogue.get_llm_dialogue()
    user_message = messages[-1]
    assert isinstance(user_message["content"], list)
    types = [block["type"] for block in user_message["content"]]
    assert "video_url" in types
    assert "text" in types


def test_tts_message_index_before_user_message(videofile):
    path, _ = videofile
    dialogue = Dialogue()
    dialogue.put(
        Message(
            role="user",
            content="[小智输出音频时段的录像，无声]",
            attachments=[{"type": "video", "kind": "tts", "path": path}],
        )
    )
    dialogue.put(
        Message(
            role="user",
            content="用户说的话",
            attachments=[{"type": "video", "kind": "user", "path": path}],
        )
    )
    messages = dialogue.get_llm_dialogue()
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) == 2
    # ⑤ 在消息列表中的索引严格小于 ④
    assert messages.index(user_messages[0]) < messages.index(user_messages[1])


def test_analysis_summary_text():
    attachments = [
        {
            "type": "video",
            "kind": "user",
            "analysis": {
                "stats": {"frames": 2},
                "frames": [
                    {
                        "timestamp_ms": 0,
                        "persons": [
                            {"person_id": 1, "speaking": True, "face_id": "face_1"},
                            {"person_id": 2, "speaking": False, "detected": False},
                        ],
                    },
                    {
                        "timestamp_ms": 100,
                        "persons": [
                            {"person_id": 1, "speaking": False},
                            # person 2 全程背对镜头：没有任何 face_id
                            {"person_id": 2, "speaking": False, "detected": False},
                        ],
                    },
                ],
            },
        }
    ]
    text = analysis_to_text(attachments)
    assert "人物1" in text
    assert "face_1" in text
    assert "未露出可识别人脸" in text  # person 2 没有 face_id


def test_missing_attachment_is_skipped(videofile):
    path, _ = videofile
    blocks = build_content_blocks(
        "hi", [{"type": "video", "kind": "user", "path": path + ".missing"}]
    )
    assert all(block["type"] != "video_url" for block in blocks)


def test_mock_video_llm_records_messages(videofile):
    from core.providers.llm.MockVideoLLM.MockVideoLLM import LLMProvider

    path, _ = videofile
    provider = LLMProvider({"reply": "收到"})
    LLMProvider.clear_records()
    dialogue = [
        {
            "role": "user",
            "content": build_content_blocks(
                "看这个", [{"type": "video", "kind": "user", "path": path}]
            ),
        }
    ]
    text = "".join(provider.response("sess_mock", dialogue))
    assert text == "收到"
    records = LLMProvider.get_records()
    assert len(records) == 1
    assert records[0]["summary"][0]["has_video"] is True
    assert "看这个" in records[0]["summary"][0]["text"]
    LLMProvider.clear_records()
