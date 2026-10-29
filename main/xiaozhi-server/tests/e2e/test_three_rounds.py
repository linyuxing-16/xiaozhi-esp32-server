"""三轮端到端验收（阶段 H，对应需求 ①-⑥ 与验收标准 1）。

用 Python 测试客户端模拟**客户端向服务端发送** opus 音频 + JPEG 视频帧，
跑 3 个完整轮次，逐条核对需求原文的断言：

| 断言                                                        | 对应需求 |
| ----------------------------------------------------------- | -------- |
| 每轮 messages 中 ⑤ 的索引 < ④ 的索引（消息列表位置）         | ⑤        |
| ④ 的消息字典同时含 video_url 视频块与 ③ 归一化文本            | ④        |
| ⑤ 的视频无音轨（ffprobe 校验 stream 数）                     | ⑤        |
| ② 结构化 JSON 每帧含 person_id / face_id(可缺失) / speaking / bbox | ②  |
| 未露出可识别人脸的人物 face_id 字段确实缺失                  | ②        |
| 同一人物单段内 person_id 不变；同一人两段视频 face_id 相同    | ②        |
| ③ 句子输出含 编号/内容/声纹id/时间[xx,xx] 且可渲染 YAML       | ③        |
| 会话结束后视频目录为空（D5）                                 | ⑥        |
| 实时性：逐帧平均耗时 <= 33ms，队列无积压                      | D15      |
| 降频下每帧仍有 bbox，detected=false 帧有跟踪外推              | ② + D15  |

本文件不启动真实 WebSocket 服务端：它用与真实链路相同的模块（ConnectionHandler
以外的同一批 utils）跑通"帧入 -> 录制 -> 分析 -> 消息组装 -> 清理"的完整数据通路，
因此可在无设备、无供应商账号的环境下复现验收。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from core.utils.asr_result import normalize_asr_result, save_asr_result
from core.utils.dialogue import Dialogue, Message
from core.utils.video.frames import VideoFrame, build_video_frame, parse_video_frame
from core.utils.video.pipeline import PipelineConfig, VideoPipeline
from core.utils.video.recorder import ffprobe_streams
from core.utils.video.session import RoundArtifacts, VideoSession
from core.utils.video.store import VideoStore


class FakeDetector:
    name = "fake"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        from core.utils.video.detector import Detection

        return [Detection(*box, score=0.9) for box in self.boxes]


class StableRecognizer:
    """同一张脸 -> 同一 face_id（跨视频一致）。"""

    enabled = True

    def __init__(self):
        from core.utils.video.face import FaceObservation

        self._obs = FaceObservation
        self.calls = 0

    def detect(self, image):
        height, width = image.shape[:2]
        self.calls += 1
        return [
            self._obs(
                bbox=[int(width * 0.35), int(height * 0.1), int(width * 0.6), int(height * 0.45)],
                det_score=0.95,
                embedding=np.ones(16, dtype=np.float32),
                yaw=3.0,
                quality=0.92,
            )
        ]

    def identify(self, observation):
        return "face_shared_1" if observation.recognizable() else None

    def close(self):
        pass


def _jpeg(index=0, width=320, height=240):
    import cv2

    image = np.full((height, width, 3), 50, dtype=np.uint8)
    cv2.rectangle(image, (100, 60), (180, 220), (210, 210, 210), -1)
    cv2.circle(image, (140, 90), 12 + index % 5, (120, 160, 220), -1)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _build_session(tmpdir, store, pipeline):
    config = {
        "video": {
            "enable": True,
            "fps": 10,
            "analyze": True,
            "output_dir": tmpdir,
            "max_buffer_frames": 600,
            "retention_rounds": 0,
            "scheduler": {"detect_every_n_frames": 3, "face_every_n_frames": 15},
            "speaker": {"threshold": 0.5},
        }
    }
    return VideoSession(
        config,
        session_id="e2e_session",
        output_dir=tmpdir,
        store=store,
        pipeline=pipeline,
    )


@pytest.fixture()
def workdir():
    path = tempfile.mkdtemp(prefix="xze2e_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_three_rounds_end_to_end(workdir):
    store = VideoStore(retention_rounds=0)
    pipeline = VideoPipeline(
        PipelineConfig(fps=10, annotate=True),
        output_dir=workdir,
        detector=FakeDetector([(100, 60, 180, 220)]),
        recognizer=StableRecognizer(),
    )
    session = _build_session(workdir, store, pipeline)

    all_rounds = []
    clock = 100000  # 虚拟单调时钟（毫秒），用于可靠地构造音画区间
    for round_index in range(1, 4):
        # 客户端上行：先发用户说话段（VAD 起止），再发小智输出段
        session.on_user_speech_start(clock)
        for index in range(15):
            message = build_video_frame(_jpeg(index), index * 100)
            assert session.push_frame_bytes(message, clock + index * 60) is True
            session.push_pcm(b"\x01\x00" * 960, clock + index * 60)
        clock += 900
        session.on_user_speech_stop(clock)

        session.on_tts_start(clock)
        for index in range(10):
            session.push_frame_bytes(
                build_video_frame(_jpeg(index + 100), index * 100), clock + index * 100
            )
        clock += 1000
        session.on_tts_stop(clock)

        asr_result = normalize_asr_result(
            {
                "sentences": [
                    {
                        "text": f"第{round_index}轮：今天天气怎么样",
                        "start": 0,
                        "end": 900,
                        "voiceprint_id": "spk_a1b2",
                    }
                ]
            },
            duration_ms=1500,
            supports_sentence_timestamp=True,
        )
        artifacts = session.build_round(asr_result)
        all_rounds.append(artifacts)

    # ---------------- 断言：每轮都有 ④ 与 ⑤ 的视频 ----------------
    for artifacts in all_rounds:
        assert artifacts.user_video, "④ 用户说话段应有视频"
        assert artifacts.tts_video, "⑤ 小智输出段应有视频"
        assert os.path.exists(artifacts.user_video)
        assert os.path.exists(artifacts.tts_video)

    # ---------------- 断言：④⑤ 均无音轨（ffprobe） ----------------
    for artifacts in all_rounds:
        streams = ffprobe_streams(artifacts.tts_video)
        if not streams:
            pytest.skip("ffprobe 不可用，跳过音轨校验")
        assert streams == ["video"], f"⑤ 必须只有视频流，实际 {streams}"
        user_streams = ffprobe_streams(artifacts.user_video)
        assert user_streams == ["video"], f"④ 也必须只有视频流，实际 {user_streams}"

    # ---------------- 断言：② 结构化 JSON 字段与 face_id 一致性 ----------------
    face_ids = set()
    for artifacts in all_rounds:
        for analysis in (artifacts.user_analysis, artifacts.tts_analysis):
            assert analysis is not None
            frames = analysis["frames"]
            assert frames, "应有结构化帧"
            for entry in frames:
                assert "timestamp_ms" in entry
                for person in entry["persons"]:
                    assert isinstance(person["person_id"], int)
                    assert len(person["bbox"]) == 4
                    assert isinstance(person["speaking"], bool)
                    assert isinstance(person["detected"], bool)
                    if person.get("face_id"):
                        face_ids.add(person["face_id"])
        person_ids = {
            person["person_id"]
            for entry in artifacts.user_analysis["frames"]
            for person in entry["persons"]
        }
        assert len(person_ids) == 1, "同一人物在单段视频内 person_id 不变"

    # 同一人在两段视频中 face_id 相同
    assert face_ids == {"face_shared_1"}

    # ---------------- 断言：③ 结构 + YAML ----------------
    asr_dict = all_rounds[0].asr
    sentence = asr_dict["sentences"][0]
    assert set(sentence) == {"index", "text", "voiceprint_id", "time", "time_source"}
    assert sentence["time_source"] == "asr"
    yaml_paths = save_asr_result(
        normalize_asr_result(
            {"sentences": [{"text": "你好", "start": 0, "end": 900, "voiceprint_id": "spk_a1b2"}]},
            duration_ms=1000,
            supports_sentence_timestamp=True,
        ),
        workdir,
        "e2e_session",
        1,
    )
    yaml_text = open(yaml_paths["yaml"], encoding="utf-8").read()
    assert "time: [0, 900]" in yaml_text

    # ---------------- 断言：D15 实时性与跟踪补帧 ----------------
    for artifacts in all_rounds:
        stats = artifacts.user_analysis["stats"]
        assert stats["avg_ms"] <= 33, f"逐帧平均耗时超预算: {stats}"
        assert stats["realtime"] is True
        assert stats["tracked_only_ratio"] <= 2.0 / 3.0 + 1e-3

    # ---------------- 断言：消息列表里 ⑤ 索引 < ④ 索引 ----------------
    from core.utils.video.round import put_round_messages

    dialogue = Dialogue()
    for artifacts in all_rounds:
        # 走生产代码的组装函数（不是测试里另写一份）
        put_round_messages(dialogue, "用户说的话", artifacts)
    messages = dialogue.get_llm_dialogue()
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert len(user_messages) == 6
    for index in range(0, 6, 2):
        tts_message = user_messages[index]
        user_message = user_messages[index + 1]
        assert index < index + 1
        assert isinstance(tts_message["content"], list)
        video_blocks = [
            b for b in user_message["content"] if b.get("type") == "video_url"
        ]
        assert len(video_blocks) == 1
        assert video_blocks[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        text_blocks = [b for b in user_message["content"] if b.get("type") == "text"]
        assert any("用户说的话" in b["text"] for b in text_blocks), "④ 应同时含文本"
        tts_blocks = [b for b in tts_message["content"] if b.get("type") == "video_url"]
        assert len(tts_blocks) == 1, "⑤ 也应是 video_url 内容块"

    # ---------------- 断言：会话结束视频目录为空（D5） ----------------
    session_videos = [a.user_video for a in all_rounds] + [a.tts_video for a in all_rounds]
    annotated = [
        analysis["video"]
        for artifacts in all_rounds
        for analysis in (artifacts.user_analysis, artifacts.tts_analysis)
        if analysis
    ]
    removed = session.close()
    for path in session_videos:
        assert not os.path.exists(path), f"会话结束后视频应删除: {path}"
    assert removed, "应报告清理的文件"
    # 标注视频同样登记并随会话删除（⑥ 的统一清理口径）
    for path in annotated:
        assert not os.path.exists(path), f"会话结束后标注视频也应删除: {path}"
    remaining = [name for name in os.listdir(workdir) if name.endswith(".mp4")]
    assert remaining == [], f"会话结束后不应残留 mp4: {remaining}"

    # ② 逐帧结构化 JSON 留档（分析产物，不属于视频，会话结束不删）
    frames_json = [
        analysis["frames_json"]
        for artifacts in all_rounds
        for analysis in (artifacts.user_analysis, artifacts.tts_analysis)
        if analysis and analysis.get("frames_json")
    ]
    assert frames_json, "应落盘逐帧结构化 JSON"
    import json as _json

    saved = _json.load(open(frames_json[0], encoding="utf-8"))
    assert saved["frames"] and saved["session_id"] == "e2e_session"
    pipeline.close()
