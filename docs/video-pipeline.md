# 视频管线（服务端）

> 本文件对应仓库根目录「需求实现计划.md」（v7）的阶段 A/C/D/E/F/G/H 落地说明。
> 需求原文逐条编号 ①-⑥ 与本文一一对应。

## 1. 总体链路

```
ESP32 (lckfb/szpi-esp32s3, JPEG 直出)
  └─ WebSocket 二进制帧 type=2 ──> ConnectionHandler._route_message
                                     └─ VideoSession.push_frame_bytes()  (有界缓冲，满丢最旧)
麦克风 opus ──> 解码 PCM ──> VideoSession.push_pcm()
VAD 上升沿 ──> on_user_speech_start()   下降沿 ──> on_user_speech_stop()      (需求 ①④)
TTS 首帧   ──> on_tts_start()           TTS 结束 ──> on_tts_stop()            (需求 ⑤ D10)
                    │
                    ▼  chat() 的线程池线程里
            VideoSession.build_round(asr_result)
                    ├─ VideoRecorder.render()  -> ④ 有音轨 mp4 / ⑤ 静音 mp4
                    ├─ VideoPipeline.analyze() -> 标注视频 + frames[] 结构化 JSON（需求 ②）
                    └─ VideoStore.register()   -> 生命周期登记（需求 ⑥ D5）
                    ▼
            Dialogue.put(⑤ 消息) -> Dialogue.put(④ 消息)   （⑤ 索引 < ④ 索引）
                    ▼
            LLM content 数组：[text, video_url(base64), text(结构化摘要)]（需求 ④ D3）
                    ▼
            会话 close() -> VideoStore.close_session() 删除全部视频（需求 ⑥ D5）
```

## 2. 模块清单

| 文件 | 职责 | 需求 |
| --- | --- | --- |
| `core/utils/video/capabilities.py` | 探测 ffmpeg / cv2 / onnxruntime / ultralytics / insightface，缺依赖整体降级并明示日志 | 阶段 A |
| `core/utils/video/frames.py` | `VideoFrame`、二进制帧解析/构造、有界 `FrameBuffer` | D11 |
| `core/utils/video/recorder.py` | 按区间切片 + ffmpeg 封装 mp4，④ 留音轨、⑤ 静音 | ①④⑤ |
| `core/utils/video/detector.py` | YOLO（ultralytics 或 ONNX）+ ByteTrack 跟踪（卡尔曼外推 + IoU 匹配） | ② |
| `core/utils/video/face.py` | InsightFace(ArcFace) + SQLite 底库，跨视频唯一 `face_id` | ② D4 |
| `core/utils/video/speaker.py` | `SpeakerDetector` 抽象 + `HeuristicSpeakerDetector`（音画加权打分） | ② D12 |
| `core/utils/video/annotator.py` | 逐帧画框 + person_id + face_id + 是否说话 | ② |
| `core/utils/video/scheduler.py` | 三级差异化实时调度 + 自适应降频 | D15 |
| `core/utils/video/pipeline.py` | 产出标注视频 + `frames[]` 结构化 JSON | ② |
| `core/utils/video/multimodal.py` | `video_url` 内容块 + base64 分块拼接 | ④ D3 |
| `core/utils/video/store.py` | 视频生命周期与孤儿文件清理 | ⑥ D5 |
| `core/utils/video/session.py` | 单连接编排（把以上接到 ConnectionHandler） | 全部 |
| `core/utils/asr_result.py` | ASR 归一化：编号/内容/声纹id/时间[xx,xx]/time_source | ③ D13 |
| `core/providers/llm/video_url/` | OpenAI 兼容网关视频适配器（`supports_video`） | ④ D7 |
| `core/providers/llm/MockVideoLLM/` | 测试提供商，记录 messages 原文 | 阶段 H |
| `core/providers/asr/mock_asr.py` | 返回带时间戳 + 声纹 id 的预定句子 | 阶段 H |

## 3. 配置（`config.yaml` 的 `video` 节）

```yaml
video:
  enable: false            # 总开关（关闭时现有音频链路行为零变化）
  output_dir: "data/video"
  fps: 10
  width: 320
  height: 240
  max_segment_ms: 60000
  max_buffer_frames: 600   # 满了丢最旧，永不积压
  analyze: true
  annotate: true
  jpeg_quality: 80
  retention_rounds: 0      # 0 = 整个会话保留（D5）；N = 只保留最近 N 轮
  orphan_ttl_seconds: 3600
  scheduler:
    detect_every_n_frames: 3
    face_every_n_frames: 15
    target_budget_ms: 33
    adaptive: true
    max_detect_interval: 6
    detector_input_size: 320
  detector:
    model_path: "yolov8n.pt"
    conf: 0.35
    imgsz: 320
  face:
    model_name: "buffalo_l"
    det_size: 320
    db_path: "data/video/faces.db"
    similarity: 0.45
  speaker:
    speaker_detector: "heuristic"   # heuristic | asd
    w_area: 0.35
    w_yaw: 0.25
    w_mouth: 0.30
    w_voiceprint: 0.10
    threshold: 0.5
  provider:
    base_url: ""
    model_name: ""
    api_key: ""
```

**不设体积上限**：D14 已按你的要求删除 20MB 保护，发送前不判体积、不截断、不降码率。
体积靠 `fps` / `width` / `height` / `max_segment_ms` 事前约束；若网关自身拒绝超大请求，
按普通 HTTP 错误记录日志。

## 4. 结构化输出（需求 ②）

```json
{
  "video": "data/video/user_xxx_round1_1712.mp4",
  "frames": [
    {
      "timestamp_ms": 1712345678,
      "persons": [
        {
          "person_id": 1,
          "bbox": [100, 60, 180, 220],
          "speaking": true,
          "detected": true,
          "face_id": "face_ab12cd34ef56",
          "face_bbox": [120, 70, 160, 120]
        }
      ]
    }
  ],
  "stats": {
    "frames": 30,
    "avg_ms": 9.4,
    "p95_ms": 18.2,
    "budget_ms": 33,
    "realtime": true,
    "tracked_only_ratio": 0.65,
    "detector_backend": "fake",
    "face_enabled": true
  }
}
```

口径说明：

- **每帧都有 bbox**：YOLO 只每 N 帧跑一次，其余帧用 ByteTrack 外推补框（`detected: false`）。
- **`face_id` 缺失即"没露出可识别人脸"**：不写空串、不写 `unknown`、不做模糊兜底（v4 澄清口径）。
- **`speaking` 为启发式结果**：权重/阈值全部在 `video.speaker`，可调参不改代码。

## 5. ASR 归一化（需求 ③）

```yaml
sentences:
  - index: 1
    text: "你好，今天天气怎么样"
    voiceprint_id: "spk_a1b2"
    time: [1200, 4300]
    time_source: asr
  - index: 2
    text: "我想出去走走"
    voiceprint_id: "spk_a1b2"
    time: [5100, 7800]
    time_source: vad
```

- `time_source: asr` = 供应商自带句级时间戳（适配器声明 `supports_sentence_timestamp`）；
- `time_source: vad` = 回退 VAD 近似；
- 无法识别声纹写 `null`（不是空串、不是 `unknown`）；
- 两种来源都必须满足"区间单调、不重叠、落在音频时长内"（`_enforce_monotonic` 兜底）。

## 6. 实时性（D15）

| 处理项 | 频率 | 说明 |
| --- | --- | --- |
| YOLO 检测 | 每 `detect_every_n_frames`=3 帧 | 最贵，必须降频 |
| ByteTrack 跟踪 | 每帧 | 只算 IoU + 外推，<1ms |
| 人脸检测+识别 | 每 `face_every_n_frames`=15 帧 | 人脸不变，无需高频 |
| 说话判定 + 画框 | 每帧 | 必须逐帧 |

超 `target_budget_ms`（33ms）自动加大检测间隔（上限 `max_detect_interval`=6），
负载回落逐步恢复；每一级都写日志（`level` 1/2/3）。
帧缓冲有界（`max_buffer_frames`），满了丢最旧，**音频链路永不被阻塞**。

## 7. 测试

```bash
cd xiaozhi-esp32-server/main/xiaozhi-server
python -m pytest tests/test_video_pipeline.py tests/core/utils/test_video_recorder.py \
  tests/core/utils/test_asr_result.py tests/core/utils/test_video_store.py \
  tests/test_video_wiring.py tests/e2e/test_three_rounds.py -v
```

| 测试文件 | 覆盖 |
| --- | --- |
| `tests/test_video_pipeline.py` | ② 每帧 bbox、`detected` 标记、person_id 不跳变、face_id 缺失口径、D15 预算 |
| `tests/core/utils/test_video_recorder.py` | ①④⑤ 音画对齐、④ 有音轨、⑤ ffprobe 只有视频流 |
| `tests/core/utils/test_asr_result.py` | ③ 三种返回形态、time_source、单调不重叠、YAML 渲染 |
| `tests/core/utils/test_video_store.py` | ⑥ 会话结束删除、retention_rounds、孤儿兜底 |
| `tests/test_video_wiring.py` | 二进制帧分流、VAD/TTS 打点、hello 能力协商、关闭时零副作用 |
| `tests/e2e/test_three_rounds.py` | 三轮验收：⑤ 索引 < ④、video_url 块、③ 结构、D5 清理、D15 预算 |

## 8. 未在本机验证、需真机/真实供应商复核

1. ESP32 真机连续推流 5 分钟不崩、音频链路无退化（阶段 B，C14 已约定由你执行）；
2. 真实 YOLO / InsightFace 权重下的检出率与耗时（本机未安装 `ultralytics` / `insightface`，
   测试用假检测器/假识别器验证链路与口径，**不代表真实模型精度**）；
3. 真实视频供应商网关对 `video_url` 内联 base64 的接受度（需 `video.provider` 填 base_url + 模型名）；
4. ESP32-S3 Wi-Fi 上 640×360/320×240 JPEG@10fps 与 opus 上行共存的实际带宽。
