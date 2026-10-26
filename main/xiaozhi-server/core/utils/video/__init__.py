"""视频能力相关工具包。

本包按需求实现计划（v7）组织：

- capabilities: 运行环境能力探测（ffmpeg / ultralytics / insightface / onnxruntime / cv2）
- frames: 视频帧数据结构与缓冲区
- recorder: 按 VAD / TTS 区间切片并用 ffmpeg 封装 mp4（①⑤）
- detector: 人物检测 + ByteTrack 跟踪（②）
- face: 人脸识别 + SQLite 底库（② D4）
- speaker: 说话人判定（音画启发式，D12 方案 A）
- annotator: 逐帧画框叠加（②）
- scheduler: 三级差异化实时调度（D15）
- pipeline: 视觉分析管线编排，产出标注视频 + frames 结构化 JSON（②）
- store: 视频生命周期与孤儿文件清理（⑥ D5）
- session: 单连接视频会话编排（把上面各层接到 ConnectionHandler）
"""
