"""逐帧画框叠加（需求 ②：每一帧的识别帧中应有如下信息）。

画框内容：person_id、face_id（可缺失）、是否说话。
输出为 BGR numpy 图像，供 pipeline 编码为 mp4。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# 画框与文字的基础样式（BGR）
PERSON_COLOR = (0, 200, 0)
PERSON_SPEAKING_COLOR = (0, 0, 255)
FACE_COLOR = (255, 160, 0)
TEXT_COLOR = (255, 255, 255)
FONT_SCALE = 0.5
THICKNESS = 2


def draw_annotations(
    image: np.ndarray,
    objects: Sequence[dict],
    font_scale: float = FONT_SCALE,
) -> np.ndarray:
    """在图像上画人物框 / 人脸框 / 文案。

    objects 的每一项形如::

        {
          "bbox": [x1, y1, x2, y2],
          "person_id": 3,
          "face_id": "face_xxx" | None,
          "speaking": True/False,
          "detected": True/False,
          "face_bbox": [x1, y1, x2, y2] | None
        }
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - 无 cv2 时不画，返回原图
        return image

    canvas = image if image.flags.writeable else image.copy()
    height, width = canvas.shape[:2]
    for obj in objects:
        bbox = obj.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        speaking = bool(obj.get("speaking"))
        detected = obj.get("detected", True)
        color = PERSON_SPEAKING_COLOR if speaking else PERSON_COLOR
        linestyle = cv2.LINE_AA if detected else cv2.LINE_8
        thickness = THICKNESS if detected else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, linestyle)

        # 标签要能被多模态模型读出来：字号随画面放大、face_id 只留短后缀、
        # 文字背后垫一块实心底板（否则小画面 + H.264 压缩后完全糊掉）。
        label_parts = [f"id={obj.get('person_id', '?')}"]
        face_id = obj.get("face_id")
        if face_id:
            label_parts.append(f"face={short_face_id(face_id)}")
        label_parts.append("speaking" if speaking else "silent")
        label = " ".join(label_parts)
        label = fit_label(label, width)

        scale, thickness = _label_style(width)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        text_x = max(0, min(x1, width - text_w - 2))
        text_y = y1 - baseline - 2
        if text_y - text_h < 0:
            text_y = min(height - baseline - 2, y1 + text_h + baseline + 2)
        cv2.rectangle(
            canvas,
            (text_x, text_y - text_h - baseline),
            (text_x + text_w + 2, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        face_bbox = obj.get("face_bbox")
        if face_bbox and len(face_bbox) == 4:
            fx1, fy1, fx2, fy2 = [int(round(v)) for v in face_bbox]
            cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), FACE_COLOR, 1, cv2.LINE_AA)

    return canvas


def _label_style(frame_width: int) -> tuple:
    """按画面宽度决定字号/线宽（320 宽用 0.55 以上才读得清）。"""
    if frame_width >= 640:
        return 0.8, 2
    if frame_width >= 480:
        return 0.7, 2
    return 0.6, 1


def short_face_id(face_id: str, keep: int = 6) -> str:
    """face_id 太长会撑出画面，只保留可辨识的后缀。"""
    text = str(face_id)
    return text if len(text) <= keep * 2 else text[-keep:]


def fit_label(label: str, frame_width: int) -> str:
    """标签超出画面宽度时截断，避免被裁掉一半。"""
    max_chars = max(8, frame_width // 9)
    return label if len(label) <= max_chars else label[: max_chars - 1] + "…"


def decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """JPEG 字节 -> BGR 图像；失败返回 None。"""
    try:
        import cv2

        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def encode_jpeg(image: np.ndarray, quality: int = 80) -> Optional[bytes]:
    """BGR 图像 -> JPEG 字节。"""
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return bytes(buffer.tobytes())
    except Exception:  # noqa: BLE001
        return None
