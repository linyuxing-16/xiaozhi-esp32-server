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

        label_parts = [f"id={obj.get('person_id', '?')}"]
        face_id = obj.get("face_id")
        if face_id:
            label_parts.append(f"face={face_id}")
        label_parts.append("speaking" if speaking else "silent")
        if not detected:
            label_parts.append("tracked")
        label = " ".join(label_parts)

        text_y = max(12, y1 - 6)
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

        face_bbox = obj.get("face_bbox")
        if face_bbox and len(face_bbox) == 4:
            fx1, fy1, fx2, fy2 = [int(round(v)) for v in face_bbox]
            cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), FACE_COLOR, 1, cv2.LINE_AA)

    return canvas


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
