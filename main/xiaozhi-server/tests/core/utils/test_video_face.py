"""人脸底库与人脸可识别口径测试（需求 ②，决策 D4）。

不依赖 insightface：用合成 embedding 验证 SQLite 底库的
"同人同 id / 新人新 id / 跨视频一致 / 侧脸与过小脸不识别"。
"""
from __future__ import annotations

import numpy as np
import pytest

from core.utils.video.face import (
    FaceDatabase,
    FaceObservation,
    associate_faces_to_persons,
)


@pytest.fixture()
def db():
    import shutil
    import tempfile

    directory = tempfile.mkdtemp(prefix="xzface_")
    database = FaceDatabase(path=f"{directory}/faces.db", similarity=0.4)
    try:
        yield database
    finally:
        database.close()
        shutil.rmtree(directory, ignore_errors=True)


def _embedding(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_new_face_gets_new_global_id(db):
    first = db.enroll(_embedding(1))
    second = db.enroll(_embedding(2))
    assert first != second
    assert first.startswith("face_")
    assert db.count() == 2


def test_same_face_matches_across_videos(db):
    enrolled = db.enroll(_embedding(7))
    # 跨视频：同一个人的特征（加一点噪声）仍应命中同一个 face_id
    noisy = _embedding(7) + np.random.default_rng(9).normal(scale=0.02, size=32).astype(np.float32)
    face_id, score = db.match(noisy)
    assert face_id == enrolled
    assert score >= 0.4


def test_different_face_does_not_match(db):
    db.enroll(_embedding(11))
    face_id, score = db.match(_embedding(12))
    assert face_id is None
    assert score < 0.4


def test_touch_updates_seen_count(db):
    face_id = db.enroll(_embedding(3))
    db.touch(face_id)
    db.touch(face_id)
    # 通过公开接口间接验证：再次匹配仍是同一个 id
    assert db.match(_embedding(3))[0] == face_id


def test_observation_recognizable_gating():
    embedding = _embedding(5)
    good = FaceObservation(bbox=[0, 0, 80, 80], det_score=0.95, embedding=embedding, yaw=5.0, quality=0.9)
    assert good.recognizable() is True

    # 侧脸（|yaw| > 45）不识别
    side = FaceObservation(bbox=[0, 0, 80, 80], det_score=0.95, embedding=embedding, yaw=70.0, quality=0.9)
    assert side.recognizable() is False

    # 太小的人脸不识别
    tiny = FaceObservation(bbox=[0, 0, 20, 20], det_score=0.95, embedding=embedding, yaw=0.0, quality=0.9)
    assert tiny.recognizable() is False

    # 置信度不足不识别
    blurry = FaceObservation(bbox=[0, 0, 80, 80], det_score=0.2, embedding=embedding, yaw=0.0, quality=0.2)
    assert blurry.recognizable() is False

    # 无特征（遮挡/检测失败）不识别
    no_embedding = FaceObservation(bbox=[0, 0, 80, 80], det_score=0.95, embedding=None, yaw=0.0, quality=0.9)
    assert no_embedding.recognizable() is False


def test_associate_faces_to_persons():
    persons = [[0, 0, 200, 300], [200, 0, 400, 300]]
    face_left = FaceObservation(bbox=[50, 40, 110, 110], det_score=0.9, embedding=_embedding(1))
    face_right = FaceObservation(bbox=[250, 40, 310, 110], det_score=0.9, embedding=_embedding(2))
    mapping = associate_faces_to_persons(persons, [face_left, face_right])
    assert mapping[0] is face_left
    assert mapping[1] is face_right


def test_face_outside_person_box_is_not_assigned():
    persons = [[0, 0, 100, 100]]
    face = FaceObservation(bbox=[300, 300, 360, 360], det_score=0.9, embedding=_embedding(3))
    assert associate_faces_to_persons(persons, [face]) == {}
