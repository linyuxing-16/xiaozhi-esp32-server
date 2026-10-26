"""视频生命周期测试（需求 ⑥ + D5）。

口径：整个会话结束才删除（retention_rounds=0）；
设为 N 时只保留最近 N 轮；孤儿文件由兜底扫描清理。
"""
from __future__ import annotations

import os
import tempfile

import pytest

from core.utils.video.store import VideoStore


@pytest.fixture()
def workdir():
    import shutil

    path = tempfile.mkdtemp(prefix="xzvstore_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _touch(directory, name):
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(b"data")
    return path


def test_session_keeps_all_videos_until_close(workdir):
    store = VideoStore(retention_rounds=0)
    paths = []
    for round_index in range(1, 4):
        for kind in ("user", "tts"):
            path = _touch(workdir, f"{kind}_{round_index}.mp4")
            store.register(path, "sess_a", round_index, kind)
            paths.append(path)

    # 3 轮之后会话内文件全在（D5）
    assert all(os.path.exists(p) for p in paths)
    assert len(store.list_session("sess_a")) == 6

    # 会话结束：目录清空
    removed = store.close_session("sess_a")
    assert len(removed) == 6
    assert not any(os.path.exists(p) for p in paths)
    assert store.list_session("sess_a") == []


def test_only_own_session_is_cleaned(workdir):
    store = VideoStore()
    a = _touch(workdir, "a.mp4")
    b = _touch(workdir, "b.mp4")
    store.register(a, "sess_a", 1, "user")
    store.register(b, "sess_b", 1, "user")

    store.close_session("sess_a")
    assert not os.path.exists(a)
    assert os.path.exists(b)


def test_retention_rounds_prunes_old_rounds(workdir):
    store = VideoStore(retention_rounds=1)
    old = _touch(workdir, "old.mp4")
    new = _touch(workdir, "new.mp4")
    store.register(old, "sess", 1, "user")
    store.register(new, "sess", 3, "user")

    store.prune_rounds("sess", current_round=3)
    assert not os.path.exists(old), "超出 retention_rounds 的旧轮次应被清理"
    assert os.path.exists(new)


def test_retention_zero_never_prunes(workdir):
    store = VideoStore(retention_rounds=0)
    path = _touch(workdir, "old.mp4")
    store.register(path, "sess", 1, "user")
    assert store.prune_rounds("sess", current_round=99) == []
    assert os.path.exists(path)


def test_sweep_orphans_removes_missing_and_expired(workdir):
    store = VideoStore(orphan_ttl_seconds=60)
    missing = os.path.join(workdir, "gone.mp4")  # 从未创建
    store.register(missing, "sess_x", 1, "user")
    stale = _touch(workdir, "stale.mp4")
    store.register(stale, "sess_y", 1, "user")

    removed = store.sweep_orphans(now=10**12)
    assert missing in removed
    assert stale in removed
    assert not os.path.exists(stale)


def test_refcount_release_deletes_file(workdir):
    store = VideoStore()
    path = _touch(workdir, "ref.mp4")
    store.register(path, "sess", 1, "user")
    store.add_ref(path)
    assert store.refcount(path) == 2
    assert store.release(path) is False
    assert os.path.exists(path)
    assert store.release(path) is True
    assert not os.path.exists(path)
