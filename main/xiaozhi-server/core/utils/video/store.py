"""视频生命周期（需求 ⑥ + 决策 D5，阶段 G）。

D5：整个会话结束才删除。所以：
- 每个视频文件登记 path / refcount / 会话 id / 创建时间；
- 会话 `close` 时统一清理该会话的全部视频；
- 定时兜底扫描孤儿文件（进程崩溃残留、超时会话残留）；
- `video.retention_rounds`：默认 0 = 整个会话保留；设为 N 则只保留最近 N 轮引用，
  改配置即可，不改代码。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


@dataclass
class VideoEntry:
    """一个视频文件的生命周期记录。"""

    path: str
    session_id: str
    round_index: int = 0
    kind: str = "user"  # user（④） / tts（⑤） / annotated（②）
    created_at: float = field(default_factory=time.time)
    refcount: int = 1

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "session_id": self.session_id,
            "round_index": self.round_index,
            "kind": self.kind,
            "created_at": self.created_at,
            "refcount": self.refcount,
        }


class VideoStore:
    """会话级视频文件登记表 + 清理。"""

    def __init__(self, retention_rounds: int = 0, orphan_ttl_seconds: int = 3600):
        self.retention_rounds = max(0, int(retention_rounds))
        self.orphan_ttl_seconds = max(60, int(orphan_ttl_seconds))
        self._entries: Dict[str, VideoEntry] = {}
        self._lock = threading.Lock()
        self._sessions: Dict[str, float] = {}

    # -------------------------------------------------------------- 登记
    def register(
        self,
        path: str,
        session_id: str,
        round_index: int = 0,
        kind: str = "user",
    ) -> VideoEntry:
        entry = VideoEntry(
            path=path, session_id=session_id, round_index=round_index, kind=kind
        )
        with self._lock:
            self._entries[path] = entry
            self._sessions[session_id] = time.time()
        logger.bind(tag=TAG).debug(f"[video] 登记视频 {kind}: {path}（会话 {session_id}）")
        return entry

    def list_session(self, session_id: str) -> List[VideoEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.session_id == session_id]

    def list_all(self) -> List[VideoEntry]:
        with self._lock:
            return list(self._entries.values())

    def refcount(self, path: str) -> int:
        with self._lock:
            entry = self._entries.get(path)
            return entry.refcount if entry else 0

    def add_ref(self, path: str) -> None:
        with self._lock:
            entry = self._entries.get(path)
            if entry:
                entry.refcount += 1

    def release(self, path: str) -> bool:
        """引用减一；降到 0 时删除文件并移除登记。返回是否已删除。"""
        with self._lock:
            entry = self._entries.get(path)
            if entry is None:
                return False
            entry.refcount = max(0, entry.refcount - 1)
            if entry.refcount > 0:
                return False
            self._entries.pop(path, None)
        return _remove_file(path)

    # -------------------------------------------------------------- 过期
    def prune_rounds(self, session_id: str, current_round: int) -> List[str]:
        """按 retention_rounds 清理旧轮次（默认 0 不清理）。"""
        if self.retention_rounds <= 0:
            return []
        threshold = current_round - self.retention_rounds
        if threshold < 0:
            return []
        removed: List[str] = []
        with self._lock:
            stale = [
                entry
                for entry in self._entries.values()
                if entry.session_id == session_id and entry.round_index < threshold
            ]
            for entry in stale:
                self._entries.pop(entry.path, None)
        for entry in stale:
            if _remove_file(entry.path):
                removed.append(entry.path)
        if removed:
            logger.bind(tag=TAG).info(
                f"[video] retention_rounds={self.retention_rounds}，清理了 {len(removed)} 个旧视频"
            )
        return removed

    def close_session(self, session_id: str) -> List[str]:
        """会话结束：删除该会话全部视频（D5）。"""
        with self._lock:
            entries = [
                entry for entry in self._entries.values() if entry.session_id == session_id
            ]
            for entry in entries:
                self._entries.pop(entry.path, None)
            self._sessions.pop(session_id, None)
        removed: List[str] = []
        for entry in entries:
            if _remove_file(entry.path):
                removed.append(entry.path)
        logger.bind(tag=TAG).info(
            f"[video] 会话 {session_id} 结束，清理视频 {len(removed)}/{len(entries)} 个（D5）"
        )
        return removed

    def sweep_orphans(self, now: Optional[float] = None) -> List[str]:
        """兜底清理（D5）：文件已消失、会话长期无活动、或已超 TTL 的登记项。"""
        now = now or time.time()
        with self._lock:
            stale: List[VideoEntry] = []
            for path, entry in list(self._entries.items()):
                exists = os.path.exists(path)
                last_active = self._sessions.get(entry.session_id, entry.created_at)
                expired = now - last_active > self.orphan_ttl_seconds
                if not exists or expired:
                    stale.append(entry)
                    self._entries.pop(path, None)
            for session_id, last_active in list(self._sessions.items()):
                if now - last_active > self.orphan_ttl_seconds:
                    self._sessions.pop(session_id, None)
        removed: List[str] = []
        for entry in stale:
            if os.path.exists(entry.path) and _remove_file(entry.path):
                removed.append(entry.path)
            else:
                removed.append(entry.path)
        if removed:
            logger.bind(tag=TAG).info(f"[video] 兜底清理孤儿视频 {len(removed)} 个")
        return removed

    def clear(self) -> None:
        """仅供测试：清空登记（不删文件）。"""
        with self._lock:
            self._entries.clear()
            self._sessions.clear()


def _remove_file(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except OSError as exc:
        logger.bind(tag=TAG).warning(f"[video] 删除视频失败 {path}: {exc}")
    return False


_store: Optional[VideoStore] = None
_store_lock = threading.Lock()


def get_video_store(retention_rounds: int = 0, orphan_ttl_seconds: int = 3600) -> VideoStore:
    """进程级单例。"""
    global _store
    with _store_lock:
        if _store is None:
            _store = VideoStore(retention_rounds, orphan_ttl_seconds)
        return _store


def reset_video_store() -> None:
    """仅供测试。"""
    global _store
    with _store_lock:
        _store = None