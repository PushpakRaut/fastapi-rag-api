"""Per-visitor document store and fixed-window rate limits, all in memory with hard caps."""
import time
from dataclasses import dataclass, field

from .rag import Chunk


@dataclass
class Session:
    docs: dict[str, list[Chunk]] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.monotonic)

    def chunks(self) -> list[Chunk]:
        return [c for doc in self.docs.values() for c in doc]


class SessionStore:
    def __init__(self, max_sessions: int, idle_seconds: int):
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self.sessions: dict[str, Session] = {}

    def get(self, sid: str) -> Session | None:
        self._evict_idle()
        s = self.sessions.get(sid)
        if s:
            s.last_seen = time.monotonic()
        return s

    def get_or_create(self, sid: str) -> Session:
        s = self.get(sid)
        if s:
            return s
        if len(self.sessions) >= self.max_sessions:
            oldest = min(self.sessions, key=lambda k: self.sessions[k].last_seen)
            del self.sessions[oldest]
        s = self.sessions[sid] = Session()
        return s

    def _evict_idle(self) -> None:
        cutoff = time.monotonic() - self.idle_seconds
        for sid in [k for k, s in self.sessions.items() if s.last_seen < cutoff]:
            del self.sessions[sid]


class RateLimiter:
    """Fixed window per key. shortcut: single-process memory, use Redis if you run several instances."""

    def __init__(self, limit: int, window_seconds: int, max_keys: int = 10_000):
        self.limit, self.window, self.max_keys = limit, window_seconds, max_keys
        self.hits: dict[str, tuple[float, int]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        start, count = self.hits.get(key, (now, 0))
        if now - start >= self.window:
            start, count = now, 0
        if count >= self.limit:
            return False
        if key not in self.hits and len(self.hits) >= self.max_keys:
            self.hits = {k: v for k, v in self.hits.items() if now - v[0] < self.window}
            if len(self.hits) >= self.max_keys:
                return False
        self.hits[key] = (start, count + 1)
        return True
