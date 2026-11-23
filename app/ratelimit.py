"""Tiny in-memory per-IP rate limiter (sliding windows: per minute and per day). Single process only."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class RateLimiter:
    def __init__(self, per_minute: int = 10, per_day: int = 50, clock: Callable[[], float] = time.monotonic,
                 max_clients: int = 10_000) -> None:
        self.per_minute, self.per_day, self.clock, self.max_clients = per_minute, per_day, clock, max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, int, str]:
        """Record a request. Returns (allowed, retry_after_seconds, which_limit)."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(ip, deque())
            while q and now - q[0] >= 86_400:
                q.popleft()
            if len(q) >= self.per_day:
                return False, int(86_400 - (now - q[0])) + 1, "day"
            recent = [t for t in q if now - t < 60]
            if len(recent) >= self.per_minute:
                return False, int(60 - (now - recent[0])) + 1, "minute"
            q.append(now)
            if len(self._hits) > self.max_clients:          # bound memory: drop idle clients
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 86_400]:
                    del self._hits[k]
            return True, 0, ""
