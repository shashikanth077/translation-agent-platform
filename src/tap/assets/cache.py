"""Cache backends and the versioned, tenant-scoped cache key.

The key includes tenant, language pair, domain, glossary version and style version, so a glossary or style
change makes old entries unreachable: invalidation by construction, not by search-and-delete.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Protocol, cast

from ..util import normalise_text, sha256_hex


def cache_key(
    source: str, pair: str, domain: str, tenant: str, glossary_v: str | None, style_v: str | None = "s1", *, kind: str = "tr"
) -> str:
    raw = "|".join([normalise_text(source), pair, domain, tenant, glossary_v or "v0", style_v or "s0"])
    return f"{kind}:{tenant}:{sha256_hex(raw)}"


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: float | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def ping(self) -> bool: ...


class MemoryCache:
    """Thread-safe LRU with optional TTL."""

    def __init__(self, max_items: int = 100_000, clock=time.monotonic) -> None:
        self._data: OrderedDict[str, tuple[str, float | None]] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()
        self._clock = clock

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if expires is not None and expires <= self._clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl_s: float | None = None) -> None:
        with self._lock:
            self._data[key] = (value, None if ttl_s is None else self._clock() + ttl_s)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def ping(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._data)


class RedisCache:
    def __init__(self, url: str, prefix: str = "tap:") -> None:
        try:
            import redis
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'redis' extra to use TAP_CACHE_BACKEND=redis") from e
        self._r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        self._prefix = prefix

    def get(self, key: str) -> str | None:
        return cast("str | None", self._r.get(self._prefix + key))

    def set(self, key: str, value: str, ttl_s: float | None = None) -> None:
        self._r.set(self._prefix + key, value, ex=None if ttl_s is None else max(1, int(ttl_s)))

    def delete(self, key: str) -> None:
        self._r.delete(self._prefix + key)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False
