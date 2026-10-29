"""Circuit breaker, retry-with-backoff and ordered fallback across engines."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ..errors import AllEnginesUnavailable, ProviderError
from ..observability import Metrics
from .base import Engine, TranslationRequest, TranslationResult

log = logging.getLogger("tap.engines")


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (reset_s elapsed) -> half-open -> closed|open."""

    def __init__(self, failure_threshold: int = 5, reset_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.reset_s, self._clock = failure_threshold, reset_s, clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._probing: set[str] = set()
        self._lock = threading.Lock()

    def is_open(self, name: str) -> bool:
        with self._lock:
            opened = self._opened_at.get(name)
            if opened is None:
                return False
            if self._clock() - opened >= self.reset_s and name not in self._probing:
                self._probing.add(name)  # let exactly one probe call through (half-open)
                return False
            return True

    def state(self, name: str) -> str:
        with self._lock:
            if name not in self._opened_at:
                return "closed"
            return "half-open" if self._clock() - self._opened_at[name] >= self.reset_s else "open"

    def record_success(self, name: str) -> None:
        with self._lock:
            self._failures.pop(name, None)
            self._opened_at.pop(name, None)
            self._probing.discard(name)

    def record_failure(self, name: str) -> None:
        with self._lock:
            self._probing.discard(name)
            n = self._failures.get(name, 0) + 1
            self._failures[name] = n
            if n >= self.threshold or name in self._opened_at:
                self._opened_at[name] = self._clock()


class EngineRegistry:
    """Ordered engines per kind. ``translate`` tries each healthy engine (with retries), then falls back."""

    def __init__(
        self,
        nmt: list[Engine],
        llm: list[Engine],
        metrics: Metrics,
        *,
        breaker: CircuitBreaker | None = None,
        retry_attempts: int = 3,
        retry_initial_s: float = 0.5,
    ) -> None:
        self._engines = {"nmt": list(nmt), "llm": list(llm)}
        self.metrics = metrics
        self.breaker = breaker or CircuitBreaker()
        self.retry_attempts, self.retry_initial_s = retry_attempts, retry_initial_s

    def engines_for(self, kind: str, src: str, tgt: str) -> list[Engine]:
        return [e for e in self._engines[kind] if e.supports(src, tgt)]

    def available(self, kind: str, src: str, tgt: str) -> bool:
        return any(self.breaker.state(e.name) != "open" for e in self.engines_for(kind, src, tgt))

    def translate(self, kind: str, req: TranslationRequest) -> TranslationResult:
        candidates = self.engines_for(kind, req.src_lang, req.tgt_lang)
        if not candidates:
            raise AllEnginesUnavailable(f"no {kind} engine supports {req.src_lang}>{req.tgt_lang}")
        last: Exception | None = None
        for engine in candidates:
            if self.breaker.is_open(engine.name):
                continue
            try:
                result = self._call(engine, req)
            except ProviderError as exc:
                last = exc
                self.breaker.record_failure(engine.name)
                self.metrics.engine_calls.labels(engine.name, "error").inc()
                log.warning("engine %s failed: %s", engine.name, exc)
                continue
            self.breaker.record_success(engine.name)
            self.metrics.engine_calls.labels(engine.name, "ok").inc()
            return result
        raise AllEnginesUnavailable(f"all {kind} engines unavailable" + (f": {last}" if last else ""))

    def _call(self, engine: Engine, req: TranslationRequest) -> TranslationResult:
        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential_jitter(initial=self.retry_initial_s, max=8) if self.retry_initial_s > 0 else (lambda _rs: 0),
            retry=retry_if_exception_type(ProviderError),
            reraise=True,
        )
        def attempt() -> TranslationResult:
            started = time.monotonic()
            res = engine.translate(req)
            res.latency_ms = (time.monotonic() - started) * 1000
            return res

        return attempt()
