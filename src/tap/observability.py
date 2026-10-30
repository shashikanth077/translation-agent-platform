"""Structured logging (no source text in logs by default) and Prometheus metrics."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("tap_log_ctx", default=None)


def bind(**fields: Any) -> contextvars.Token:
    """Attach fields (job_id, tenant_id, request_id...) to every log line in this context."""
    return _ctx.set({**(_ctx.get() or {}), **fields})


def unbind(token: contextvars.Token) -> None:
    _ctx.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **(_ctx.get() or {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        if getattr(h, "_tap", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler._tap = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


class Metrics:
    """One registry per platform instance so tests can create many platforms in a process."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.segments = Counter("tap_segments_total", "Segments completed", ["path", "decision"], registry=self.registry)
        self.cache_lookups = Counter("tap_cache_lookups_total", "Cache/TM lookups", ["result"], registry=self.registry)
        self.engine_calls = Counter("tap_engine_calls_total", "Engine calls", ["engine", "outcome"], registry=self.registry)
        self.gateway_calls = Counter("tap_gateway_calls_total", "MCP gateway calls", ["tool", "status"], registry=self.registry)
        self.segment_seconds = Histogram(
            "tap_segment_seconds",
            "Segment workflow latency",
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.http_seconds = Histogram("tap_http_request_seconds", "HTTP latency", ["method", "route", "status"], registry=self.registry)
        self.review_queue = Gauge("tap_review_queue_depth", "Pending review items", registry=self.registry)
        self.dead_letters = Gauge("tap_event_dead_letters", "Events parked in the DLQ", registry=self.registry)
