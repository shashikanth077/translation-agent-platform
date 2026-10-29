"""Transactional outbox + in-process event bus with idempotent, retried consumers.

``publish`` writes to the ``outbox`` table in the caller's transaction (so a state change and its event
commit atomically). ``flush`` (called by a relay thread and by tests) delivers each event to every
subscribed consumer at-least-once; consumers are idempotent by ``(event_id, consumer)``. After
``max_attempts`` failures a delivery is parked as ``dead`` (the dead-letter queue is a status, not a table).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .db import Database
from .util import dumps, loads, new_id, now_iso

log = logging.getLogger("tap.events")

SCHEMA = [
    """
    CREATE TABLE outbox (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT NOT NULL UNIQUE,
        topic TEXT NOT NULL,
        tenant_id TEXT,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
    );
    CREATE INDEX idx_outbox_status ON outbox(status, seq);
    CREATE TABLE deliveries (
        event_id TEXT NOT NULL,
        consumer TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        PRIMARY KEY (event_id, consumer)
    )
    """
]


@dataclass(frozen=True)
class Event:
    id: str
    topic: str
    tenant_id: str | None
    payload: dict[str, Any]
    created_at: str = field(default_factory=now_iso)


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, db: Database, max_attempts: int = 5) -> None:
        self.db = db
        self.max_attempts = max_attempts
        self._subs: list[tuple[str, tuple[str, ...], Handler]] = []
        self._flush_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        db.migrate("events", SCHEMA)

    def subscribe(self, consumer: str, topics: str | tuple[str, ...], handler: Handler) -> None:
        patterns = (topics,) if isinstance(topics, str) else tuple(topics)
        self._subs.append((consumer, patterns, handler))

    def publish(self, topic: str, payload: dict[str, Any], tenant_id: str | None = None) -> str:
        event_id = new_id("evt_")
        self.db.execute(
            "INSERT INTO outbox(id, topic, tenant_id, payload, created_at) VALUES(?,?,?,?,?)",
            (event_id, topic, tenant_id, dumps(payload), now_iso()),
        )
        return event_id

    @staticmethod
    def _matches(topic: str, patterns: tuple[str, ...]) -> bool:
        return any(p == "*" or p == topic or (p.endswith(".*") and topic.startswith(p[:-1])) for p in patterns)

    def flush(self, limit: int = 200) -> int:
        """Deliver pending events. Safe to call concurrently and re-entrantly. Returns events completed."""
        done = 0
        with self._flush_lock:
            while True:
                rows = self.db.query("SELECT * FROM outbox WHERE status='pending' ORDER BY seq LIMIT ?", (limit,))
                if not rows:
                    return done
                progressed = False
                for row in rows:
                    event = Event(row["id"], row["topic"], row["tenant_id"], loads(row["payload"], {}), row["created_at"])
                    if self._deliver(event):
                        self.db.execute("UPDATE outbox SET status='done' WHERE id=?", (event.id,))
                        done += 1
                        progressed = True
                if not progressed:
                    return done

    def _deliver(self, event: Event) -> bool:
        complete = True
        for consumer, patterns, handler in self._subs:
            if not self._matches(event.topic, patterns):
                continue
            state = self.db.one("SELECT status, attempts FROM deliveries WHERE event_id=? AND consumer=?", (event.id, consumer))
            if state and state["status"] in ("done", "dead"):
                continue
            attempts = state["attempts"] if state else 0
            try:
                handler(event)
            except Exception as exc:
                attempts += 1
                status = "dead" if attempts >= self.max_attempts else "retry"
                log.warning("consumer %s failed on %s (attempt %d): %s", consumer, event.topic, attempts, exc)
                self._record(event.id, consumer, status, attempts, repr(exc))
                complete = complete and status == "dead"
            else:
                self._record(event.id, consumer, "done", attempts + 1, None)
        return complete

    def _record(self, event_id: str, consumer: str, status: str, attempts: int, error: str | None) -> None:
        self.db.execute(
            "INSERT INTO deliveries(event_id, consumer, status, attempts, last_error) VALUES(?,?,?,?,?) "
            "ON CONFLICT(event_id, consumer) DO UPDATE SET status=excluded.status, "
            "attempts=excluded.attempts, last_error=excluded.last_error",
            (event_id, consumer, status, attempts, error),
        )

    def dead_letters(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT d.event_id, d.consumer, d.attempts, d.last_error, o.topic FROM deliveries d "
            "JOIN outbox o ON o.id=d.event_id WHERE d.status='dead'"
        )
        return [dict(r) for r in rows]

    def pending_count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM outbox WHERE status='pending'") or 0)

    # -- relay thread -------------------------------------------------------------------------
    def start_relay(self, interval_s: float = 0.25) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.flush()
                except Exception:
                    log.exception("event relay iteration failed")

        self._thread = threading.Thread(target=loop, name="tap-event-relay", daemon=True)
        self._thread.start()

    def stop_relay(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
