"""Notification service: stores operator-facing notifications (review requested, approval needed, job done...)."""

from __future__ import annotations

import logging
from typing import Any

from ..db import Database
from ..events import Event
from ..util import dumps, loads

log = logging.getLogger("tap.notifications")

SCHEMA = [
    """
    CREATE TABLE notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT NOT NULL UNIQUE,
        tenant_id TEXT,
        topic TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
]
NOTIFY_TOPICS = ("review.requested", "job.awaiting_approval", "job.completed", "job.failed", "job.needs_attention")


class NotificationService:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("notifications", SCHEMA)

    def on_event(self, ev: Event) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO notifications(event_id, tenant_id, topic, payload, created_at) VALUES(?,?,?,?,?)",
            (ev.id, ev.tenant_id, ev.topic, dumps(ev.payload), ev.created_at),
        )
        log.info("notification %s tenant=%s", ev.topic, ev.tenant_id)

    def list_recent(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {**dict(r), "payload": loads(r["payload"], {})}
            for r in self.db.query("SELECT * FROM notifications WHERE tenant_id=? ORDER BY id DESC LIMIT ?", (tenant_id, limit))
        ]
