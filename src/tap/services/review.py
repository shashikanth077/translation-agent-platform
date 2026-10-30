"""Review workbench service: human-in-the-loop queue (blocking review items and advisory LQA samples)."""

from __future__ import annotations

from typing import Any

from ..db import Database
from ..errors import ConflictError, NotFoundError
from ..util import dumps, loads, new_id, now_iso

SCHEMA = [
    """
    CREATE TABLE review_items (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        seg_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 1,
        payload TEXT NOT NULL,
        assignee TEXT,
        resolution TEXT,
        created_at TEXT NOT NULL,
        sla_due TEXT,
        resolved_at TEXT
    );
    CREATE INDEX idx_review_queue ON review_items(tenant_id, status, priority DESC, created_at);
    CREATE UNIQUE INDEX idx_review_open ON review_items(job_id, lang, seg_id, kind) WHERE status IN ('pending','claimed')
    """
]


class ReviewService:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("review", SCHEMA)

    def create(
        self,
        tenant_id: str,
        job_id: str,
        lang: str,
        seg_id: str,
        thread_id: str,
        payload: dict[str, Any],
        *,
        kind: str = "review",
        priority: int = 1,
        sla_due: str | None = None,
    ) -> dict[str, Any]:
        """Idempotent: an open item for the same segment is returned instead of duplicated."""
        with self.db.tx():
            existing = self.db.one(
                "SELECT * FROM review_items WHERE job_id=? AND lang=? AND seg_id=? AND kind=? AND status IN ('pending','claimed')",
                (job_id, lang, seg_id, kind),
            )
            if existing:
                return self._row(existing)
            item_id = new_id("rev_")
            self.db.execute(
                "INSERT INTO review_items(id, tenant_id, job_id, lang, seg_id, thread_id, kind, status, priority, payload,"
                " created_at, sla_due)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, tenant_id, job_id, lang, seg_id, thread_id, kind, "pending", priority, dumps(payload), now_iso(), sla_due),
            )
        return self.get(item_id)

    @staticmethod
    def _row(r: Any) -> dict[str, Any]:
        d = dict(r)
        d["payload"] = loads(d["payload"], {})
        d["resolution"] = loads(d.get("resolution"))
        return d

    def get(self, item_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        r = self.db.one(
            "SELECT * FROM review_items WHERE id=?" + (" AND tenant_id=?" if tenant_id else ""),
            (item_id, tenant_id) if tenant_id else (item_id,),
        )
        if r is None:
            raise NotFoundError(f"review item {item_id} not found")
        return self._row(r)

    def list_items(
        self, tenant_id: str, *, status: str | None = None, kind: str | None = None, job_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM review_items WHERE tenant_id=?", [tenant_id]
        for col, val in (("status", status), ("kind", kind), ("job_id", job_id)):
            if val:
                sql += f" AND {col}=?"
                params.append(val)
        rows = self.db.query(sql + " ORDER BY priority DESC, created_at LIMIT ?", (*params, limit))
        return [self._row(r) for r in rows]

    def claim(self, item_id: str, tenant_id: str, user: str) -> dict[str, Any]:
        with self.db.tx():
            item = self.get(item_id, tenant_id)
            if item["status"] == "claimed" and item["assignee"] != user:
                raise ConflictError(f"already claimed by {item['assignee']}")
            if item["status"] not in ("pending", "claimed"):
                raise ConflictError(f"item is {item['status']}")
            self.db.execute("UPDATE review_items SET status='claimed', assignee=? WHERE id=?", (user, item_id))
        return self.get(item_id)

    def resolve(self, item_id: str, tenant_id: str, user: str, resolution: dict[str, Any]) -> dict[str, Any]:
        with self.db.tx():
            item = self.get(item_id, tenant_id)
            if item["status"] not in ("pending", "claimed"):
                raise ConflictError(f"item is already {item['status']}")
            if item["assignee"] and item["assignee"] != user:
                raise ConflictError(f"item is claimed by {item['assignee']}")
            self.db.execute(
                "UPDATE review_items SET status='done', assignee=?, resolution=?, resolved_at=? WHERE id=?",
                (user, dumps(resolution), now_iso(), item_id),
            )
        return self.get(item_id)

    def reopen(self, item_id: str) -> None:
        """Used when resuming the graph failed: the reviewer's decision must not be lost."""
        self.db.execute("UPDATE review_items SET status='pending', resolved_at=NULL WHERE id=?", (item_id,))

    def depth(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM review_items WHERE status IN ('pending','claimed') AND kind='review'") or 0)

    def open_blocking(self, job_id: str) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM review_items WHERE job_id=? AND kind='review' AND status IN ('pending','claimed')", (job_id,)
            )
            or 0
        )
