"""Approvals for high-risk tools. An approval is bound to (tenant, tool, exact arguments) and is single-use."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..db import Database
from ..errors import ConflictError, Forbidden, NotFoundError
from ..util import canonical_json, new_id, now_iso, sha256_hex, utcnow

SCHEMA = [
    """
    CREATE TABLE approvals (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        tool TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        args TEXT NOT NULL,
        status TEXT NOT NULL,
        requested_by TEXT NOT NULL,
        decided_by TEXT,
        reason TEXT,
        created_at TEXT NOT NULL,
        decided_at TEXT
    );
    CREATE INDEX idx_approvals_lookup ON approvals(tenant_id, tool, args_hash, status)
    """
]


def args_hash(tool: str, args: dict[str, Any]) -> str:
    return sha256_hex(tool + canonical_json(args))


class Approvals:
    def __init__(self, db: Database, ttl_hours: float = 24.0) -> None:
        self.db = db
        self.ttl = timedelta(hours=ttl_hours)
        db.migrate("approvals", SCHEMA)

    def request(self, tenant_id: str, tool: str, args: dict[str, Any], requested_by: str) -> str:
        h = args_hash(tool, args)
        existing = self.db.one(
            "SELECT id FROM approvals WHERE tenant_id=? AND tool=? AND args_hash=? AND status IN ('pending','approved')",
            (tenant_id, tool, h),
        )
        if existing:
            return existing["id"]
        aid = new_id("apr_")
        self.db.execute(
            "INSERT INTO approvals(id, tenant_id, tool, args_hash, args, status, requested_by, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (aid, tenant_id, tool, h, canonical_json(args), "pending", requested_by, now_iso()),
        )
        return aid

    def grant(self, tenant_id: str, tool: str, args: dict[str, Any], decided_by: str, reason: str = "") -> str:
        """Record an already-decided approval (e.g. a reviewer clicked "approve" on a job)."""
        aid = self.request(tenant_id, tool, args, decided_by)
        self.decide(tenant_id, aid, approve=True, decided_by=decided_by, reason=reason, allow_self=True)
        return aid

    def decide(
        self, tenant_id: str, approval_id: str, *, approve: bool, decided_by: str, reason: str = "", allow_self: bool = False
    ) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM approvals WHERE id=? AND tenant_id=?", (approval_id, tenant_id))
        if row is None:
            raise NotFoundError(f"approval {approval_id} not found")
        if row["status"] == "approved" and approve and row["decided_by"] == decided_by:
            return dict(row)  # idempotent
        if row["status"] != "pending":
            raise ConflictError(f"approval already {row['status']}")
        if not allow_self and row["requested_by"] == decided_by:
            raise Forbidden("four-eyes rule: the requester cannot approve their own request")
        self.db.execute(
            "UPDATE approvals SET status=?, decided_by=?, reason=?, decided_at=? WHERE id=?",
            ("approved" if approve else "rejected", decided_by, reason, now_iso(), approval_id),
        )
        return dict(self.db.one("SELECT * FROM approvals WHERE id=?", (approval_id,)))  # type: ignore[arg-type]

    def cutoff(self) -> str:
        """Approvals decided before this instant have expired: a stale 'yes' must not authorise a later action."""
        return (utcnow() - self.ttl).isoformat(timespec="milliseconds")

    def consume(self, tenant_id: str, tool: str, args: dict[str, Any]) -> str | None:
        """Atomically spend one matching approval. Returns its id, or None if there is none."""
        h = args_hash(tool, args)
        with self.db.tx():
            row = self.db.one(
                "SELECT id FROM approvals WHERE tenant_id=? AND tool=? AND args_hash=? AND status='approved' AND decided_at>=? "
                "ORDER BY created_at LIMIT 1",
                (tenant_id, tool, h, self.cutoff()),
            )
            if row is None:
                return None
            self.db.execute("UPDATE approvals SET status='consumed' WHERE id=?", (row["id"],))
            return str(row["id"])

    def list_all(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM approvals WHERE tenant_id=?", [tenant_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        return [dict(r) for r in self.db.query(sql + " ORDER BY created_at DESC LIMIT 200", tuple(params))]
