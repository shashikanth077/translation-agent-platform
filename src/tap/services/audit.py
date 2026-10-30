"""Append-only, hash-chained audit log. Every record's hash covers the previous record's hash, so any
edit, deletion or reordering is detectable with ``verify()``. This is tamper-*evident*, not tamper-proof:
ship the head hash to external WORM storage for stronger guarantees."""

from __future__ import annotations

import threading
from typing import Any

from ..db import Database
from ..util import canonical_json, dumps, loads, now_iso, sha256_hex

SCHEMA = [
    """
    CREATE TABLE audit_log (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        tenant_id TEXT,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        subject TEXT,
        details TEXT NOT NULL,
        prev_hash TEXT NOT NULL,
        hash TEXT NOT NULL
    );
    CREATE INDEX idx_audit_tenant ON audit_log(tenant_id, seq);
    CREATE INDEX idx_audit_subject ON audit_log(subject, seq)
    """
]
GENESIS = "0" * 64


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.Lock()
        db.migrate("audit", SCHEMA)

    def append(
        self,
        actor: str,
        action: str,
        *,
        tenant_id: str | None = None,
        subject: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        details = details or {}
        with self._lock, self.db.tx():
            prev = self.db.scalar("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1") or GENESIS
            ts = now_iso()
            digest = self._hash(prev, ts, tenant_id, actor, action, subject, details)
            cur = self.db.execute(
                "INSERT INTO audit_log(ts, tenant_id, actor, action, subject, details, prev_hash, hash) VALUES(?,?,?,?,?,?,?,?)",
                (ts, tenant_id, actor, action, subject, dumps(details), prev, digest),
            )
            return int(cur.lastrowid or 0)

    @staticmethod
    def _hash(prev: str, ts: str, tenant: str | None, actor: str, action: str, subject: str | None, details: dict) -> str:
        return sha256_hex(prev + canonical_json([ts, tenant, actor, action, subject, details]))

    def verify(self) -> dict[str, Any]:
        prev = GENESIS
        count = 0
        for row in self.db.query("SELECT * FROM audit_log ORDER BY seq"):
            expected = self._hash(prev, row["ts"], row["tenant_id"], row["actor"], row["action"], row["subject"], loads(row["details"], {}))
            if row["prev_hash"] != prev or row["hash"] != expected:
                return {"ok": False, "checked": count, "first_bad_seq": row["seq"], "head": prev}
            prev = row["hash"]
            count += 1
        return {"ok": True, "checked": count, "head": prev}

    def search(
        self, *, tenant_id: str | None = None, subject: str | None = None, action: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        clauses, params = ["1=1"], []
        for col, val in (("tenant_id", tenant_id), ("subject", subject), ("action", action)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        rows = self.db.query(f"SELECT * FROM audit_log WHERE {' AND '.join(clauses)} ORDER BY seq DESC LIMIT ?", (*params, limit))
        return [{**dict(r), "details": loads(r["details"], {})} for r in rows]
