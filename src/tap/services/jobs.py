"""Job service: system of record for documents, jobs, per-language segment results and assembled outputs."""

from __future__ import annotations

from typing import Any

from ..db import Database
from ..domain.models import TERMINAL_JOB_STATES, DocFormat, JobSpec, JobStatus
from ..errors import ConflictError, NotFoundError
from ..util import dumps, loads, new_id, now_iso

SCHEMA = [
    """
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        format TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_documents_tenant ON documents(tenant_id, created_at);
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        idem_key TEXT NOT NULL,
        status TEXT NOT NULL,
        spec TEXT NOT NULL,
        document_id TEXT NOT NULL,
        policy TEXT NOT NULL,
        glossary_version TEXT NOT NULL,
        created_by TEXT NOT NULL,
        analysis TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (tenant_id, idem_key)
    );
    CREATE INDEX idx_jobs_tenant ON jobs(tenant_id, created_at);
    CREATE TABLE job_segments (
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        seg_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        source TEXT NOT NULL,
        words INTEGER NOT NULL,
        risk TEXT NOT NULL,
        repeat_of TEXT,
        status TEXT NOT NULL,
        target TEXT,
        path TEXT,
        decision TEXT,
        composite REAL,
        reason TEXT,
        cost_units REAL NOT NULL DEFAULT 0,
        cacheable INTEGER NOT NULL DEFAULT 0,
        approved_by TEXT,
        detail TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (job_id, lang, seg_id)
    );
    CREATE TABLE job_outputs (
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        content TEXT NOT NULL,
        checks_ok INTEGER NOT NULL,
        issues TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (job_id, lang)
    )
    """
]

# legal status transitions; anything else is a bug and raises
_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.ANALYZING, JobStatus.FAILED},
    JobStatus.ANALYZING: {JobStatus.TRANSLATING, JobStatus.FAILED},
    JobStatus.TRANSLATING: {JobStatus.AWAITING_REVIEW, JobStatus.AWAITING_APPROVAL, JobStatus.NEEDS_ATTENTION, JobStatus.FAILED},
    JobStatus.AWAITING_REVIEW: {JobStatus.TRANSLATING, JobStatus.AWAITING_APPROVAL, JobStatus.NEEDS_ATTENTION, JobStatus.FAILED},
    JobStatus.AWAITING_APPROVAL: {JobStatus.DELIVERED, JobStatus.NEEDS_ATTENTION, JobStatus.AWAITING_REVIEW, JobStatus.FAILED},
    JobStatus.NEEDS_ATTENTION: {JobStatus.TRANSLATING, JobStatus.AWAITING_APPROVAL, JobStatus.FAILED},
    JobStatus.DELIVERED: set(),
    JobStatus.FAILED: set(),
}


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("jobs", SCHEMA)

    # -- documents ------------------------------------------------------------------------------
    def create_document(self, tenant_id: str, filename: str, fmt: DocFormat, content: str) -> str:
        doc_id = new_id("doc_")
        self.db.execute(
            "INSERT INTO documents(id, tenant_id, filename, format, content, created_at) VALUES(?,?,?,?,?,?)",
            (doc_id, tenant_id, filename, fmt.value, content, now_iso()),
        )
        return doc_id

    def get_document(self, tenant_id: str, doc_id: str) -> dict[str, Any]:
        r = self.db.one("SELECT * FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id))
        if r is None:
            raise NotFoundError(f"document {doc_id} not found")
        return dict(r)

    def list_documents(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, filename, format, created_at, length(content) AS size FROM documents "
            "WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [dict(r) for r in rows]

    def find_document(self, tenant_id: str, name: str) -> dict[str, Any] | None:
        """Match by id or by (case-insensitive) filename/stem; used by conversational intake."""
        name = name.strip().lower()
        for d in self.list_documents(tenant_id, 200):
            stem = d["filename"].rsplit(".", 1)[0].lower()
            if name in (d["id"].lower(), d["filename"].lower(), stem):
                return d
        return None

    # -- jobs -----------------------------------------------------------------------------------
    def create_job(self, spec: JobSpec, policy: str, idem_key: str, created_by: str, glossary_version: str) -> tuple[dict[str, Any], bool]:
        """Returns (job, created). A repeated idempotency key returns the original job (safe client retries)."""
        with self.db.tx():
            existing = self.db.one("SELECT * FROM jobs WHERE tenant_id=? AND idem_key=?", (spec.tenant_id, idem_key))
            if existing:
                if loads(existing["spec"]) != loads(spec.model_dump_json()):
                    raise ConflictError("Idempotency-Key was already used with a different job specification")
                return self._row(existing), False
            job_id = new_id("job_")
            now = now_iso()
            self.db.execute(
                "INSERT INTO jobs(id, tenant_id, idem_key, status, spec, document_id, policy, glossary_version, created_by,"
                " created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    spec.tenant_id,
                    idem_key,
                    JobStatus.QUEUED.value,
                    spec.model_dump_json(),
                    spec.document_id,
                    policy,
                    glossary_version,
                    created_by,
                    now,
                    now,
                ),
            )
        return self.get(job_id), True

    @staticmethod
    def _row(r: Any) -> dict[str, Any]:
        d = dict(r)
        d["spec"] = loads(d["spec"], {})
        d["analysis"] = loads(d.get("analysis"))
        return d

    def get(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        r = self.db.one(
            "SELECT * FROM jobs WHERE id=?" + (" AND tenant_id=?" if tenant_id else ""), (job_id, tenant_id) if tenant_id else (job_id,)
        )
        if r is None:
            raise NotFoundError(f"job {job_id} not found")
        return self._row(r)

    def list_jobs(self, tenant_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM jobs WHERE tenant_id=?", [tenant_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        return [self._row(r) for r in self.db.query(sql + " ORDER BY created_at DESC LIMIT ?", (*params, limit))]

    def set_status(self, job_id: str, new: JobStatus, *, error: str | None = None) -> None:
        with self.db.tx():
            cur = JobStatus(self.db.scalar("SELECT status FROM jobs WHERE id=?", (job_id,)))
            if cur == new:
                return
            if new not in _TRANSITIONS[cur]:
                raise ConflictError(f"illegal job transition {cur.value} -> {new.value}")
            self.db.execute(
                "UPDATE jobs SET status=?, error=COALESCE(?, error), updated_at=? WHERE id=?", (new.value, error, now_iso(), job_id)
            )

    def set_analysis(self, job_id: str, analysis: dict[str, Any]) -> None:
        self.db.execute("UPDATE jobs SET analysis=?, updated_at=? WHERE id=?", (dumps(analysis), now_iso(), job_id))

    def non_terminal_job_ids(self) -> list[str]:
        marks = ",".join("?" for _ in TERMINAL_JOB_STATES)
        return [
            r["id"]
            for r in self.db.query(
                f"SELECT id FROM jobs WHERE status NOT IN ({marks}) ORDER BY created_at", tuple(s.value for s in TERMINAL_JOB_STATES)
            )
        ]

    # -- segments -------------------------------------------------------------------------------
    def add_segments(self, job_id: str, rows: list[dict[str, Any]]) -> None:
        with self.db.tx():
            for r in rows:
                self.db.execute(
                    "INSERT OR IGNORE INTO job_segments(job_id, lang, seg_id, idx, source, words, risk, repeat_of, status, updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        r["lang"],
                        r["seg_id"],
                        r["idx"],
                        r["source"],
                        r["words"],
                        r["risk"],
                        r.get("repeat_of"),
                        "pending",
                        now_iso(),
                    ),
                )

    def segments(self, job_id: str, lang: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM job_segments WHERE job_id=?", [job_id]
        if lang:
            sql += " AND lang=?"
            params.append(lang)
        if status:
            sql += " AND status=?"
            params.append(status)
        out = []
        for r in self.db.query(sql + " ORDER BY lang, idx", tuple(params)):
            d = dict(r)
            d["detail"] = loads(d.get("detail"), {})
            out.append(d)
        return out

    def get_segment(self, job_id: str, lang: str, seg_id: str) -> dict[str, Any]:
        r = self.db.one("SELECT * FROM job_segments WHERE job_id=? AND lang=? AND seg_id=?", (job_id, lang, seg_id))
        if r is None:
            raise NotFoundError(f"segment {job_id}/{lang}/{seg_id} not found")
        d = dict(r)
        d["detail"] = loads(d.get("detail"), {})
        return d

    def set_segment_status(self, job_id: str, lang: str, seg_id: str, status: str) -> None:
        self.db.execute(
            "UPDATE job_segments SET status=?, updated_at=? WHERE job_id=? AND lang=? AND seg_id=?",
            (status, now_iso(), job_id, lang, seg_id),
        )

    def complete_segment(self, job_id: str, lang: str, seg_id: str, result: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE job_segments SET status='done', target=?, path=?, decision=?, composite=?, reason=?, cost_units=?, cacheable=?,"
            " approved_by=?, detail=?, updated_at=? WHERE job_id=? AND lang=? AND seg_id=?",
            (
                result["text"],
                result["path"],
                result["decision"],
                result.get("composite"),
                result.get("reason"),
                result["cost_units"],
                int(result.get("cacheable", False)),
                result.get("approved_by"),
                dumps(result.get("detail", {})),
                now_iso(),
                job_id,
                lang,
                seg_id,
            ),
        )

    # -- outputs --------------------------------------------------------------------------------
    def save_output(self, job_id: str, lang: str, content: str, checks_ok: bool, issues: list[str]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO job_outputs(job_id, lang, content, checks_ok, issues, created_at) VALUES(?,?,?,?,?,?)",
            (job_id, lang, content, int(checks_ok), dumps(issues), now_iso()),
        )

    def outputs(self, job_id: str) -> list[dict[str, Any]]:
        return [
            {**dict(r), "issues": loads(r["issues"], [])}
            for r in self.db.query("SELECT * FROM job_outputs WHERE job_id=? ORDER BY lang", (job_id,))
        ]

    def progress(self, job_id: str) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM job_segments WHERE job_id=? GROUP BY status", (job_id,))
        return {r["status"]: r["n"] for r in rows}
