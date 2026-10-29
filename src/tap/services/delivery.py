"""Delivery service: final format validation and idempotent publication through a connector."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from ..db import Database
from ..domain.models import DocFormat
from ..errors import ValidationFailed
from ..parse import formats
from ..util import now_iso, sha256_hex

SCHEMA = [
    """
    CREATE TABLE deliveries_log (
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        location TEXT NOT NULL,
        delivered_at TEXT NOT NULL,
        PRIMARY KEY (job_id, lang)
    )
    """
]
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Connector(Protocol):
    def deliver(self, tenant_id: str, job_id: str, lang: str, filename: str, content: str) -> str: ...


class FilesystemConnector:
    """Writes ``<root>/<tenant>/<job>/<lang>/<filename>``. Stands in for client-system connectors (SFTP, CMS, S3...)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def deliver(self, tenant_id: str, job_id: str, lang: str, filename: str, content: str) -> str:
        parts = [_SAFE.sub("_", p) for p in (tenant_id, job_id, lang, Path(filename).name)]
        dest = self.root.joinpath(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(dest)  # atomic: a reader never sees a half-written file
        return str(dest)


def format_check(fmt: DocFormat, source_content: str, output: str, src_lang: str) -> list[str]:
    """Round-trip sanity: the assembled file must re-parse to the same structure as the source."""
    issues: list[str] = []
    try:
        src = formats.parse(fmt, source_content, src_lang)
        out = formats.parse(fmt, output, src_lang)
    except ValidationFailed as e:
        return [f"output does not re-parse: {e.message}"]
    if fmt is DocFormat.TXT:
        return issues  # sentence segmentation of translated text is not guaranteed to match
    if len(src.segments) != len(out.segments):
        issues.append(f"segment count changed: {len(src.segments)} -> {len(out.segments)}")
    if fmt is DocFormat.JSON and list(src.skeleton["paths"].values()) != list(out.skeleton["paths"].values()):
        issues.append("JSON structure (keys) changed")
    return issues


class DeliveryService:
    def __init__(self, db: Database, connector: Connector) -> None:
        self.db, self.connector = db, connector
        db.migrate("delivery", SCHEMA)

    def deliver(self, tenant_id: str, job_id: str, filename: str, outputs: list[dict[str, Any]]) -> list[dict[str, str]]:
        done = []
        for out in outputs:
            h = sha256_hex(out["content"])
            row = self.db.one("SELECT * FROM deliveries_log WHERE job_id=? AND lang=?", (job_id, out["lang"]))
            if row and row["content_hash"] == h:
                done.append({"lang": out["lang"], "location": row["location"], "duplicate": "true"})  # idempotent
                continue
            location = self.connector.deliver(tenant_id, job_id, out["lang"], filename, out["content"])
            self.db.execute(
                "INSERT OR REPLACE INTO deliveries_log(job_id, lang, content_hash, location, delivered_at) VALUES(?,?,?,?,?)",
                (job_id, out["lang"], h, location, now_iso()),
            )
            done.append({"lang": out["lang"], "location": location, "duplicate": "false"})
        return done

    def delivered(self, job_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM deliveries_log WHERE job_id=? ORDER BY lang", (job_id,))]
