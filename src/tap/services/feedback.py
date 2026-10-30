"""Feedback service: turns reviewer edits into evaluation data and glossary suggestions (closing the loop).

Post-edit distance is the headline signal: it is a direct measure of how much human effort a machine
translation still needed, and a better quality proxy than any single automatic metric. Glossary
suggestions are *suggestions*: a human admin must approve them through ``update_glossary`` (approval-gated).
"""

from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from ..db import Database
from ..events import Event
from ..util import now_iso

SCHEMA = [
    """
    CREATE TABLE post_edits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        seg_id TEXT NOT NULL,
        domain TEXT,
        source TEXT NOT NULL,
        machine TEXT NOT NULL,
        final TEXT NOT NULL,
        distance REAL NOT NULL,
        engine TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (job_id, lang, seg_id)
    );
    CREATE TABLE term_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        machine_term TEXT NOT NULL,
        human_term TEXT NOT NULL,
        occurrences INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'pending',
        UNIQUE (tenant_id, lang, machine_term, human_term)
    )
    """
]


def edit_distance(a: str, b: str) -> float:
    """Normalised Levenshtein distance in [0, 1]: 0 = identical, 1 = completely rewritten."""
    return Levenshtein.normalized_distance(a, b)


def replaced_terms(machine: str, final: str) -> list[tuple[str, str]]:
    """Word-level replacements the reviewer made (candidate glossary suggestions)."""
    a, b = machine.split(), final.split()
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and (i2 - i1) <= 3 and (j2 - j1) <= 3:
            out.append((" ".join(a[i1:i2]).strip(".,;:"), " ".join(b[j1:j2]).strip(".,;:")))
    return [(m, h) for m, h in out if m and h and m.lower() != h.lower()]


class FeedbackService:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("feedback", SCHEMA)

    def on_review_resolved(self, ev: Event) -> None:
        p = ev.payload
        if p.get("kind") != "review" or p.get("machine") is None:
            return
        d = edit_distance(p["machine"], p["final"])
        self.db.execute(
            "INSERT OR REPLACE INTO post_edits(tenant_id, job_id, lang, seg_id, domain, source, machine, final, distance,"
            " engine, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                ev.tenant_id,
                p["job_id"],
                p["lang"],
                p["seg_id"],
                p.get("domain"),
                p["source"],
                p["machine"],
                p["final"],
                d,
                p.get("engine"),
                now_iso(),
            ),
        )
        if d > 0:
            for m, h in replaced_terms(p["machine"], p["final"]):
                self.db.execute(
                    "INSERT INTO term_suggestions(tenant_id, lang, machine_term, human_term) VALUES(?,?,?,?) "
                    "ON CONFLICT(tenant_id, lang, machine_term, human_term) DO UPDATE SET occurrences=occurrences+1",
                    (ev.tenant_id, p["lang"], m, h),
                )

    def suggestions(self, tenant_id: str, min_occurrences: int = 2) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM term_suggestions WHERE tenant_id=? AND status='pending' AND occurrences>=? ORDER BY occurrences DESC LIMIT 100",
            (tenant_id, min_occurrences),
        )
        return [dict(r) for r in rows]

    def stats(self, tenant_id: str) -> dict[str, Any]:
        rows = self.db.query("SELECT distance, engine FROM post_edits WHERE tenant_id=?", (tenant_id,))
        if not rows:
            return {"edits": 0, "mean_distance": None, "unchanged_rate": None, "by_engine": {}}
        by_engine: dict[str, list[float]] = {}
        for r in rows:
            by_engine.setdefault(r["engine"] or "unknown", []).append(r["distance"])
        return {
            "edits": len(rows),
            "mean_distance": sum(r["distance"] for r in rows) / len(rows),
            "unchanged_rate": sum(1 for r in rows if r["distance"] == 0) / len(rows),
            "by_engine": {k: {"n": len(v), "mean_distance": sum(v) / len(v)} for k, v in by_engine.items()},
        }

    def export_golden(self, tenant_id: str, path: Path) -> int:
        """Human-approved translations become regression examples for ``tap eval`` (JSONL)."""
        rows = self.db.query("SELECT source, final, lang, domain, job_id, seg_id FROM post_edits WHERE tenant_id=?", (tenant_id,))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(
                    json.dumps(
                        {
                            "id": f"{r['job_id']}:{r['lang']}:{r['seg_id']}",
                            "source": r["source"],
                            "reference": r["final"],
                            "src_lang": "en",
                            "tgt_lang": r["lang"],
                            "domain": r["domain"] or "clinical",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(rows)

    def count_terms(self, tenant_id: str) -> Counter[str]:  # pragma: no cover - small convenience
        return Counter(
            {
                r["machine_term"]: r["occurrences"]
                for r in self.db.query("SELECT machine_term, occurrences FROM term_suggestions WHERE tenant_id=?", (tenant_id,))
            }
        )
