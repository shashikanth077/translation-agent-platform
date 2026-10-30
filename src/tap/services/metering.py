"""Metering and cost governance: usage per path, savings versus the legacy baseline, budgets.

The cost model is explicit and configurable (``Settings.prices``): each processing step has a price per 1,000
words. ``baseline_cost_per_1k_words`` is what the legacy workflow would have charged for the same words.
Reported savings are therefore only as good as those two inputs: replace the defaults with measured values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..db import Database
from ..errors import BudgetExceeded
from ..events import Event
from ..util import dumps, loads, now_iso

SCHEMA = [
    """
    CREATE TABLE usage_events (
        event_id TEXT PRIMARY KEY,
        ts TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        lang TEXT NOT NULL,
        seg_id TEXT NOT NULL,
        domain TEXT,
        path TEXT NOT NULL,
        decision TEXT,
        words INTEGER NOT NULL,
        cost_units REAL NOT NULL,
        baseline_units REAL NOT NULL,
        composite REAL,
        meta TEXT
    );
    CREATE INDEX idx_usage_tenant ON usage_events(tenant_id, ts);
    CREATE INDEX idx_usage_job ON usage_events(job_id)
    """
]


class Metering:
    def __init__(self, db: Database, prices: dict[str, float], baseline_per_1k: float, default_job_budget: float) -> None:
        self.db, self.prices, self.baseline_per_1k = db, prices, baseline_per_1k
        self.default_job_budget = default_job_budget
        db.migrate("metering", SCHEMA)

    # -- pricing --------------------------------------------------------------------------------
    def cost(self, steps: list[str], words: int) -> float:
        return round(sum(self.prices.get(s, 0.0) for s in steps) * words / 1000, 6)

    def baseline(self, words: int) -> float:
        return round(self.baseline_per_1k * words / 1000, 6)

    # -- event consumer -------------------------------------------------------------------------
    def on_segment_completed(self, ev: Event) -> None:
        p = ev.payload
        self.db.execute(
            "INSERT OR IGNORE INTO usage_events(event_id, ts, tenant_id, job_id, lang, seg_id, domain, path, decision, words,"
            " cost_units, baseline_units, composite, meta) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ev.id,
                ev.created_at,
                ev.tenant_id,
                p["job_id"],
                p["lang"],
                p["seg_id"],
                p.get("domain"),
                p["path"],
                p.get("decision"),
                p["words"],
                p["cost_units"],
                self.baseline(p["words"]),
                p.get("composite"),
                dumps(p.get("meta", {})),
            ),
        )

    # -- governance -----------------------------------------------------------------------------
    def job_spend(self, job_id: str) -> float:
        return float(self.db.scalar("SELECT COALESCE(SUM(cost_units),0) FROM usage_events WHERE job_id=?", (job_id,)) or 0.0)

    def check_job_budget(self, job_id: str, planned_units: float, budget: float | None = None) -> None:
        budget = budget or self.default_job_budget
        if self.job_spend(job_id) + planned_units > budget:
            raise BudgetExceeded(f"job {job_id} would exceed its budget of {budget:.2f} units")

    def job_has_budget(self, job_id: str, planned_units: float, budget: float | None = None) -> bool:
        try:
            self.check_job_budget(job_id, planned_units, budget)
            return True
        except BudgetExceeded:
            return False

    # -- reporting ------------------------------------------------------------------------------
    def report(self, tenant_id: str, *, job_id: str | None = None, since: str | None = None, until: str | None = None) -> dict[str, Any]:
        sql = "SELECT * FROM usage_events WHERE tenant_id=?"
        params: list[Any] = [tenant_id]
        for clause, val in (("job_id=?", job_id), ("ts>=?", since), ("ts<?", until)):
            if val:
                sql += f" AND {clause}"
                params.append(val)
        rows = self.db.query(sql, tuple(params))
        total_words = sum(r["words"] for r in rows)
        total_cost = sum(r["cost_units"] for r in rows)
        total_base = sum(r["baseline_units"] for r in rows)
        by_path: dict[str, dict[str, float]] = defaultdict(lambda: {"segments": 0, "words": 0, "cost_units": 0.0})
        for r in rows:
            agg = by_path[r["path"]]
            agg["segments"] += 1
            agg["words"] += r["words"]
            agg["cost_units"] += r["cost_units"]
        reuse_words = sum(a["words"] for p, a in by_path.items() if p.startswith("reuse"))
        return {
            "tenant_id": tenant_id,
            "segments": len(rows),
            "words": total_words,
            "cost_units": round(total_cost, 4),
            "baseline_cost_units": round(total_base, 4),
            "savings_pct": round(100 * (1 - total_cost / total_base), 2) if total_base else None,
            "reuse_word_share_pct": round(100 * reuse_words / total_words, 2) if total_words else None,
            "cost_per_1k_words": round(1000 * total_cost / total_words, 4) if total_words else None,
            "by_path": {p: {k: round(v, 4) for k, v in a.items()} for p, a in sorted(by_path.items())},
            "assumptions": {
                "prices_per_1k_words": self.prices,
                "baseline_per_1k_words": self.baseline_per_1k,
                "note": "Savings are computed against the configured baseline; replace the defaults with measured values.",
            },
            "generated_at": now_iso(),
        }

    def qe_history(self, tenant_id: str, pair: str, domain: str) -> tuple[float | None, int]:
        """Mean composite of the *first NMT candidate* (whether or not it was later repaired) and its sample size.

        Using every first attempt avoids survivorship bias: averaging only segments NMT got right would hide failures.
        """
        row = self.db.one(
            "SELECT AVG(json_extract(meta,'$.nmt_composite')) AS m, COUNT(json_extract(meta,'$.nmt_composite')) AS n "
            "FROM usage_events WHERE tenant_id=? AND domain=? AND json_extract(meta,'$.pair')=?",
            (tenant_id, domain, pair),
        )
        return (row["m"], int(row["n"])) if row and row["n"] else (None, 0)

    def job_summary(self, job_id: str) -> dict[str, Any]:
        rows = self.db.query("SELECT path, decision, composite FROM usage_events WHERE job_id=?", (job_id,))
        n = len(rows)
        return {
            "segments": n,
            "auto_accept_rate": (sum(1 for r in rows if "human" not in r["path"]) / n) if n else 0.0,
            "human_rate": (sum(1 for r in rows if "human" in r["path"]) / n) if n else 0.0,
            "mean_composite": (
                sum(r["composite"] for r in rows if r["composite"] is not None) / max(1, sum(1 for r in rows if r["composite"] is not None))
            ),
            "cost_units": self.job_spend(job_id),
        }


def parse_meta(raw: str | None) -> dict[str, Any]:
    return loads(raw, {})
