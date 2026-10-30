"""A/B experiments: deterministic bucketing, outcome logging and statistics.

Rules baked in (from the guide): randomise by *job/document* (segments inside one document are not
independent), keep quality as a non-inferiority guardrail while testing cost or speed, and refuse to call a
winner below the minimum sample size.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from ..db import Database
from ..errors import ConflictError, NotFoundError, ValidationFailed
from ..util import dumps, loads, now_iso

SCHEMA = [
    """
    CREATE TABLE experiments (
        name TEXT PRIMARY KEY,
        variants TEXT NOT NULL,
        traffic_pct INTEGER NOT NULL DEFAULT 100,
        status TEXT NOT NULL DEFAULT 'running',
        primary_metric TEXT NOT NULL,
        guardrail_metric TEXT,
        guardrail_margin REAL NOT NULL DEFAULT 0.02,
        min_samples INTEGER NOT NULL DEFAULT 200,
        created_at TEXT NOT NULL
    );
    CREATE TABLE assignments (
        experiment TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        variant TEXT NOT NULL,
        ts TEXT NOT NULL,
        PRIMARY KEY (experiment, unit_id)
    );
    CREATE TABLE outcomes (
        experiment TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        metric TEXT NOT NULL,
        value REAL NOT NULL,
        PRIMARY KEY (experiment, unit_id, metric)
    )
    """
]


def bucket(unit_id: str, experiment: str, variants: list[str], traffic_pct: int = 100) -> str | None:
    """Deterministic assignment: the same unit always lands in the same variant. ``None`` = not in experiment."""
    h = int(hashlib.sha256(f"{experiment}:{unit_id}".encode()).hexdigest(), 16)
    if h % 100 >= traffic_pct:
        return None
    return variants[(h // 100) % len(variants)]


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_prop_z(xa: int, na: int, xb: int, nb: int) -> tuple[float, float]:
    if na == 0 or nb == 0:
        return 0.0, 1.0
    pa, pb = xa / na, xb / nb
    p = (xa + xb) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    if se == 0:
        return 0.0, 1.0
    z = (pb - pa) / se
    return z, 2 * (1 - norm_cdf(abs(z)))


def noninferiority_z(xa: int, na: int, xb: int, nb: int, margin: float) -> tuple[float, float]:
    """H0: pB - pA <= -margin  vs  H1: pB - pA > -margin (one-sided). Small p => B is not worse than A by more than margin."""
    pa, pb = xa / na, xb / nb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    if se == 0:
        return (math.inf if pb - pa > -margin else -math.inf), 0.0 if pb - pa > -margin else 1.0
    z = (pb - pa + margin) / se
    return z, 1 - norm_cdf(z)


def welch_z(a: list[float], b: list[float]) -> tuple[float, float]:
    """Large-sample Welch comparison (normal approximation; valid for the sample sizes we require)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:  # no variance at all: any difference in means is exact, equal means are indistinguishable
        return (0.0, 1.0) if ma == mb else (math.copysign(math.inf, mb - ma), 0.0)
    z = (mb - ma) / se
    return z, 2 * (1 - norm_cdf(abs(z)))


def sample_size_per_arm(baseline_pct: float, mde_points: float, z_alpha: float = 1.96, z_power: float = 0.8416) -> int:
    p1 = baseline_pct / 100
    p2 = p1 + mde_points / 100
    if not 0 < p1 < 1 or not 0 < p2 < 1 or mde_points == 0:
        raise ValidationFailed("baseline and baseline+lift must be strictly between 0 and 100 percent, lift non-zero")
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


@dataclass
class Experiment:
    name: str
    variants: dict[str, dict[str, Any]]
    traffic_pct: int
    status: str
    primary_metric: str
    guardrail_metric: str | None
    guardrail_margin: float
    min_samples: int


class ExperimentService:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("experiments", SCHEMA)

    def create(
        self,
        name: str,
        variants: dict[str, dict[str, Any]],
        *,
        primary_metric: str,
        traffic_pct: int = 100,
        guardrail_metric: str | None = None,
        guardrail_margin: float = 0.02,
        min_samples: int = 200,
    ) -> Experiment:
        if len(variants) < 2:
            raise ValidationFailed("an experiment needs at least two variants")
        if not 1 <= traffic_pct <= 100:
            raise ValidationFailed("traffic_pct must be 1..100")
        try:
            self.db.execute(
                "INSERT INTO experiments(name, variants, traffic_pct, primary_metric, guardrail_metric, guardrail_margin,"
                " min_samples, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (name, dumps(variants), traffic_pct, primary_metric, guardrail_metric, guardrail_margin, min_samples, now_iso()),
            )
        except Exception as e:
            if "UNIQUE" in str(e):
                raise ConflictError(f"experiment {name!r} already exists") from e
            raise
        return self.get(name)

    def get(self, name: str) -> Experiment:
        r = self.db.one("SELECT * FROM experiments WHERE name=?", (name,))
        if r is None:
            raise NotFoundError(f"experiment {name!r} not found")
        return Experiment(
            r["name"],
            loads(r["variants"], {}),
            r["traffic_pct"],
            r["status"],
            r["primary_metric"],
            r["guardrail_metric"],
            r["guardrail_margin"],
            r["min_samples"],
        )

    def list_all(self) -> list[Experiment]:
        return [self.get(r["name"]) for r in self.db.query("SELECT name FROM experiments ORDER BY created_at")]

    def stop(self, name: str) -> None:
        self.get(name)
        self.db.execute("UPDATE experiments SET status='stopped' WHERE name=?", (name,))

    def assign(self, name: str, unit_id: str) -> tuple[str, dict[str, Any]] | None:
        """Variant name + its parameters for ``unit_id``, or None if the experiment is absent/stopped/out of traffic."""
        r = self.db.one("SELECT * FROM experiments WHERE name=? AND status='running'", (name,))
        if r is None:
            return None
        variants = loads(r["variants"], {})
        v = bucket(unit_id, name, sorted(variants), r["traffic_pct"])
        if v is None:
            return None
        self.db.execute(
            "INSERT OR IGNORE INTO assignments(experiment, unit_id, variant, ts) VALUES(?,?,?,?)", (name, unit_id, v, now_iso())
        )
        return v, variants[v]

    def record(self, name: str, unit_id: str, metrics: dict[str, float]) -> None:
        if self.db.one("SELECT 1 FROM assignments WHERE experiment=? AND unit_id=?", (name, unit_id)) is None:
            return  # unit was not in the experiment
        with self.db.tx():
            for metric, value in metrics.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO outcomes(experiment, unit_id, metric, value) VALUES(?,?,?,?)",
                    (name, unit_id, metric, float(value)),
                )

    def analysis(self, name: str) -> dict[str, Any]:
        exp = self.get(name)
        rows = self.db.query(
            "SELECT a.variant, o.metric, o.value FROM assignments a JOIN outcomes o ON o.experiment=a.experiment AND o.unit_id=a.unit_id "
            "WHERE a.experiment=?",
            (name,),
        )
        data: dict[str, dict[str, list[float]]] = {v: {} for v in exp.variants}
        for r in rows:
            data[r["variant"]].setdefault(r["metric"], []).append(r["value"])
        variants = sorted(exp.variants)
        out: dict[str, Any] = {"experiment": name, "primary_metric": exp.primary_metric, "variants": {}, "verdict": "insufficient_data"}
        for v in variants:
            vals = data[v].get(exp.primary_metric, [])
            out["variants"][v] = {
                "n": len(vals),
                "mean": (sum(vals) / len(vals)) if vals else None,
                "metrics": {m: {"n": len(x), "mean": sum(x) / len(x)} for m, x in data[v].items() if x},
            }
        if len(variants) == 2:
            a, b = variants
            va, vb = data[a].get(exp.primary_metric, []), data[b].get(exp.primary_metric, [])
            if min(len(va), len(vb)) >= exp.min_samples:
                z, p = welch_z(va, vb)
                out["primary"] = {"z": z, "p_value": p, "diff": sum(vb) / len(vb) - sum(va) / len(va)}
                out["verdict"] = "significant" if p < 0.05 else "no_significant_difference"
            if exp.guardrail_metric:
                ga, gb = data[a].get(exp.guardrail_metric, []), data[b].get(exp.guardrail_metric, [])
                if min(len(ga), len(gb)) >= exp.min_samples:
                    xa, xb = sum(1 for x in ga if x >= 0.5), sum(1 for x in gb if x >= 0.5)
                    z, p = noninferiority_z(xa, len(ga), xb, len(gb), exp.guardrail_margin)
                    out["guardrail"] = {
                        "metric": exp.guardrail_metric,
                        "margin": exp.guardrail_margin,
                        "z": z,
                        "p_value": p,
                        "non_inferior": p < 0.05,
                    }
                    if not out["guardrail"]["non_inferior"] and out["verdict"] == "significant":
                        out["verdict"] = "guardrail_failed"
        out["min_samples"] = exp.min_samples
        return out
