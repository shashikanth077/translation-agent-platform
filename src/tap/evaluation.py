"""Offline evaluation against a golden set: the CI gate that decides whether a prompt/model/threshold change ships.

Each example runs through the real segment workflow (cache lookup, routing, translation, checks, scoring,
consensus) in an isolated evaluation tenant, so a regression anywhere in that chain fails the build.
Human-review outcomes are counted as such; the review interrupt is never resumed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .domain.models import Domain
from .parse.tokens import numbers_and_units
from .platform import Platform
from .services.policy import classify_risk
from .util import new_id

EVAL_TENANT = "__eval__"


@dataclass
class Example:
    id: str
    source: str
    src_lang: str = "en"
    tgt_lang: str = "de"
    domain: str = "clinical"
    reference: str | None = None
    policy: str = "fast"


@dataclass
class EvalReport:
    prefix: str
    n: int
    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "n": self.n, "metrics": self.metrics, "per_example": self.per_example}


def load_dataset(path: str | Path) -> list[Example]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Example(**json.loads(line)))
    return out


def run_eval(platform: Platform, examples: list[Example], *, prefix: str = "run") -> EvalReport:
    run_id = new_id("eval_")
    rows: list[dict[str, Any]] = []
    for ex in examples:
        job_id = f"{run_id}-{ex.id}"
        out = platform.workflow.start(
            {
                "job_id": job_id,
                "tenant_id": EVAL_TENANT,
                "seg_id": ex.id,
                "source": ex.source,
                "src_lang": ex.src_lang,
                "tgt_lang": ex.tgt_lang,
                "domain": ex.domain,
                "glossary_version": "v0",
                "policy": ex.policy,
                "risk": classify_risk(ex.source, Domain(ex.domain)).value,
                "words": max(1, len(ex.source.split())),
            }
        )
        if out.final is not None:
            text, decision, path, composite = out.final["text"], out.final["decision"], out.final["path"], out.final.get("composite")
        else:  # the review interrupt is counted as a human outcome and never resumed in evaluation
            text, decision, path = (out.interrupt or {}).get("candidate", ""), "human", "human"
            composite = (out.state or {}).get("decision", {}).get("composite")
        ref_sim = fuzz.ratio(text, ex.reference) / 100 if ex.reference else None
        rows.append(
            {
                "id": ex.id,
                "decision": decision,
                "path": path,
                "composite": composite,
                "numbers_ok": numbers_and_units(ex.source)[0] == numbers_and_units(text)[0],
                "reference_similarity": ref_sim,
            }
        )
    n = len(rows) or 1
    comps = [r["composite"] for r in rows if r["composite"] is not None]
    refs = [r["reference_similarity"] for r in rows if r["reference_similarity"] is not None]
    metrics = {
        "numbers_ok_rate": sum(r["numbers_ok"] for r in rows) / n,
        "mean_composite": sum(comps) / len(comps) if comps else 0.0,
        "human_rate": sum(1 for r in rows if r["decision"] == "human") / n,
        "mean_reference_similarity": sum(refs) / len(refs) if refs else 0.0,
    }
    return EvalReport(prefix, len(rows), metrics, rows)


def compare(
    report: EvalReport, baseline: dict[str, Any], *, composite_tol: float = 0.02, human_tol: float = 0.10, reference_tol: float = 0.03
) -> list[str]:
    """Return regression messages (empty = pass). A single wrong number is always a failure, baseline or not."""
    fails = []
    m, b = report.metrics, baseline["metrics"]
    if m["numbers_ok_rate"] < 1.0:
        bad = [r["id"] for r in report.per_example if not r["numbers_ok"]]
        fails.append(f"numbers_ok_rate {m['numbers_ok_rate']:.3f} < 1.0 (failing: {', '.join(bad)})")
    if m["mean_composite"] < b["mean_composite"] - composite_tol:
        fails.append(f"mean_composite dropped {b['mean_composite']:.3f} -> {m['mean_composite']:.3f}")
    if m["human_rate"] > b["human_rate"] + human_tol:
        fails.append(f"human_rate rose {b['human_rate']:.3f} -> {m['human_rate']:.3f}")
    if b.get("mean_reference_similarity") and m["mean_reference_similarity"] < b["mean_reference_similarity"] - reference_tol:
        fails.append(f"mean_reference_similarity dropped {b['mean_reference_similarity']:.3f} -> {m['mean_reference_similarity']:.3f}")
    return fails
