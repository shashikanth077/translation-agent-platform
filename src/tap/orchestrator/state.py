"""Typed LangGraph state for one (segment, target language) workflow run.

Only JSON-serialisable values live here: the state is checkpointed after every node, so it must stay small
(no job-level data, no model objects) and replayable.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class SegState(TypedDict, total=False):
    # --- inputs (set once) ---
    job_id: str
    tenant_id: str
    seg_id: str
    source: str
    src_lang: str
    tgt_lang: str
    domain: str
    glossary_version: str
    policy: str
    risk: str
    words: int
    # --- working state ---
    terms: list[dict[str, Any]]
    tm_examples: list[dict[str, Any]]
    reuse: dict[str, Any] | None
    route: str  # nmt | llm | human
    route_reason: str
    next: str  # explicit successor chosen by a node (tool failures fall back / escalate)
    candidates: Annotated[list[dict[str, Any]], operator.add]  # reducer: append only
    steps: Annotated[list[str], operator.add]  # priced processing steps, e.g. ["nmt", "score", "llm", "score"]
    trace: Annotated[list[dict[str, Any]], operator.add]  # reason-coded decision log (no source text)
    checks: dict[str, bool]
    findings: list[str]
    scores: dict[str, float]
    raw_scores: dict[str, float]
    scored_n: int  # number of candidates that existed when scores were computed (freshness marker)
    nmt_composite: float | None
    attempts: int
    decision: dict[str, Any]
    force_human: bool
    human: dict[str, Any] | None
    # --- output ---
    final: dict[str, Any]
