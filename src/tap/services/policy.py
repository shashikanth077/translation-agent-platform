"""Workflow policies and the supervisor/router.

The router is deliberately *rules first, model second*: every decision is a pure function of segment
features, policy, measured quality history and the A/B variant, and returns a reason code that is logged
with the trace. That makes routing explainable, testable and cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from ..domain.models import Audience, Domain, JobSpec, PolicyName, Risk
from ..util import utcnow

HumanMode = Literal["escalation", "high_risk", "all_changed", "all"]


@dataclass(frozen=True)
class Policy:
    name: PolicyName
    human: HumanMode  # when a human must sign off even if scores pass
    llm_first: bool  # skip NMT, translate with the LLM and full context
    fuzzy_draft: bool  # may use a fuzzy TM match as a draft
    accept: float | None  # override accept threshold (None = global default)
    cache_write: bool  # may approved output be written to TM/cache
    lqa_sample_rate: float  # advisory human sampling of auto-accepted/reused segments
    approval: Literal["auto", "manual"]
    priority: int  # review queue priority (higher first)

    def requires_human(self, risk: Risk, *, reused_exact: bool) -> bool:
        if reused_exact:
            return False
        return self.human == "all" or self.human == "all_changed" or (self.human == "high_risk" and risk is Risk.HIGH)


POLICIES: dict[PolicyName, Policy] = {
    PolicyName.FAST: Policy(PolicyName.FAST, "escalation", False, True, None, True, 0.02, "auto", 1),
    PolicyName.BALANCED: Policy(PolicyName.BALANCED, "high_risk", False, True, None, True, 0.05, "manual", 2),
    PolicyName.REGULATORY_STRICT: Policy(PolicyName.REGULATORY_STRICT, "all_changed", False, True, 0.90, True, 0.10, "manual", 3),
    PolicyName.SAFETY_URGENT: Policy(PolicyName.SAFETY_URGENT, "all", True, False, 0.90, False, 0.0, "manual", 9),
}


def derive_policy(spec: JobSpec, now: datetime | None = None, urgent_hours: float = 6.0) -> Policy:
    """Pick the processing policy from the job's risk profile unless the caller pinned one."""
    if spec.policy is not None:
        return POLICIES[spec.policy]
    now = now or utcnow()
    urgent = spec.deadline is not None and spec.deadline <= now + timedelta(hours=urgent_hours)
    if urgent and spec.domain in (Domain.CLINICAL, Domain.REGULATORY):
        return POLICIES[PolicyName.SAFETY_URGENT]
    if spec.domain is Domain.REGULATORY or spec.regulatory_use or spec.audience is Audience.PATIENTS:
        return POLICIES[PolicyName.REGULATORY_STRICT]
    if spec.domain in (Domain.UI, Domain.MARKETING):
        return POLICIES[PolicyName.FAST]
    return POLICIES[PolicyName.BALANCED]


# -- risk classification ----------------------------------------------------------------------------
_HIGH = re.compile(
    r"\b(do not|must not|never|contraindicat\w*|allerg\w*|adverse|serious|fatal|death|overdos\w*|pregnan\w*|"
    r"warning|IMP|titrat\w*|dose[- ]escalation|withdraw\w*|discontinu\w*|emergency|suicid\w*)\b",
    re.IGNORECASE,
)
_MEDIUM = re.compile(
    r"\b(dose|dosage|tablet|capsule|mg|ml|daily|injection|infusion|administ\w*|symptom\w*|side effect\w*)\b|\d", re.IGNORECASE
)


def classify_risk(source: str, domain: Domain) -> Risk:
    if domain in (Domain.UI, Domain.MARKETING):
        return Risk.LOW
    if _HIGH.search(source):
        return Risk.HIGH
    if _MEDIUM.search(source):
        return Risk.MEDIUM
    return Risk.LOW


# -- routing ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteFeatures:
    risk: Risk
    policy: Policy
    pair: str
    domain: str
    nmt_available: bool
    llm_available: bool
    nmt_mean_composite: float | None  # historical quality of the NMT path for this pair+domain
    nmt_samples: int
    accept_threshold: float
    variant: dict[str, str] | None = None  # A/B parameters, e.g. {"medium_risk_route": "llm"}


@dataclass(frozen=True)
class Route:
    engine: Literal["nmt", "llm", "human"]
    reason: str


MIN_HISTORY = 30


def route(f: RouteFeatures) -> Route:
    """Choose the cheapest adequate engine. Reason codes are stable identifiers used in traces and dashboards."""
    if not f.nmt_available and not f.llm_available:
        return Route("human", "no_engine_available")
    if f.policy.llm_first and f.llm_available:
        return Route("llm", "policy_llm_first")
    if not f.nmt_available:
        return Route("llm", "nmt_unavailable")
    if not f.llm_available:
        return Route("nmt", "llm_unavailable")
    # learn from history: if NMT has been consistently below the bar for this pair+domain, don't pay twice
    if f.nmt_samples >= MIN_HISTORY and f.nmt_mean_composite is not None and f.nmt_mean_composite < f.accept_threshold - 0.05:
        return Route("llm", "nmt_history_below_threshold")
    if f.variant:
        chosen = f.variant.get(f"{f.risk.value}_risk_route")
        if chosen in ("nmt", "llm"):
            return Route(chosen, f"ab_variant_{chosen}")  # type: ignore[arg-type]
    return Route("nmt", "default_nmt_first")
