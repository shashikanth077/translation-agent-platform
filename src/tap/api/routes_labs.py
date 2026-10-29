"""Assets, quality, MCP, experiments, metering and audit endpoints (these double as the guide's "labs")."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Response

from ..assets.reuse import ReuseContext
from ..assets.terminology import TermChange
from ..domain.models import Principal
from ..errors import ValidationFailed
from ..mcp.registry import Caller
from ..platform import Platform
from ..quality.base import ScoreRequest
from ..quality.consensus import ConsensusConfig, decide
from ..services.experiments import sample_size_per_arm
from .deps import get_platform, require
from .schemas import (
    ApprovalDecision,
    CacheLookupIn,
    ConsensusIn,
    ExperimentIn,
    GlossaryCommit,
    McpCall,
    SampleSizeIn,
    ScoreIn,
    TMEntryIn,
    TMXImport,
)

router = APIRouter()


# -- TM / glossary / cache -------------------------------------------------------------------------
@router.get("/tm/search", tags=["assets"])
def tm_search(
    source: str,
    src_lang: str = "en",
    tgt_lang: str = "de",
    min_match: int = Query(75, ge=50, le=100),
    domain: str | None = None,
    principal: Principal = Depends(require("assets:read")),
    p: Platform = Depends(get_platform),
) -> list[dict[str, Any]]:
    return [m.as_dict() for m in p.tm.search(principal.tenant_id, src_lang, tgt_lang, source, min_pct=min_match, domain=domain)]


@router.post("/tm/entries", status_code=201, tags=["assets"])
def tm_add(body: TMEntryIn, principal: Principal = Depends(require("assets:write")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    gv = f"v{p.terminology.latest_version(principal.tenant_id)}"
    entry_id = p.tm.add(
        principal.tenant_id,
        body.src_lang,
        body.tgt_lang,
        body.domain,
        body.source,
        body.target,
        glossary_version=gv,
        origin="human",
        approved_by=principal.subject,
    )
    p.audit.append(principal.subject, "tm.add", tenant_id=principal.tenant_id, subject=str(entry_id))
    return {"id": entry_id}


@router.post("/tm/import", tags=["assets"])
def tm_import(
    body: TMXImport, principal: Principal = Depends(require("assets:write")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    res = p.tm.import_tmx(
        principal.tenant_id,
        body.content,
        default_domain=body.domain,
        glossary_version=f"v{p.terminology.latest_version(principal.tenant_id)}",
        approved_by=principal.subject,
    )
    p.audit.append(principal.subject, "tm.import", tenant_id=principal.tenant_id, details=res)
    return res


@router.get("/tm/export", tags=["assets"])
def tm_export(
    src_lang: str = "en", tgt_lang: str = "de", principal: Principal = Depends(require("assets:read")), p: Platform = Depends(get_platform)
) -> Response:
    return Response(p.tm.export_tmx(principal.tenant_id, src_lang, tgt_lang), media_type="application/xml")


@router.get("/glossary", tags=["assets"])
def glossary(
    src_lang: str = "en",
    tgt_lang: str = "de",
    version: int | None = None,
    principal: Principal = Depends(require("assets:read")),
    p: Platform = Depends(get_platform),
) -> dict[str, Any]:
    latest = p.terminology.latest_version(principal.tenant_id)
    terms = p.terminology.terms_at(principal.tenant_id, src_lang, tgt_lang, version)
    return {"version": f"v{version if version is not None else latest}", "latest": f"v{latest}", "terms": [t.as_dict() for t in terms]}


@router.post("/glossary/commit", status_code=201, tags=["assets"])
def glossary_commit(
    body: GlossaryCommit, principal: Principal = Depends(require("assets:write")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    """Publish a new glossary version (admin API). Agents change terms only via the approval-gated ``update_glossary`` tool."""
    version = p.terminology.commit(
        principal.tenant_id,
        [TermChange(c.op, c.src_lang, c.tgt_lang, c.term, c.translation, c.kind) for c in body.changes],
        note=body.note,
        actor=principal.subject,
    )
    impacted: dict[str, Any] = {}
    for c in body.changes:
        impacted.setdefault(c.tgt_lang, [])
        impacted[c.tgt_lang] += p.terminology.impact(principal.tenant_id, p.tm, c.src_lang, c.tgt_lang, [c.term])
    p.audit.append(
        principal.subject, "glossary.commit", tenant_id=principal.tenant_id, subject=f"v{version}", details={"changes": len(body.changes)}
    )
    return {
        "version": f"v{version}",
        "note": "cache keys include the glossary version: entries cached under older versions are no longer reachable",
        "impacted_tm_entries": {k: len(v) for k, v in impacted.items()},
    }


@router.get("/glossary/suggestions", tags=["assets"])
def glossary_suggestions(
    min_occurrences: int = 2, principal: Principal = Depends(require("assets:write")), p: Platform = Depends(get_platform)
) -> list[dict[str, Any]]:
    return p.feedback.suggestions(principal.tenant_id, min_occurrences)


@router.post("/cache/lookup", tags=["assets"])
def cache_lookup(
    body: CacheLookupIn, principal: Principal = Depends(require("assets:read")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    gv = body.glossary_version or f"v{p.terminology.latest_version(principal.tenant_id)}"
    ctx = ReuseContext(principal.tenant_id, body.src_lang, body.tgt_lang, body.domain, gv)
    hit = p.reuse.lookup(body.source, ctx)
    action = {
        "exact": "reuse, no model call (after sanity checks)",
        "fuzzy": "use as draft if >=90% and checks pass; otherwise as few-shot context" if hit.pct >= 90 else "use as context for NMT/LLM",
        "semantic": "suggest as context only; never auto-reuse",
        "none": "fresh translation via the router",
    }[hit.type]
    return {**hit.as_dict(), "key": hit.key, "glossary_version": gv, "recommended_action": action}


# -- quality ---------------------------------------------------------------------------------------
@router.post("/quality/score", tags=["quality"])
def quality_score(
    body: ScoreIn, principal: Principal = Depends(require("assets:read")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    from ..quality.checks import run_checks

    b = p.quality.score(ScoreRequest(body.source, body.target, body.src_lang, body.tgt_lang, body.domain))
    terms = p.terminology.terms_at(principal.tenant_id, body.src_lang, body.tgt_lang)
    rep = run_checks(body.source, body.target, src_lang=body.src_lang, tgt_lang=body.tgt_lang, terms=terms, terminology=p.terminology)
    d = decide(b.scores, rep.passed, p.consensus, can_repair=True)
    return {
        "scores": b.scores,
        "raw": b.raw,
        "findings": b.findings,
        "versions": b.versions,
        "checks": rep.passed,
        "check_findings": rep.findings,
        "decision": d.model_dump(),
    }


@router.post("/quality/consensus", tags=["quality"])
def quality_consensus(
    body: ConsensusIn, _: Principal = Depends(require("assets:read")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    cfg = ConsensusConfig(
        body.weights or p.consensus.weights,
        body.accept if body.accept is not None else p.consensus.accept,
        body.critical_semantic if body.critical_semantic is not None else p.consensus.critical_semantic,
        body.max_disagreement if body.max_disagreement is not None else p.consensus.max_disagreement,
    )
    if abs(sum(cfg.weights.values()) - 1) > 1e-6:
        raise ValidationFailed("weights must sum to 1.0")
    if body.scores is not None and set(body.scores) != set(cfg.weights):
        raise ValidationFailed(f"scores must have exactly the keys {sorted(cfg.weights)}")
    return decide(body.scores, body.checks, cfg, force_human=body.force_human, can_repair=body.can_repair).model_dump()


# -- MCP gateway -----------------------------------------------------------------------------------
def _caller(body: McpCall, principal: Principal) -> Caller:
    return (
        Caller.agent(body.agent, principal.tenant_id)
        if body.caller == "agent"
        else Caller.user(principal.subject, principal.tenant_id, principal.roles)
    )


@router.get("/mcp/tools", tags=["mcp"])
def mcp_tools(principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)) -> list[dict[str, Any]]:
    return p.gateway.describe()


@router.post("/mcp/check", tags=["mcp"])
def mcp_check(body: McpCall, principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    """Dry-run the gateway pipeline: shows every check, executes nothing, writes no audit record."""
    return p.gateway.call(_caller(body, principal), body.tool, body.arguments, dry_run=True).as_dict()


@router.post("/mcp/call", tags=["mcp"])
def mcp_call(
    body: McpCall, principal: Principal = Depends(require("admin", "review")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    return p.gateway.call(_caller(body, principal), body.tool, body.arguments).as_dict()


@router.get("/mcp/approvals", tags=["mcp"])
def mcp_approvals(
    status: str | None = None, principal: Principal = Depends(require("approvals")), p: Platform = Depends(get_platform)
) -> list[dict[str, Any]]:
    return p.approvals.list_all(principal.tenant_id, status)


@router.post("/mcp/approvals/{approval_id}/decide", tags=["mcp"])
def mcp_decide(
    approval_id: str, body: ApprovalDecision, principal: Principal = Depends(require("approvals")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    res = p.approvals.decide(
        principal.tenant_id, approval_id, approve=body.approve, decided_by=f"user:{principal.subject}", reason=body.reason
    )
    p.audit.append(
        principal.subject, "approval.decided", tenant_id=principal.tenant_id, subject=approval_id, details={"approve": body.approve}
    )
    return res


# -- experiments -----------------------------------------------------------------------------------
@router.get("/experiments", tags=["experiments"])
def experiments(_: Principal = Depends(require("experiments")), p: Platform = Depends(get_platform)) -> list[dict[str, Any]]:
    return [e.__dict__ for e in p.experiments.list_all()]


@router.post("/experiments", status_code=201, tags=["experiments"])
def experiment_create(
    body: ExperimentIn, principal: Principal = Depends(require("experiments")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    exp = p.experiments.create(
        body.name,
        body.variants,
        primary_metric=body.primary_metric,
        traffic_pct=body.traffic_pct,
        guardrail_metric=body.guardrail_metric,
        guardrail_margin=body.guardrail_margin,
        min_samples=body.min_samples,
    )
    p.audit.append(principal.subject, "experiment.create", tenant_id=principal.tenant_id, subject=exp.name)
    return exp.__dict__


@router.get("/experiments/{name}/analysis", tags=["experiments"])
def experiment_analysis(name: str, _: Principal = Depends(require("experiments")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.experiments.analysis(name)


@router.post("/experiments/{name}/stop", tags=["experiments"])
def experiment_stop(
    name: str, principal: Principal = Depends(require("experiments")), p: Platform = Depends(get_platform)
) -> dict[str, str]:
    p.experiments.stop(name)
    p.audit.append(principal.subject, "experiment.stop", tenant_id=principal.tenant_id, subject=name)
    return {"status": "stopped"}


@router.post("/experiments/sample-size", tags=["experiments"])
def sample_size(body: SampleSizeIn, _: Principal = Depends(require("experiments"))) -> dict[str, Any]:
    z_alpha = {0.05: 1.96, 0.01: 2.576}.get(body.alpha)
    z_power = {0.8: 0.8416, 0.9: 1.2816}.get(body.power)
    if z_alpha is None or z_power is None:
        raise ValidationFailed("alpha must be 0.05 or 0.01 and power 0.8 or 0.9")
    n = sample_size_per_arm(body.baseline_pct, body.mde_points, z_alpha, z_power)
    return {"per_arm": n, "total": 2 * n, "unit": "jobs/documents (randomise by document, not by segment)"}


# -- metering / audit / feedback ------------------------------------------------------------------
@router.get("/metering/report", tags=["metering"])
def metering_report(
    since: str | None = None,
    until: str | None = None,
    principal: Principal = Depends(require("metering:read")),
    p: Platform = Depends(get_platform),
) -> dict[str, Any]:
    return p.metering.report(principal.tenant_id, since=since, until=until)


@router.get("/feedback/stats", tags=["metering"])
def feedback_stats(principal: Principal = Depends(require("metering:read")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.feedback.stats(principal.tenant_id)


@router.get("/audit", tags=["audit"])
def audit_search(
    subject: str | None = None,
    action: str | None = None,
    limit: int = Query(100, le=500),
    principal: Principal = Depends(require("audit:read")),
    p: Platform = Depends(get_platform),
) -> list[dict[str, Any]]:
    return p.audit.search(tenant_id=principal.tenant_id, subject=subject, action=action, limit=limit)


@router.get("/audit/verify", tags=["audit"])
def audit_verify(_: Principal = Depends(require("audit:read")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.audit.verify()
