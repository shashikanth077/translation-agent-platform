"""Tool handlers: what each registered MCP tool actually does. Bound to services at platform assembly.

Handlers trust the gateway to have validated schema, scope, tenant and approval, but still derive security-relevant
values (tenant, glossary content) from the *caller*, never from model-supplied arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..assets.reuse import ReuseService
from ..assets.terminology import TermChange, Terminology, parse_version
from ..assets.tm import TMMatch, TranslationMemory
from ..domain.models import JobStatus
from ..engines.base import RepairInfo, TranslationRequest
from ..engines.resilience import EngineRegistry
from ..errors import BudgetExceeded
from ..parse.tokens import count_words
from ..quality.base import ScoreRequest
from ..quality.service import QualityService
from ..services.jobs import JobStore
from ..services.metering import Metering
from ..services.review import ReviewService
from .gateway import MCPGateway
from .registry import Caller, ToolSpec

Publisher = Callable[[Caller, str], dict[str, Any]]


class ToolHandlers:
    def __init__(
        self,
        *,
        tm: TranslationMemory,
        terminology: Terminology,
        engines: EngineRegistry,
        quality: QualityService,
        reuse: ReuseService,
        review: ReviewService,
        jobs: JobStore,
        metering: Metering,
        publisher: Publisher,
    ) -> None:
        self.tm, self.terminology, self.engines, self.quality = tm, terminology, engines, quality
        self.reuse, self.review, self.jobs, self.metering, self.publisher = reuse, review, jobs, metering, publisher

    def bind(self, gw: MCPGateway) -> None:
        for name in (
            "tm_lookup",
            "glossary_get",
            "translate_nmt",
            "translate_llm",
            "score_quality",
            "submit_for_review",
            "update_glossary",
            "publish_translation",
        ):
            gw.register(name, getattr(self, name))
        gw.register_precondition("job_awaiting_approval", self._job_awaiting_approval)
        gw._budget_check = self.budget_check

    # -- gateway hooks --------------------------------------------------------------------------
    def budget_check(self, caller: Caller, spec: ToolSpec, args: dict[str, Any], ctx: dict[str, Any]) -> None:
        job_id = ctx.get("job_id")
        if not (spec.cost_kind and job_id):
            return
        words = max(1, count_words(args.get("text") or args.get("source") or ""))
        planned = self.metering.cost([spec.cost_kind], words)
        self.metering.check_job_budget(job_id, planned)
        if planned < 0:  # pragma: no cover - defensive
            raise BudgetExceeded("negative cost")

    def _job_awaiting_approval(self, caller: Caller, args: dict[str, Any]) -> str | None:
        try:
            job = self.jobs.get(args["job_id"], caller.tenant_id)
        except Exception:
            return "job not found for this tenant"
        if job["status"] != JobStatus.AWAITING_APPROVAL.value:
            return f"job is {job['status']}, not awaiting approval"
        if self.review.open_blocking(job["id"]):
            return "job still has open review items"
        bad = [o["lang"] for o in self.jobs.outputs(job["id"]) if not o["checks_ok"]]
        if bad:
            return f"final format checks failed for: {', '.join(bad)}"
        return None

    # -- read tools -----------------------------------------------------------------------------
    def tm_lookup(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        matches = self.tm.search(
            caller.tenant_id, a["src_lang"], a["tgt_lang"], a["source"], min_pct=a.get("min_match", 75), domain=a.get("domain"), limit=5
        )
        return [m.as_dict() for m in matches]

    def glossary_get(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        latest = self.terminology.latest_version(caller.tenant_id)
        version = parse_version(a.get("version"), latest)
        return [t.as_dict() for t in self.terminology.terms_at(caller.tenant_id, a["src_lang"], a["tgt_lang"], version)]

    # -- execution tools ------------------------------------------------------------------------
    def translate_nmt(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        res = self.engines.translate("nmt", TranslationRequest(a["text"], a["src_lang"], a["tgt_lang"], a.get("domain", "clinical")))
        return {"text": res.text, "engine": res.engine, "meta": res.meta}

    def translate_llm(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        version = parse_version(a["glossary_version"], self.terminology.latest_version(caller.tenant_id))
        glossary = self.terminology.terms_at(caller.tenant_id, a["src_lang"], a["tgt_lang"], version)  # server-side, tenant-scoped
        examples = [TMMatch(0, e["source"], e["target"], int(e.get("pct", 0)), "", "", "tm", False) for e in a.get("tm_examples", [])]
        repair = RepairInfo(a["repair"]["previous"], a["repair"]["findings"]) if a.get("repair") else None
        res = self.engines.translate(
            "llm",
            TranslationRequest(
                a["text"], a["src_lang"], a["tgt_lang"], a.get("domain", "clinical"), glossary, examples, a.get("context"), repair
            ),
        )
        return {"text": res.text, "engine": res.engine, "meta": res.meta}

    def score_quality(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        b = self.quality.score(
            ScoreRequest(a["source"], a["translation"], a.get("src_lang", "en"), a["tgt_lang"], a.get("domain", "clinical"))
        )
        return {"scores": b.scores, "raw": b.raw, "findings": b.findings, "versions": b.versions, "cached": b.cached}

    # -- write tools ----------------------------------------------------------------------------
    def submit_for_review(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.get(a["job_id"], caller.tenant_id)  # raises NotFound for other tenants' jobs
        interrupts: dict[str, dict[str, Any]] = ctx.get("interrupts", {})
        items = []
        for seg_id in a["segment_ids"]:
            payload = interrupts.get(seg_id)
            if payload is None:
                continue
            item = self.review.create(
                job["tenant_id"],
                job["id"],
                payload["lang"],
                seg_id,
                f"{job['id']}:{payload['lang']}:{seg_id}",
                payload,
                priority=ctx.get("priority", 1),
                sla_due=ctx.get("sla_due"),
            )
            items.append(item["id"])
        return {"items": items}

    def update_glossary(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        src, tgt = a.get("src_lang", "en"), a.get("tgt_lang", "de")
        version = self.terminology.commit(
            caller.tenant_id, [TermChange("add", src, tgt, a["term"], a["translation"])], note="update_glossary tool", actor=caller.id
        )
        impacted = self.terminology.impact(caller.tenant_id, self.tm, src, tgt, [a["term"]])
        return {"version": f"v{version}", "impacted_tm_entries": len(impacted)}

    def publish_translation(self, caller: Caller, a: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        return self.publisher(caller, a["job_id"])
