"""Job-level orchestration: intake -> analysis -> per-segment LangGraph runs -> review -> assembly -> approval -> delivery.

The job runner is deliberately a plain, restartable state machine over the database (not a second graph): all
durable state is in ``jobs``/``job_segments``/checkpoints, every step is idempotent, and ``recover()`` simply
re-enqueues unfinished jobs after a crash. Per-segment reasoning (the part that benefits from a graph) lives in
``SegmentWorkflow`` and is checkpointed per ``(job, language, segment)`` thread.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..assets.reuse import ReuseContext, ReuseService
from ..assets.terminology import Terminology
from ..config import Settings
from ..domain.models import DocFormat, JobSpec, JobStatus, PolicyName, Principal
from ..errors import Forbidden, NotFoundError, ValidationFailed
from ..events import EventBus
from ..mcp.approvals import Approvals
from ..mcp.gateway import MCPGateway
from ..mcp.registry import Caller
from ..observability import Metrics, bind, unbind
from ..parse import formats
from ..parse.tokens import count_words
from ..quality.checks import run_checks
from ..services.audit import AuditLog
from ..services.delivery import DeliveryService, format_check
from ..services.identity import IdentityService
from ..services.jobs import JobStore
from ..services.metering import Metering
from ..services.policy import POLICIES, Policy, classify_risk, derive_policy
from ..services.review import ReviewService
from ..util import normalise_text, now_iso
from .tools import ToolClient
from .workflow import SegmentOutcome, SegmentWorkflow

log = logging.getLogger("tap.runner")


class JobRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        jobs: JobStore,
        workflow: SegmentWorkflow,
        reuse: ReuseService,
        terminology: Terminology,
        review: ReviewService,
        metering: Metering,
        delivery: DeliveryService,
        approvals: Approvals,
        gateway: MCPGateway,
        audit: AuditLog,
        bus: EventBus,
        identity: IdentityService,
        tools: ToolClient,
        metrics: Metrics,
    ) -> None:
        self.s, self.jobs, self.wf, self.reuse, self.terminology = settings, jobs, workflow, reuse, terminology
        self.review, self.metering, self.delivery, self.approvals = review, metering, delivery, approvals
        self.gateway, self.audit, self.bus, self.identity, self.tools, self.metrics = gateway, audit, bus, identity, tools, metrics
        self._job_pool = ThreadPoolExecutor(max_workers=settings.worker_threads, thread_name_prefix="tap-job")
        self._seg_pool = ThreadPoolExecutor(max_workers=settings.segment_threads, thread_name_prefix="tap-seg")
        self._guard = threading.Lock()
        self._running: set[str] = set()
        self._rerun: set[str] = set()
        self._idle = threading.Condition(self._guard)

    # ------------------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------
    def submit(self, principal: Principal, spec: JobSpec, *, idem_key: str) -> dict[str, Any]:
        if spec.tenant_id != principal.tenant_id:
            raise Forbidden("job tenant does not match the caller's tenant")
        self.identity.check_entitlements(spec)
        doc = self.jobs.get_document(spec.tenant_id, spec.document_id)  # NotFound for foreign/unknown documents
        parsed = formats.parse(DocFormat(doc["format"]), doc["content"], spec.source_lang)  # fail fast on bad files
        glossary_version = spec.glossary_version or f"v{self.terminology.latest_version(spec.tenant_id)}"
        policy = derive_policy(spec, urgent_hours=self.s.urgent_hours)
        job, created = self.jobs.create_job(spec, policy.name.value, idem_key, principal.subject, glossary_version)
        if created:
            self.audit.append(
                principal.subject,
                "job.created",
                tenant_id=spec.tenant_id,
                subject=job["id"],
                details={
                    "policy": policy.name.value,
                    "langs": spec.target_langs,
                    "segments": len(parsed.segments),
                    "glossary_version": glossary_version,
                },
            )
            self.bus.publish("job.created", {"job_id": job["id"], "policy": policy.name.value}, spec.tenant_id)
            self.enqueue(job["id"])
        return job

    # ------------------------------------------------------------------------------------------
    # scheduling (one runner per job at a time; extra requests coalesce into one re-run)
    # ------------------------------------------------------------------------------------------
    def enqueue(self, job_id: str) -> None:
        with self._guard:
            if job_id in self._running:
                self._rerun.add(job_id)
                return
            self._running.add(job_id)
        self._job_pool.submit(self._loop, job_id)

    def _loop(self, job_id: str) -> None:
        token = bind(job_id=job_id)
        try:
            while True:
                try:
                    self._advance(job_id)
                except Exception as exc:
                    log.exception("job %s failed", job_id)
                    self._fail(job_id, exc)
                with self._guard:
                    if job_id in self._rerun:
                        self._rerun.discard(job_id)
                        continue
                    self._running.discard(job_id)
                    self._idle.notify_all()
                    return
        finally:
            unbind(token)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no job is running (tests, graceful shutdown)."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        self.bus.flush()
        return True

    def recover(self) -> int:
        """After a restart: repair review items lost in a crash window, then re-enqueue unfinished jobs."""
        ids = self.jobs.non_terminal_job_ids()
        for job_id in ids:
            self._reconcile_reviews(self.jobs.get(job_id))
            self.enqueue(job_id)
        return len(ids)

    def _reconcile_reviews(self, job: dict[str, Any]) -> None:
        spec, policy, _doc = self._context(job)
        open_keys = {
            (i["lang"], i["seg_id"])
            for i in self.review.list_items(job["tenant_id"], status=None, kind="review", job_id=job["id"])
            if i["status"] in ("pending", "claimed")
        }
        for row in self.jobs.segments(job["id"], status="in_review"):
            if (row["lang"], row["seg_id"]) in open_keys:
                continue
            info = self.wf.inspect(job["id"], row["lang"], row["seg_id"])
            if info["status"] == "interrupted":
                self._record(job, spec, policy, row["lang"], row, SegmentOutcome("interrupted", interrupt=info["interrupt"]))
            elif info["status"] == "finished":
                self._record(job, spec, policy, row["lang"], row, SegmentOutcome("done", final=info["final"]))
            else:
                self.jobs.set_segment_status(job["id"], row["lang"], row["seg_id"], "pending")

    def shutdown(self) -> None:
        self._job_pool.shutdown(wait=True, cancel_futures=False)
        self._seg_pool.shutdown(wait=True, cancel_futures=False)

    def _fail(self, job_id: str, exc: Exception) -> None:
        try:
            job = self.jobs.get(job_id)
            self.jobs.set_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:500])
            self.audit.append("system", "job.failed", tenant_id=job["tenant_id"], subject=job_id, details={"error": type(exc).__name__})
            self.bus.publish("job.failed", {"job_id": job_id, "error": type(exc).__name__}, job["tenant_id"])
        except Exception:  # pragma: no cover - last resort: never let the failure handler kill the worker
            log.exception("could not record failure of job %s", job_id)

    # ------------------------------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------------------------------
    def _advance(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        status = JobStatus(job["status"])
        if status == JobStatus.QUEUED:
            self.jobs.set_status(job_id, JobStatus.ANALYZING)
            status = JobStatus.ANALYZING
        if status == JobStatus.ANALYZING:
            self._analyze(job)
            self.jobs.set_status(job_id, JobStatus.TRANSLATING)
            status = JobStatus.TRANSLATING
        if status == JobStatus.AWAITING_REVIEW:
            self.jobs.set_status(job_id, JobStatus.TRANSLATING)
            status = JobStatus.TRANSLATING
        if status == JobStatus.TRANSLATING:
            self._translate(self.jobs.get(job_id))
            self._after_translation(job_id)

    def _context(self, job: dict[str, Any]) -> tuple[JobSpec, Policy, dict[str, Any]]:
        spec = JobSpec.model_validate(job["spec"])
        return spec, POLICIES[PolicyName(job["policy"])], self.jobs.get_document(job["tenant_id"], job["document_id"])

    # -- analysis ----------------------------------------------------------------------------------
    def _analyze(self, job: dict[str, Any]) -> None:
        spec, _policy, doc = self._context(job)
        parsed = formats.parse(DocFormat(doc["format"]), doc["content"], spec.source_lang)
        seen: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        unique: list[Any] = []
        for idx, seg in enumerate(parsed.segments):
            norm = normalise_text(seg.source)
            repeat_of = seen.get(norm)
            if repeat_of is None:
                seen[norm] = seg.id
                unique.append(seg)
            for lang in spec.target_langs:
                rows.append(
                    {
                        "lang": lang,
                        "seg_id": seg.id,
                        "idx": idx,
                        "source": seg.source,
                        "words": count_words(seg.source, spec.source_lang),
                        "risk": classify_risk(seg.source, spec.domain).value,
                        "repeat_of": repeat_of,
                    }
                )
        self.jobs.add_segments(job["id"], rows)
        self.jobs.set_analysis(job["id"], self._estimate(job, spec, unique, len(parsed.segments)))
        self.bus.publish("job.analyzed", {"job_id": job["id"]}, job["tenant_id"])

    def _estimate(self, job: dict[str, Any], spec: JobSpec, unique: list[Any], total_segments: int) -> dict[str, Any]:
        """Pre-analysis quote: how much can be reused, and what it should cost versus the legacy baseline."""
        per_lang: dict[str, Any] = {}
        est_cost = base_cost = 0.0
        total_words = 0
        for lang in spec.target_langs:
            counts: dict[str, int] = defaultdict(int)
            for seg in unique:
                words = count_words(seg.source, spec.source_lang)
                hit = self.reuse.lookup(
                    seg.source, ReuseContext(spec.tenant_id, spec.source_lang, lang, spec.domain.value, job["glossary_version"])
                )
                if hit.type == "exact":
                    kind, steps = "exact", ["reuse_exact"]
                elif hit.type == "fuzzy" and hit.pct >= self.s.tm_draft_min_pct:
                    kind, steps = "fuzzy_draft", ["reuse_fuzzy", "score"]
                elif hit.type == "fuzzy":
                    kind, steps = "fuzzy_context", ["nmt", "score"]
                else:
                    kind, steps = "new", ["nmt", "score"]
                counts[kind] += 1
                est_cost += self.metering.cost(steps, words)
                total_words += words
            base_cost += self.metering.baseline(sum(count_words(seg.source, spec.source_lang) for seg in unique))
            per_lang[lang] = dict(counts)
        return {
            "segments": total_segments,
            "unique_segments": len(unique),
            "repeated_segments": total_segments - len(unique),
            "words_unique": total_words // max(1, len(spec.target_langs)),
            "reuse_by_language": per_lang,
            "estimated_cost_units": round(est_cost, 4),
            "baseline_cost_units": round(base_cost, 4),
            "estimated_savings_pct": round(100 * (1 - est_cost / base_cost), 1) if base_cost else None,
            "note": "estimate excludes repair and human review; actual cost is reported by /metering",
        }

    # -- translation -------------------------------------------------------------------------------
    def _translate(self, job: dict[str, Any]) -> None:
        spec, policy, _doc = self._context(job)
        todo: list[tuple[str, dict[str, Any]]] = []
        for lang in spec.target_langs:
            for row in self.jobs.segments(job["id"], lang):
                # in_review segments belong to the review flow: re-processing them here races with a reviewer
                if row["repeat_of"] or row["status"] in ("done", "failed", "in_review"):
                    continue
                todo.append((lang, row))
        futures = [self._seg_pool.submit(self._process_segment, job, spec, policy, lang, row) for lang, row in todo]
        errors = []
        for f in futures:
            try:
                f.result()
            except Exception as exc:
                errors.append(exc)
                log.exception("segment failed")
        self._propagate_repeats(job["id"])
        if errors:
            self.jobs.set_status(job["id"], JobStatus.NEEDS_ATTENTION, error=f"{len(errors)} segment(s) failed: {type(errors[0]).__name__}")
            self.bus.publish("job.needs_attention", {"job_id": job["id"], "failed_segments": len(errors)}, job["tenant_id"])

    def _initial_state(self, job: dict[str, Any], spec: JobSpec, policy: Policy, lang: str, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["id"],
            "tenant_id": job["tenant_id"],
            "seg_id": row["seg_id"],
            "source": row["source"],
            "src_lang": spec.source_lang,
            "tgt_lang": lang,
            "domain": spec.domain.value,
            "glossary_version": job["glossary_version"],
            "policy": policy.name.value,
            "risk": row["risk"],
            "words": row["words"],
        }

    def _process_segment(self, job: dict[str, Any], spec: JobSpec, policy: Policy, lang: str, row: dict[str, Any]) -> None:
        started = time.monotonic()
        info = self.wf.inspect(job["id"], lang, row["seg_id"])
        if info["status"] == "finished":
            outcome = SegmentOutcome("done", final=info["final"])
        elif info["status"] == "interrupted":
            outcome = SegmentOutcome("interrupted", interrupt=info["interrupt"])
        elif info["status"] == "crashed":
            outcome = self.wf.continue_run(job["id"], lang, row["seg_id"])
        else:
            outcome = self.wf.start(self._initial_state(job, spec, policy, lang, row))  # type: ignore[arg-type]
        self.metrics.segment_seconds.observe(time.monotonic() - started)
        self._record(job, spec, policy, lang, row, outcome)

    def _record(self, job: dict[str, Any], spec: JobSpec, policy: Policy, lang: str, row: dict[str, Any], outcome: SegmentOutcome) -> None:
        job_id, seg_id = job["id"], row["seg_id"]
        if outcome.status == "interrupted":
            payload = outcome.interrupt or {}
            self.jobs.set_segment_status(job_id, lang, seg_id, "in_review")
            self.tools.call(
                "submit_for_review",
                {"job_id": job_id, "segment_ids": [seg_id], "lang": lang},
                tenant_id=job["tenant_id"],
                context={"interrupts": {seg_id: payload}, "priority": policy.priority, "job_id": job_id},
            )
            self.bus.publish(
                "review.requested",
                {"job_id": job_id, "lang": lang, "seg_id": seg_id, "reason": payload.get("reason_code"), "priority": policy.priority},
                job["tenant_id"],
            )
            return
        final = outcome.final or {}
        self.jobs.complete_segment(job_id, lang, seg_id, final)
        detail = final.get("detail", {})
        self.metrics.segments.labels(final["path"], final["decision"]).inc()
        self.bus.publish(
            "segment.completed",
            {
                "job_id": job_id,
                "lang": lang,
                "seg_id": seg_id,
                "domain": spec.domain.value,
                "path": final["path"],
                "decision": final["decision"],
                "words": row["words"],
                "cost_units": final["cost_units"],
                "composite": final.get("composite"),
                "meta": {"pair": f"{spec.source_lang}>{lang}", "nmt_composite": detail.get("nmt_composite"), "engine": final.get("engine")},
            },
            job["tenant_id"],
        )

    def _propagate_repeats(self, job_id: str) -> None:
        """Repetitions inside one job are translated once; identical segments copy the result at zero cost."""
        done = {(r["lang"], r["seg_id"]): r for r in self.jobs.segments(job_id, status="done")}
        for row in self.jobs.segments(job_id):
            origin = row["repeat_of"]
            if not origin or row["status"] == "done":
                continue
            src = done.get((row["lang"], origin))
            if src is None:
                continue
            self.jobs.complete_segment(
                job_id,
                row["lang"],
                row["seg_id"],
                {
                    "text": src["target"],
                    "path": "reuse_repeat",
                    "decision": "accept",
                    "composite": src["composite"],
                    "reason": "repetition",
                    "cost_units": 0.0,
                    "cacheable": False,
                    "approved_by": src["approved_by"],
                    "detail": {"repeat_of": origin},
                },
            )

    def _after_translation(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if JobStatus(job["status"]) != JobStatus.TRANSLATING:
            return  # already moved (needs_attention / failed)
        if self.review.open_blocking(job_id) or self.jobs.progress(job_id).get("in_review"):
            self.jobs.set_status(job_id, JobStatus.AWAITING_REVIEW)
            return
        self._finalize(job)

    # -- human review ------------------------------------------------------------------------------
    def submit_review(
        self, principal: Principal, item_id: str, *, text: str, note: str | None = None, override: bool = False
    ) -> dict[str, Any]:
        item = self.review.get(item_id, principal.tenant_id)
        if item["kind"] == "lqa":
            res = self.review.resolve(
                item_id, principal.tenant_id, principal.subject, {"verdict": "issue" if note else "ok", "note": note, "text": text}
            )
            self.audit.append(
                principal.subject,
                "lqa.resolved",
                tenant_id=principal.tenant_id,
                subject=item["job_id"],
                details={"seg_id": item["seg_id"], "lang": item["lang"]},
            )
            return res
        job = self.jobs.get(item["job_id"], principal.tenant_id)
        spec = JobSpec.model_validate(job["spec"])
        payload = item["payload"]
        # reviewer edits get the same deterministic checks as machine output: an edit can introduce a wrong number too
        terms = self.terminology.terms_at(job["tenant_id"], spec.source_lang, item["lang"], int(job["glossary_version"][1:]))
        rep = run_checks(
            payload["source"], text, src_lang=spec.source_lang, tgt_lang=item["lang"], terms=terms, terminology=self.terminology
        )
        if not rep.ok and not override:
            raise ValidationFailed(
                "the edited translation fails deterministic checks; fix it or resubmit with override and a note",
                details={"findings": rep.findings},
            )
        if override and not (note and note.strip()):
            raise ValidationFailed("an override requires a note explaining why")
        if override and not principal.has_role("admin", "linguist"):
            raise Forbidden("only linguists and admins can override failed checks")
        resolved = self.review.resolve(
            item_id,
            principal.tenant_id,
            principal.subject,
            {"text": text, "note": note, "override": override and not rep.ok, "findings": rep.findings},
        )
        try:
            self._resume_segment(job, item, principal.subject, text, note)
        except Exception:
            self.review.reopen(item_id)  # the reviewer's work must not be lost nor the item stranded
            raise
        self.audit.append(
            principal.subject,
            "review.resolved",
            tenant_id=principal.tenant_id,
            subject=job["id"],
            details={
                "seg_id": item["seg_id"],
                "lang": item["lang"],
                "override": override and not rep.ok,
                "changed": text != payload.get("candidate"),
            },
        )
        return resolved

    def _resume_segment(self, job: dict[str, Any], item: dict[str, Any], reviewer: str, text: str, note: str | None) -> None:
        spec, policy, _doc = self._context(job)
        lang, seg_id = item["lang"], item["seg_id"]
        outcome = self.wf.resume(job["id"], lang, seg_id, {"text": text, "reviewer": reviewer, "note": note})
        row = self.jobs.get_segment(job["id"], lang, seg_id)
        self._record(job, spec, policy, lang, row, outcome)
        payload = item["payload"]
        self.bus.publish(
            "review.resolved",
            {
                "kind": "review",
                "job_id": job["id"],
                "lang": lang,
                "seg_id": seg_id,
                "domain": spec.domain.value,
                "source": payload["source"],
                "machine": payload.get("candidate") or "",
                "final": text,
                "engine": payload.get("engine"),
            },
            job["tenant_id"],
        )
        self._propagate_repeats(job["id"])
        self.enqueue(job["id"])

    # -- finalisation ------------------------------------------------------------------------------
    def _finalize(self, job: dict[str, Any]) -> None:
        spec, policy, doc = self._context(job)
        job_id = job["id"]
        fmt = DocFormat(doc["format"])
        parsed = formats.parse(fmt, doc["content"], spec.source_lang)
        problems = False
        for lang in spec.target_langs:
            segs = self.jobs.segments(job_id, lang)
            unfinished = [s["seg_id"] for s in segs if s["status"] != "done"]
            if unfinished:
                self.jobs.set_status(job_id, JobStatus.NEEDS_ATTENTION, error=f"{len(unfinished)} segment(s) not finished for {lang}")
                self.bus.publish("job.needs_attention", {"job_id": job_id, "lang": lang}, job["tenant_id"])
                return
            try:
                content = formats.assemble(fmt, parsed.skeleton, {s["seg_id"]: s["target"] for s in segs}, target_lang=lang)
                issues = format_check(fmt, doc["content"], content, spec.source_lang)
            except ValidationFailed as e:
                content, issues = "", [e.message]
            self.jobs.save_output(job_id, lang, content, not issues, issues)
            problems = problems or bool(issues)
            self._sample_for_lqa(job, policy, lang, segs)
        if problems:
            self.jobs.set_status(job_id, JobStatus.NEEDS_ATTENTION, error="final format checks failed")
            self.bus.publish("job.needs_attention", {"job_id": job_id, "reason": "format"}, job["tenant_id"])
            return
        self.jobs.set_status(job_id, JobStatus.AWAITING_APPROVAL)
        self.audit.append(
            "system", "job.awaiting_approval", tenant_id=job["tenant_id"], subject=job_id, details={"policy": policy.name.value}
        )
        self.bus.publish("job.awaiting_approval", {"job_id": job_id, "policy": policy.name.value}, job["tenant_id"])
        if policy.approval == "auto":
            self._auto_approve(job, policy)

    def _sample_for_lqa(self, job: dict[str, Any], policy: Policy, lang: str, segs: list[dict[str, Any]]) -> None:
        """Advisory human sampling of segments that no human has seen. Deterministic, so reruns pick the same ones."""
        if policy.lqa_sample_rate <= 0:
            return
        threshold = int(policy.lqa_sample_rate * 2**32)
        for s in segs:
            if "human" in (s["path"] or "") or s["path"] == "reuse_repeat":
                continue
            h = int.from_bytes(hashlib.sha256(f"{job['id']}:{lang}:{s['seg_id']}".encode()).digest()[:4], "big")
            if h < threshold:
                self.review.create(
                    job["tenant_id"],
                    job["id"],
                    lang,
                    s["seg_id"],
                    f"{job['id']}:{lang}:{s['seg_id']}",
                    {
                        "job_id": job["id"],
                        "lang": lang,
                        "seg_id": s["seg_id"],
                        "source": s["source"],
                        "candidate": s["target"],
                        "engine": s["path"],
                        "scores": {},
                        "why_flagged": "advisory LQA sample",
                    },
                    kind="lqa",
                    priority=0,
                )

    # -- approval & delivery -----------------------------------------------------------------------
    def approve(self, principal: Principal, job_id: str, *, reason: str = "") -> dict[str, Any]:
        job = self.jobs.get(job_id, principal.tenant_id)
        caller = Caller.user(principal.subject, principal.tenant_id, principal.roles)
        self.approvals.grant(principal.tenant_id, "publish_translation", {"job_id": job_id}, caller.id, reason or "approved via API")
        res = self.gateway.call(caller, "publish_translation", {"job_id": job_id})
        if res.status != "allowed":
            raise Forbidden(f"publish refused: {res.reason}", details={"checks": [c.__dict__ for c in res.checks]})
        return self.jobs.get(job["id"], principal.tenant_id)

    def _auto_approve(self, job: dict[str, Any], policy: Policy) -> None:
        args = {"job_id": job["id"]}
        self.approvals.grant(
            job["tenant_id"], "publish_translation", args, "policy:auto-approve", f"policy {policy.name.value} allows auto approval"
        )
        res = self.gateway.call(Caller.agent("delivery", job["tenant_id"]), "publish_translation", args)
        if res.status != "allowed":
            log.warning("auto-approval of %s refused: %s", job["id"], res.reason)

    def publish(self, caller: Caller, job_id: str) -> dict[str, Any]:
        """Handler of the ``publish_translation`` tool (runs only after the gateway approved the call)."""
        job = self.jobs.get(job_id, caller.tenant_id)
        spec, _policy, doc = self._context(job)
        delivered = self.delivery.deliver(job["tenant_id"], job_id, doc["filename"], self.jobs.outputs(job_id))
        committed = self._commit_assets(job, spec)
        self.jobs.set_status(job_id, JobStatus.DELIVERED)
        self.audit.append(
            caller.id,
            "job.delivered",
            tenant_id=job["tenant_id"],
            subject=job_id,
            details={"languages": [d["lang"] for d in delivered], "assets_committed": committed},
        )
        self.bus.publish("job.completed", {"job_id": job_id, "assets_committed": committed}, job["tenant_id"])
        return {"delivered": delivered, "assets_committed": committed, "at": now_iso()}

    def _commit_assets(self, job: dict[str, Any], spec: JobSpec) -> int:
        """Write approved translations to TM + cache. Happens only after approval, so unreviewed output never poisons reuse."""
        n = 0
        for seg in self.jobs.segments(job["id"], status="done"):
            if not seg["cacheable"] or seg["path"] in ("reuse_exact", "reuse_repeat"):
                continue
            ctx = ReuseContext(job["tenant_id"], spec.source_lang, seg["lang"], spec.domain.value, job["glossary_version"])
            self.reuse.store(
                seg["source"], seg["target"], ctx, origin="human" if seg["approved_by"] else "auto_accepted", approved_by=seg["approved_by"]
            )
            n += 1
        return n

    # -- operations --------------------------------------------------------------------------------
    def retry(self, principal: Principal, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id, principal.tenant_id)
        if job["status"] != JobStatus.NEEDS_ATTENTION.value:
            raise ValidationFailed(f"job is {job['status']}; only needs_attention jobs can be retried")
        self.jobs.set_status(job_id, JobStatus.TRANSLATING)
        self.audit.append(principal.subject, "job.retry", tenant_id=principal.tenant_id, subject=job_id)
        self.enqueue(job_id)
        return self.jobs.get(job_id)

    def result(self, principal: Principal, job_id: str, lang: str) -> str:
        self.jobs.get(job_id, principal.tenant_id)
        for out in self.jobs.outputs(job_id):
            if out["lang"] == lang:
                return str(out["content"])
        raise NotFoundError(f"no output for {lang} yet")


__all__ = ["JobRunner"]
