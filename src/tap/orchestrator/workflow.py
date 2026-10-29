"""The per-segment LangGraph workflow (the heart of the platform).

    START -> cache_lookup --hit--------------------------------------------> END
                 |  \\--draft (fuzzy TM match)--> check
                 \\--miss--> route --nmt--> translate_nmt --> check --> score --> decide
                                  \\--llm--> translate_llm -/                          |
                                  \\--human-> human_review                  accept -> stage -> END
                                                   ^                         repair -> translate_llm
                                                   |                         human  -> human_review
                                                   \\-- resume with the reviewer's text (interrupt)

Design rules (each is covered by tests):
  * deterministic checks run before any paid scoring; a failed check is never averaged away;
  * repair is capped (``max_repair_attempts``) and then escalates to a human: no infinite loops;
  * nodes that call tools are idempotent and code before ``interrupt()`` is side-effect free (the node re-runs on resume);
  * the gateway can refuse any model/engine call (injection, budget, scope): the segment escalates instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..assets.reuse import ReuseContext, ReuseService
from ..assets.terminology import Term, Terminology
from ..assets.tm import TMMatch
from ..config import Settings
from ..domain.models import PolicyName, Risk
from ..engines.resilience import EngineRegistry
from ..observability import Metrics
from ..quality.checks import run_checks
from ..quality.consensus import ConsensusConfig, composite, decide
from ..services.experiments import ExperimentService
from ..services.metering import Metering
from ..services.policy import POLICIES, RouteFeatures, route
from .state import SegState
from .tools import ToolClient, ToolRefused

ROUTE_EXPERIMENT = "route-policy"


@dataclass
class WorkflowDeps:
    settings: Settings
    reuse: ReuseService
    terminology: Terminology
    tools: ToolClient
    engines: EngineRegistry
    metering: Metering
    experiments: ExperimentService
    consensus: ConsensusConfig
    metrics: Metrics


@dataclass
class SegmentOutcome:
    status: str  # "done" | "interrupted"
    final: dict[str, Any] | None = None
    interrupt: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


def _terms(state: SegState) -> list[Term]:
    return [Term(**t) for t in state.get("terms", [])]


def _tm_matches(state: SegState) -> list[TMMatch]:
    return [TMMatch(**m) for m in state.get("tm_examples", [])]


def _last(state: SegState) -> dict[str, Any]:
    return state["candidates"][-1]


class SegmentWorkflow:
    def __init__(self, deps: WorkflowDeps, checkpointer: BaseCheckpointSaver) -> None:
        self.d = deps
        self.app: Any = self._build().compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def thread_id(job_id: str, lang: str, seg_id: str) -> str:
        return f"{job_id}:{lang}:{seg_id}"

    def _config(self, state_or_ids: dict[str, Any] | tuple[str, str, str]) -> dict[str, Any]:
        job_id, lang, seg_id = (
            state_or_ids if isinstance(state_or_ids, tuple) else (state_or_ids["job_id"], state_or_ids["tgt_lang"], state_or_ids["seg_id"])
        )
        return {
            "configurable": {"thread_id": self.thread_id(job_id, lang, seg_id)},
            "recursion_limit": 60,
            # picked up by LangSmith when LANGSMITH_TRACING=true; never put source text in metadata
            "tags": ["tap", "segment"],
            "metadata": {"job_id": job_id, "lang": lang, "seg_id": seg_id},
        }

    def start(self, initial: SegState) -> SegmentOutcome:
        init: SegState = {**initial, "candidates": [], "steps": [], "trace": [], "attempts": 0}
        return self._outcome(self.app.invoke(init, self._config(dict(initial))), self._config(dict(initial)))

    def resume(self, job_id: str, lang: str, seg_id: str, answer: dict[str, Any]) -> SegmentOutcome:
        cfg = self._config((job_id, lang, seg_id))
        return self._outcome(self.app.invoke(Command(resume=answer), cfg), cfg)

    def continue_run(self, job_id: str, lang: str, seg_id: str) -> SegmentOutcome:
        """Resume a run that was interrupted by a crash (not by ``interrupt()``): replay from the last checkpoint."""
        cfg = self._config((job_id, lang, seg_id))
        return self._outcome(self.app.invoke(None, cfg), cfg)

    def inspect(self, job_id: str, lang: str, seg_id: str) -> dict[str, Any]:
        """Checkpoint status: ``fresh`` (never started), ``interrupted``, ``crashed`` (mid-graph) or ``finished``."""
        snap = self.app.get_state(self._config((job_id, lang, seg_id)))
        if not snap.values:
            return {"status": "fresh"}
        if not snap.next:
            return {"status": "finished", "final": snap.values.get("final")}
        interrupts = [i for t in snap.tasks for i in t.interrupts]
        if interrupts:
            return {"status": "interrupted", "interrupt": interrupts[0].value}
        return {"status": "crashed", "next": list(snap.next)}

    def _outcome(self, result: dict[str, Any], cfg: dict[str, Any]) -> SegmentOutcome:
        if result.get("__interrupt__"):
            return SegmentOutcome("interrupted", interrupt=result["__interrupt__"][0].value, state=_public(result))
        return SegmentOutcome("done", final=result["final"], state=_public(result))

    # ------------------------------------------------------------------------------------------
    # graph definition
    # ------------------------------------------------------------------------------------------
    def _build(self) -> Any:
        g: Any = StateGraph(SegState)
        for name, fn in (
            ("cache_lookup", self.cache_lookup),
            ("route", self.route_node),
            ("translate_nmt", self.translate_nmt),
            ("translate_llm", self.translate_llm),
            ("check", self.check),
            ("score", self.score),
            ("decide", self.decide),
            ("human_review", self.human_review),
            ("stage", self.stage),
        ):
            g.add_node(name, fn)
        g.add_edge(START, "cache_lookup")
        g.add_conditional_edges("cache_lookup", lambda s: s["next"], {"stage": "stage", "check": "check", "route": "route"})
        g.add_conditional_edges("route", lambda s: s["route"], {"nmt": "translate_nmt", "llm": "translate_llm", "human": "human_review"})
        g.add_conditional_edges("translate_nmt", lambda s: s["next"], {"check": "check", "llm": "translate_llm", "human": "human_review"})
        g.add_conditional_edges("translate_llm", lambda s: s["next"], {"check": "check", "human": "human_review"})
        g.add_conditional_edges("check", lambda s: s["next"], {"score": "score", "decide": "decide"})
        g.add_conditional_edges("score", lambda s: s["next"], {"decide": "decide", "human": "human_review"})
        g.add_conditional_edges(
            "decide", lambda s: s["next"], {"stage": "stage", "translate_llm": "translate_llm", "human": "human_review"}
        )
        g.add_edge("human_review", "stage")
        g.add_edge("stage", END)
        return g

    # ------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------
    def _ctx(self, s: SegState) -> ReuseContext:
        return ReuseContext(s["tenant_id"], s["src_lang"], s["tgt_lang"], s["domain"], s["glossary_version"])

    def _call(self, s: SegState, tool: str, args: dict[str, Any]) -> Any:
        return self.d.tools.call(tool, args, tenant_id=s["tenant_id"], context={"job_id": s["job_id"]})

    def _escalate(self, s: SegState, reason: str, node: str, detail: str = "") -> dict[str, Any]:
        """A refused/failed tool call must never fail the segment: it goes to a human with the reason attached."""
        return {
            "next": "human",
            "force_human": True,
            "trace": [{"node": node, "event": "escalate", "reason": reason, "detail": detail[:200]}],
        }

    # ------------------------------------------------------------------------------------------
    # nodes
    # ------------------------------------------------------------------------------------------
    def cache_lookup(self, s: SegState) -> dict[str, Any]:
        policy = POLICIES[PolicyName(s["policy"])]
        try:
            terms = self._call(
                s,
                "glossary_get",
                {
                    "client_id": s["tenant_id"],
                    "src_lang": s["src_lang"].split("-")[0],
                    "tgt_lang": s["tgt_lang"].split("-")[0],
                    "version": s["glossary_version"],
                },
            )
        except ToolRefused as e:
            return {"terms": [], "next": "route", "trace": [{"node": "cache_lookup", "event": "glossary_unavailable", "reason": e.reason}]}
        hit = self.d.reuse.lookup(s["source"], self._ctx(s))
        out: dict[str, Any] = {"terms": terms, "reuse": {k: v for k, v in hit.as_dict().items() if k != "suggestions"}, "next": "route"}
        examples: list[dict[str, Any]] = []
        trace = [{"node": "cache_lookup", "event": hit.type, "pct": hit.pct}]

        if hit.type == "exact":
            rep = run_checks(
                s["source"],
                hit.target or "",
                src_lang=s["src_lang"],
                tgt_lang=s["tgt_lang"],
                terms=_terms({"terms": terms}),
                terminology=self.d.terminology,
            )
            if rep.ok:
                out.update(
                    candidates=[{"engine": "tm", "text": hit.target, "meta": {"reuse": "exact"}}],
                    checks=rep.passed,
                    findings=[],
                    scores={},
                    next="stage",
                    steps=["reuse_exact"],
                    decision={
                        "action": "accept",
                        "reason": "exact_reuse",
                        "composite": None,
                        "disagreement": None,
                        "explanation": "exact cache/TM hit passed deterministic sanity checks",
                    },
                )
                out["trace"] = [*trace, {"node": "cache_lookup", "event": "reuse_exact"}]
                return out
            # a cached entry that fails sanity checks is stale or poisoned: never reuse it blindly
            trace.append({"node": "cache_lookup", "event": "cache_entry_failed_checks", "reason": "; ".join(rep.findings)[:200]})
            out.update(
                candidates=[{"engine": "tm", "text": hit.target, "meta": {"reuse": "exact_failed"}}], next="check", steps=["reuse_exact"]
            )
        elif hit.type == "fuzzy" and policy.fuzzy_draft and hit.pct >= self.d.settings.tm_draft_min_pct and hit.target:
            out.update(
                candidates=[
                    {"engine": "tm", "text": hit.target, "meta": {"reuse": "fuzzy", "pct": hit.pct, "numbers_differ": hit.numbers_differ}}
                ],
                next="check",
                steps=["reuse_fuzzy"],
            )
            trace.append({"node": "cache_lookup", "event": "fuzzy_draft", "pct": hit.pct, "numbers_differ": hit.numbers_differ})
        elif hit.type == "fuzzy" and hit.pct >= self.d.settings.tm_context_min_pct and hit.target:
            examples.append(
                {
                    "id": 0,
                    "source": hit.tm_source or "",
                    "target": hit.target,
                    "pct": hit.pct,
                    "domain": s["domain"],
                    "glossary_version": s["glossary_version"],
                    "origin": "tm",
                    "numbers_differ": hit.numbers_differ,
                }
            )
        # semantic suggestions are context only, never reused
        for sug in (
            self.d.reuse.semantic.suggest(s["tenant_id"], s["src_lang"], s["tgt_lang"], s["source"], limit=2) if hit.type != "exact" else []
        ):
            if all(e["source"] != sug["source"] for e in examples):
                examples.append(
                    {
                        "id": sug["id"],
                        "source": sug["source"],
                        "target": sug["target"],
                        "pct": int(sug["similarity"] * 100),
                        "domain": s["domain"],
                        "glossary_version": "",
                        "origin": "semantic",
                        "numbers_differ": False,
                    }
                )
        out["tm_examples"] = examples[:3]
        out["trace"] = trace
        return out

    def route_node(self, s: SegState) -> dict[str, Any]:
        policy = POLICIES[PolicyName(s["policy"])]
        src, tgt = s["src_lang"], s["tgt_lang"]
        mean, n = self.d.metering.qe_history(s["tenant_id"], f"{src}>{tgt}", s["domain"])
        assigned = self.d.experiments.assign(ROUTE_EXPERIMENT, s["job_id"])
        accept = policy.accept if policy.accept is not None else self.d.consensus.accept
        r = route(
            RouteFeatures(
                risk=Risk(s["risk"]),
                policy=policy,
                pair=f"{src}>{tgt}",
                domain=s["domain"],
                nmt_available=self.d.engines.available("nmt", src, tgt),
                llm_available=self.d.engines.available("llm", src, tgt),
                nmt_mean_composite=mean,
                nmt_samples=n,
                accept_threshold=accept,
                variant=assigned[1] if assigned else None,
            )
        )
        out: dict[str, Any] = {
            "route": r.engine,
            "route_reason": r.reason,
            "trace": [
                {
                    "node": "route",
                    "event": r.engine,
                    "reason": r.reason,
                    **({"experiment": f"{ROUTE_EXPERIMENT}:{assigned[0]}"} if assigned else {}),
                }
            ],
        }
        if r.engine == "human":
            out["force_human"] = True
        return out

    def translate_nmt(self, s: SegState) -> dict[str, Any]:
        try:
            res = self._call(
                s, "translate_nmt", {"text": s["source"], "src_lang": s["src_lang"], "tgt_lang": s["tgt_lang"], "domain": s["domain"]}
            )
        except ToolRefused as e:
            if e.failed_step in ("injection", "budget", "schema", "rate_limit"):
                return self._escalate(s, f"gateway_{e.failed_step}", "translate_nmt", e.reason)
            return {"next": "llm", "trace": [{"node": "translate_nmt", "event": "fallback_llm", "reason": e.reason[:200]}]}
        return {
            "candidates": [{"engine": res["engine"], "text": res["text"], "meta": res.get("meta", {})}],
            "steps": ["nmt"],
            "next": "check",
            "trace": [{"node": "translate_nmt", "event": "translated", "engine": res["engine"]}],
        }

    def translate_llm(self, s: SegState) -> dict[str, Any]:
        repair = None
        attempts = s.get("attempts", 0)
        if s.get("candidates"):  # a previous candidate exists, so this is a repair, not a fresh translation
            findings = list(s.get("findings", [])) or [s.get("decision", {}).get("explanation") or "quality below threshold"]
            repair = {"previous": _last(s)["text"], "findings": findings[:10]}
            attempts += 1
        args: dict[str, Any] = {
            "text": s["source"],
            "src_lang": s["src_lang"],
            "tgt_lang": s["tgt_lang"],
            "glossary_version": s["glossary_version"],
            "domain": s["domain"],
        }
        if repair:
            args["repair"] = repair
        examples = [{"source": m["source"], "target": m["target"], "pct": m["pct"]} for m in s.get("tm_examples", [])]
        if examples:
            args["tm_examples"] = examples
        try:
            res = self._call(s, "translate_llm", args)
        except ToolRefused as e:
            return self._escalate(s, f"gateway_{e.failed_step}", "translate_llm", e.reason)
        meta = {**res.get("meta", {}), "repair": bool(repair)}
        return {
            "candidates": [{"engine": res["engine"], "text": res["text"], "meta": meta}],
            "steps": ["llm"],
            "attempts": attempts,
            "next": "check",
            "trace": [
                {"node": "translate_llm", "event": "repair" if repair else "translated", "engine": res["engine"], "attempt": attempts}
            ],
        }

    def check(self, s: SegState) -> dict[str, Any]:
        cand = _last(s)
        skip = frozenset({"polarity"}) if cand.get("meta", {}).get("pseudo") else frozenset()
        rep = run_checks(
            s["source"],
            cand["text"],
            src_lang=s["src_lang"],
            tgt_lang=s["tgt_lang"],
            terms=_terms(s),
            terminology=self.d.terminology,
            skip=skip,
        )
        can_repair = self._can_repair(s)
        # skip paid scoring when a repair will follow anyway: the failed check is already decisive
        nxt = "decide" if (not rep.ok and can_repair) else "score"
        return {
            "checks": rep.passed,
            "findings": rep.findings,
            "next": nxt,
            "trace": [{"node": "check", "event": "pass" if rep.ok else "fail", "failed": rep.failed()}],
        }

    def _can_repair(self, s: SegState) -> bool:
        return (
            s.get("attempts", 0) < self.d.settings.max_repair_attempts
            and self.d.engines.available("llm", s["src_lang"], s["tgt_lang"])
            and self.d.metering.job_has_budget(s["job_id"], self.d.metering.cost(["llm", "score"], s.get("words", 1)))
        )

    def score(self, s: SegState) -> dict[str, Any]:
        cand = _last(s)
        try:
            res = self._call(
                s,
                "score_quality",
                {
                    "source": s["source"],
                    "translation": cand["text"],
                    "src_lang": s["src_lang"],
                    "tgt_lang": s["tgt_lang"],
                    "domain": s["domain"],
                },
            )
        except ToolRefused as e:
            return self._escalate(s, f"gateway_{e.failed_step}", "score", e.reason) | {"next": "human"}
        scores = res["scores"]
        out: dict[str, Any] = {
            "scores": scores,
            "raw_scores": res.get("raw", {}),
            "steps": ["score"],
            "next": "decide",
            "scored_n": len(s.get("candidates", [])),
            "findings": [*s.get("findings", []), *res.get("findings", [])],
            "trace": [{"node": "score", "event": "scored", "scores": scores, "cached": res.get("cached", False)}],
        }
        if cand["engine"].startswith("nmt:") and s.get("nmt_composite") is None:
            out["nmt_composite"] = composite(scores, self.d.consensus.weights)
        return out

    def decide(self, s: SegState) -> dict[str, Any]:
        policy = POLICIES[PolicyName(s["policy"])]
        cfg = self.d.consensus.with_accept(policy.accept)
        reused = bool(s.get("candidates")) and _last(s).get("meta", {}).get("reuse") == "exact"
        force = s.get("force_human", False) or policy.requires_human(Risk(s["risk"]), reused_exact=reused)
        # scores of an earlier candidate must never be applied to a newer one
        fresh = s.get("scored_n") == len(s.get("candidates", []))
        dec = decide(s.get("scores") if fresh else None, s.get("checks", {}), cfg, force_human=force, can_repair=self._can_repair(s))
        nxt = {"accept": "stage", "repair": "translate_llm", "human": "human"}[dec.action]
        return {
            "decision": dec.model_dump(),
            "next": nxt,
            "trace": [
                {"node": "decide", "event": dec.action, "reason": dec.reason, "composite": dec.composite, "disagreement": dec.disagreement}
            ],
        }

    def human_review(self, s: SegState) -> dict[str, Any]:
        # Everything above interrupt() re-runs on resume, so it is pure.
        cand = _last(s) if s.get("candidates") else {"engine": "none", "text": "", "meta": {}}
        why = s.get("decision", {}).get("explanation") or s.get("route_reason") or "human review required"
        payload = {
            "job_id": s["job_id"],
            "lang": s["tgt_lang"],
            "seg_id": s["seg_id"],
            "source": s["source"],
            "candidate": cand["text"],
            "engine": cand["engine"],
            "scores": s.get("scores", {}),
            "checks": s.get("checks", {}),
            "findings": s.get("findings", []),
            "why_flagged": why,
            "risk": s["risk"],
            "policy": s["policy"],
            "reason_code": s.get("decision", {}).get("reason") or s.get("route_reason"),
            "terms": s.get("terms", []),
            "tm_examples": s.get("tm_examples", []),
        }
        answer = interrupt(payload)
        text = str(answer["text"])
        return {
            "candidates": [
                {
                    "engine": "human",
                    "text": text,
                    "meta": {"reviewer": answer.get("reviewer"), "machine": cand["text"], "machine_engine": cand["engine"]},
                }
            ],
            "steps": ["human"],
            "human": {"reviewer": answer.get("reviewer"), "note": answer.get("note"), "changed": text != cand["text"]},
            "decision": {
                "action": "accept",
                "reason": "human_approved",
                "composite": s.get("decision", {}).get("composite"),
                "disagreement": s.get("decision", {}).get("disagreement"),
                "explanation": "approved by reviewer",
            },
            "trace": [{"node": "human_review", "event": "resolved", "changed": text != cand["text"]}],
        }

    def stage(self, s: SegState) -> dict[str, Any]:
        """Final bookkeeping. TM/cache are written only after job approval (see JobRunner), never mid-flight;
        this node decides *whether* the result may be cached and prices the path."""
        policy = POLICIES[PolicyName(s["policy"])]
        cand = _last(s)
        human = s.get("human")
        steps = s.get("steps", [])
        engines = [c["engine"] for c in s["candidates"]]
        first = s["candidates"][0]
        if first["engine"] == "tm" and first.get("meta", {}).get("reuse") == "exact" and len(engines) == 1:
            path = "reuse_exact"
        else:
            if first["engine"] == "tm" and first.get("meta", {}).get("reuse") == "fuzzy":
                base = "reuse_fuzzy"
            elif any(e.startswith("nmt:") for e in engines):
                base = "nmt"
            elif any(e.startswith("llm:") for e in engines):
                base = "llm"
            else:
                base = "human"
            path = base + ("+repair" if s.get("attempts", 0) > 0 else "")
            if human and base != "human":
                path += "+human"
        domain = s["domain"]
        llm_used = any(c["engine"].startswith("llm:") for c in s["candidates"])
        if not policy.cache_write:
            cacheable = False
        elif human:
            cacheable = True
        elif domain == "regulatory":
            cacheable = False  # regulated content enters the TM only after human approval
        elif llm_used:
            cacheable = domain in ("ui", "marketing")
        else:
            cacheable = True
        dec = s.get("decision", {})
        final = {
            "text": cand["text"],
            "path": path,
            "decision": dec.get("action", "accept"),
            "reason": dec.get("reason"),
            "composite": dec.get("composite"),
            "cacheable": cacheable,
            "approved_by": (human or {}).get("reviewer"),
            "engine": cand["engine"],
            "cost_units": self.d.metering.cost(steps, s.get("words", 0)),
            "detail": {
                "checks": s.get("checks", {}),
                "findings": s.get("findings", []),
                "scores": s.get("scores", {}),
                "route_reason": s.get("route_reason"),
                "attempts": s.get("attempts", 0),
                "trace": s.get("trace", []),
                "machine": (cand.get("meta") or {}).get("machine") if human else None,
                "nmt_composite": s.get("nmt_composite"),
                "steps": steps,
            },
        }
        return {"final": final}


def _public(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "__interrupt__"}


__all__ = ["SegmentOutcome", "SegmentWorkflow", "WorkflowDeps"]
