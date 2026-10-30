"""Composition root: builds every service from ``Settings`` and wires them together.

Everything is constructed here and nowhere else, so tests can build a platform with fakes by passing overrides,
and the module boundaries (which service may call which) are visible in one file.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .assets.cache import CacheBackend, MemoryCache, RedisCache
from .assets.reuse import ReuseService
from .assets.semantic import SemanticIndex
from .assets.terminology import Terminology
from .assets.tm import TranslationMemory
from .config import Settings
from .db import Database
from .engines.base import Engine
from .engines.demo import DemoLLM, DemoNMT
from .engines.libretranslate import LibreTranslateEngine
from .engines.llm_client import AnthropicClient, LLMClient
from .engines.llm_engine import LLMTranslator
from .engines.resilience import CircuitBreaker, EngineRegistry
from .events import EventBus
from .mcp.approvals import Approvals
from .mcp.gateway import MCPGateway
from .mcp.handlers import ToolHandlers
from .observability import Metrics
from .orchestrator.runner import JobRunner
from .orchestrator.tools import ToolClient
from .orchestrator.workflow import SegmentWorkflow, WorkflowDeps
from .quality.base import Evaluator
from .quality.calibration import Calibration
from .quality.consensus import ConsensusConfig
from .quality.demo import DemoEvaluator
from .quality.heuristics import HeuristicFluency, HeuristicSemantic, QeProxy
from .quality.llm_judge import LLMJudge
from .quality.service import QualityService
from .services.audit import AuditLog
from .services.delivery import DeliveryService, FilesystemConnector
from .services.experiments import ExperimentService
from .services.feedback import FeedbackService
from .services.identity import IdentityService, TenantProfile
from .services.intake import IntakeService, LLMExtractor, RuleExtractor
from .services.jobs import JobStore
from .services.metering import Metering
from .services.notifications import NOTIFY_TOPICS, NotificationService
from .services.review import ReviewService

log = logging.getLogger("tap.platform")


@dataclass
class Platform:
    settings: Settings
    db: Database
    metrics: Metrics
    bus: EventBus
    audit: AuditLog
    identity: IdentityService
    cache: CacheBackend
    tm: TranslationMemory
    terminology: Terminology
    reuse: ReuseService
    engines: EngineRegistry
    quality: QualityService
    consensus: ConsensusConfig
    experiments: ExperimentService
    metering: Metering
    approvals: Approvals
    gateway: MCPGateway
    jobs: JobStore
    review: ReviewService
    delivery: DeliveryService
    feedback: FeedbackService
    notifications: NotificationService
    intake: IntakeService
    workflow: SegmentWorkflow
    runner: JobRunner
    llm: LLMClient | None
    _checkpoint_conn: sqlite3.Connection

    # ------------------------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        *,
        llm_client: LLMClient | None = None,
        engines: EngineRegistry | None = None,
        evaluators: list[Evaluator] | None = None,
        tenants: list[TenantProfile] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> Platform:
        s = settings or Settings()
        s.data_dir.mkdir(parents=True, exist_ok=True)
        metrics = Metrics()
        db = Database(s.database_file)
        bus = EventBus(db)
        audit = AuditLog(db)
        identity = IdentityService(s, tenants)
        cache: CacheBackend = RedisCache(s.redis_url) if s.cache_backend == "redis" and s.redis_url else MemoryCache()
        tm, terminology = TranslationMemory(db), Terminology(db)
        reuse = ReuseService(cache, tm, SemanticIndex(db), metrics, context_min_pct=s.tm_context_min_pct)

        llm = llm_client or cls._llm_client(s)
        registry = engines or cls._engines(s, llm, metrics)
        quality = QualityService(evaluators or cls._evaluators(s, llm), Calibration(), cache)
        consensus = ConsensusConfig(dict(s.weights), s.accept_threshold, s.critical_semantic, s.max_disagreement)

        experiments = ExperimentService(db)
        metering = Metering(db, s.prices, s.baseline_cost_per_1k_words, s.default_job_budget_units)
        approvals = Approvals(db)
        gateway = MCPGateway(approvals, audit, metrics, rate_per_min=s.gateway_rate_limit_per_min, injection_mode=s.injection_mode)
        jobs, review = JobStore(db), ReviewService(db)
        delivery = DeliveryService(db, FilesystemConnector(s.data_dir / "delivered"))
        feedback, notifications = FeedbackService(db), NotificationService(db)

        conn = sqlite3.connect(str(s.checkpoint_file), check_same_thread=False)
        saver = checkpointer or SqliteSaver(conn)
        tools = ToolClient(gateway)
        workflow = SegmentWorkflow(WorkflowDeps(s, reuse, terminology, tools, registry, metering, experiments, consensus, metrics), saver)
        runner = JobRunner(
            settings=s,
            jobs=jobs,
            workflow=workflow,
            reuse=reuse,
            terminology=terminology,
            review=review,
            metering=metering,
            delivery=delivery,
            approvals=approvals,
            gateway=gateway,
            audit=audit,
            bus=bus,
            identity=identity,
            tools=tools,
            metrics=metrics,
        )
        ToolHandlers(
            tm=tm,
            terminology=terminology,
            engines=registry,
            quality=quality,
            reuse=reuse,
            review=review,
            jobs=jobs,
            metering=metering,
            publisher=runner.publish,
        ).bind(gateway)

        extractor = LLMExtractor(llm) if (s.intake_backend == "llm" and llm) else RuleExtractor()
        intake = IntakeService(db, jobs, identity, extractor, urgent_hours=s.urgent_hours)

        p = cls(
            s,
            db,
            metrics,
            bus,
            audit,
            identity,
            cache,
            tm,
            terminology,
            reuse,
            registry,
            quality,
            consensus,
            experiments,
            metering,
            approvals,
            gateway,
            jobs,
            review,
            delivery,
            feedback,
            notifications,
            intake,
            workflow,
            runner,
            llm,
            conn,
        )
        p._subscribe()
        return p

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _llm_client(s: Settings) -> LLMClient | None:
        if s.llm_backend == "anthropic" and s.anthropic_api_key:
            return AnthropicClient(s.anthropic_api_key.get_secret_value(), s.llm_model, timeout_s=s.llm_timeout_s)
        return None

    @staticmethod
    def _engines(s: Settings, llm: LLMClient | None, metrics: Metrics) -> EngineRegistry:
        nmt: list[Engine] = (
            [DemoNMT()]
            if s.nmt_backend == "demo"
            else [
                LibreTranslateEngine(
                    s.libretranslate_url, s.libretranslate_api_key.get_secret_value() if s.libretranslate_api_key else None
                )
            ]
        )
        llms: list[Engine] = [DemoLLM()] if s.llm_backend == "demo" or llm is None else [LLMTranslator(llm)]
        return EngineRegistry(
            nmt,
            llms,
            metrics,
            breaker=CircuitBreaker(s.breaker_failure_threshold, s.breaker_reset_s),
            retry_attempts=s.engine_retry_attempts,
            retry_initial_s=s.engine_retry_initial_s,
        )

    @staticmethod
    def _evaluators(s: Settings, llm: LLMClient | None) -> list[Evaluator]:
        qe, sem, flu = QeProxy(), HeuristicSemantic(), HeuristicFluency()
        if s.quality_backend == "demo":
            return [DemoEvaluator("comet", qe), DemoEvaluator("semantic", sem), DemoEvaluator("fluency", flu)]
        if s.quality_backend == "heuristic":
            return [qe, sem, flu]
        if s.quality_backend == "llm":
            if llm is None:
                raise ValueError("TAP_QUALITY_BACKEND=llm needs an LLM client (TAP_LLM_BACKEND=anthropic)")
            return [qe, LLMJudge("semantic", llm, model=s.llm_judge_model), LLMJudge("fluency", llm, model=s.llm_judge_model)]
        from .quality.ml import CometKiwiEvaluator, LabseSemanticEvaluator  # optional heavy dependencies

        return [CometKiwiEvaluator(), LabseSemanticEvaluator(), LLMJudge("fluency", llm, model=s.llm_judge_model) if llm else flu]

    def _subscribe(self) -> None:
        self.bus.subscribe("metering", "segment.completed", self.metering.on_segment_completed)
        self.bus.subscribe("feedback", "review.resolved", self.feedback.on_review_resolved)
        self.bus.subscribe("notifications", NOTIFY_TOPICS, self.notifications.on_event)
        self.bus.subscribe("experiments", "job.completed", self._record_experiments)

    def _record_experiments(self, ev: Any) -> None:
        job_id = ev.payload["job_id"]
        summary = self.metering.job_summary(job_id)
        job = self.jobs.get(job_id)
        words = sum(s["words"] for s in self.jobs.segments(job_id)) or 1
        metrics = {
            "cost_per_1k_words": 1000 * summary["cost_units"] / words,
            "auto_accept_rate": summary["auto_accept_rate"],
            "human_rate": summary["human_rate"],
            "mean_composite": summary["mean_composite"],
            "quality_pass": float(summary["mean_composite"] >= self.settings.accept_threshold),
        }
        for exp in self.experiments.list_all():
            if exp.status == "running":
                self.experiments.record(exp.name, job["id"], metrics)

    # ------------------------------------------------------------------------------------------
    def start(self) -> None:
        self.bus.start_relay(self.settings.relay_interval_s)
        recovered = self.runner.recover()
        if recovered:
            log.info("re-enqueued %d unfinished job(s) after restart", recovered)

    def stop(self) -> None:
        self.runner.wait_idle(timeout=30)
        self.bus.stop_relay()
        self.bus.flush()
        self.runner.shutdown()
        self._checkpoint_conn.close()
        self.db.close()

    def seed_demo(self, tenant_id: str = "acme") -> dict[str, int]:
        from .demo.seed import seed

        return seed(self, tenant_id)


def default_data_dir() -> Path:
    return Path(".tap-data")
