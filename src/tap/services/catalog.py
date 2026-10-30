"""Service catalogue: maps the reference decomposition (guide, "Microservices" tab) to the code that implements it.

Everything runs in one process by default (a modular monolith with enforced seams); each entry names the module
that owns the capability and its data, so a service can be lifted into its own deployable without redesign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ServiceEntry:
    category: str
    name: str
    responsibility: str
    module: str
    owns_tables: tuple[str, ...] = ()
    status: str = "implemented"  # implemented | partial | external
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


CATALOG: list[ServiceEntry] = [
    ServiceEntry("Access", "API gateway", "Auth, tenant context, rate limits, request validation", "tap.api.app"),
    ServiceEntry(
        "Access",
        "Identity and tenant service",
        "API keys, roles, tenant entitlements",
        "tap.services.identity",
        note="API-key auth; plug OIDC/JWT in front for production SSO",
    ),
    ServiceEntry(
        "Agents", "Intake agent service", "Conversational job intake to a validated JobSpec", "tap.services.intake", ("intake_sessions",)
    ),
    ServiceEntry(
        "Agents",
        "Orchestrator service",
        "LangGraph segment workflow, job state machine, checkpoints, interrupts",
        "tap.orchestrator.workflow",
    ),
    ServiceEntry("Agents", "Supervisor / router service", "Per-segment route decisions with reason codes", "tap.services.policy"),
    ServiceEntry("Agents", "MCP gateway", "Allow-list, scopes, schema, approvals, audit for tool calls", "tap.mcp.gateway", ("approvals",)),
    ServiceEntry(
        "Agents",
        "Prompt management service",
        "Versioned prompts and templates",
        "tap.engines.prompts",
        status="partial",
        note="prompts are versioned in code and gated by `tap eval`; no runtime prompt registry yet",
    ),
    ServiceEntry(
        "Agents",
        "Feedback and learning service",
        "Post-edit deltas to datasets and glossary suggestions",
        "tap.services.feedback",
        ("post_edits", "term_suggestions"),
    ),
    ServiceEntry(
        "Content",
        "Job service",
        "Job lifecycle, documents, segment results",
        "tap.services.jobs",
        ("documents", "jobs", "job_segments", "job_outputs"),
    ),
    ServiceEntry(
        "Content",
        "File parse and assembly service",
        "Parse/rebuild TXT, XLIFF 1.2, JSON resources",
        "tap.parse.formats",
        note="DOCX/PDF need external converters",
    ),
    ServiceEntry("Content", "Segmentation service", "Sentence splitting, repetition analysis", "tap.parse.segmentation"),
    ServiceEntry("Assets", "Cache service", "Versioned, tenant-scoped exact cache (memory or Redis)", "tap.assets.cache"),
    ServiceEntry("Assets", "TM service", "Fuzzy matching, TM storage, TMX import/export", "tap.assets.tm", ("tm_entries",)),
    ServiceEntry(
        "Assets",
        "Terminology service",
        "Versioned glossaries, term checks, impact analysis",
        "tap.assets.terminology",
        ("terms", "glossary_versions"),
    ),
    ServiceEntry(
        "Assets",
        "Semantic index service",
        "Embedding nearest-neighbour suggestions (context only)",
        "tap.assets.semantic",
        status="partial",
        note="hashing embedder by default; sentence-transformers via the `ml` extra",
    ),
    ServiceEntry("Engines", "NMT adapter service", "Uniform NMT interface with retry, breaker and fallback", "tap.engines.resilience"),
    ServiceEntry("Engines", "LLM adapter service", "LLM translation/repair behind a structured-output client", "tap.engines.llm_engine"),
    ServiceEntry("Quality", "Deterministic check service", "Numbers, units, tags, terms, polarity, length", "tap.quality.checks"),
    ServiceEntry(
        "Quality",
        "QE service (COMET)",
        "COMET-Kiwi quality estimation",
        "tap.quality.ml",
        status="partial",
        note="optional `ml` extra; default is a heuristic proxy",
    ),
    ServiceEntry("Quality", "Semantic accuracy service", "Embedding/LLM-judge semantic accuracy with number caps", "tap.quality.llm_judge"),
    ServiceEntry("Quality", "Fluency service", "Target-side fluency scoring", "tap.quality.heuristics"),
    ServiceEntry("Quality", "Consensus service", "Weights, hard overrides, disagreement, decision", "tap.quality.consensus"),
    ServiceEntry(
        "Human",
        "Review workbench service",
        "Review queue, claim/submit with checks, LQA sampling",
        "tap.services.review",
        ("review_items",),
        note="API only; the bundled console is a minimal reviewer UI",
    ),
    ServiceEntry(
        "Human",
        "Vendor and linguist management",
        "Assignments, availability, ratings",
        "-",
        status="external",
        note="out of scope: reviewers are API users with the linguist role",
    ),
    ServiceEntry(
        "Delivery",
        "Delivery and connector service",
        "Final format check, idempotent publish via connector",
        "tap.services.delivery",
        ("deliveries_log",),
    ),
    ServiceEntry(
        "Experiments",
        "A/B and feature-flag service",
        "Deterministic bucketing, outcomes, significance, guardrails",
        "tap.services.experiments",
        ("experiments", "assignments", "outcomes"),
    ),
    ServiceEntry(
        "Business", "Metering and cost service", "Usage per path, savings vs baseline, budgets", "tap.services.metering", ("usage_events",)
    ),
    ServiceEntry(
        "Business",
        "Billing and reporting service",
        "Invoices, dashboards",
        "tap.services.metering",
        status="partial",
        note="usage report API; invoicing is out of scope",
    ),
    ServiceEntry(
        "Compliance",
        "Audit and compliance service",
        "Append-only, hash-chained decisions and approvals",
        "tap.services.audit",
        ("audit_log",),
    ),
    ServiceEntry(
        "Platform", "Notification service", "Operator notifications from events", "tap.services.notifications", ("notifications",)
    ),
    ServiceEntry("Platform", "Evaluation and dataset service", "Golden-set regression runs (CI gate)", "tap.evaluation"),
    ServiceEntry("Platform", "Event bus", "Transactional outbox, idempotent consumers, DLQ", "tap.events", ("outbox", "deliveries")),
]
