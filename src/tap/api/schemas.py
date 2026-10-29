"""Request bodies. Responses are plain dicts built by the services (kept explicit and versionable)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.models import Audience, Domain, PolicyName


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=5_000_000)
    format: Literal["txt", "xliff", "json"] | None = None


class JobRequest(BaseModel):
    """A JobSpec without the tenant: the tenant always comes from the API key, never from the body."""

    model_config = ConfigDict(extra="forbid")
    document_id: str
    source_lang: str = "en"
    target_langs: list[str] = Field(min_length=1)
    domain: Domain
    audience: Audience | None = None
    glossary_version: str | None = None
    back_translation: bool = False
    deadline: datetime | None = None
    regulatory_use: bool = False
    scope: Literal["full", "changes_only"] = "full"
    base_version: str | None = None
    policy: PolicyName | None = None
    notes: str | None = None


class MessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ReviewSubmit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000)
    note: str | None = Field(default=None, max_length=2000)
    override: bool = False


class ApprovalDecision(BaseModel):
    approve: bool
    reason: str = ""


class TMEntryIn(BaseModel):
    src_lang: str
    tgt_lang: str
    domain: str
    source: str
    target: str


class TMXImport(BaseModel):
    content: str = Field(min_length=1, max_length=10_000_000)
    domain: str = "imported"


class TermIn(BaseModel):
    op: Literal["add", "remove"] = "add"
    src_lang: str = "en"
    tgt_lang: str
    term: str
    translation: str = ""
    kind: Literal["preferred", "forbidden", "dnt"] = "preferred"


class GlossaryCommit(BaseModel):
    changes: list[TermIn] = Field(min_length=1, max_length=1000)
    note: str = ""


class CacheLookupIn(BaseModel):
    source: str
    src_lang: str = "en"
    tgt_lang: str
    domain: str = "regulatory"
    glossary_version: str | None = None


class ScoreIn(BaseModel):
    source: str
    target: str
    src_lang: str = "en"
    tgt_lang: str
    domain: str = "clinical"


class ConsensusIn(BaseModel):
    scores: dict[str, float] | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    weights: dict[str, float] | None = None
    accept: float | None = None
    critical_semantic: float | None = None
    max_disagreement: float | None = None
    force_human: bool = False
    can_repair: bool = True


class McpCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    caller: Literal["agent", "me"] = "me"
    agent: Literal["orchestrator", "evaluator", "delivery"] = "orchestrator"


class ExperimentIn(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,60}$")
    variants: dict[str, dict[str, Any]]
    primary_metric: str = "cost_per_1k_words"
    traffic_pct: int = 100
    guardrail_metric: str | None = "quality_pass"
    guardrail_margin: float = 0.02
    min_samples: int = 200


class SampleSizeIn(BaseModel):
    baseline_pct: float
    mde_points: float
    alpha: float = 0.05
    power: float = 0.8
