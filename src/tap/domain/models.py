"""Typed domain models. These are the contracts shared by services; the JobSpec is the intake contract."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


class Domain(StrEnum):
    REGULATORY = "regulatory"
    CLINICAL = "clinical"
    UI = "ui"
    MARKETING = "marketing"


class Audience(StrEnum):
    PATIENTS = "patients"
    SITES = "sites"
    INTERNAL = "internal"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyName(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    REGULATORY_STRICT = "regulatory-strict"
    SAFETY_URGENT = "safety-urgent"


class DocFormat(StrEnum):
    TXT = "txt"
    XLIFF = "xliff"
    JSON = "json"


class JobStatus(StrEnum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    AWAITING_REVIEW = "awaiting_review"
    AWAITING_APPROVAL = "awaiting_approval"
    DELIVERED = "delivered"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"


TERMINAL_JOB_STATES = {JobStatus.DELIVERED, JobStatus.FAILED}


def normalise_lang(code: str) -> str:
    code = code.strip().replace("_", "-")
    head, *rest = code.split("-")
    return "-".join([head.lower(), *[r.upper() if len(r) == 2 else r.title() for r in rest]])


class JobSpec(BaseModel):
    """The validated output of intake. Nothing downstream accepts free text."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    document_id: str
    source_lang: str
    target_langs: list[str] = Field(min_length=1, max_length=50)
    domain: Domain
    audience: Audience | None = None
    glossary_version: str | None = None  # "v7"; None means "latest published"
    back_translation: bool = False
    deadline: datetime | None = None
    regulatory_use: bool = False
    scope: Literal["full", "changes_only"] = "full"
    base_version: str | None = None
    policy: PolicyName | None = None  # explicit override; normally derived from domain/audience/deadline
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("source_lang")
    @classmethod
    def _src(cls, v: str) -> str:
        v = normalise_lang(v)
        if not LANG_RE.match(v):
            raise ValueError(f"invalid language code {v!r}")
        return v

    @field_validator("target_langs")
    @classmethod
    def _tgts(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for code in v:
            code = normalise_lang(code)
            if not LANG_RE.match(code):
                raise ValueError(f"invalid language code {code!r}")
            if code not in out:
                out.append(code)
        return out

    @field_validator("glossary_version")
    @classmethod
    def _gv(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"v\d+", v):
            raise ValueError("glossary_version must look like 'v7'")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.source_lang in self.target_langs:
            raise ValueError("target_langs must not contain the source language")


class Candidate(BaseModel):
    """One candidate translation for a segment."""

    engine: str  # nmt:<name> | llm:<name> | tm | human
    text: str
    scores: dict[str, float] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    cost_units: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    action: Literal["accept", "repair", "human"]
    reason: str
    composite: float | None = None
    disagreement: float | None = None
    explanation: str = ""


class SegmentStatus(StrEnum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    DONE = "done"
    FAILED = "failed"


class Principal(BaseModel):
    """An authenticated caller: a user acting through the API or a service identity."""

    subject: str
    tenant_id: str
    roles: frozenset[str] = frozenset()
    kind: Literal["user", "agent", "system"] = "user"

    def has_role(self, *roles: str) -> bool:
        return bool(self.roles & set(roles))
