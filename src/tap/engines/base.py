"""Engine contracts shared by NMT and LLM adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..assets.terminology import Term
from ..assets.tm import TMMatch
from ..errors import ProviderError

__all__ = ["Engine", "ProviderError", "RepairInfo", "TranslationRequest", "TranslationResult"]


@dataclass(frozen=True)
class RepairInfo:
    """What the evaluators found wrong with the previous candidate; drives the repair prompt."""

    previous: str
    findings: list[str]


@dataclass(frozen=True)
class TranslationRequest:
    text: str
    src_lang: str
    tgt_lang: str
    domain: str = "clinical"
    glossary: list[Term] = field(default_factory=list)
    tm_examples: list[TMMatch] = field(default_factory=list)
    context: str | None = None
    repair: RepairInfo | None = None
    prompt_version: str | None = None  # set by the A/B service to pick a prompt variant


@dataclass
class TranslationResult:
    text: str
    engine: str  # "nmt:<name>" | "llm:<name>"
    model_version: str = ""
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    name: str
    kind: Literal["nmt", "llm"]
    supports_glossary: bool

    def supports(self, src_lang: str, tgt_lang: str) -> bool: ...
    def translate(self, req: TranslationRequest) -> TranslationResult: ...
