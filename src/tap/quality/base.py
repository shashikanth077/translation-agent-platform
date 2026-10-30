"""Evaluator contracts. An evaluator fills one *slot* (comet / semantic / fluency) of the consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Slot = Literal["comet", "semantic", "fluency"]


@dataclass(frozen=True)
class ScoreRequest:
    source: str
    target: str
    src_lang: str
    tgt_lang: str
    domain: str = "clinical"
    reference: str | None = None


@dataclass
class EvalResult:
    score: float  # raw metric value; calibration normalises it to 0..1
    findings: list[str] = field(default_factory=list)


class Evaluator(Protocol):
    slot: Slot
    name: str
    version: str

    def evaluate(self, req: ScoreRequest) -> EvalResult: ...


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))
