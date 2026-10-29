"""Demo evaluators: scripted scores for the reference scenarios, heuristics for everything else."""

from __future__ import annotations

from ..demo.corpus import PSEUDO_SCORES, is_pseudo, scripted_scores
from .base import EvalResult, Evaluator, ScoreRequest, Slot


class DemoEvaluator:
    def __init__(self, slot: Slot, fallback: Evaluator) -> None:
        self.slot = slot
        self.fallback = fallback
        self.name = f"demo-{slot}"
        self.version = "1"

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        scripted = scripted_scores(req.source, req.target)
        if scripted is not None:
            return EvalResult(scripted[self.slot], [])
        if is_pseudo(req.target):
            return EvalResult(PSEUDO_SCORES[self.slot], [])
        return self.fallback.evaluate(req)
