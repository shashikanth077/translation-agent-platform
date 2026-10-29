"""Consensus decision: weighted composite, hard overrides and disagreement detection.

Order of precedence (first match wins):
  1. failed deterministic check          -> repair   (never averaged away)
  2. critical semantic score             -> repair
  3. judges disagree                     -> human    (a split vote is a signal, not noise)
  4. composite >= accept threshold       -> accept
  5. otherwise                           -> repair
Policy flags (e.g. ``force_human``) are applied last so they can only make the outcome *safer*.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..domain.models import Decision


@dataclass(frozen=True)
class ConsensusConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"comet": 0.4, "semantic": 0.4, "fluency": 0.2})
    accept: float = 0.85
    critical_semantic: float = 0.60
    max_disagreement: float = 0.25

    def with_accept(self, accept: float | None) -> ConsensusConfig:
        return self if accept is None else replace(self, accept=accept)


def composite(scores: dict[str, float], weights: dict[str, float]) -> float:
    return sum(w * scores[k] for k, w in weights.items())


def disagreement(scores: dict[str, float]) -> float:
    vals = list(scores.values())
    return max(vals) - min(vals) if vals else 0.0


def decide(
    scores: dict[str, float] | None,
    checks: dict[str, bool],
    cfg: ConsensusConfig,
    *,
    force_human: bool = False,
    can_repair: bool = True,
) -> Decision:
    failed = [k for k, v in checks.items() if not v]
    comp = composite(scores, cfg.weights) if scores else None
    dis = disagreement(scores) if scores else None

    def out(action: str, reason: str, why: str) -> Decision:
        if force_human and action == "accept":
            return Decision(
                action="human",
                reason="policy_requires_human",
                composite=comp,
                disagreement=dis,
                explanation=f"{why}; policy requires human sign-off",
            )
        return Decision(action=action, reason=reason, composite=comp, disagreement=dis, explanation=why)

    if failed:
        return out("repair" if can_repair else "human", "deterministic_check_failed", f"deterministic check failed: {', '.join(failed)}")
    if not scores:
        return out("repair" if can_repair else "human", "no_scores", "no quality scores available")
    if scores.get("semantic", 1.0) < cfg.critical_semantic:
        return out(
            "repair" if can_repair else "human",
            "critical_semantic_score",
            f"semantic accuracy {scores['semantic']:.2f} below critical {cfg.critical_semantic:.2f}",
        )
    if dis is not None and dis > cfg.max_disagreement:
        return out("human", "judges_disagree", f"judges disagree by {dis:.2f} (max {cfg.max_disagreement:.2f})")
    if comp is not None and comp >= cfg.accept:
        return out("accept", "above_threshold", f"composite {comp:.2f} >= {cfg.accept:.2f}")
    return out("repair" if can_repair else "human", "below_threshold", f"composite {comp:.2f} below {cfg.accept:.2f}")
