"""Score normalisation. Metrics live on different scales (COMET-Kiwi ~0.3-0.95, a judge 0-1, cosine 0-1) and
differ per language pair, so raw values are mapped to 0..1 with per-(evaluator, language pair) anchors before
weighting. ``fit_anchors`` derives anchors from human-rated samples (MQM-labelled good vs bad)."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .base import clamp


@dataclass(frozen=True)
class Anchors:
    bad: float = 0.0  # raw value of an unacceptable translation  -> maps to 0.0
    good: float = 1.0  # raw value of a clearly good translation   -> maps to 1.0

    def apply(self, raw: float) -> float:
        span = self.good - self.bad
        return clamp(raw) if span <= 1e-9 else clamp((raw - self.bad) / span)


class Calibration:
    def __init__(self, table: dict[str, Anchors] | None = None) -> None:
        self._table = table or {}

    @staticmethod
    def key(evaluator: str, pair: str) -> str:
        return f"{evaluator}|{pair}"

    def normalise(self, evaluator: str, pair: str, raw: float) -> float:
        a = self._table.get(self.key(evaluator, pair)) or self._table.get(self.key(evaluator, "*")) or Anchors()
        return a.apply(raw)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text())
        return cls({k: Anchors(**v) for k, v in data.items()})

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({k: {"bad": a.bad, "good": a.good} for k, a in self._table.items()}, indent=2))

    def set(self, evaluator: str, pair: str, anchors: Anchors) -> None:
        self._table[self.key(evaluator, pair)] = anchors


def fit_anchors(good_raw: list[float], bad_raw: list[float]) -> Anchors:
    """Median of human-approved samples -> 1.0, median of human-rejected samples -> 0.0."""
    if not good_raw or not bad_raw:
        raise ValueError("need both good and bad labelled samples")
    return Anchors(bad=statistics.median(bad_raw), good=statistics.median(good_raw))
