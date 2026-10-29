"""Quality service: runs the evaluators in parallel, normalises, caches by (source, target, evaluator versions)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..assets.cache import CacheBackend, cache_key
from .base import Evaluator, ScoreRequest
from .calibration import Calibration


@dataclass
class ScoreBundle:
    scores: dict[str, float] = field(default_factory=dict)  # normalised 0..1 per slot
    raw: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    cached: bool = False


class QualityService:
    def __init__(self, evaluators: Iterable[Evaluator], calibration: Calibration | None = None, cache: CacheBackend | None = None) -> None:
        self.evaluators = {e.slot: e for e in evaluators}
        self.calibration = calibration or Calibration()
        self.cache = cache
        self._pool = ThreadPoolExecutor(max_workers=max(3, len(self.evaluators)), thread_name_prefix="tap-eval")

    def versions(self) -> dict[str, str]:
        return {slot: f"{e.name}@{e.version}" for slot, e in self.evaluators.items()}

    def score(self, req: ScoreRequest, *, use_cache: bool = True) -> ScoreBundle:
        pair = f"{req.src_lang}>{req.tgt_lang}"
        key = cache_key(req.source + "\x1f" + req.target, pair, req.domain, "scores", "|".join(sorted(self.versions().values())), kind="sc")
        if use_cache and self.cache is not None and (hit := self.cache.get(key)):
            bundle = ScoreBundle(**json.loads(hit))
            bundle.cached = True
            return bundle
        futures = {slot: self._pool.submit(e.evaluate, req) for slot, e in self.evaluators.items()}
        bundle = ScoreBundle(versions=self.versions())
        for slot, fut in futures.items():
            res = fut.result()
            bundle.raw[slot] = res.score
            bundle.scores[slot] = self.calibration.normalise(self.evaluators[slot].name, pair, res.score)
            bundle.findings += [f"{slot}: {f}" for f in res.findings]
        if self.cache is not None:
            self.cache.set(key, json.dumps({k: v for k, v in bundle.__dict__.items() if k != "cached"}), ttl_s=86400)
        return bundle
