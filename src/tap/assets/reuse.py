"""Tiered reuse lookup: exact cache -> fuzzy TM -> semantic suggestions.

Returns a ``ReuseHit`` describing what was found and what the caller may safely do with it. This service never
decides acceptance: a hit is a candidate that still goes through deterministic checks (and, for fuzzy matches,
scoring) in the workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..observability import Metrics
from ..util import now_iso
from .cache import CacheBackend, cache_key
from .semantic import SemanticIndex
from .tm import TranslationMemory


@dataclass(frozen=True)
class ReuseContext:
    tenant_id: str
    src_lang: str
    tgt_lang: str
    domain: str
    glossary_version: str  # "v7"
    style_version: str = "s1"

    @property
    def pair(self) -> str:
        return f"{self.src_lang}>{self.tgt_lang}"


@dataclass
class ReuseHit:
    type: Literal["exact", "fuzzy", "semantic", "none"] = "none"
    pct: int = 0
    target: str | None = None
    tm_source: str | None = None
    numbers_differ: bool = False
    context_mismatch: bool = False  # 100% text match but different domain or glossary version
    suggestions: list[dict[str, Any]] = field(default_factory=list)  # fuzzy matches below draft threshold + semantic
    key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReuseService:
    def __init__(
        self, cache: CacheBackend, tm: TranslationMemory, semantic: SemanticIndex, metrics: Metrics, *, context_min_pct: int = 75
    ) -> None:
        self.cache, self.tm, self.semantic, self.metrics = cache, tm, semantic, metrics
        self.context_min_pct = context_min_pct

    def key(self, source: str, ctx: ReuseContext) -> str:
        return cache_key(source, ctx.pair, ctx.domain, ctx.tenant_id, ctx.glossary_version, ctx.style_version)

    def lookup(self, source: str, ctx: ReuseContext) -> ReuseHit:
        key = self.key(source, ctx)
        cached = self.cache.get(key)
        if cached:
            self.metrics.cache_lookups.labels("exact").inc()
            return ReuseHit("exact", 100, json.loads(cached)["target"], key=key)
        entry = self.tm.exact(ctx.tenant_id, ctx.src_lang, ctx.tgt_lang, ctx.domain, ctx.glossary_version, source)
        if entry:  # the cache was cold (restart, eviction): backfill from the durable TM
            self._remember(key, entry.target, "tm")
            self.metrics.cache_lookups.labels("exact").inc()
            return ReuseHit("exact", 100, entry.target, tm_source=entry.source, key=key)

        matches = self.tm.search(ctx.tenant_id, ctx.src_lang, ctx.tgt_lang, source, min_pct=self.context_min_pct, limit=5)
        if matches:
            best = matches[0]
            self.metrics.cache_lookups.labels("fuzzy").inc()
            return ReuseHit(
                "fuzzy",
                best.pct,
                best.target,
                best.source,
                best.numbers_differ,
                context_mismatch=best.pct == 100,
                key=key,
                suggestions=[m.as_dict() for m in matches[1:]],
            )
        sem = self.semantic.suggest(ctx.tenant_id, ctx.src_lang, ctx.tgt_lang, source)
        if sem:
            self.metrics.cache_lookups.labels("semantic").inc()
            return ReuseHit("semantic", 0, None, None, key=key, suggestions=sem)
        self.metrics.cache_lookups.labels("miss").inc()
        return ReuseHit(key=key)

    def store(self, source: str, target: str, ctx: ReuseContext, *, origin: str, approved_by: str | None) -> None:
        """Persist an *approved* translation to the TM (durable) and the cache (fast path)."""
        self.tm.add(
            ctx.tenant_id,
            ctx.src_lang,
            ctx.tgt_lang,
            ctx.domain,
            source,
            target,
            glossary_version=ctx.glossary_version,
            origin=origin,
            approved_by=approved_by,
        )
        self._remember(self.key(source, ctx), target, origin)

    def _remember(self, key: str, target: str, origin: str) -> None:
        self.cache.set(key, json.dumps({"target": target, "origin": origin, "ts": now_iso()}))
