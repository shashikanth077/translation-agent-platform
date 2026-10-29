"""Semantic suggestions: nearest TM entries by embedding similarity.

Suggestions are *context only*: they are never auto-reused, because two sentences can be close in embedding
space and differ in a critical detail. The default embedder is a dependency-free character-trigram hasher;
swap in ``SentenceTransformerEmbedder`` (extra ``ml``) for real multilingual embeddings such as LaBSE.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol

from ..db import Database
from ..util import normalise_text


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> dict[int, float] | list[float]: ...
    def similarity(self, a: dict[int, float] | list[float], b: dict[int, float] | list[float]) -> float: ...


class HashingEmbedder:
    name = "char3-hash"

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def embed(self, text: str) -> dict[int, float]:
        t = f"  {normalise_text(text).lower()}  "
        counts = Counter(zlib.crc32(t[i : i + 3].encode()) % self.dim for i in range(len(t) - 2))
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    def similarity(self, a, b) -> float:
        if len(a) > len(b):
            a, b = b, a
        return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


class SentenceTransformerEmbedder:  # pragma: no cover - needs the optional 'ml' extra and model download
    name = "labse"

    def __init__(self, model: str = "sentence-transformers/LaBSE") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def similarity(self, a, b) -> float:
        return float(sum(x * y for x, y in zip(a, b, strict=False)))


class SemanticIndex:
    def __init__(self, db: Database, embedder: Embedder | None = None, max_scan: int = 5000) -> None:
        self.db = db
        self.embedder = embedder or HashingEmbedder()
        self.max_scan = max_scan

    def suggest(self, tenant_id: str, src_lang: str, tgt_lang: str, source: str, *, limit: int = 3, min_sim: float = 0.6) -> list[dict]:
        q = self.embedder.embed(source)
        rows = self.db.query(
            "SELECT id, source, target FROM tm_entries WHERE tenant_id=? AND src_lang=? AND tgt_lang=? LIMIT ?",
            (tenant_id, src_lang, tgt_lang, self.max_scan),
        )
        scored = [(self.embedder.similarity(q, self.embedder.embed(r["source"])), r) for r in rows]
        scored = [(s, r) for s, r in scored if s >= min_sim]
        scored.sort(key=lambda x: -x[0])
        return [{"id": r["id"], "source": r["source"], "target": r["target"], "similarity": round(s, 3)} for s, r in scored[:limit]]
