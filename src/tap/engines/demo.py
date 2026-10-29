"""Deterministic offline engines. They exist so the *platform* (routing, checks, scoring, HITL, caching, MCP,
metering) can be run and tested end to end without API keys. They are not translation systems."""

from __future__ import annotations

from typing import Literal

from ..demo.corpus import entry_for, pseudo
from .base import TranslationRequest, TranslationResult


class _DemoBase:
    supports_glossary = True
    kind: Literal["nmt", "llm"]

    def supports(self, src_lang: str, tgt_lang: str) -> bool:
        return True

    def _lookup(self, req: TranslationRequest, table: str) -> str | None:
        entry = entry_for(req.text)
        if entry is None:
            return None
        return getattr(entry, table).get(req.tgt_lang.split("-")[0])


class DemoNMT(_DemoBase):
    name = "demo-nmt"
    kind: Literal["nmt", "llm"] = "nmt"

    def translate(self, req: TranslationRequest) -> TranslationResult:
        text = self._lookup(req, "nmt")
        if text is not None:
            return TranslationResult(text, f"nmt:{self.name}", "demo-1", meta={"scripted": True})
        return TranslationResult(pseudo(req.text, req.tgt_lang, "nmt"), f"nmt:{self.name}", "demo-1", meta={"pseudo": True})


class DemoLLM(_DemoBase):
    name = "demo-llm"
    kind: Literal["nmt", "llm"] = "llm"

    def translate(self, req: TranslationRequest) -> TranslationResult:
        text = self._lookup(req, "llm")
        if text is not None:
            return TranslationResult(text, f"llm:{self.name}", "demo-1", meta={"scripted": True, "repair": bool(req.repair)})
        return TranslationResult(
            pseudo(req.text, req.tgt_lang, "llm"), f"llm:{self.name}", "demo-1", meta={"pseudo": True, "repair": bool(req.repair)}
        )
