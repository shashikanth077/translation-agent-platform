"""LibreTranslate NMT adapter (self-hostable, open source). One real NMT option that needs no cloud account."""

from __future__ import annotations

from typing import Literal

import httpx

from ..errors import ProviderError
from .base import TranslationRequest, TranslationResult


class LibreTranslateEngine:
    name = "libretranslate"
    kind: Literal["nmt", "llm"] = "nmt"
    supports_glossary = False  # glossary adherence is enforced by the term check + LLM repair instead

    def __init__(self, base_url: str, api_key: str | None = None, *, timeout_s: float = 15.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._api_key = api_key

    def supports(self, src_lang: str, tgt_lang: str) -> bool:
        return True

    def translate(self, req: TranslationRequest) -> TranslationResult:
        payload = {"q": req.text, "source": req.src_lang.split("-")[0], "target": req.tgt_lang.split("-")[0], "format": "text"}
        if self._api_key:
            payload["api_key"] = self._api_key
        try:
            r = self._client.post("/translate", json=payload)
        except httpx.HTTPError as e:
            raise ProviderError(f"libretranslate unreachable: {e}") from e
        if r.status_code >= 500 or r.status_code == 429:
            raise ProviderError(f"libretranslate returned {r.status_code}")
        if r.status_code >= 400:
            # 4xx is our request's fault: retrying will not help, but it still counts as an engine failure
            raise ProviderError(f"libretranslate rejected the request ({r.status_code}): {r.text[:200]}")
        try:
            text = r.json()["translatedText"]
        except (ValueError, KeyError) as e:
            raise ProviderError("libretranslate returned an unexpected payload") from e
        return TranslationResult(text, f"nmt:{self.name}", "libretranslate")
