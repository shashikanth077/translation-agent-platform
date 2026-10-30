"""LLM translation / repair engine on top of an ``LLMClient``."""

from __future__ import annotations

from typing import Literal

from ..errors import ProviderError
from . import prompts
from .base import TranslationRequest, TranslationResult
from .llm_client import LLMClient

SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string", "description": "The complete translation, nothing else."}},
    "required": ["translation"],
    "additionalProperties": False,
}


class LLMTranslator:
    kind: Literal["nmt", "llm"] = "llm"
    supports_glossary = True

    def __init__(self, client: LLMClient, name: str = "claude") -> None:
        self.client = client
        self.name = name

    def supports(self, src_lang: str, tgt_lang: str) -> bool:
        return True

    def translate(self, req: TranslationRequest) -> TranslationResult:
        version = prompts.REPAIR_V1 if req.repair else (req.prompt_version or prompts.TRANSLATE_V1)
        resp = self.client.call_tool(
            system=prompts.system_prompt(req.src_lang, req.tgt_lang, req.domain),
            user=prompts.user_prompt(req.text, req.glossary, req.tm_examples, req.context, req.repair),
            tool_name="submit_translation",
            tool_description="Submit the final translation.",
            schema=SUBMIT_SCHEMA,
            max_tokens=max(512, len(req.text) * 3),
        )
        text = resp.data.get("translation")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("LLM returned an empty translation")
        return TranslationResult(
            text.strip(),
            f"llm:{self.name}",
            resp.model,
            meta={
                "prompt_version": version,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "repair": bool(req.repair),
            },
        )
