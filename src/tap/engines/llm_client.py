"""Minimal structured-output LLM client used by the LLM translator, the judges and the intake agent.

Keeping this behind a Protocol means the rest of the platform never imports a vendor SDK, tests inject a fake,
and swapping providers touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import ProviderError


@dataclass
class LLMResponse:
    data: dict[str, Any]  # arguments of the forced tool call
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMClient(Protocol):
    model: str

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str, *, timeout_s: float = 30.0, client: Any | None = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - optional dependency
                raise RuntimeError("install the 'anthropic' extra to use TAP_LLM_BACKEND=anthropic") from e
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)  # retries live in EngineRegistry
        self._client: Any = client
        self.model = model

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> LLMResponse:
        try:
            msg = self._client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": tool_name, "description": tool_description, "input_schema": schema}],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as e:  # SDK raises several types (rate limit, connection, 5xx): all are retryable provider errors
            raise ProviderError(f"anthropic call failed: {type(e).__name__}: {e}") from e
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                usage = getattr(msg, "usage", None)
                return LLMResponse(
                    dict(block.input),
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                    getattr(msg, "model", model or self.model),
                )
        raise ProviderError("model did not return the requested tool call")
