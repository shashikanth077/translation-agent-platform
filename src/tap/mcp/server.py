"""Standalone MCP server (stdio) exposing the platform's tools to any MCP client (an IDE, Claude Desktop, an agent).

Every call goes through the same ``MCPGateway`` as the internal workflow: the server holds a *service identity*
(tenant + agent role) and cannot do anything that identity may not do, whatever the connected model asks for.

    TAP_MCP_TENANT=acme python -m tap.mcp.server

Works with MCP SDK v1 (``FastMCP``) and v2 (``MCPServer``).
"""

from __future__ import annotations

import os
from typing import Any

from ..config import Settings
from ..platform import Platform
from .registry import Caller


def _server_class() -> Any:
    import importlib

    try:
        return importlib.import_module("mcp.server.mcpserver").MCPServer  # mcp >= 2
    except (ImportError, AttributeError):
        return importlib.import_module("mcp.server.fastmcp").FastMCP  # mcp 1.x


def build_server(platform: Platform, tenant_id: str, agent: str = "orchestrator") -> Any:
    server = _server_class()("translation-platform")
    caller = Caller.agent(agent, tenant_id)

    def invoke(tool: str, **args: Any) -> Any:
        args = {k: v for k, v in args.items() if v is not None}
        res = platform.gateway.call(caller, tool, args)
        if res.status == "allowed":
            return res.output
        extra = f" (approval {res.approval_id} is pending)" if res.approval_id else ""
        raise ValueError(f"{tool} {res.status}: {res.reason}{extra}")

    @server.tool()
    def tm_lookup(source: str, src_lang: str, tgt_lang: str, min_match: int = 75, domain: str | None = None) -> list[dict]:
        """Fuzzy search the translation memory. Returns up to 5 matches with match percentage."""
        return invoke("tm_lookup", source=source, src_lang=src_lang, tgt_lang=tgt_lang, min_match=min_match, domain=domain)

    @server.tool()
    def glossary_get(client_id: str, src_lang: str, tgt_lang: str, version: str | None = None) -> list[dict]:
        """Return approved terminology for a client and language pair."""
        return invoke("glossary_get", client_id=client_id, src_lang=src_lang, tgt_lang=tgt_lang, version=version)

    @server.tool()
    def translate_nmt(text: str, src_lang: str, tgt_lang: str, domain: str | None = None) -> dict:
        """Machine-translate text with the configured NMT engine."""
        return invoke("translate_nmt", text=text, src_lang=src_lang, tgt_lang=tgt_lang, domain=domain)

    @server.tool()
    def translate_llm(
        text: str, src_lang: str, tgt_lang: str, glossary_version: str, domain: str | None = None, context: str | None = None
    ) -> dict:
        """LLM translation with glossary enforcement (costly). Use after NMT fails quality checks."""
        return invoke(
            "translate_llm",
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            glossary_version=glossary_version,
            domain=domain,
            context=context,
        )

    @server.tool()
    def score_quality(source: str, translation: str, tgt_lang: str, src_lang: str | None = None, domain: str | None = None) -> dict:
        """Return COMET-slot, semantic-accuracy and fluency scores (0..1)."""
        return invoke("score_quality", source=source, translation=translation, tgt_lang=tgt_lang, src_lang=src_lang, domain=domain)

    @server.tool()
    def submit_for_review(job_id: str, segment_ids: list[str], lang: str | None = None) -> dict:
        """Send segments to human review."""
        return invoke("submit_for_review", job_id=job_id, segment_ids=segment_ids, lang=lang)

    @server.tool()
    def update_glossary(client_id: str, term: str, translation: str, src_lang: str | None = None, tgt_lang: str | None = None) -> dict:
        """Change an approved glossary entry. Requires admin role and an approval."""
        return invoke("update_glossary", client_id=client_id, term=term, translation=translation, src_lang=src_lang, tgt_lang=tgt_lang)

    @server.tool()
    def publish_translation(job_id: str) -> dict:
        """Deliver an approved translation to the client system. Requires an approval and an approved job state."""
        return invoke("publish_translation", job_id=job_id)

    return server


def main() -> None:  # pragma: no cover - process entry point
    tenant = os.environ.get("TAP_MCP_TENANT")
    if not tenant:
        raise SystemExit("set TAP_MCP_TENANT to the tenant this MCP server acts for")
    platform = Platform.create(Settings())
    platform.start()
    try:
        build_server(platform, tenant, os.environ.get("TAP_MCP_AGENT", "orchestrator")).run()
    finally:
        platform.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
