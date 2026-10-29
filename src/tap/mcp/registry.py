"""MCP tool registry: the only way agents reach platform capabilities.

Mirrors the guide's tool table. Anything destructive (delete TM, bulk overwrite, raw SQL) is *not registered*,
so no prompt can make an agent call it. Write tools require approval and (for publish) an approved job state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

_LANG = {"type": "string", "pattern": r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$", "maxLength": 16}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 8000}
_ID = {"type": "string", "pattern": r"^[A-Za-z0-9_.:-]{1,80}$"}

Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    scope: str
    risk: Risk
    requires_approval: bool
    roles: frozenset[str]
    input_schema: dict[str, Any]
    cost_kind: str | None = None  # key into the price table; used for budget checks
    tenant_arg: str | None = None  # argument that must equal the caller's tenant
    precondition: str | None = None  # named precondition evaluated by the handler layer
    scan_args: frozenset[str] | None = None  # args scanned for injection (None = all string args)


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_ALL = frozenset({"agent", "linguist", "admin"})

TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "tm_lookup",
            "Fuzzy search the translation memory",
            "read:tm",
            "low",
            False,
            _ALL,
            _schema(
                {
                    "source": _TEXT,
                    "src_lang": _LANG,
                    "tgt_lang": _LANG,
                    "min_match": {"type": "integer", "minimum": 50, "maximum": 100},
                    "domain": {"type": "string", "maxLength": 40},
                },
                ["source", "src_lang", "tgt_lang"],
            ),
        ),
        ToolSpec(
            "glossary_get",
            "Get approved terminology",
            "read:terms",
            "low",
            False,
            _ALL,
            _schema(
                {"client_id": _ID, "src_lang": _LANG, "tgt_lang": _LANG, "version": {"type": "string", "pattern": r"^v\d+$"}},
                ["client_id", "src_lang", "tgt_lang"],
            ),
            tenant_arg="client_id",
        ),
        ToolSpec(
            "translate_nmt",
            "Machine-translate text with the configured engine",
            "exec:translate",
            "low",
            False,
            _ALL,
            _schema(
                {"text": _TEXT, "src_lang": _LANG, "tgt_lang": _LANG, "domain": {"type": "string", "maxLength": 40}},
                ["text", "src_lang", "tgt_lang"],
            ),
            cost_kind="nmt",
            scan_args=frozenset({"text"}),
        ),
        ToolSpec(
            "translate_llm",
            "LLM translation or repair (costly)",
            "exec:translate",
            "medium",
            False,
            frozenset({"agent", "admin"}),
            _schema(
                {
                    "text": _TEXT,
                    "src_lang": _LANG,
                    "tgt_lang": _LANG,
                    "glossary_version": {"type": "string", "pattern": r"^v\d+$"},
                    "domain": {"type": "string", "maxLength": 40},
                    "context": {"type": "string", "maxLength": 4000},
                    "repair": _schema(
                        {"previous": _TEXT, "findings": {"type": "array", "items": {"type": "string", "maxLength": 400}, "maxItems": 20}},
                        ["previous", "findings"],
                    ),
                    "tm_examples": {
                        "type": "array",
                        "maxItems": 3,
                        "items": _schema({"source": _TEXT, "target": _TEXT, "pct": {"type": "integer"}}, ["source", "target"]),
                    },
                },
                ["text", "src_lang", "tgt_lang", "glossary_version"],
            ),
            cost_kind="llm",
            scan_args=frozenset({"text", "context"}),
        ),
        ToolSpec(
            "score_quality",
            "Return COMET, semantic and fluency scores",
            "exec:score",
            "low",
            False,
            _ALL,
            _schema(
                {
                    "source": _TEXT,
                    "translation": _TEXT,
                    "tgt_lang": _LANG,
                    "src_lang": _LANG,
                    "domain": {"type": "string", "maxLength": 40},
                },
                ["source", "translation", "tgt_lang"],
            ),
            cost_kind="score",
            scan_args=frozenset(),
        ),
        ToolSpec(
            "submit_for_review",
            "Send segments to human review",
            "write:review",
            "medium",
            False,
            _ALL,
            _schema(
                {"job_id": _ID, "segment_ids": {"type": "array", "items": _ID, "minItems": 1, "maxItems": 500}, "lang": _LANG},
                ["job_id", "segment_ids"],
            ),
        ),
        ToolSpec(
            "update_glossary",
            "Change an approved glossary entry",
            "write:terms",
            "high",
            True,
            frozenset({"admin"}),
            _schema(
                {
                    "client_id": _ID,
                    "term": {"type": "string", "minLength": 1, "maxLength": 200},
                    "translation": {"type": "string", "minLength": 1, "maxLength": 200},
                    "src_lang": _LANG,
                    "tgt_lang": _LANG,
                },
                ["client_id", "term", "translation"],
            ),
            tenant_arg="client_id",
        ),
        ToolSpec(
            "publish_translation",
            "Deliver approved translation to the client system",
            "write:publish",
            "high",
            True,
            frozenset({"linguist", "admin", "delivery-agent"}),
            _schema({"job_id": _ID}, ["job_id"]),
            precondition="job_awaiting_approval",
        ),
    ]
}

# Scopes granted per role. Scopes are what a *caller* may do; ToolSpec.scope is what a *tool* needs.
SCOPES_BY_ROLE: dict[str, frozenset[str]] = {
    "agent": frozenset({"read:tm", "read:terms", "exec:translate", "exec:score", "write:review"}),
    "delivery-agent": frozenset({"write:publish"}),
    "linguist": frozenset({"read:tm", "read:terms", "exec:translate", "exec:score", "write:review", "write:publish"}),
    "admin": frozenset({"read:tm", "read:terms", "exec:translate", "exec:score", "write:review", "write:terms", "write:publish"}),
}

# Least-privilege agent identities: the evaluator cannot publish, the delivery agent cannot translate.
AGENT_TOOLS: dict[str, frozenset[str]] = {
    "orchestrator": frozenset({"tm_lookup", "glossary_get", "translate_nmt", "translate_llm", "score_quality", "submit_for_review"}),
    "evaluator": frozenset({"score_quality"}),
    "delivery": frozenset({"publish_translation"}),
}


@dataclass(frozen=True)
class Caller:
    id: str
    tenant_id: str
    kind: Literal["agent", "user", "system"]
    roles: frozenset[str]
    allowed_tools: frozenset[str] | None = None  # None = every tool the roles allow
    scopes: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def agent(cls, name: str, tenant_id: str) -> Caller:
        role = "delivery-agent" if name == "delivery" else "agent"
        return cls(f"agent:{name}", tenant_id, "agent", frozenset({role}), AGENT_TOOLS[name], SCOPES_BY_ROLE[role])

    @classmethod
    def user(cls, subject: str, tenant_id: str, roles: frozenset[str]) -> Caller:
        scopes: frozenset[str] = frozenset().union(*(SCOPES_BY_ROLE.get(r, frozenset()) for r in roles)) if roles else frozenset()
        return cls(f"user:{subject}", tenant_id, "user", roles, None, scopes)
