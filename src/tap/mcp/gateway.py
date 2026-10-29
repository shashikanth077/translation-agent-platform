"""The MCP gateway: policy enforcement between LLM-proposed tool calls and platform capabilities.

Pipeline (each step is reported so the "MCP safety lab" can show it):
  registry -> allow_list -> role -> scope -> schema -> tenant_scope -> rate_limit -> budget -> injection
  -> approval -> precondition -> execute -> audit
The first failing step stops the call. Tool *output* is also untrusted: it is returned as data and never
interpreted as instructions by the platform.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import jsonschema

from ..errors import BudgetExceeded
from ..observability import Metrics
from ..services.audit import AuditLog
from .approvals import Approvals, args_hash
from .registry import TOOLS, Caller, ToolSpec

INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)"
    r"|(?:call|invoke|run|execute)\s+\w+\s+(?:for|on)\s+all\b"
    r"|\bsystem\s*:|you\s+are\s+now\b|reveal\s+(?:your\s+)?(?:system\s+)?prompt|<\|im_start\|>|\[/?INST\]",
    re.IGNORECASE,
)

Status = Literal["allowed", "denied", "pending_approval"]
Handler = Callable[[Caller, dict[str, Any], dict[str, Any]], Any]
Precondition = Callable[[Caller, dict[str, Any]], str | None]


@dataclass
class CheckStep:
    name: str
    status: Literal["pass", "fail", "pending", "skipped", "flagged"]
    detail: str = ""


@dataclass
class GatewayResult:
    status: Status
    reason: str = ""
    checks: list[CheckStep] = field(default_factory=list)
    output: Any = None
    approval_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RateLimiter:
    """Token bucket per caller."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate, self.capacity, self._clock = per_minute / 60.0, float(per_minute), clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, self._clock()))
            now = self._clock()
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1, now)
            return True


class MCPGateway:
    PIPELINE = (
        "registry",
        "allow_list",
        "role",
        "scope",
        "schema",
        "tenant_scope",
        "rate_limit",
        "budget",
        "injection",
        "approval",
        "precondition",
        "execute",
        "audit",
    )

    def __init__(
        self,
        approvals: Approvals,
        audit: AuditLog,
        metrics: Metrics,
        *,
        rate_per_min: int = 1200,
        injection_mode: Literal["deny", "flag"] = "deny",
        tools: dict[str, ToolSpec] | None = None,
        budget_check: Callable[[Caller, ToolSpec, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self.tools = tools or TOOLS
        self.approvals, self.audit, self.metrics = approvals, audit, metrics
        self.injection_mode = injection_mode
        self.limiter = RateLimiter(rate_per_min)
        self._handlers: dict[str, Handler] = {}
        self._preconditions: dict[str, Precondition] = {}
        self._budget_check = budget_check

    def register(self, tool: str, handler: Handler) -> None:
        if tool not in self.tools:
            raise KeyError(f"unknown tool {tool}")
        self._handlers[tool] = handler

    def register_precondition(self, name: str, fn: Precondition) -> None:
        self._preconditions[name] = fn

    def describe(self, caller: Caller | None = None) -> list[dict[str, Any]]:
        out = []
        for t in self.tools.values():
            visible = caller is None or ((caller.allowed_tools is None or t.name in caller.allowed_tools) and bool(caller.roles & t.roles))
            if visible:
                out.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "scope": t.scope,
                        "risk": t.risk,
                        "requires_approval": t.requires_approval,
                        "roles": sorted(t.roles),
                        "input_schema": t.input_schema,
                    }
                )
        return out

    # -- the pipeline -------------------------------------------------------------------------------
    def call(
        self, caller: Caller, tool: str, args: dict[str, Any], *, context: dict[str, Any] | None = None, dry_run: bool = False
    ) -> GatewayResult:
        ctx = context or {}
        res = GatewayResult("allowed")
        steps: dict[str, CheckStep] = {n: CheckStep(n, "skipped") for n in self.PIPELINE}
        res.checks = list(steps.values())
        spec = self.tools.get(tool)

        def fail(step: str, detail: str, status: Status = "denied") -> GatewayResult:
            steps[step].status = "pending" if status == "pending_approval" else "fail"
            steps[step].detail = detail
            res.status, res.reason = status, detail
            return self._finish(res, caller, tool, args, steps, dry_run)

        def ok(step: str, detail: str = "") -> None:
            steps[step].status, steps[step].detail = "pass", detail

        if spec is None:
            return fail("registry", f"unknown tool {tool!r}")
        ok("registry")
        if caller.allowed_tools is not None and tool not in caller.allowed_tools:
            return fail("allow_list", f"{caller.id} may not call {tool}")
        ok("allow_list")
        if not caller.roles & spec.roles:
            return fail("role", f"roles {sorted(caller.roles)} cannot call {tool} (needs one of {sorted(spec.roles)})")
        ok("role")
        if spec.scope not in caller.scopes:
            return fail("scope", f"missing scope {spec.scope}")
        ok("scope")
        try:
            jsonschema.validate(args, spec.input_schema)
        except jsonschema.ValidationError as e:
            where = "/".join(str(p) for p in e.absolute_path) or "arguments"
            return fail("schema", f"{where}: {e.message[:200]}")
        ok("schema")
        if spec.tenant_arg and args.get(spec.tenant_arg) != caller.tenant_id:
            return fail("tenant_scope", f"{spec.tenant_arg} does not match the caller's tenant")
        ok("tenant_scope")
        if not dry_run and not self.limiter.allow(caller.id):
            return fail("rate_limit", "rate limit exceeded")
        ok("rate_limit")
        if self._budget_check:
            try:
                self._budget_check(caller, spec, args, ctx)
            except BudgetExceeded as e:
                return fail("budget", e.message)
        ok("budget")
        hit = self._scan(spec, args)
        if hit:
            if self.injection_mode == "deny":
                return fail("injection", f"suspicious content in argument {hit!r}")
            steps["injection"].status, steps["injection"].detail = "flagged", f"suspicious content in {hit!r} (flag mode)"
        else:
            ok("injection")

        approval_id: str | None = None
        if spec.requires_approval:
            if dry_run:
                has = self.approvals.db.one(
                    "SELECT id FROM approvals WHERE tenant_id=? AND tool=? AND status='approved' AND args_hash=?",
                    (caller.tenant_id, tool, args_hash(tool, args)),
                )
                if has is None:
                    return fail("approval", "approval required", "pending_approval")
                ok("approval", "approved approval on file (dry run: not consumed)")
            else:
                approval_id = self.approvals.consume(caller.tenant_id, tool, args)
                if approval_id is None:
                    res.approval_id = self.approvals.request(caller.tenant_id, tool, args, caller.id)
                    return fail("approval", "approval required", "pending_approval")
                ok("approval", approval_id)
        else:
            ok("approval", "not required")
        if spec.precondition:
            problem = self._preconditions[spec.precondition](caller, args) if spec.precondition in self._preconditions else None
            if problem:
                if approval_id:  # don't burn an approval on a call that could not run: hand it back
                    self.approvals.db.execute("UPDATE approvals SET status='approved' WHERE id=?", (approval_id,))
                return fail("precondition", problem)
        ok("precondition")

        if dry_run:
            steps["execute"].detail = "dry run: not executed"
            return self._finish(res, caller, tool, args, steps, dry_run)
        handler = self._handlers.get(tool)
        if handler is None:
            return fail("execute", f"no handler registered for {tool}")
        try:
            res.output = handler(caller, args, ctx)
        except Exception as e:  # tool failures are data for the agent, not crashes of the gateway
            steps["execute"].status, steps["execute"].detail = "fail", f"{type(e).__name__}: {e}"
            res.status, res.reason = "denied", f"tool error: {type(e).__name__}: {e}"
            res.output = None
            return self._finish(res, caller, tool, args, steps, dry_run, errored=True)
        ok("execute")
        return self._finish(res, caller, tool, args, steps, dry_run)

    def _scan(self, spec: ToolSpec, args: dict[str, Any]) -> str | None:
        for key, value in args.items():
            if spec.scan_args is not None and key not in spec.scan_args:
                continue
            if self._contains_injection(value):
                return key
        return None

    def _contains_injection(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(INJECTION.search(value))
        if isinstance(value, dict):
            return any(self._contains_injection(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_injection(v) for v in value)
        return False

    def _finish(
        self,
        res: GatewayResult,
        caller: Caller,
        tool: str,
        args: dict[str, Any],
        steps: dict[str, CheckStep],
        dry_run: bool,
        errored: bool = False,
    ) -> GatewayResult:
        self.metrics.gateway_calls.labels(tool if tool in self.tools else "unknown", res.status).inc()
        if not dry_run:
            # arguments are summarised, never stored verbatim: they can contain regulated source text
            summary = {
                k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 80 else v)
                for k, v in args.items()
                if k not in ("text", "source", "translation", "context", "repair", "tm_examples")
            }
            self.audit.append(
                caller.id,
                f"tool.{res.status}",
                tenant_id=caller.tenant_id,
                subject=tool,
                details={"args": summary, "reason": res.reason, "approval_id": res.approval_id, "errored": errored},
            )
            steps["audit"].status, steps["audit"].detail = "pass", "recorded"
        else:
            steps["audit"].detail = "dry run: not recorded"
        return res
