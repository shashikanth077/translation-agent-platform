"""Agent-side tool client: every model/engine/scoring action of the workflow goes through the MCP gateway."""

from __future__ import annotations

from typing import Any

from ..mcp.gateway import GatewayResult, MCPGateway
from ..mcp.registry import Caller


class ToolRefused(Exception):
    """The gateway did not allow the call. ``reason`` is safe to log and to show to reviewers."""

    def __init__(self, tool: str, result: GatewayResult) -> None:
        super().__init__(f"{tool}: {result.status}: {result.reason}")
        self.tool, self.result = tool, result

    @property
    def reason(self) -> str:
        return self.result.reason

    @property
    def failed_step(self) -> str:
        return next((c.name for c in self.result.checks if c.status in ("fail", "pending")), "unknown")


class ToolClient:
    def __init__(self, gateway: MCPGateway, agent: str = "orchestrator") -> None:
        self.gateway, self.agent = gateway, agent

    def call(self, tool: str, args: dict[str, Any], *, tenant_id: str, context: dict[str, Any] | None = None) -> Any:
        result = self.gateway.call(Caller.agent(self.agent, tenant_id), tool, args, context=context)
        if result.status != "allowed":
            raise ToolRefused(tool, result)
        return result.output
