"""Identity and tenant service: API-key authentication, role scopes and tenant entitlements."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..domain.models import Domain, JobSpec, Principal
from ..errors import EntitlementError, Unauthorized

# Roles -> API permissions. The MCP tool scopes (read:tm, ...) are separate and live in tap.mcp.registry.
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "client": frozenset({"jobs:write", "jobs:read", "assets:read", "metering:read", "intake"}),
    "linguist": frozenset({"jobs:read", "review", "assets:read", "publish"}),
    "admin": frozenset(
        {
            "jobs:write",
            "jobs:read",
            "assets:read",
            "assets:write",
            "metering:read",
            "intake",
            "review",
            "publish",
            "admin",
            "experiments",
            "audit:read",
            "approvals",
        }
    ),
}


@dataclass
class TenantProfile:
    """Entitlements checked at intake: what a client may ask the platform to do."""

    tenant_id: str
    name: str
    source_langs: set[str] | None = None  # None = any
    target_langs: set[str] | None = None
    domains: set[str] = field(default_factory=lambda: {d.value for d in Domain})
    allows_personal_data: bool = False
    job_budget_units: float | None = None


DEV_TENANTS = [
    TenantProfile("acme", "Acme Pharma (dev)", target_langs={"de", "fr", "es", "it", "pt", "nl", "ja"}),
    TenantProfile("beta", "Beta Biotech (dev)", target_langs={"de", "fr"}, domains={"ui", "marketing"}),
]
DEV_KEYS: list[dict[str, Any]] = [
    {"key": "dev-admin", "subject": "admin@acme", "tenant_id": "acme", "roles": ["admin"]},
    {"key": "dev-client", "subject": "pm@acme", "tenant_id": "acme", "roles": ["client"]},
    {"key": "dev-linguist", "subject": "linguist@acme", "tenant_id": "acme", "roles": ["linguist"]},
    {"key": "dev-beta", "subject": "pm@beta", "tenant_id": "beta", "roles": ["client"]},
]


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class IdentityService:
    def __init__(self, settings: Settings, tenants: list[TenantProfile] | None = None) -> None:
        keys = settings.parsed_api_keys() or (DEV_KEYS if settings.env != "prod" else [])
        self._keys = {
            _digest(k["key"]): Principal(subject=k["subject"], tenant_id=k["tenant_id"], roles=frozenset(k["roles"]), kind="user")
            for k in keys
        }
        self._tenants = {t.tenant_id: t for t in (tenants if tenants is not None else DEV_TENANTS)}
        for k in keys:  # unknown tenants referenced by keys get a permissive default profile
            self._tenants.setdefault(k["tenant_id"], TenantProfile(k["tenant_id"], k["tenant_id"]))

    def authenticate(self, api_key: str | None) -> Principal:
        if not api_key:
            raise Unauthorized("missing API key")
        digest = _digest(api_key)
        for known, principal in self._keys.items():
            if hmac.compare_digest(known, digest):
                return principal
        raise Unauthorized("invalid API key")

    @staticmethod
    def permissions(principal: Principal) -> frozenset[str]:
        perms: set[str] = set()
        for role in principal.roles:
            perms |= ROLE_PERMISSIONS.get(role, frozenset())
        return frozenset(perms)

    def tenant(self, tenant_id: str) -> TenantProfile:
        return self._tenants.get(tenant_id) or TenantProfile(tenant_id, tenant_id)

    def check_entitlements(self, spec: JobSpec) -> None:
        """Reject early with a clear reason: cheaper than failing mid-workflow."""
        t = self.tenant(spec.tenant_id)
        if t.source_langs is not None and spec.source_lang.split("-")[0] not in t.source_langs:
            raise EntitlementError(f"source language {spec.source_lang!r} is not enabled for this client")
        if t.target_langs is not None:
            bad = [x for x in spec.target_langs if x.split("-")[0] not in t.target_langs]
            if bad:
                raise EntitlementError(f"target languages not enabled for this client: {', '.join(bad)}")
        if spec.domain.value not in t.domains:
            raise EntitlementError(f"domain {spec.domain.value!r} is not enabled for this client")
