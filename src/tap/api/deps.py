"""FastAPI dependencies: platform access, authentication, authorisation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi import Depends, Header, Request

from ..domain.models import Principal
from ..errors import Forbidden
from ..platform import Platform
from ..services.identity import IdentityService


def get_platform(request: Request) -> Platform:
    return cast(Platform, request.app.state.platform)


def get_principal(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key"), p: Platform = Depends(get_platform)
) -> Principal:
    principal = p.identity.authenticate(x_api_key)
    request.state.principal = principal
    return principal


def require(*perms: str) -> Callable[..., Principal]:
    """Authorise if the caller has *any* of the listed permissions."""

    def dep(principal: Principal = Depends(get_principal)) -> Principal:
        have = IdentityService.permissions(principal)
        if perms and not (have & set(perms)):
            raise Forbidden(f"requires one of: {', '.join(perms)}")
        return principal

    return dep
