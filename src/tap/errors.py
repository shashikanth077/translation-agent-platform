"""Exception hierarchy shared by all services."""

from __future__ import annotations


class TapError(Exception):
    """Base class for all platform errors."""

    code = "tap_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(TapError):
    code = "not_found"
    http_status = 404


class ConflictError(TapError):
    code = "conflict"
    http_status = 409


class ValidationFailed(TapError):
    code = "validation_failed"
    http_status = 422


class Unauthorized(TapError):
    code = "unauthorized"
    http_status = 401


class Forbidden(TapError):
    code = "forbidden"
    http_status = 403


class EntitlementError(Forbidden):
    code = "entitlement_denied"


class BudgetExceeded(TapError):
    code = "budget_exceeded"
    http_status = 402


class ProviderError(TapError):
    """A translation or scoring provider failed (retryable)."""

    code = "provider_error"
    http_status = 502


class AllEnginesUnavailable(ProviderError):
    code = "all_engines_unavailable"
    http_status = 503
