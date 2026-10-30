"""Runtime configuration. Every setting can be overridden with a ``TAP_`` environment variable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Price per 1,000 words for each processing step, in abstract "cost units" (read: currency).
# These are illustrative defaults: replace them with your contracted vendor and infrastructure rates.
DEFAULT_PRICES: dict[str, float] = {
    "reuse_exact": 0.0,
    "reuse_fuzzy": 2.0,
    "nmt": 4.0,
    "llm": 25.0,
    "score": 1.0,
    "human": 30.0,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAP_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    data_dir: Path = Path(".tap-data")
    db_path: Path | None = None
    checkpoint_path: Path | None = None

    # --- backends -------------------------------------------------------------------------
    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str | None = None
    # "demo" runs fully offline with deterministic stand-ins so the whole platform can be exercised
    # without API keys or GPUs. Switch to real providers for production use.
    nmt_backend: Literal["demo", "libretranslate"] = "demo"
    llm_backend: Literal["demo", "anthropic"] = "demo"
    quality_backend: Literal["demo", "heuristic", "llm", "ml"] = "demo"
    intake_backend: Literal["rules", "llm"] = "rules"
    libretranslate_url: str = "http://localhost:5000"
    libretranslate_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_judge_model: str | None = None  # defaults to llm_model
    llm_timeout_s: float = 30.0

    # --- resilience -----------------------------------------------------------------------
    engine_retry_attempts: int = 3
    engine_retry_initial_s: float = 0.5
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0

    # --- workflow policy ------------------------------------------------------------------
    max_repair_attempts: int = 2
    tm_draft_min_pct: int = 90
    tm_context_min_pct: int = 75
    weights: dict[str, float] = {"comet": 0.4, "semantic": 0.4, "fluency": 0.2}
    accept_threshold: float = 0.85
    critical_semantic: float = 0.60
    max_disagreement: float = 0.25
    urgent_hours: float = 6.0
    worker_threads: int = 4
    segment_threads: int = 4
    default_job_budget_units: float = 5000.0

    # --- cost model -----------------------------------------------------------------------
    prices: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PRICES))
    baseline_cost_per_1k_words: float = 40.0

    # --- security -------------------------------------------------------------------------
    # JSON list of {"key": ..., "subject": ..., "tenant_id": ..., "roles": [...]}. Required in prod.
    api_keys_json: str | None = None
    api_rate_limit_per_min: int = 600
    injection_mode: Literal["deny", "flag"] = "deny"
    gateway_rate_limit_per_min: int = 1200

    # --- misc -----------------------------------------------------------------------------
    cors_origins: list[str] = []
    relay_interval_s: float = 0.25
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.env == "prod":
            if not self.api_keys_json:
                raise ValueError("TAP_API_KEYS_JSON must be set when TAP_ENV=prod (no built-in dev keys in prod)")
            if self.llm_backend == "demo" or self.nmt_backend == "demo" or self.quality_backend == "demo":
                raise ValueError("demo backends are not allowed when TAP_ENV=prod")
        if self.cache_backend == "redis" and not self.redis_url:
            raise ValueError("TAP_REDIS_URL is required when TAP_CACHE_BACKEND=redis")
        if self.llm_backend == "anthropic" and not self.anthropic_api_key:
            raise ValueError("TAP_ANTHROPIC_API_KEY is required when TAP_LLM_BACKEND=anthropic")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("TAP_WEIGHTS must sum to 1.0")
        return self

    @property
    def database_file(self) -> Path:
        return self.db_path or self.data_dir / "tap.db"

    @property
    def checkpoint_file(self) -> Path:
        return self.checkpoint_path or self.data_dir / "checkpoints.db"

    def parsed_api_keys(self) -> list[dict]:
        return json.loads(self.api_keys_json) if self.api_keys_json else []
