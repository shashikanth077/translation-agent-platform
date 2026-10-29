"""FastAPI application factory: middleware, error mapping, health/metrics, static console."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from .. import __version__
from ..config import Settings
from ..errors import TapError
from ..mcp.gateway import RateLimiter
from ..observability import bind, configure_logging, unbind
from ..platform import Platform
from . import routes_core, routes_labs

log = logging.getLogger("tap.api")
STATIC = Path(__file__).parent / "static"
API = "/api/v1"


def create_app(platform: Platform | None = None, settings: Settings | None = None) -> FastAPI:
    owned = platform is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        s = settings or Settings()
        configure_logging(s.log_level)
        app.state.platform = platform or Platform.create(s)
        if owned:
            app.state.platform.start()
        try:
            yield
        finally:
            if owned:
                app.state.platform.stop()

    app = FastAPI(
        title="Translation Agent Platform",
        version=__version__,
        lifespan=lifespan,
        description="Agentic NMT + LLM translation: intake -> LangGraph workflow -> consensus QA -> review -> approved delivery.",
    )
    if platform is not None:  # tests construct the platform themselves; available before lifespan too
        app.state.platform = platform
    limiter = RateLimiter((settings or (platform.settings if platform else Settings())).api_rate_limit_per_min)
    cors = (settings or (platform.settings if platform else Settings())).cors_origins
    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=cors, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = bind(request_id=rid)
        started = time.monotonic()
        key = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
        if request.url.path.startswith(API) and not limiter.allow(key):
            unbind(token)
            return JSONResponse(
                {"error": {"code": "rate_limited", "message": "too many requests"}}, status_code=429, headers={"Retry-After": "1"}
            )
        try:
            response = await call_next(request)
        finally:
            unbind(token)
        response.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        plat = getattr(request.app.state, "platform", None)
        if plat and route is not None:
            plat.metrics.http_seconds.labels(request.method, getattr(route, "path", "unknown"), str(response.status_code)).observe(
                time.monotonic() - started
            )
        return response

    @app.exception_handler(TapError)
    async def tap_error(_: Request, exc: TapError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, status_code=exc.http_status)

    @app.exception_handler(ValidationError)
    async def pydantic_error(_: Request, exc: ValidationError) -> JSONResponse:
        errs = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            {"error": {"code": "validation_failed", "message": "invalid request", "details": {"errors": errs}}}, status_code=422
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": "internal_error", "message": "internal error"}}, status_code=500)

    app.include_router(routes_core.router, prefix=API)
    app.include_router(routes_labs.router, prefix=API)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz(request: Request) -> JSONResponse:
        p: Platform = request.app.state.platform
        checks = {"database": bool(p.db.scalar("SELECT 1")), "cache": p.cache.ping(), "event_dead_letters": len(p.bus.dead_letters()) == 0}
        ok = checks["database"] and checks["cache"]
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics", tags=["ops"])
    def metrics(request: Request) -> Response:
        p: Platform = request.app.state.platform
        p.metrics.review_queue.set(p.review.depth())
        p.metrics.dead_letters.set(len(p.bus.dead_letters()))
        return Response(generate_latest(p.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    if STATIC.exists():
        app.mount("/console", StaticFiles(directory=STATIC, html=True), name="console")

        @app.get("/", include_in_schema=False)
        def root() -> Response:
            return Response(status_code=307, headers={"Location": "/console/"})

    return app
