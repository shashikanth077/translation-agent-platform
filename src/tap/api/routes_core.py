"""Core workflow endpoints: documents, intake chat, jobs, review, approval."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Response

from ..domain.models import DocFormat, JobSpec, Principal
from ..errors import ValidationFailed
from ..parse.formats import detect_format
from ..platform import Platform
from ..services.catalog import CATALOG
from .deps import get_platform, require
from .schemas import DocumentIn, JobRequest, MessageIn, ReviewSubmit

router = APIRouter()


def _job_view(p: Platform, job: dict[str, Any]) -> dict[str, Any]:
    progress = p.jobs.progress(job["id"])
    return {
        "id": job["id"],
        "status": job["status"],
        "policy": job["policy"],
        "glossary_version": job["glossary_version"],
        "spec": job["spec"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "created_by": job["created_by"],
        "error": job["error"],
        "analysis": job["analysis"],
        "progress": progress,
        "open_review_items": p.review.open_blocking(job["id"]),
        "outputs": [{"lang": o["lang"], "checks_ok": bool(o["checks_ok"]), "issues": o["issues"]} for o in p.jobs.outputs(job["id"])],
    }


# -- documents -------------------------------------------------------------------------------------
@router.post("/documents", status_code=201, tags=["documents"])
def upload_document(
    body: DocumentIn, principal: Principal = Depends(require("jobs:write")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    fmt = DocFormat(body.format) if body.format else detect_format(body.filename, body.content)
    from ..parse import formats

    parsed = formats.parse(fmt, body.content)  # reject unparseable/empty files at upload, not mid-workflow
    doc_id = p.jobs.create_document(principal.tenant_id, body.filename, fmt, body.content)
    return {"id": doc_id, "filename": body.filename, "format": fmt.value, "segments": len(parsed.segments)}


@router.get("/documents", tags=["documents"])
def list_documents(principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)) -> list[dict[str, Any]]:
    return p.jobs.list_documents(principal.tenant_id)


# -- conversational intake -------------------------------------------------------------------------
@router.post("/intake/sessions", status_code=201, tags=["intake"])
def intake_create(principal: Principal = Depends(require("intake")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.intake.create(principal)


@router.get("/intake/sessions/{sid}", tags=["intake"])
def intake_get(sid: str, principal: Principal = Depends(require("intake")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.intake.view(sid, principal.tenant_id)


@router.post("/intake/sessions/{sid}/messages", tags=["intake"])
def intake_message(
    sid: str, body: MessageIn, principal: Principal = Depends(require("intake")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    return p.intake.message(sid, principal, body.message)


@router.post("/intake/sessions/{sid}/start", status_code=202, tags=["intake"])
def intake_start(sid: str, principal: Principal = Depends(require("jobs:write")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    spec = p.intake.confirmed_spec(sid, principal)
    job = p.runner.submit(principal, spec, idem_key=f"intake:{sid}")  # replaying "start" returns the same job
    p.intake.attach_job(sid, job["id"])
    return _job_view(p, job)


# -- jobs ------------------------------------------------------------------------------------------
@router.post("/jobs", status_code=202, tags=["jobs"])
def create_job(
    body: JobRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    principal: Principal = Depends(require("jobs:write")),
    p: Platform = Depends(get_platform),
) -> dict[str, Any]:
    if not idempotency_key or len(idempotency_key) < 8:
        raise ValidationFailed("Idempotency-Key header (min 8 characters) is required so retries cannot create duplicate jobs")
    spec = JobSpec(tenant_id=principal.tenant_id, **body.model_dump())
    job = p.runner.submit(principal, spec, idem_key=idempotency_key)
    return _job_view(p, job)


@router.get("/jobs", tags=["jobs"])
def list_jobs(
    status: str | None = None, principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)
) -> list[dict[str, Any]]:
    return [_job_view(p, j) for j in p.jobs.list_jobs(principal.tenant_id, status)]


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str, principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return _job_view(p, p.jobs.get(job_id, principal.tenant_id))


@router.get("/jobs/{job_id}/segments", tags=["jobs"])
def job_segments(
    job_id: str,
    lang: str | None = None,
    status: str | None = None,
    include_trace: bool = False,
    principal: Principal = Depends(require("jobs:read")),
    p: Platform = Depends(get_platform),
) -> list[dict[str, Any]]:
    p.jobs.get(job_id, principal.tenant_id)
    out = []
    for s in p.jobs.segments(job_id, lang, status):
        d = {
            k: s[k]
            for k in (
                "lang",
                "seg_id",
                "idx",
                "source",
                "words",
                "risk",
                "status",
                "target",
                "path",
                "decision",
                "composite",
                "reason",
                "cost_units",
                "cacheable",
                "approved_by",
                "repeat_of",
            )
        }
        detail = s["detail"] or {}
        d["checks"], d["findings"], d["scores"] = detail.get("checks"), detail.get("findings"), detail.get("scores")
        if include_trace:
            d["trace"] = detail.get("trace")
        out.append(d)
    return out


@router.get("/jobs/{job_id}/output/{lang}", tags=["jobs"])
def job_output(
    job_id: str, lang: str, principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)
) -> Response:
    return Response(p.runner.result(principal, job_id, lang), media_type="text/plain; charset=utf-8")


@router.post("/jobs/{job_id}/approve", tags=["jobs"])
def approve_job(
    job_id: str, reason: str = "", principal: Principal = Depends(require("publish")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    return _job_view(p, p.runner.approve(principal, job_id, reason=reason))


@router.post("/jobs/{job_id}/retry", tags=["jobs"])
def retry_job(job_id: str, principal: Principal = Depends(require("admin")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return _job_view(p, p.runner.retry(principal, job_id))


@router.get("/jobs/{job_id}/report", tags=["jobs"])
def job_report(
    job_id: str, principal: Principal = Depends(require("metering:read")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    p.jobs.get(job_id, principal.tenant_id)
    return p.metering.report(principal.tenant_id, job_id=job_id)


# -- review ----------------------------------------------------------------------------------------
@router.get("/review/items", tags=["review"])
def review_items(
    status: str | None = "pending",
    kind: str | None = None,
    job_id: str | None = None,
    principal: Principal = Depends(require("review")),
    p: Platform = Depends(get_platform),
) -> list[dict[str, Any]]:
    return p.review.list_items(principal.tenant_id, status=status, kind=kind, job_id=job_id)


@router.get("/review/items/{item_id}", tags=["review"])
def review_item(item_id: str, principal: Principal = Depends(require("review")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.review.get(item_id, principal.tenant_id)


@router.post("/review/items/{item_id}/claim", tags=["review"])
def review_claim(item_id: str, principal: Principal = Depends(require("review")), p: Platform = Depends(get_platform)) -> dict[str, Any]:
    return p.review.claim(item_id, principal.tenant_id, principal.subject)


@router.post("/review/items/{item_id}/submit", tags=["review"])
def review_submit(
    item_id: str, body: ReviewSubmit, principal: Principal = Depends(require("review")), p: Platform = Depends(get_platform)
) -> dict[str, Any]:
    return p.runner.submit_review(principal, item_id, text=body.text, note=body.note, override=body.override)


# -- catalogue -------------------------------------------------------------------------------------
@router.get("/platform/services", tags=["platform"])
def services(_: Principal = Depends(require("jobs:read", "admin"))) -> list[dict[str, Any]]:
    return [e.as_dict() for e in CATALOG]


@router.get("/notifications", tags=["platform"])
def notifications(
    limit: int = Query(50, le=200), principal: Principal = Depends(require("jobs:read")), p: Platform = Depends(get_platform)
) -> list[dict[str, Any]]:
    return p.notifications.list_recent(principal.tenant_id, limit)
