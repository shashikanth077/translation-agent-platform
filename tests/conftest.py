from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tap.config import Settings
from tap.domain.models import DocFormat, Domain, JobSpec, PolicyName, Principal
from tap.platform import Platform


def principal(role: str = "admin", tenant: str = "acme", subject: str | None = None) -> Principal:
    return Principal(subject=subject or f"{role}@{tenant}", tenant_id=tenant, roles=frozenset({role}))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", engine_retry_initial_s=0, relay_interval_s=0.05)


@pytest.fixture
def platform(settings: Settings) -> Iterator[Platform]:
    p = Platform.create(settings)
    p.seed_demo("acme")
    yield p
    p.stop()


@pytest.fixture
def admin() -> Principal:
    return principal("admin")


@pytest.fixture
def linguist() -> Principal:
    return principal("linguist")


@pytest.fixture
def client_user() -> Principal:
    return principal("client")


def make_job(
    p: Platform,
    text: str,
    *,
    tenant: str = "acme",
    langs: list[str] | None = None,
    domain: Domain = Domain.CLINICAL,
    policy: PolicyName | None = PolicyName.BALANCED,
    fmt: DocFormat = DocFormat.TXT,
    filename: str = "doc.txt",
    key: str = "idem-key-1",
    who: Principal | None = None,
    **extra,
) -> dict:  # noqa: ANN003
    who = who or principal("admin", tenant)
    doc = p.jobs.create_document(tenant, filename, fmt, text)
    spec = JobSpec(tenant_id=tenant, document_id=doc, source_lang="en", target_langs=langs or ["de"], domain=domain, policy=policy, **extra)
    job = p.runner.submit(who, spec, idem_key=key)
    assert p.runner.wait_idle(60)
    return p.jobs.get(job["id"])


def resolve_all(p: Platform, who: Principal, tenant: str = "acme", edit=None) -> int:  # noqa: ANN001
    """Resolve every pending blocking review item (optionally editing text via ``edit(payload) -> str``)."""
    n = 0
    for it in p.review.list_items(tenant, status="pending", kind="review"):
        text = edit(it["payload"]) if edit else it["payload"]["candidate"]
        p.review.claim(it["id"], tenant, who.subject)
        p.runner.submit_review(who, it["id"], text=text)
        n += 1
    assert p.runner.wait_idle(60)
    return n
