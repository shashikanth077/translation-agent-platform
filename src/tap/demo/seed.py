"""Seed a tenant with demo TM entries and a glossary so the reuse and terminology paths are visible."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..assets.terminology import TermChange
from .corpus import GLOSSARY_SEED, TM_SEED

if TYPE_CHECKING:
    from ..platform import Platform


def seed(p: Platform, tenant_id: str = "acme") -> dict[str, int]:
    version = p.terminology.latest_version(tenant_id)
    if version == 0:
        version = p.terminology.commit(
            tenant_id, [TermChange("add", s, t, term, tr) for s, t, term, tr in GLOSSARY_SEED], note="demo glossary", actor="seed"
        )
    added = 0
    for domain, src, tgt, lang in TM_SEED:
        p.tm.add(tenant_id, "en", lang, domain, src, tgt, glossary_version=f"v{version}", origin="human", approved_by="seed")
        added += 1
    return {"glossary_version": version, "tm_entries": added}
