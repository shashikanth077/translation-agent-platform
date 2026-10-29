"""Translation memory: tenant-scoped storage, fuzzy matching with rapidfuzz, TMX import/export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as DET
from rapidfuzz import fuzz, process

from ..db import Database
from ..errors import ValidationFailed
from ..parse.tokens import numbers_and_units
from ..util import normalise_text, now_iso

SCHEMA = [
    """
    CREATE TABLE tm_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        src_lang TEXT NOT NULL,
        tgt_lang TEXT NOT NULL,
        domain TEXT NOT NULL,
        glossary_version TEXT NOT NULL,
        source TEXT NOT NULL,
        source_norm TEXT NOT NULL,
        target TEXT NOT NULL,
        origin TEXT NOT NULL,
        approved_by TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (tenant_id, src_lang, tgt_lang, domain, glossary_version, source_norm)
    );
    CREATE INDEX idx_tm_lookup ON tm_entries(tenant_id, src_lang, tgt_lang)
    """
]
MAX_SCAN = 20_000


@dataclass(frozen=True)
class TMMatch:
    id: int
    source: str
    target: str
    pct: int
    domain: str
    glossary_version: str
    origin: str
    numbers_differ: bool  # fuzzy match whose numbers differ from the query: the dangerous kind

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TranslationMemory:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("tm", SCHEMA)

    def add(
        self,
        tenant_id: str,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        source: str,
        target: str,
        *,
        glossary_version: str = "v0",
        origin: str = "human",
        approved_by: str | None = None,
    ) -> int:
        norm = normalise_text(source)
        if not norm or not target.strip():
            raise ValidationFailed("TM entries need non-empty source and target")
        self.db.execute(
            "INSERT INTO tm_entries(tenant_id, src_lang, tgt_lang, domain, glossary_version, source, source_norm,"
            " target, origin, approved_by, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(tenant_id, src_lang, tgt_lang, domain, glossary_version, source_norm) "
            "DO UPDATE SET target=excluded.target, origin=excluded.origin, approved_by=excluded.approved_by, "
            "created_at=excluded.created_at",
            (tenant_id, src_lang, tgt_lang, domain, glossary_version, source, norm, target, origin, approved_by, now_iso()),
        )
        return int(
            self.db.scalar(
                "SELECT id FROM tm_entries WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND domain=? AND "
                "glossary_version=? AND source_norm=?",
                (tenant_id, src_lang, tgt_lang, domain, glossary_version, norm),
            )
        )

    def exact(self, tenant_id: str, src_lang: str, tgt_lang: str, domain: str, glossary_version: str, source: str) -> TMMatch | None:
        row = self.db.one(
            "SELECT * FROM tm_entries WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND domain=? AND "
            "glossary_version=? AND source_norm=?",
            (tenant_id, src_lang, tgt_lang, domain, glossary_version, normalise_text(source)),
        )
        return (
            None
            if row is None
            else TMMatch(row["id"], row["source"], row["target"], 100, row["domain"], row["glossary_version"], row["origin"], False)
        )

    def search(
        self,
        tenant_id: str,
        src_lang: str,
        tgt_lang: str,
        source: str,
        *,
        min_pct: int = 75,
        limit: int = 5,
        domain: str | None = None,
    ) -> list[TMMatch]:
        query = normalise_text(source)
        n = len(query)
        sql = (
            "SELECT id, source, source_norm, target, domain, glossary_version, origin FROM tm_entries "
            "WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND length(source_norm) BETWEEN ? AND ?"
        )
        params: list[Any] = [tenant_id, src_lang, tgt_lang, int(n * (min_pct / 100)) - 1, int(n / max(min_pct / 100, 0.01)) + 1]
        if domain:
            sql += " AND domain=?"
            params.append(domain)
        rows = self.db.query(sql + f" LIMIT {MAX_SCAN}", tuple(params))
        if not rows:
            return []
        hits = process.extract(query, [r["source_norm"] for r in rows], scorer=fuzz.ratio, score_cutoff=min_pct, limit=limit * 3)
        q_nums = numbers_and_units(source)[0]
        out: list[TMMatch] = []
        for _text, score, idx in hits:
            r = rows[idx]
            out.append(
                TMMatch(
                    r["id"],
                    r["source"],
                    r["target"],
                    int(score),
                    r["domain"],
                    r["glossary_version"],
                    r["origin"],
                    numbers_and_units(r["source"])[0] != q_nums,
                )
            )
        # best first; prefer same-domain / human-approved on ties
        out.sort(key=lambda m: (-m.pct, m.origin != "human", m.id))
        return out[:limit]

    def count(self, tenant_id: str) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM tm_entries WHERE tenant_id=?", (tenant_id,)) or 0)

    def containing(self, tenant_id: str, src_lang: str, tgt_lang: str, needle: str, limit: int = 1000) -> list[dict[str, Any]]:
        like = f"%{needle.lower().replace('%', '').replace('_', '')}%"
        rows = self.db.query(
            "SELECT id, source, target, glossary_version FROM tm_entries WHERE tenant_id=? AND src_lang=? AND tgt_lang=? "
            "AND lower(source_norm) LIKE ? LIMIT ?",
            (tenant_id, src_lang, tgt_lang, like, limit),
        )
        return [dict(r) for r in rows]

    # -- TMX 1.4b ---------------------------------------------------------------------------------
    def export_tmx(self, tenant_id: str, src_lang: str, tgt_lang: str) -> str:
        tmx = ET.Element("tmx", version="1.4")
        ET.SubElement(
            tmx,
            "header",
            creationtool="translation-agent-platform",
            creationtoolversion="0.1",
            segtype="sentence",
            adminlang="en",
            srclang=src_lang,
            datatype="plaintext",
            o_tmf="tap",
        )
        body = ET.SubElement(tmx, "body")
        for r in self.db.query(
            "SELECT source, target, domain FROM tm_entries WHERE tenant_id=? AND src_lang=? AND tgt_lang=? ORDER BY id",
            (tenant_id, src_lang, tgt_lang),
        ):
            tu = ET.SubElement(body, "tu")
            ET.SubElement(tu, "prop", type="x-domain").text = r["domain"]
            for lang, text in ((src_lang, r["source"]), (tgt_lang, r["target"])):
                tuv = ET.SubElement(tu, "tuv", {"xml:lang": lang})
                ET.SubElement(tuv, "seg").text = text
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(tmx, encoding="unicode") + "\n"

    def import_tmx(
        self,
        tenant_id: str,
        content: str,
        *,
        default_domain: str = "imported",
        glossary_version: str = "v0",
        approved_by: str | None = None,
    ) -> dict[str, int]:
        try:
            root = DET.fromstring(content.encode("utf-8"))
        except Exception as e:
            raise ValidationFailed(f"invalid TMX: {e}") from e
        xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
        header = root.find("header")
        src_default = (header.get("srclang") if header is not None else None) or "en"
        added = skipped = 0
        for tu in root.iter("tu"):
            domain = next((p.text for p in tu.findall("prop") if p.get("type") == "x-domain" and p.text), default_domain)
            segs = {
                (tuv.get(xml_lang) or tuv.get("lang") or "").split("-")[0].lower(): "".join(tuv.find("seg").itertext()).strip()
                for tuv in tu.findall("tuv")
                if tuv.find("seg") is not None
            }
            src = segs.get(src_default.split("-")[0].lower())
            others = {k: v for k, v in segs.items() if k != src_default.split("-")[0].lower() and v}
            if not src or not others:
                skipped += 1
                continue
            for lang, tgt in others.items():
                self.add(
                    tenant_id,
                    src_default,
                    lang,
                    domain,
                    src,
                    tgt,
                    glossary_version=glossary_version,
                    origin="import",
                    approved_by=approved_by,
                )
                added += 1
        return {"added": added, "skipped": skipped}
