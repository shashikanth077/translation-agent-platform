"""Terminology service: versioned glossaries, required/forbidden/do-not-translate terms and term checks.

Every change goes through ``commit`` and produces a new glossary version. Versions are part of the cache
key, so a term update invalidates dependent cache entries by construction; ``impact`` lists the TM entries
that contain a changed term so they can be re-translated or flagged for review.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rapidfuzz import fuzz

from ..db import Database
from ..errors import ValidationFailed
from ..parse.tokens import strip_accents
from ..util import now_iso

SCHEMA = [
    """
    CREATE TABLE glossary_versions (
        tenant_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        note TEXT,
        actor TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (tenant_id, version)
    );
    CREATE TABLE terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        src_lang TEXT NOT NULL,
        tgt_lang TEXT NOT NULL,
        term TEXT NOT NULL,
        translation TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'preferred',
        added_in INTEGER NOT NULL,
        removed_in INTEGER
    );
    CREATE INDEX idx_terms_lookup ON terms(tenant_id, src_lang, tgt_lang)
    """
]
Kind = Literal["preferred", "forbidden", "dnt"]


@dataclass(frozen=True)
class Term:
    term: str
    translation: str
    kind: Kind = "preferred"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TermChange:
    op: Literal["add", "remove"]
    src_lang: str
    tgt_lang: str
    term: str
    translation: str = ""
    kind: Kind = "preferred"


def parse_version(label: str | None, latest: int) -> int:
    if label is None:
        return latest
    if not re.fullmatch(r"v\d+", label):
        raise ValidationFailed(f"bad glossary version {label!r}")
    return int(label[1:])


def _term_regex(term: str) -> re.Pattern[str]:
    flags = 0 if (term.isupper() and len(term) <= 6) else re.IGNORECASE  # acronyms are case-sensitive
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags)


def _contains_term(target: str, translation: str) -> bool:
    """Inflection-tolerant containment: German/Slavic terms change endings, so allow near-matches."""
    t, tr = strip_accents(target.lower()), strip_accents(translation.lower())
    if tr in t:
        return True
    words = tr.split()
    toks = re.findall(r"[\w'-]+", t)
    n = len(words)
    return any(fuzz.ratio(" ".join(toks[i : i + n]), tr) >= 88 for i in range(len(toks) - n + 1))


class Terminology:
    def __init__(self, db: Database) -> None:
        self.db = db
        db.migrate("terminology", SCHEMA)

    def latest_version(self, tenant_id: str) -> int:
        return int(self.db.scalar("SELECT MAX(version) FROM glossary_versions WHERE tenant_id=?", (tenant_id,)) or 0)

    def commit(self, tenant_id: str, changes: list[TermChange], *, note: str = "", actor: str = "system") -> int:
        if not changes:
            raise ValidationFailed("no term changes")
        with self.db.tx():
            version = self.latest_version(tenant_id) + 1
            self.db.execute(
                "INSERT INTO glossary_versions(tenant_id, version, note, actor, created_at) VALUES(?,?,?,?,?)",
                (tenant_id, version, note, actor, now_iso()),
            )
            for c in changes:
                if c.op == "remove":
                    self.db.execute(
                        "UPDATE terms SET removed_in=? WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND lower(term)=lower(?) "
                        "AND removed_in IS NULL",
                        (version, tenant_id, c.src_lang, c.tgt_lang, c.term),
                    )
                    continue
                if not c.term.strip() or (c.kind != "dnt" and not c.translation.strip()):
                    raise ValidationFailed("terms need a term and a translation")
                # an update is a remove + add within one version
                self.db.execute(
                    "UPDATE terms SET removed_in=? WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND lower(term)=lower(?) "
                    "AND kind=? AND removed_in IS NULL",
                    (version, tenant_id, c.src_lang, c.tgt_lang, c.term, c.kind),
                )
                self.db.execute(
                    "INSERT INTO terms(tenant_id, src_lang, tgt_lang, term, translation, kind, added_in) VALUES(?,?,?,?,?,?,?)",
                    (tenant_id, c.src_lang, c.tgt_lang, c.term, c.translation or c.term, c.kind, version),
                )
        return version

    def terms_at(self, tenant_id: str, src_lang: str, tgt_lang: str, version: int | None = None) -> list[Term]:
        v = self.latest_version(tenant_id) if version is None else version
        rows = self.db.query(
            "SELECT term, translation, kind FROM terms WHERE tenant_id=? AND src_lang=? AND tgt_lang=? AND added_in<=? "
            "AND (removed_in IS NULL OR removed_in>?) ORDER BY length(term) DESC, term",
            (tenant_id, src_lang.split("-")[0], tgt_lang.split("-")[0], v, v),
        )
        return [Term(r["term"], r["translation"], r["kind"]) for r in rows]

    def applicable(self, source: str, terms: list[Term]) -> list[Term]:
        """Terms whose source side occurs in ``source``."""
        return [t for t in terms if _term_regex(t.term).search(source)]

    def check(self, source: str, target: str, terms: list[Term]) -> list[str]:
        findings: list[str] = []
        for t in self.applicable(source, terms):
            if t.kind == "preferred" and not _contains_term(target, t.translation):
                findings.append(f"required term missing: '{t.term}' -> '{t.translation}'")
            elif t.kind == "dnt" and t.term.lower() not in target.lower():
                findings.append(f"do-not-translate term altered: '{t.term}'")
            elif t.kind == "forbidden" and _contains_term(target, t.translation):
                findings.append(f"forbidden term used: '{t.translation}'")
        return findings

    def impact(self, tenant_id: str, tm: Any, src_lang: str, tgt_lang: str, changed_terms: list[str]) -> list[dict[str, Any]]:
        """TM entries whose source contains a changed term (candidates for re-translation)."""
        seen: dict[int, dict[str, Any]] = {}
        for term in changed_terms:
            for row in tm.containing(tenant_id, src_lang, tgt_lang, term):
                seen.setdefault(row["id"], {**row, "term": term})
        return list(seen.values())

    def export_csv_rows(self, tenant_id: str, version: int | None = None) -> list[dict[str, str]]:
        v = self.latest_version(tenant_id) if version is None else version
        rows = self.db.query(
            "SELECT src_lang, tgt_lang, term, translation, kind FROM terms WHERE tenant_id=? AND added_in<=? "
            "AND (removed_in IS NULL OR removed_in>?) ORDER BY src_lang, tgt_lang, term",
            (tenant_id, v, v),
        )
        return [dict(r) for r in rows]
