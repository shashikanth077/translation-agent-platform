"""Deterministic checks: numbers, units, tags/placeholders, terms, polarity and length.

These run before any model-based scoring. They are cheap, explainable and catch the most dangerous
errors in clinical text (a changed dose, a dropped "not"). A failed check is a *hard* signal: the
consensus step never lets a high composite score override it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..assets.terminology import Term, Terminology
from ..parse.tokens import TAG, numbers_and_units, protected_multiset

CHECK_NAMES = ("numbers", "units", "tags", "terms", "polarity", "length")

_NEG_SRC = re.compile(r"\b(?:not|no|never|without|cannot|none|neither|nor|n't|avoid|prohibited|contraindicated)\b|n't\b", re.I)
_NEG_TGT: dict[str, re.Pattern[str]] = {
    "de": re.compile(r"\b(?:nicht|kein\w*|ohne|nie|niemals|weder|nirgends|verboten|vermeiden?|kontraindiziert)\b", re.I),
    "fr": re.compile(r"\b(?:ne|n'|pas|jamais|sans|aucun\w*|ni|non|éviter|interdit|contre-indiqu\w+)\b|\bn['’]", re.I),
    "es": re.compile(r"\b(?:no|nunca|jamás|sin|ning\w+|ni|evitar?|prohibido|contraindicado)\b", re.I),
    "it": re.compile(r"\b(?:non|mai|senza|nessun\w*|né|evitare|vietato|controindicato)\b", re.I),
    "pt": re.compile(r"\b(?:não|nunca|jamais|sem|nenhum\w*|nem|evitar|proibido|contraindicado)\b", re.I),
    "nl": re.compile(r"\b(?:niet|geen\w*|nooit|zonder|noch|vermijd\w*|verboden|gecontra-indiceerd)\b", re.I),
}
# plausible target/source character-length ratios (loose on purpose: this catches omissions/hallucinations,
# not style). (hard_min, soft_min, soft_max, hard_max)
_LEN_BANDS: dict[str, tuple[float, float, float, float]] = {
    "ja": (0.15, 0.25, 1.1, 2.0),
    "zh": (0.15, 0.25, 1.1, 2.0),
    "ko": (0.2, 0.3, 1.2, 2.0),
}
_DEFAULT_BAND = (0.4, 0.6, 1.8, 3.0)


@dataclass
class CheckReport:
    passed: dict[str, bool] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(self.passed.values())

    def failed(self) -> list[str]:
        return [k for k, v in self.passed.items() if not v]


def _diff(a, b) -> str:
    missing, extra = a - b, b - a
    parts = []
    if missing:
        parts.append("missing " + ", ".join(sorted(missing)))
    if extra:
        parts.append("unexpected " + ", ".join(sorted(extra)))
    return "; ".join(parts)


def run_checks(
    source: str,
    target: str,
    *,
    src_lang: str,
    tgt_lang: str,
    terms: list[Term] | None = None,
    terminology: Terminology | None = None,
    skip: frozenset[str] = frozenset(),
) -> CheckReport:
    rep = CheckReport()
    if not target.strip():
        rep.passed = dict.fromkeys(CHECK_NAMES, False)
        rep.findings.append("empty translation")
        return rep

    s_nums, s_units = numbers_and_units(source)
    t_nums, t_units = numbers_and_units(target)
    rep.passed["numbers"] = s_nums == t_nums
    if not rep.passed["numbers"]:
        rep.findings.append("number mismatch: " + _diff(s_nums, t_nums))
    rep.passed["units"] = s_units == t_units or not rep.passed["numbers"]  # number errors already reported
    if not rep.passed["units"]:
        rep.findings.append("unit mismatch: " + _diff(s_units, t_units))

    s_prot, t_prot = protected_multiset(source), protected_multiset(target)
    rep.passed["tags"] = s_prot == t_prot
    if not rep.passed["tags"]:
        rep.findings.append("tag/placeholder mismatch: " + _diff(s_prot, t_prot))
    elif _tag_order(source) != _tag_order(target) and not _is_reorderable(source):
        rep.warnings.append("tag order changed")

    findings = terminology.check(source, target, terms or []) if terminology and terms else []
    rep.passed["terms"] = not findings
    rep.findings.extend(findings)

    neg_pat = _NEG_TGT.get(tgt_lang.split("-")[0])
    if neg_pat is not None and src_lang.split("-")[0] == "en":
        has_src, has_tgt = bool(_NEG_SRC.search(_untag(source))), bool(neg_pat.search(_untag(target)))
        rep.passed["polarity"] = not (has_src and not has_tgt)
        if not rep.passed["polarity"]:
            rep.findings.append("negation in source not found in translation (possible polarity inversion)")
    else:
        rep.passed["polarity"] = True

    ratio = len(_untag(target)) / max(len(_untag(source)), 1)
    hard_min, soft_min, soft_max, hard_max = _LEN_BANDS.get(tgt_lang.split("-")[0], _DEFAULT_BAND)
    short = len(_untag(source)) < 12  # ratios are meaningless for very short strings
    rep.passed["length"] = short or hard_min <= ratio <= hard_max
    if not rep.passed["length"]:
        rep.findings.append(f"implausible length ratio {ratio:.2f} (possible omission or hallucination)")
    elif not short and not soft_min <= ratio <= soft_max:
        rep.warnings.append(f"unusual length ratio {ratio:.2f}")
    if target.strip() == source.strip() and len(source) > 20 and not _only_protected(source):
        rep.passed["length"] = False
        rep.findings.append("translation identical to source (untranslated)")
    for name in skip:  # e.g. pseudo-translations from the offline demo engine
        rep.passed[name] = True
    return rep


def _untag(s: str) -> str:
    return TAG.sub(" ", s)


def _tag_order(s: str) -> list[str]:
    return TAG.findall(s)


def _is_reorderable(s: str) -> bool:
    return len(set(TAG.findall(s))) <= 1


def _only_protected(s: str) -> bool:
    return not re.sub(r"\W+", "", protected_stripped(s))


def protected_stripped(s: str) -> str:
    from ..parse.tokens import strip_protected

    return strip_protected(s)
