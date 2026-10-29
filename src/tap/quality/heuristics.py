"""Dependency-free reference evaluators.

These are intentionally simple, explainable and cheap. They are a safety net and a fallback for
development, **not** substitutes for COMET or a calibrated LLM judge: use ``quality_backend=llm|ml`` in
production and calibrate against human MQM ratings.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..parse.tokens import numbers_and_units, protected_multiset, strip_protected
from .base import EvalResult, ScoreRequest, Slot, clamp

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
_PAIRS = {"(": ")", "[": "]", "{": "}", '"': '"', "«": "»", "„": "“"}


def _content_words(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(strip_protected(s))]


class QeProxy:
    """Reference-free "does this look like a translation of that?" proxy (stands in for the COMET slot)."""

    slot: Slot = "comet"
    name = "qe-proxy"
    version = "1"

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        src, tgt = strip_protected(req.source), strip_protected(req.target)
        findings: list[str] = []
        ratio = len(tgt) / max(len(src), 1)
        score = 1.0
        if ratio < 0.6:
            score -= min(0.5, (0.6 - ratio) * 1.2)
            findings.append(f"translation much shorter than source (ratio {ratio:.2f})")
        elif ratio > 1.8:
            score -= min(0.4, (ratio - 1.8) * 0.3)
            findings.append(f"translation much longer than source (ratio {ratio:.2f})")
        sw, tw = set(_content_words(src)), set(_content_words(tgt))
        if sw and len(sw & tw) / len(sw) > 0.6 and req.src_lang.split("-")[0] != req.tgt_lang.split("-")[0]:
            score -= 0.4
            findings.append("target shares most words with source (likely untranslated)")
        if protected_multiset(req.source) != protected_multiset(req.target):
            score -= 0.15
        return EvalResult(clamp(score), findings)


class HeuristicSemantic:
    name = "semantic-heuristic"
    slot: Slot = "semantic"
    version = "1"

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        findings: list[str] = []
        s_nums, s_units = numbers_and_units(req.source)
        t_nums, t_units = numbers_and_units(req.target)
        score = 1.0
        if s_nums != t_nums:  # a wrong number caps the score regardless of anything else
            findings.append("numbers differ between source and translation")
            score = min(score, 0.4)
        elif s_units != t_units:
            findings.append("units differ between source and translation")
            score = min(score, 0.5)
        src_len, tgt_len = len(strip_protected(req.source)), len(strip_protected(req.target))
        ratio = tgt_len / max(src_len, 1)
        if src_len > 40 and ratio < 0.7:  # long sentence with a short translation: probable omission
            score -= min(0.4, (0.7 - ratio) * 1.2)
            findings.append(f"possible omission (length ratio {ratio:.2f})")
        return EvalResult(clamp(score), findings)


class HeuristicFluency:
    name = "fluency-heuristic"
    slot: Slot = "fluency"
    version = "1"

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        t = req.target.strip()
        findings: list[str] = []
        score = 1.0
        if _REPEAT.search(t):
            score -= 0.25
            findings.append("repeated word")
        for o, c in _PAIRS.items():
            if o != c and t.count(o) != t.count(c):
                score -= 0.1
                findings.append(f"unbalanced {o}{c}")
        if req.source.rstrip()[-1:] in ".!?" and t[-1:] not in '.!?:;)"”»':
            score -= 0.08
            findings.append("missing final punctuation")
        if re.search(r"\s{2,}|\s[,.;:!?]", t):
            score -= 0.08
            findings.append("spacing/punctuation irregularity")
        if t and t[0].islower() and req.source[:1].isupper():
            score -= 0.05
        # similarity to the source in a different language usually means copy-through
        if SequenceMatcher(None, req.source.lower(), t.lower()).ratio() > 0.9 and req.src_lang != req.tgt_lang and len(t) > 20:
            score -= 0.4
            findings.append("target is nearly identical to source")
        return EvalResult(clamp(score), findings)
