"""Rule-based sentence segmentation with abbreviation and decimal protection.

Rules are intentionally close to SRX defaults. Use the same segmentation for TM import and for new content,
otherwise fuzzy matching degrades. Swap in pySBD/ICU behind ``split_sentences`` if you need more languages.
"""

from __future__ import annotations

import re

ABBREVIATIONS = {
    "e.g",
    "i.e",
    "etc",
    "vs",
    "cf",
    "approx",
    "ca",
    "no",
    "nos",
    "fig",
    "figs",
    "tab",
    "eq",
    "ref",
    "refs",
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "jr",
    "inc",
    "ltd",
    "co",
    "corp",
    "st",
    "al",
    "resp",
    "min",
    "max",
    "z.b",
    "d.h",
    "bzw",
    "usw",
    "ggf",
    "nr",
    "abs",
    "vol",
    "p",
    "pp",
    "u.a",
    "o.ä",
}
_BOUNDARY = re.compile(r"""(?<=[.!?…。！？])["'”’)\]]*\s+(?=["'“‘(\[]?[A-ZÀ-ÖØ-Þ0-9぀-鿿])""")
_CJK_END = re.compile(r"(?<=[。！？])")


def _ends_with_abbreviation(chunk: str) -> bool:
    m = re.search(r"([A-Za-zÀ-ÿ.]+)\.[\"'”’)\]]*$", chunk.rstrip())
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    # single capital letter initials ("J. Smith") and listed abbreviations
    return word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()) or "." in word


def split_sentences(text: str, lang: str = "en") -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = [p for p in _CJK_END.split(text) if p.strip()] if re.search(r"[。！？]", text) else [text]
    out: list[str] = []
    for piece in pieces:
        start = 0
        buf: list[str] = []
        for m in _BOUNDARY.finditer(piece):
            chunk = piece[start : m.start()]
            tail = piece[m.start() : m.end()]
            candidate = chunk + tail.rstrip()
            # a period after an abbreviation or inside a numbered list marker is not a boundary
            if _ends_with_abbreviation(candidate) and not candidate.rstrip().endswith(("!", "?")):
                continue
            buf.append(piece[start : m.end()].strip())
            start = m.end()
        buf.append(piece[start:].strip())
        out.extend(b for b in buf if b)
    return out
