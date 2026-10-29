"""Protected tokens (placeholders, tags, URLs), number/unit extraction and word counting.

These are the "deterministic where it matters" primitives used by segmentation, checks and risk rules.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Things that must survive translation byte-for-byte.
PROTECTED = re.compile(
    r"""
      <[^<>\n]+>                    # xml/html tags
    | \{\{[^{}]+\}\}                # {{mustache}}
    | \{[^{}\n]*\}                  # {name} / ICU argument
    | %\([A-Za-z_]\w*\)[sdif]       # %(name)s
    | %(?:\d+\$)?[sdifx]            # %s %1$s %d
    | \$\{[^}]+\}                   # ${var}
    | https?://[^\s<>"']+           # URLs
    | [\w.+-]+@[\w-]+\.[\w.-]+      # e-mail
    """,
    re.VERBOSE,
)
TAG = re.compile(r"<[^<>\n]+>")

# Only symbolic units: words such as "hours"/"tablets" are translated, so comparing them across languages would
# raise false alarms. Numbers themselves are always compared (see ``numbers_and_units``).
_UNITS = "mg|µg|μg|mcg|ng|g|kg|ml|mL|l|L|dl|mmol|mol|IU|U|mm|cm|m|km|mmHg|°C|°F|%|h|min"
# number with optional thousands/decimal separators, optional trailing unit (hyphen or space separated)
NUMBER_UNIT = re.compile(
    rf"(?<![\w.])(?P<num>\d{{1,3}}(?:[ ,.  ]\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    rf"(?:[\s -]?(?P<unit>{_UNITS})(?!\w))?",
    re.IGNORECASE,
)

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def protected_tokens(text: str) -> list[str]:
    return PROTECTED.findall(text)


def protected_multiset(text: str) -> Counter[str]:
    return Counter(protected_tokens(text))


def strip_protected(text: str) -> str:
    return PROTECTED.sub(" ", text)


def normalise_number(raw: str) -> str:
    """'1,000.5' / '1.000,5' / '1 000,5' / '25,5' -> canonical '1000.5' / '25.5'."""
    s = raw.replace(" ", "").replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        # a lone separator followed by exactly 3 digits and a group of <=3 leading digits is a thousands mark
        if len(parts) > 2 or (len(parts[-1]) == 3 and 1 <= len(parts[0]) <= 3 and parts[0] != "0"):
            s = "".join(parts)
        else:
            s = ".".join(parts)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def numbers_and_units(text: str) -> tuple[Counter[str], Counter[str]]:
    """Return (numbers, number+unit pairs) found in ``text``, protected tokens excluded."""
    body = strip_protected(text)
    nums: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    for m in NUMBER_UNIT.finditer(body):
        n = normalise_number(m.group("num"))
        nums[n] += 1
        unit = m.group("unit")
        if unit:
            pairs[f"{n} {normalise_unit(unit)}"] += 1
    return nums, pairs


_UNIT_ALIASES = {"μg": "µg", "mcg": "µg"}


def normalise_unit(unit: str) -> str:
    u = unit.lower() if unit.lower() not in ("iu",) else "IU"
    return _UNIT_ALIASES.get(u, u)


def count_words(text: str, lang: str = "en") -> int:
    """Whitespace words for alphabetic scripts; characters for CJK (a standard localisation convention)."""
    cleaned = strip_protected(text)
    cjk = len(_CJK.findall(cleaned))
    if cjk:
        return cjk + len(_CJK.sub(" ", cleaned).split())
    return len(cleaned.split())


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
