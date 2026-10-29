"""Offline demo corpus. Reproduces the five scenarios of the reference guide's workflow lab.

``DemoEntry.nmt`` / ``.llm`` hold the (deliberately flawed or fixed) outputs the demo engines return, and
``scores`` the scripted metric values the demo evaluators report for them. Anything not in this table is
"pseudo-translated" (``[DE] text``) so the platform stays usable offline. None of this is used when real
backends are configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..util import normalise_text

Scores = dict[str, float]


@dataclass(frozen=True)
class DemoEntry:
    src: str
    nmt: dict[str, str]
    llm: dict[str, str]
    nmt_scores: dict[str, Scores]
    llm_scores: dict[str, Scores] = field(default_factory=dict)


def _s(c: float, s: float, f: float) -> Scores:
    return {"comet": c, "semantic": s, "fluency": f}


ENTRIES: list[DemoEntry] = [
    # S1: straightforward dosing sentence; NMT is good enough.
    DemoEntry(
        "Take one 25 mg tablet by mouth once daily with food.",
        nmt={
            "de": "Nehmen Sie einmal täglich eine 25-mg-Tablette zum Essen ein.",
            "fr": "Prenez un comprimé de 25 mg par voie orale une fois par jour avec de la nourriture.",
        },
        llm={
            "de": "Nehmen Sie einmal täglich eine 25-mg-Tablette oral zusammen mit einer Mahlzeit ein.",
            "fr": "Prenez un comprimé de 25 mg par voie orale, une fois par jour, au cours d'un repas.",
        },
        nmt_scores={"de": _s(0.91, 0.96, 0.93), "fr": _s(0.90, 0.95, 0.92)},
        llm_scores={"de": _s(0.93, 0.97, 0.95), "fr": _s(0.92, 0.96, 0.94)},
    ),
    # S2: NMT drops the second clause (omission). Fluency looks fine; COMET dips; semantic accuracy catches it.
    DemoEntry(
        "Do not take this medicine if you are allergic to penicillin or have had a severe reaction after any antibiotic.",
        nmt={"de": "Nehmen Sie dieses Arzneimittel nicht ein, wenn Sie allergisch gegen Penicillin sind."},
        llm={
            "de": "Nehmen Sie dieses Arzneimittel nicht ein, wenn Sie allergisch gegen Penicillin sind oder nach der "
            "Anwendung eines Antibiotikums eine schwere Reaktion hatten."
        },
        nmt_scores={"de": _s(0.78, 0.55, 0.90)},
        llm_scores={"de": _s(0.90, 0.95, 0.94)},
    ),
    # S3: UI string (normally an exact TM hit; the engine output is only used on a cold TM).
    DemoEntry(
        "Click Save to continue.",
        nmt={"de": "Klicken Sie auf Speichern, um fortzufahren.", "fr": "Cliquez sur Enregistrer pour continuer."},
        llm={"de": "Klicken Sie auf Speichern, um fortzufahren.", "fr": "Cliquez sur Enregistrer pour continuer."},
        nmt_scores={"de": _s(0.95, 0.98, 0.96), "fr": _s(0.94, 0.97, 0.95)},
        llm_scores={"de": _s(0.95, 0.98, 0.96), "fr": _s(0.94, 0.97, 0.95)},
    ),
    # S4: the TM holds the 48-hour version; only the number differs, which the number check must catch.
    DemoEntry(
        "Participants must report any serious adverse event within 24 hours.",
        nmt={"de": "Teilnehmer müssen jedes schwerwiegende unerwünschte Ereignis innerhalb von 24 Stunden melden."},
        llm={"de": "Teilnehmer müssen jedes schwerwiegende unerwünschte Ereignis innerhalb von 24 Stunden melden."},
        nmt_scores={"de": _s(0.92, 0.97, 0.94)},
        llm_scores={"de": _s(0.92, 0.97, 0.94)},
    ),
    # S5: NMT leaves "IMP" untranslated (required-term check). The LLM repair uses the term but changes the
    # meaning of "titrate", so the judges disagree and a human decides.
    DemoEntry(
        "The IMP should be titrated per the dose-escalation scheme in Table 4.",
        nmt={"de": "Das IMP sollte gemäß dem Dosiseskalationsschema in Tabelle 4 titriert werden."},
        llm={"de": "Das Prüfpräparat sollte gemäß dem Dosiseskalationsschema in Tabelle 4 angepasst werden."},
        nmt_scores={"de": _s(0.84, 0.88, 0.90)},
        llm_scores={"de": _s(0.90, 0.62, 0.94)},
    ),
]

# Stand-ins for approved TM content so reuse paths are visible in the demo (tenant "acme").
TM_SEED: list[tuple[str, str, str, str]] = [
    # (domain, src, tgt, tgt_lang)
    ("ui", "Click Save to continue.", "Klicken Sie auf Speichern, um fortzufahren.", "de"),
    (
        "regulatory",
        "Participants must report any serious adverse event within 48 hours.",
        "Teilnehmer müssen jedes schwerwiegende unerwünschte Ereignis innerhalb von 48 Stunden melden.",
        "de",
    ),
    ("regulatory", "Store below 25 degrees C in the original package.", "Unter 25 °C in der Originalverpackung lagern.", "de"),
    ("regulatory", "Contact your study doctor if you have any questions.", "Wenden Sie sich bei Fragen an Ihren Studienarzt.", "de"),
    (
        "regulatory",
        "Do not use this medicine after the expiry date.",
        "Verwenden Sie dieses Arzneimittel nicht nach dem Verfallsdatum.",
        "de",
    ),
    ("regulatory", "Keep out of the sight and reach of children.", "Arzneimittel für Kinder unzugänglich aufbewahren.", "de"),
]

GLOSSARY_SEED = [  # (src, tgt, term, translation)
    ("en", "de", "IMP", "Prüfpräparat"),
    ("en", "de", "adverse event", "unerwünschtes Ereignis"),
    ("en", "de", "study doctor", "Studienarzt"),
    ("en", "de", "dose-escalation", "Dosiseskalation"),
]

_BY_SRC = {normalise_text(e.src): e for e in ENTRIES}
_SCORES: dict[tuple[str, str], Scores] = {}
for _e in ENTRIES:
    for _lang, _t in _e.nmt.items():
        _SCORES[(normalise_text(_e.src), normalise_text(_t))] = _e.nmt_scores[_lang]
    for _lang, _t in _e.llm.items():
        _SCORES.setdefault((normalise_text(_e.src), normalise_text(_t)), _e.llm_scores.get(_lang, _e.nmt_scores[_lang]))
        _SCORES[(normalise_text(_e.src), normalise_text(_t))] = _e.llm_scores.get(_lang, _e.nmt_scores[_lang])
# Scores for the stale TM draft in S4 (kept to show that high scores alone must not accept a wrong number).
_SCORES[
    (
        normalise_text(ENTRIES[3].src),
        normalise_text("Teilnehmer müssen jedes schwerwiegende unerwünschte Ereignis innerhalb von 48 Stunden melden."),
    )
] = _s(0.93, 0.60, 0.94)

PSEUDO_SCORES: Scores = _s(0.92, 0.95, 0.93)


def entry_for(source: str) -> DemoEntry | None:
    return _BY_SRC.get(normalise_text(source))


def scripted_scores(source: str, target: str) -> Scores | None:
    return _SCORES.get((normalise_text(source), normalise_text(target)))


def pseudo(text: str, tgt_lang: str, tag: str) -> str:
    return f"[{tgt_lang.split('-')[0].upper()}:{tag}] {text}"


def is_pseudo(target: str) -> bool:
    return target.startswith("[") and "] " in target[:12]
