from __future__ import annotations

import pytest

from tap.assets.terminology import Term, Terminology
from tap.db import Database
from tap.demo.corpus import ENTRIES, scripted_scores
from tap.quality.checks import run_checks
from tap.quality.consensus import ConsensusConfig, composite, decide, disagreement

CFG = ConsensusConfig()
OK = {"numbers": True, "terms": True, "tags": True}


def chk(src, tgt, tgt_lang="de", terms=None):
    return run_checks(src, tgt, src_lang="en", tgt_lang=tgt_lang, terms=terms, terminology=Terminology(Database()) if terms else None)


# --- deterministic checks -------------------------------------------------------------------
def test_changed_number_fails():
    r = chk("Report within 24 hours.", "Melden Sie innerhalb von 48 Stunden.")
    assert not r.passed["numbers"] and "missing 24" in r.findings[0]


def test_locale_number_formats_are_equal():
    assert chk("Pay 1,000.50 now.", "Zahlen Sie jetzt 1.000,50.").passed["numbers"]


def test_unit_mismatch_detected():
    r = chk("Take 25 mg daily.", "Nehmen Sie täglich 25 ml ein.")
    assert r.passed["numbers"] and not r.passed["units"]


def test_tags_and_placeholders_must_match():
    assert chk("Hello {name}, <b>click</b>.", "Hallo {name}, <b>klicken</b>.").passed["tags"]
    assert not chk("Hello {name}.", "Hallo {nom}.").passed["tags"]
    assert not chk("Click <b>here</b>.", "Klicken Sie hier.").passed["tags"]


def test_dropped_negation_is_a_polarity_failure():
    assert not chk("Do not take this medicine with alcohol.", "Nehmen Sie dieses Arzneimittel mit Alkohol ein.").passed["polarity"]
    assert chk("Do not take this medicine with alcohol.", "Nehmen Sie dieses Arzneimittel nicht mit Alkohol ein.").passed["polarity"]
    assert chk("Do not take this medicine.", "Ne prenez pas ce médicament.", "fr").passed["polarity"]


def test_implausible_length_and_untranslated_and_empty():
    assert not chk("This is a long enough source sentence to measure.", "Kurz.").passed["length"]
    assert not chk("This is a long enough source sentence to measure.", "This is a long enough source sentence to measure.").passed[
        "length"
    ]
    assert not any(chk("Hello there my friend.", "  ").passed.values())


def test_required_forbidden_and_dnt_terms():
    terms = [Term("IMP", "Prüfpräparat"), Term("study", "Studie", "forbidden"), Term("ClinPro", "ClinPro", "dnt")]
    assert not chk("The IMP is stored.", "Das IMP wird gelagert.", terms=terms).passed["terms"]
    assert chk("The IMP is stored.", "Das Prüfpräparat wird gelagert.", terms=terms).passed["terms"]
    assert chk("The IMP is stored.", "Die Prüfpräparate werden gelagert.", terms=terms).passed["terms"]  # inflection tolerated
    assert not chk("Use ClinPro now.", "Verwenden Sie jetzt ClinPrint.", terms=terms).passed["terms"]


def test_skip_removes_named_checks():
    r = run_checks("Do not run.", "[DE] Do not run.", src_lang="en", tgt_lang="de", skip=frozenset({"polarity"}))
    assert r.passed["polarity"]


# --- consensus: the guide's scripted lab table -----------------------------------------------
@pytest.mark.parametrize(
    ("scores", "action", "reason"),
    [
        (dict(comet=0.91, semantic=0.96, fluency=0.93), "accept", "above_threshold"),  # S1 NMT
        (dict(comet=0.78, semantic=0.55, fluency=0.90), "repair", "critical_semantic_score"),  # S2 NMT: omission
        (dict(comet=0.90, semantic=0.95, fluency=0.94), "accept", "above_threshold"),  # S2 LLM
        (dict(comet=0.90, semantic=0.62, fluency=0.94), "human", "judges_disagree"),  # S5 LLM
        (dict(comet=0.70, semantic=0.72, fluency=0.70), "repair", "below_threshold"),
    ],
)
def test_consensus_table(scores, action, reason):
    d = decide(scores, OK, CFG)
    assert (d.action, d.reason) == (action, reason)


def test_failed_check_beats_a_perfect_composite():
    d = decide(dict(comet=1, semantic=1, fluency=1), {"numbers": False}, CFG)
    assert d.action == "repair" and d.reason == "deterministic_check_failed"


def test_cannot_repair_escalates_instead():
    assert decide(dict(comet=0.78, semantic=0.55, fluency=0.9), OK, CFG, can_repair=False).action == "human"
    assert decide(None, {"numbers": False}, CFG, can_repair=False).action == "human"


def test_policy_can_only_make_outcome_safer():
    d = decide(dict(comet=0.95, semantic=0.95, fluency=0.95), OK, CFG, force_human=True)
    assert d.action == "human" and d.reason == "policy_requires_human"
    assert decide(dict(comet=0.5, semantic=0.5, fluency=0.5), OK, CFG, force_human=True).action == "repair"


def test_composite_and_disagreement_math():
    assert composite(dict(comet=1, semantic=0.5, fluency=0), CFG.weights) == pytest.approx(0.6)
    assert disagreement(dict(a=0.2, b=0.9)) == pytest.approx(0.7)


def test_scripted_demo_scores_exist_for_every_entry():
    for e in ENTRIES:
        for lang, text in e.nmt.items():
            assert scripted_scores(e.src, text) is not None, (e.src, lang)
