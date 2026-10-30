from __future__ import annotations

from collections import Counter
from datetime import timedelta

import pytest

from tap.db import Database
from tap.domain.models import Audience, Domain, JobSpec, PolicyName, Risk
from tap.errors import ConflictError, NotFoundError, ValidationFailed
from tap.services.experiments import ExperimentService, bucket, noninferiority_z, sample_size_per_arm, two_prop_z
from tap.services.policy import POLICIES, RouteFeatures, classify_risk, derive_policy, route
from tap.util import utcnow


def spec(**kw):
    base = dict(tenant_id="acme", document_id="d", source_lang="en", target_langs=["de"], domain=Domain.CLINICAL)
    return JobSpec(**{**base, **kw})


def test_policy_derivation():
    now = utcnow()
    assert derive_policy(spec(domain=Domain.REGULATORY)).name == PolicyName.REGULATORY_STRICT
    assert derive_policy(spec(audience=Audience.PATIENTS)).name == PolicyName.REGULATORY_STRICT
    assert derive_policy(spec(domain=Domain.UI)).name == PolicyName.FAST
    assert derive_policy(spec()).name == PolicyName.BALANCED
    assert derive_policy(spec(deadline=now + timedelta(hours=3)), now=now).name == PolicyName.SAFETY_URGENT
    assert derive_policy(spec(deadline=now + timedelta(days=3)), now=now).name == PolicyName.BALANCED
    assert (
        derive_policy(spec(domain=Domain.UI, deadline=now + timedelta(hours=1)), now=now).name == PolicyName.FAST
    )  # urgency only escalates regulated content
    assert (
        derive_policy(spec(domain=Domain.UI, policy=PolicyName.REGULATORY_STRICT)).name == PolicyName.REGULATORY_STRICT
    )  # explicit pin wins


def test_jobspec_validation():
    with pytest.raises(ValueError):
        spec(target_langs=["en"])
    with pytest.raises(ValueError):
        spec(target_langs=["de", "not a lang!"])
    with pytest.raises(ValueError):
        spec(glossary_version="7")
    assert spec(target_langs=["DE", "de", "pt_br"]).target_langs == ["de", "pt-BR"]
    with pytest.raises(ValueError):
        JobSpec(tenant_id="a", document_id="d", source_lang="en", target_langs=["de"], domain="clinical", surprise=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("text", "domain", "risk"),
    [
        ("Take one 25 mg tablet by mouth once daily with food.", Domain.CLINICAL, Risk.MEDIUM),
        ("Do not take this medicine if you are allergic to penicillin.", Domain.CLINICAL, Risk.HIGH),
        ("Participants must report any serious adverse event within 24 hours.", Domain.REGULATORY, Risk.HIGH),
        ("The IMP should be titrated per the scheme.", Domain.CLINICAL, Risk.HIGH),
        ("Click Save to continue.", Domain.UI, Risk.LOW),
        ("Do not click here!", Domain.UI, Risk.LOW),
        ("Thank you for visiting.", Domain.CLINICAL, Risk.LOW),
    ],
)
def test_risk_classification(text, domain, risk):
    assert classify_risk(text, domain) == risk


def feats(**kw):
    base = dict(
        risk=Risk.MEDIUM,
        policy=POLICIES[PolicyName.BALANCED],
        pair="en>de",
        domain="clinical",
        nmt_available=True,
        llm_available=True,
        nmt_mean_composite=None,
        nmt_samples=0,
        accept_threshold=0.85,
    )
    return RouteFeatures(**{**base, **kw})


def test_routing_rules_and_reason_codes():
    assert route(feats()).reason == "default_nmt_first"
    assert route(feats(policy=POLICIES[PolicyName.SAFETY_URGENT])).engine == "llm"
    assert route(feats(nmt_available=False)).reason == "nmt_unavailable"
    assert route(feats(llm_available=False)).reason == "llm_unavailable"
    assert route(feats(nmt_available=False, llm_available=False)).engine == "human"
    # history must be large enough to count, and bad enough to matter
    assert route(feats(nmt_mean_composite=0.5, nmt_samples=5)).engine == "nmt"
    assert route(feats(nmt_mean_composite=0.5, nmt_samples=100)).reason == "nmt_history_below_threshold"
    assert route(feats(nmt_mean_composite=0.84, nmt_samples=100)).engine == "nmt"
    assert route(feats(variant={"medium_risk_route": "llm"})).reason == "ab_variant_llm"
    assert route(feats(variant={"high_risk_route": "llm"})).engine == "nmt"


def test_policy_human_requirements():
    assert POLICIES[PolicyName.REGULATORY_STRICT].requires_human(Risk.LOW, reused_exact=False)
    assert not POLICIES[PolicyName.REGULATORY_STRICT].requires_human(Risk.LOW, reused_exact=True)
    assert POLICIES[PolicyName.BALANCED].requires_human(Risk.HIGH, reused_exact=False)
    assert not POLICIES[PolicyName.BALANCED].requires_human(Risk.MEDIUM, reused_exact=False)
    assert not POLICIES[PolicyName.FAST].requires_human(Risk.HIGH, reused_exact=False)


# --- experiments ---------------------------------------------------------------------------------
def test_bucketing_is_deterministic_and_roughly_balanced():
    ids = [f"job-{i}" for i in range(4000)]
    a = [bucket(i, "exp", ["A", "B"]) for i in ids]
    assert a == [bucket(i, "exp", ["A", "B"]) for i in ids]
    c = Counter(a)
    assert 1850 < c["A"] < 2150
    assert Counter(bucket(i, "exp", ["A", "B"], traffic_pct=10) for i in ids)[None] > 3400
    assert [bucket(i, "other", ["A", "B"]) for i in ids] != a  # different experiments are independent


def test_z_test_matches_guide_example_and_planner():
    z, p = two_prop_z(3160, 4000, 3280, 4000)  # 79.0% vs 82.0%
    assert z > 3 and p < 0.01
    assert two_prop_z(100, 200, 100, 200) == (0.0, 1.0)
    assert sample_size_per_arm(79, 2) == 6278  # +2 points at 79%, alpha 0.05, power 80%: thousands per arm
    with pytest.raises(ValidationFailed):
        sample_size_per_arm(99, 5)


def test_noninferiority():
    _, p_ok = noninferiority_z(800, 1000, 790, 1000, margin=0.05)
    _, p_bad = noninferiority_z(800, 1000, 700, 1000, margin=0.05)
    assert p_ok < 0.05 < p_bad


@pytest.fixture
def svc():
    return ExperimentService(Database())


def test_experiment_lifecycle_and_analysis(svc):
    svc.create(
        "route-policy",
        {"A": {"medium_risk_route": "nmt"}, "B": {"medium_risk_route": "llm"}},
        primary_metric="cost_per_1k_words",
        guardrail_metric="quality_pass",
        min_samples=30,
    )
    with pytest.raises(ConflictError):
        svc.create("route-policy", {"A": {}, "B": {}}, primary_metric="x")
    with pytest.raises(ValidationFailed):
        svc.create("solo", {"A": {}}, primary_metric="x")
    assert svc.assign("nope", "j1") is None
    assignments = {f"j{i}": svc.assign("route-policy", f"j{i}") for i in range(200)}
    assert {a[0] for a in assignments.values()} == {"A", "B"}
    assert svc.assign("route-policy", "j1") == assignments["j1"]  # stable
    for unit, (variant, _) in assignments.items():
        cost = 10.0 if variant == "A" else 20.0
        svc.record("route-policy", unit, {"cost_per_1k_words": cost + (hash(unit) % 3), "quality_pass": 1.0})
    svc.record("route-policy", "never-assigned", {"cost_per_1k_words": 1})  # ignored
    res = svc.analysis("route-policy")
    assert res["verdict"] == "significant" and res["primary"]["diff"] > 5
    assert res["guardrail"]["metric"] == "quality_pass"
    svc.stop("route-policy")
    assert svc.assign("route-policy", "new-job") is None
    with pytest.raises(NotFoundError):
        svc.get("missing")


def test_analysis_refuses_to_call_a_winner_below_min_samples(svc):
    svc.create("e", {"A": {}, "B": {}}, primary_metric="m", min_samples=100)
    for i in range(20):
        svc.assign("e", f"u{i}")
        svc.record("e", f"u{i}", {"m": float(i)})
    assert svc.analysis("e")["verdict"] == "insufficient_data"


def test_guardrail_failure_overrides_a_cost_win(svc):
    svc.create("g", {"A": {}, "B": {}}, primary_metric="cost", guardrail_metric="quality_pass", guardrail_margin=0.02, min_samples=50)
    for i in range(600):
        v, _ = svc.assign("g", f"u{i}")  # type: ignore[misc]
        svc.record("g", f"u{i}", {"cost": 5.0 if v == "B" else 10.0, "quality_pass": 0.0 if (v == "B" and i % 3 == 0) else 1.0})
    assert svc.analysis("g")["verdict"] == "guardrail_failed"
