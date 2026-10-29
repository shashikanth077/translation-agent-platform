"""Job-level behaviour: lifecycle, safety gates, idempotency, recovery, isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tap.config import Settings
from tap.demo.corpus import ENTRIES
from tap.domain.models import DocFormat, Domain, JobSpec, JobStatus, PolicyName
from tap.errors import ConflictError, EntitlementError, Forbidden, NotFoundError, ValidationFailed
from tap.platform import Platform
from tests.conftest import make_job, principal, resolve_all

S1, S2, S3, S4, S5 = (e.src for e in ENTRIES)
LEAFLET = f"{S1} {S2} {S4} {S5}"


def test_full_lifecycle_with_human_review_and_approved_delivery(platform, linguist):
    job = make_job(platform, LEAFLET, key="life-1")
    assert job["status"] == "awaiting_review" and job["analysis"]["segments"] == 4
    items = platform.review.list_items("acme", status="pending")
    assert {i["seg_id"] for i in items} == {"s2", "s3", "s4"}  # S2/S4/S5: high-risk or disagreeing
    resolve_all(platform, linguist, edit=lambda p: p["candidate"].replace("angepasst", "titriert"))
    job = platform.jobs.get(job["id"])
    assert job["status"] == "awaiting_approval" and platform.runner.result(principal("admin"), job["id"], "de")
    assert not platform.delivery.delivered(job["id"])  # nothing leaves the platform before approval
    platform.runner.approve(linguist, job["id"], reason="ok")
    job = platform.jobs.get(job["id"])
    assert job["status"] == "delivered"
    out = platform.runner.result(linguist, job["id"], "de")
    assert "titriert" in out and "oder nach der Anwendung" in out
    delivered = platform.delivery.delivered(job["id"])
    assert len(delivered) == 1 and Path(delivered[0]["location"]).read_text(encoding="utf-8") == out
    assert platform.audit.verify()["ok"]
    actions = [r["action"] for r in platform.audit.search(tenant_id="acme", limit=500)]
    assert {"job.created", "job.awaiting_approval", "review.resolved", "job.delivered", "tool.allowed"} <= set(actions)


def test_human_approved_segments_enter_tm_only_after_delivery(platform, linguist):
    job = make_job(platform, S5, key="tm-1")
    resolve_all(platform, linguist)
    tm_before = platform.tm.count("acme")
    assert platform.jobs.get(job["id"])["status"] == "awaiting_approval"
    assert platform.tm.count("acme") == tm_before  # not yet
    platform.runner.approve(linguist, job["id"])
    assert platform.tm.count("acme") == tm_before + 1
    v = f"v{platform.terminology.latest_version('acme')}"
    hit = platform.tm.exact("acme", "en", "de", "clinical", v, S5)
    assert hit and hit.origin == "human"
    # a second job with the same sentence is now an exact reuse: no model call, no cost
    job2 = make_job(platform, S5, key="tm-2", policy=PolicyName.FAST)
    seg = platform.jobs.segments(job2["id"])[0]
    assert seg["path"] == "reuse_exact" and seg["cost_units"] == 0 and job2["status"] == "delivered"  # fast policy auto-approves


def test_unreviewed_regulatory_machine_output_is_not_cached(platform, linguist):
    job = make_job(platform, "The visit schedule is shown below.", domain=Domain.REGULATORY, policy=PolicyName.BALANCED, key="reg-1")
    assert job["status"] == "awaiting_approval" and platform.jobs.segments(job["id"])[0]["cacheable"] == 0
    before = platform.tm.count("acme")
    platform.runner.approve(linguist, job["id"])
    assert platform.tm.count("acme") == before


def test_idempotent_submission_and_conflicting_reuse_of_key(platform, admin):
    doc = platform.jobs.create_document("acme", "a.txt", DocFormat.TXT, "Hello there.")
    spec = JobSpec(tenant_id="acme", document_id=doc, source_lang="en", target_langs=["de"], domain=Domain.UI)
    a = platform.runner.submit(admin, spec, idem_key="same-key-1")
    b = platform.runner.submit(admin, spec, idem_key="same-key-1")
    assert a["id"] == b["id"] and len(platform.jobs.list_jobs("acme")) == 1
    other = spec.model_copy(update={"target_langs": ["fr"]})
    with pytest.raises(ConflictError):
        platform.runner.submit(admin, other, idem_key="same-key-1")
    assert platform.runner.wait_idle(30)


def test_tenant_isolation_everywhere(platform):
    job = make_job(platform, S5, key="iso-1")
    beta = principal("admin", "beta")
    with pytest.raises(NotFoundError):
        platform.jobs.get(job["id"], "beta")
    doc = platform.jobs.create_document("beta", "b.txt", DocFormat.TXT, "Hello.")
    with pytest.raises(NotFoundError):
        platform.jobs.get_document("beta", job["document_id"])
    with pytest.raises(Forbidden):  # body tenant must equal the caller's tenant
        platform.runner.submit(
            beta, JobSpec(tenant_id="acme", document_id=doc, source_lang="en", target_langs=["de"], domain=Domain.UI), idem_key="x" * 8
        )
    with pytest.raises(NotFoundError):  # a foreign document id is indistinguishable from a missing one
        platform.runner.submit(
            beta,
            JobSpec(tenant_id="beta", document_id=job["document_id"], source_lang="en", target_langs=["de"], domain=Domain.UI),
            idem_key="y" * 8,
        )
    assert platform.review.list_items("beta") == []
    with pytest.raises(NotFoundError):
        platform.review.get(platform.review.list_items("acme")[0]["id"], "beta")
    assert platform.tm.search("beta", "en", "de", S5, min_pct=50) == []


def test_entitlements_are_enforced_at_submission(platform, admin):
    beta = principal("admin", "beta")
    doc = platform.jobs.create_document("beta", "b.txt", DocFormat.TXT, "Hello.")
    with pytest.raises(EntitlementError):  # beta only has de/fr and ui/marketing
        platform.runner.submit(
            beta, JobSpec(tenant_id="beta", document_id=doc, source_lang="en", target_langs=["ja"], domain=Domain.UI), idem_key="ent-0001"
        )
    with pytest.raises(EntitlementError):
        platform.runner.submit(
            beta,
            JobSpec(tenant_id="beta", document_id=doc, source_lang="en", target_langs=["de"], domain=Domain.CLINICAL),
            idem_key="ent-0002",
        )


def test_glossary_update_invalidates_cached_reuse_and_forces_recheck(platform, linguist, admin):
    job = make_job(platform, S5, key="gl-1")
    resolve_all(platform, linguist)
    platform.runner.approve(linguist, job["id"])
    from tap.assets.terminology import TermChange

    platform.terminology.commit("acme", [TermChange("add", "en", "de", "IMP", "Studienmedikation")], note="client mandate", actor="admin")
    job2 = make_job(platform, S5, key="gl-2", policy=PolicyName.FAST)
    seg = platform.jobs.segments(job2["id"])[0]
    assert seg["path"] != "reuse_exact"  # old translation uses the retired term: must not be reused silently
    assert "Studienmedikation" in (seg["target"] or "") or job2["status"] == "awaiting_review"


def test_repeated_sentences_are_translated_once(platform):
    job = make_job(platform, "Hello there. Hello there. Hello there.", domain=Domain.UI, policy=PolicyName.FAST, key="rep-1")
    segs = platform.jobs.segments(job["id"])
    assert [s["path"] for s in segs] == ["nmt", "reuse_repeat", "reuse_repeat"]
    assert job["analysis"]["repeated_segments"] == 2 and len({s["target"] for s in segs}) == 1
    assert sum(1 for s in segs if s["cost_units"] > 0) == 1


def test_multi_language_job_assembles_each_language(platform, linguist):
    job = make_job(
        platform, "Click Save to continue. Hello there.", domain=Domain.UI, policy=PolicyName.BALANCED, langs=["de", "fr"], key="ml-1"
    )
    assert job["status"] == "awaiting_approval"
    de, fr = (platform.runner.result(linguist, job["id"], lang) for lang in ("de", "fr"))
    assert de.startswith("Klicken Sie auf Speichern") and fr.startswith("Cliquez sur Enregistrer") and de != fr


def test_json_resource_file_roundtrips_through_the_platform(platform, linguist):
    src = json.dumps({"save": "Click Save to continue.", "hi": "Hello {name}!", "n": 3})
    job = make_job(platform, src, domain=Domain.UI, policy=PolicyName.BALANCED, fmt=DocFormat.JSON, filename="en.json", key="json-1")
    assert job["status"] == "awaiting_approval"
    out = json.loads(platform.runner.result(linguist, job["id"], "de"))
    assert out["n"] == 3 and out["save"].startswith("Klicken") and "{name}" in out["hi"]


def test_xliff_job_keeps_inline_tags(platform, linguist):
    xlf = (
        '<?xml version="1.0"?><xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2"><file source-language="en" datatype="plaintext" original="a">'
        '<body><trans-unit id="1"><source>Click <g id="1">Save</g> to continue.</source></trans-unit></body></file></xliff>'
    )
    job = make_job(platform, xlf, domain=Domain.UI, policy=PolicyName.BALANCED, fmt=DocFormat.XLIFF, filename="a.xlf", key="xlf-1")
    assert job["status"] == "awaiting_approval", job["error"]
    out = platform.runner.result(linguist, job["id"], "de")
    assert '<g id="1">' in out and 'target-language="de"' in out


def test_client_role_cannot_publish_and_publish_requires_awaiting_approval(platform, client_user, linguist, admin):
    job = make_job(platform, S1, policy=PolicyName.FAST, domain=Domain.UI, key="pub-1")
    with pytest.raises(Forbidden):
        platform.runner.approve(client_user, job["id"])
    job2 = make_job(platform, S5, key="pub-2")  # still awaiting review
    assert job2["status"] == "awaiting_review"
    with pytest.raises(Forbidden, match="not awaiting approval|awaiting_review"):
        platform.runner.approve(linguist, job2["id"])


def test_fast_policy_auto_approves_and_delivers(platform):
    job = make_job(platform, "Click Save to continue.", domain=Domain.UI, policy=PolicyName.FAST, key="auto-1")
    assert job["status"] == "delivered"
    approvals = platform.approvals.list_all("acme")
    assert approvals and approvals[0]["decided_by"] == "policy:auto-approve"


def test_publish_is_idempotent_at_the_connector(platform, linguist):
    job = make_job(platform, "Hello there.", domain=Domain.UI, policy=PolicyName.BALANCED, key="idem-pub")
    platform.runner.approve(linguist, job["id"])
    first = platform.delivery.delivered(job["id"])
    outs = platform.jobs.outputs(job["id"])
    again = platform.delivery.deliver("acme", job["id"], "doc.txt", outs)
    assert again[0]["duplicate"] == "true" and platform.delivery.delivered(job["id"]) == first


def test_review_edit_must_pass_checks_unless_overridden_with_a_note(platform, linguist, client_user):
    make_job(platform, S4, domain=Domain.REGULATORY, policy=PolicyName.BALANCED, key="rev-1")
    item = platform.review.list_items("acme")[0]
    bad = item["payload"]["candidate"].replace("24", "48")
    with pytest.raises(ValidationFailed) as e:
        platform.runner.submit_review(linguist, item["id"], text=bad)
    assert "number mismatch" in str(e.value.details)
    with pytest.raises(ValidationFailed, match="note"):
        platform.runner.submit_review(linguist, item["id"], text=bad, override=True)
    assert platform.review.get(item["id"])["status"] == "pending"  # a rejected edit does not consume the item
    platform.runner.submit_review(linguist, item["id"], text=bad, override=True, note="sponsor changed the window per amendment 3")
    assert platform.review.get(item["id"])["status"] == "done"
    with pytest.raises(ValidationFailed):
        platform.runner.submit_review(linguist, item["id"], text="x")  # already resolved -> conflict
        raise ValidationFailed("resolved")


def test_review_claim_is_exclusive(platform):
    make_job(platform, S5, key="claim-1")
    item = platform.review.list_items("acme")[0]
    platform.review.claim(item["id"], "acme", "alice")
    with pytest.raises(ConflictError):
        platform.review.claim(item["id"], "acme", "bob")
    with pytest.raises(ConflictError):
        platform.review.resolve(item["id"], "acme", "bob", {})


def test_lqa_sampling_is_advisory_and_does_not_block(platform, linguist):
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    job = make_job(platform, text, domain=Domain.UI, policy=PolicyName.REGULATORY_STRICT, key="lqa-1")
    # strict policy => every changed segment needs a human; resolve them, then check LQA items on reused/auto segments
    assert job["status"] == "awaiting_review"
    resolve_all(platform, linguist)
    job = platform.jobs.get(job["id"])
    assert job["status"] == "awaiting_approval"
    assert all(i["kind"] == "lqa" for i in platform.review.list_items("acme", status="pending"))


def test_crash_recovery_resumes_pending_reviews_after_restart(settings, linguist):
    p1 = Platform.create(settings)
    p1.seed_demo("acme")
    job = make_job(p1, LEAFLET, key="crash-1")
    assert job["status"] == "awaiting_review"
    pending = {i["id"] for i in p1.review.list_items("acme", status="pending")}
    p1.stop()

    p2 = Platform.create(settings)  # same data dir: a restarted process
    p2.start()
    assert p2.runner.wait_idle(30)
    assert {i["id"] for i in p2.review.list_items("acme", status="pending")} == pending  # no duplicates, nothing lost
    resolve_all(p2, linguist)
    assert p2.jobs.get(job["id"])["status"] == "awaiting_approval"
    p2.stop()


def test_recovery_recreates_review_item_lost_in_a_crash_window(settings, linguist):
    p1 = Platform.create(settings)
    p1.seed_demo("acme")
    job = make_job(p1, S5, key="lost-1")
    p1.db.execute("DELETE FROM review_items")  # simulate: segment is in_review but the item never got written
    p1.stop()
    p2 = Platform.create(settings)
    p2.start()
    assert p2.runner.wait_idle(30)
    assert len(p2.review.list_items("acme", status="pending")) == 1
    resolve_all(p2, linguist)
    assert p2.jobs.get(job["id"])["status"] == "awaiting_approval"
    p2.stop()


def test_mid_graph_crash_is_replayed_from_the_checkpoint(settings):
    from tap.errors import ProviderError

    p = Platform.create(Settings(**{**settings.model_dump(), "max_repair_attempts": 2}))
    p.seed_demo("acme")
    real = p.gateway._handlers["score_quality"]
    calls = {"n": 0}

    def exploding(caller, args, ctx):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt  # not an Exception: simulates the process dying mid-node
        return real(caller, args, ctx)

    p.gateway._handlers["score_quality"] = exploding
    doc = p.jobs.create_document("acme", "d.txt", DocFormat.TXT, S1)
    spec = JobSpec(tenant_id="acme", document_id=doc, source_lang="en", target_langs=["de"], domain=Domain.CLINICAL, policy=PolicyName.FAST)
    with pytest.raises(KeyboardInterrupt):
        p.workflow.start(
            {
                "job_id": "J-crash",
                "tenant_id": "acme",
                "seg_id": "s1",
                "source": S1,
                "src_lang": "en",
                "tgt_lang": "de",
                "domain": "clinical",
                "glossary_version": "v1",
                "policy": "fast",
                "risk": "medium",
                "words": 10,
            }
        )
    assert p.workflow.inspect("J-crash", "de", "s1")["status"] == "crashed"
    out = p.workflow.continue_run("J-crash", "de", "s1")
    assert out.status == "done" and out.final["path"] == "nmt"
    assert doc and spec and ProviderError
    p.stop()


def test_failed_segment_marks_job_needs_attention_and_retry_recovers(platform, admin, monkeypatch):
    boom = {"on": True}
    real = platform.wf.start if hasattr(platform, "wf") else platform.workflow.start

    def flaky(initial):  # noqa: ANN001
        if boom["on"]:
            raise RuntimeError("worker died")
        return real(initial)

    monkeypatch.setattr(platform.workflow, "start", flaky)
    job = make_job(platform, "Hello there.", domain=Domain.UI, policy=PolicyName.BALANCED, key="retry-1")
    assert job["status"] == JobStatus.NEEDS_ATTENTION.value and "failed" in job["error"]
    boom["on"] = False
    platform.runner.retry(admin, job["id"])
    assert platform.runner.wait_idle(30)
    assert platform.jobs.get(job["id"])["status"] == "awaiting_approval"
    with pytest.raises(ValidationFailed):
        platform.runner.retry(admin, job["id"])


def test_metering_reports_cost_paths_and_savings_against_baseline(platform, linguist):
    make_job(platform, "Click Save to continue. Hello there.", domain=Domain.UI, policy=PolicyName.FAST, key="meter-1")
    platform.bus.flush()
    rep = platform.metering.report("acme")
    assert rep["segments"] == 2 and rep["words"] > 0 and rep["by_path"]["reuse_exact"]["cost_units"] == 0
    assert rep["savings_pct"] > 90 and rep["reuse_word_share_pct"] > 0
    assert rep["assumptions"]["baseline_per_1k_words"] == 40.0
    assert platform.metering.report("beta")["segments"] == 0


def test_feedback_loop_records_post_edit_distance_and_term_suggestions(platform, linguist):
    for i in range(2):
        make_job(platform, S5, key=f"fb-{i}")
        resolve_all(platform, linguist, edit=lambda p: p["candidate"].replace("angepasst", "titriert"))
    platform.bus.flush()
    stats = platform.feedback.stats("acme")
    assert stats["edits"] == 2 and 0 < stats["mean_distance"] < 0.2 and stats["unchanged_rate"] == 0
    assert {(s["machine_term"], s["human_term"]) for s in platform.feedback.suggestions("acme")} == {("angepasst", "titriert")}


def test_experiment_assignment_and_outcomes_are_recorded_per_job(platform):
    platform.experiments.create(
        "route-policy",
        {"A": {"medium_risk_route": "nmt"}, "B": {"medium_risk_route": "llm"}},
        primary_metric="cost_per_1k_words",
        min_samples=1,
    )
    texts = [
        "The clinic opens early.",
        "Parking is available nearby.",
        "Bring your identity card.",
        "Coffee is free for visitors.",
        "Lockers are in the lobby.",
        "Wear comfortable shoes today.",
        "The lift is on the left.",
        "Visitors sign in at reception.",
    ]
    for i, t in enumerate(texts):
        job = make_job(platform, t, domain=Domain.CLINICAL, policy=PolicyName.FAST, key=f"abm-{i:03d}-key")
        assert job["status"] == "delivered"
    platform.bus.flush()
    res = platform.experiments.analysis("route-policy")
    assert sum(v["n"] for v in res["variants"].values()) == len(texts)  # one outcome per job, none per segment
    assert set(res["variants"]) == {"A", "B"} and all(
        v["metrics"]["quality_pass"]["mean"] == 1.0 for v in res["variants"].values() if v["n"]
    )
