"""``tap`` command line: serve the API, run the offline demo, seed demo data, evaluate against a golden set."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import Settings
from .domain.models import DocFormat, Domain, JobSpec, PolicyName, Principal
from .platform import Platform

DEMO_TEXT = (
    "Take one 25 mg tablet by mouth once daily with food. "
    "Do not take this medicine if you are allergic to penicillin or have had a severe reaction after any antibiotic. "
    "Participants must report any serious adverse event within 24 hours. "
    "The IMP should be titrated per the dose-escalation scheme in Table 4. "
    "Contact your study doctor if you have any questions."
)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    settings = Settings()
    platform = Platform.create(settings)
    if args.seed_demo:
        print("seeded:", platform.seed_demo(args.tenant))
    platform.start()
    try:
        uvicorn.run(create_app(platform, settings), host=args.host, port=args.port, log_level=settings.log_level.lower())
    finally:
        platform.stop()
    return 0


def _seed(args: argparse.Namespace) -> int:
    platform = Platform.create(Settings())
    print(json.dumps(platform.seed_demo(args.tenant)))
    platform.stop()
    return 0


def _row(*cols: object, widths: tuple[int, ...]) -> str:
    return "  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths, strict=False))


def _demo(_: argparse.Namespace) -> int:
    """Run the guide's workflow-lab scenarios end to end with offline stand-in engines, including a human pause/resume."""
    with tempfile.TemporaryDirectory() as tmp:
        platform = Platform.create(Settings(data_dir=Path(tmp), engine_retry_initial_s=0))
        platform.start()
        print("seeded:", platform.seed_demo("acme"), "\n")
        admin = Principal(subject="demo-admin", tenant_id="acme", roles=frozenset({"admin"}))
        linguist = Principal(subject="demo-linguist", tenant_id="acme", roles=frozenset({"linguist"}))
        doc = platform.jobs.create_document("acme", "leaflet.txt", DocFormat.TXT, DEMO_TEXT)
        spec = JobSpec(
            tenant_id="acme", document_id=doc, source_lang="en", target_langs=["de"], domain=Domain.CLINICAL, policy=PolicyName.BALANCED
        )
        job = platform.runner.submit(admin, spec, idem_key="demo-job-1")
        platform.runner.wait_idle(60)
        job = platform.jobs.get(job["id"])
        print(f"job {job['id']} -> {job['status']} (policy {job['policy']})\nanalysis: {json.dumps(job['analysis'])}\n")
        w = (4, 6, 22, 9, 24, 7)
        print(_row("seg", "risk", "path", "decision", "reason", "cost", widths=w))
        for s in platform.jobs.segments(job["id"]):
            print(
                _row(
                    s["seg_id"],
                    s["risk"],
                    s["path"] or s["status"],
                    s["decision"] or "-",
                    s["reason"] or "-",
                    f"{s['cost_units']:.3f}",
                    widths=w,
                )
            )

        items = platform.review.list_items("acme", status="pending", kind="review")
        print(f"\n{len(items)} segment(s) waiting for a human; resolving them as the linguist:")
        for it in items:
            payload = it["payload"]
            text = payload["candidate"]
            if "Prüfpräparat" in text and "angepasst" in text:  # S5: the LLM changed the meaning of "titrate"
                text = text.replace("angepasst", "titriert")
            print(
                f"  - {it['seg_id']}: {payload['why_flagged']}" + ("  [reviewer fixes the meaning]" if text != payload["candidate"] else "")
            )
            platform.review.claim(it["id"], "acme", linguist.subject)
            platform.runner.submit_review(linguist, it["id"], text=text)
        platform.runner.wait_idle(60)
        job = platform.jobs.get(job["id"])
        print(f"\njob -> {job['status']}")
        platform.runner.approve(linguist, job["id"], reason="demo approval")
        job = platform.jobs.get(job["id"])
        print(f"job -> {job['status']}; delivered files: {[d['location'] for d in platform.delivery.delivered(job['id'])]}\n")
        print(platform.runner.result(linguist, job["id"], "de"))
        platform.bus.flush()
        rep = platform.metering.report("acme", job_id=job["id"])
        print(
            f"\njob 1 (high-risk leaflet): cost {rep['cost_units']} vs baseline {rep['baseline_cost_units']}"
            f" -> savings {rep['savings_pct']}%"
        )
        print("  small and safety-critical: three of five segments need human sign-off, so reuse cannot save much here.\n")

        # job 2: UI strings, the profile where reuse and cheap routing pay off
        ui = json.dumps(
            {
                "save": "Click Save to continue.",
                "confirm": "Click Save to continue.",
                "retry": "Click Save to continue.",
                "title": "Visit schedule",
                "help": "Contact your study doctor if you have any questions.",
                "hint": "Keep out of the sight and reach of children.",
            }
        )
        doc2 = platform.jobs.create_document("acme", "strings.json", DocFormat.JSON, ui)
        spec2 = JobSpec(tenant_id="acme", document_id=doc2, source_lang="en", target_langs=["de"], domain=Domain.UI)
        job2 = platform.runner.submit(admin, spec2, idem_key="demo-job-2")
        platform.runner.wait_idle(60)
        job2 = platform.jobs.get(job2["id"])
        print(f"job 2 (UI strings, policy {job2['policy']}) -> {job2['status']}")
        for s in platform.jobs.segments(job2["id"]):
            print("  " + _row(s["seg_id"], s["path"], f"{s['cost_units']:.3f}", s["source"], widths=(4, 14, 7, 60)))
        platform.bus.flush()
        rep2 = platform.metering.report("acme", job_id=job2["id"])
        print(
            f"  cost {rep2['cost_units']} vs baseline {rep2['baseline_cost_units']} -> savings {rep2['savings_pct']}%"
            f" (reuse share {rep2['reuse_word_share_pct']}%)"
        )
        rep = platform.metering.report("acme")
        print(f"\nboth jobs: cost {rep['cost_units']} vs baseline {rep['baseline_cost_units']} -> savings {rep['savings_pct']}%")
        print("(prices and baseline are illustrative defaults; see `assumptions` in the metering report)")
        print("audit chain:", platform.audit.verify())
        platform.stop()
    return 0


def _eval(args: argparse.Namespace) -> int:
    from .evaluation import compare, load_dataset, load_glossary, run_eval

    with tempfile.TemporaryDirectory() as tmp:
        platform = Platform.create(Settings(data_dir=Path(tmp), engine_retry_initial_s=0))
        report = run_eval(
            platform, load_dataset(args.dataset), prefix=args.prefix, glossary=load_glossary(args.glossary) if args.glossary else None
        )
        platform.stop()
    print(json.dumps(report.metrics, indent=2))
    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n")
        print("baseline written to", args.write_baseline)
    if args.baseline:
        fails = compare(report, json.loads(Path(args.baseline).read_text()))
        for f in fails:
            print("REGRESSION:", f, file=sys.stderr)
        if fails and args.fail_on_regression:
            return 1
    return 0


def _verify_audit(_: argparse.Namespace) -> int:
    platform = Platform.create(Settings())
    res = platform.audit.verify()
    print(json.dumps(res))
    platform.stop()
    return 0 if res["ok"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tap", description="Translation Agent Platform")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP API and web console")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--seed-demo", action="store_true", help="load demo TM and glossary for the 'acme' tenant")
    s.add_argument("--tenant", default="acme")
    s.set_defaults(fn=_serve)
    sub.add_parser("demo", help="run the offline end-to-end demo (no keys needed)").set_defaults(fn=_demo)
    sd = sub.add_parser("seed", help="load demo TM and glossary into the configured database")
    sd.add_argument("--tenant", default="acme")
    sd.set_defaults(fn=_seed)
    e = sub.add_parser("eval", help="evaluate the workflow on a golden set (CI gate)")
    e.add_argument("--dataset", default="data/golden/de-clinical-golden.jsonl")
    e.add_argument("--glossary", default="data/golden/glossary.json", help="required terms applied to the evaluation tenant")
    e.add_argument("--prefix", default="run")
    e.add_argument("--baseline")
    e.add_argument("--write-baseline")
    e.add_argument("--fail-on-regression", action="store_true")
    e.set_defaults(fn=_eval)
    sub.add_parser("verify-audit", help="verify the audit hash chain").set_defaults(fn=_verify_audit)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
