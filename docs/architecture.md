# Architecture

## 1. Shape: a modular monolith with real seams

The guide describes 25+ microservices. This repository implements the *boundaries* of that decomposition
(`GET /api/v1/platform/services` lists all 32 with owner module and tables) but ships them as one process, because that is the
cheapest thing that can be run, debugged and tested end to end on a laptop. The rules that keep the seams honest:

1. **One owner per table.** Each service creates its own tables through `Database.migrate(service, steps)` (versioned, append-only
   steps). No service queries another service's tables; it calls the service. (`metering.qe_history` reads only metering tables;
   `semantic.suggest` reads TM rows through the same schema owner's query shape and is the one documented shortcut.)
2. **Cross-service effects go through events.** `segment.completed` → metering; `review.resolved` → feedback;
   `job.completed` → experiment outcomes; `review.requested`/`job.*` → notifications. Events are written to an outbox in the
   caller's transaction and delivered at-least-once to idempotent consumers; after 5 failures a delivery is parked
   (`/readyz` reports dead letters, `/metrics` exposes the count).
3. **Agents act only through the MCP gateway** (translate, score, review submission, glossary change, publish). Deterministic
   in-process services (cache/TM lookup, checks, consensus) are called directly, as in the guide's roster.
4. **Composition happens in one file** (`platform.py`), so the dependency graph is visible and tests can substitute fakes.

### Splitting a service out
Pick a service, put its interface behind HTTP/gRPC, give it its own database, and replace the in-process call with a client.
The event bus becomes Kafka/SQS; `EventBus.publish/subscribe` is the only API consumers use. The hardest seams are the
segment workflow (needs a shared checkpointer: swap `SqliteSaver` for the Postgres saver) and the job runner (needs a queue
instead of thread pools).

## 2. Data ownership

| Service | Tables | Notes |
|---|---|---|
| events | `outbox`, `deliveries` | status `pending/done/dead` |
| audit | `audit_log` | hash chain: `hash = sha256(prev_hash ‖ canonical(record))` |
| jobs | `documents`, `jobs`, `job_segments`, `job_outputs` | `UNIQUE(tenant, idem_key)`; legal status transitions enforced |
| tm | `tm_entries` | `UNIQUE(tenant, pair, domain, glossary_version, source_norm)` |
| terminology | `glossary_versions`, `terms` | point-in-time terms: `added_in ≤ v < removed_in` |
| review | `review_items` | partial unique index: one open item per segment+kind |
| approvals | `approvals` | bound to `(tenant, tool, sha256(args))`, single use |
| metering | `usage_events` | one row per processed segment (event id is the primary key → idempotent) |
| experiments | `experiments`, `assignments`, `outcomes` | assignment is a pure hash of `(experiment, unit)` |
| feedback | `post_edits`, `term_suggestions` | |
| delivery | `deliveries_log` | idempotent by content hash |
| intake / notifications | `intake_sessions`, `notifications` | |
| LangGraph | `checkpoints.db` (separate file) | per-`(job, lang, segment)` thread |

## 3. The segment workflow

Source: `orchestrator/workflow.py`. State (`orchestrator/state.py`) is JSON-only and append-only where it matters
(`candidates`, `steps`, `trace` use the `operator.add` reducer) so a checkpoint is small and replayable.

* **cache_lookup** – glossary via the gateway tool, then exact (cache → TM backfill), fuzzy, semantic. An exact hit is reused
  only if it passes the deterministic checks (a poisoned/stale entry falls through to a draft). A fuzzy hit ≥ `tm_draft_min_pct`
  becomes a *draft candidate*; 75–89% and semantic hits become few-shot context only.
* **route** – `services/policy.route`: rules first (policy → availability → measured QE history → A/B variant → default NMT),
  always with a reason code. History uses the *first NMT attempt* of every segment to avoid survivorship bias.
* **translate_nmt / translate_llm** – gateway tools. An NMT outage falls back to the LLM; a refusal (injection, budget, schema,
  rate limit) escalates the segment to a human with the reason recorded.
* **check → score → decide** – checks first; if a repair will follow anyway, paid scoring is skipped. Scores are applied only
  to the candidate they were computed for (`scored_n` marker). `quality/consensus.decide` is a pure function.
* **human_review** – `interrupt()` with the full context (source, candidate, scores, findings, glossary, TM examples); code before
  the interrupt is pure because the node re-runs on resume.
* **stage** – decides cacheability and prices the path. **TM/cache are written only after approval and delivery** (`JobRunner._commit_assets`),
  never mid-flight: regulated content is cached only when a human approved it; LLM output for clinical content needs a human too;
  UI/marketing may cache auto-accepted output.

### Policies (`services/policy.py`)

| Policy | Human sign-off | Engine | Cache write | Approval |
|---|---|---|---|---|
| `fast` | only on escalation | NMT first | yes | auto (recorded as `policy:auto-approve`) |
| `balanced` | high-risk segments | NMT first | yes | manual |
| `regulatory-strict` | every changed segment, accept ≥ 0.90 | NMT first | after human | manual, 10% advisory LQA sample |
| `safety-urgent` | every segment | LLM first, no fuzzy drafts | **never** | manual, top review priority |

Derived from the job: regulatory domain / patient audience / regulatory use → strict; clinical or regulatory with a deadline
< `urgent_hours` → safety-urgent; UI/marketing → fast; otherwise balanced. An explicit `policy` in the request overrides.

## 4. Job state machine

`queued → analyzing → translating ⇄ awaiting_review → awaiting_approval → delivered`, with `needs_attention` (a segment crashed
or the format check failed; `POST /jobs/{id}/retry`) and `failed`. Transitions are validated in `JobStore.set_status`.

* **Idempotent everywhere**: `Idempotency-Key` on submit; segment results are upserts; usage events keyed by event id; delivery
  is idempotent by content hash; review items are unique per open segment.
* **Crash recovery** (`JobRunner.recover`): re-enqueues unfinished jobs; segments mid-graph replay from the checkpoint
  (`continue_run`), interrupted ones keep their review item, and an item lost in the crash window is recreated.
* **Concurrency**: one runner per job at a time (extra requests coalesce into one re-run); segments run in a bounded pool;
  the reviewer flow and the job loop are separated (`in_review` segments are never reprocessed by the loop; see the race test).
* **Repetitions** inside a job are translated once and copied at zero cost (`reuse_repeat`).

## 5. MCP gateway

Pipeline (`mcp/gateway.py`, every step reported by the dry-run endpoint): `registry → allow_list → role → scope → schema →
tenant_scope → rate_limit → budget → injection → approval → precondition → execute → audit`.

* Destructive tools are **not registered**; there is no way to call them.
* Agent identities are least-privilege (`orchestrator`, `evaluator`, `delivery`); a user's role grants scopes; both must allow a tool.
* Tenant-scoped arguments (`client_id`) must equal the caller's tenant; glossary content for LLM prompts is loaded server-side
  from the caller's tenant, never from model-supplied arguments.
* Approvals are bound to the exact arguments, single use, four-eyes for the requester, and returned to the pool if a
  precondition fails after consumption.
* Audit records summarise arguments and **never store source text**.
* Injection scanning is a tripwire (regex), not a guarantee; it is one layer next to schema validation, least privilege and
  approvals. Default mode denies; `TAP_INJECTION_MODE=flag` marks instead.

## 6. Quality

* **Deterministic checks** (`quality/checks.py`): locale-independent numbers, symbolic units (not translated words such as
  "hours"), protected tokens (placeholders/tags/URLs), required/forbidden/do-not-translate terms with inflection tolerance,
  negation preserved (EN→de/fr/es/it/pt/nl), length-ratio bands, untranslated copy-through.
* **Evaluators fill three slots** (`comet`, `semantic`, `fluency`): demo (scripted), heuristic, LLM judges (structured verdicts,
  text wrapped as data), or `ml` (COMET-Kiwi, LaBSE + number cap). Per-(evaluator, pair) calibration anchors normalise raw
  scores (`quality/calibration.py`, `fit_anchors` from human-labelled samples).
* **Consensus precedence**: failed check → repair; critical semantic score → repair; disagreement → human; composite ≥ accept →
  accept; else repair. Policy can only make an outcome safer (accept → human).

## 7. Deviations from the guide (and why)

| Guide | Here | Reason |
|---|---|---|
| Cache write inside the graph (`cache_write`) | `stage` node decides, `_commit_assets` writes after approval | unreviewed output must not poison reuse; a rejected job must leave no assets behind |
| Job orchestration as a graph | job level is a restartable state machine over the DB; segment level is the LangGraph | the job level is long-lived and human-paced; DB state + `recover()` is simpler to operate than nested interrupts |
| 25+ separate deployables | one process, 32 seams | see §1 |
| `temperature` low | not set | current Anthropic SDK/models do not accept sampling parameters; determinism comes from forced structured output, versioned prompts and the eval gate |
| `translate_*` injection check flags source text | default deny → segment goes to a human | an instruction hidden in a source document is exactly the case a human should see |
| 80% cost reduction | reported from configured prices | the number depends on measured baseline and reuse; the platform computes it, it does not assume it |

## 8. Test strategy

* `test_workflow.py` – the guide's five scenarios and the failure paths (engine outage, budget, injection, poisoned cache, repair cap).
* `test_runner_e2e.py` – lifecycle, TM-after-approval, idempotency, tenant isolation, entitlements, glossary invalidation,
  formats, auto-approval, review edit checks, crash recovery (restart on the same data dir, lost review item, mid-graph crash), retry.
* `test_misc.py::test_many_jobs_in_parallel…` – 10 concurrent jobs × 2 languages: no lost/duplicated usage events, valid audit chain.
* `test_gateway.py`, `test_infra.py`, `test_assets.py`, `test_parse.py` (incl. XML bombs / XXE), `test_api.py` (incl. the guide's intake transcript).
