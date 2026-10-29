# Operations guide

## Running

| Mode | Command | Notes |
|---|---|---|
| Local demo | `tap demo` | temp data dir, offline backends |
| API + console | `tap serve [--seed-demo]` | `/console`, `/docs`, `/healthz`, `/readyz`, `/metrics` |
| MCP server (stdio) | `TAP_MCP_TENANT=acme python -m tap.mcp.server` | service identity = orchestrator agent; same gateway, same limits |
| Compose | `docker compose up --build` | API + Redis (+ LibreTranslate with `--profile real`) |

State lives in `TAP_DATA_DIR` (`tap.db`, `checkpoints.db`, `delivered/`). Back up both SQLite files together
(`sqlite3 tap.db ".backup ..."` is safe while running; WAL mode). Restoring one without the other is safe for jobs
(recovery re-reads checkpoints) but may orphan review items; run `tap serve` once and let `recover()` reconcile.

## Signals

| Signal | Where | Alert on |
|---|---|---|
| Cache/TM hit mix | `tap_cache_lookups_total{result}` | exact-hit share drops after a deploy or glossary change |
| Routing / decisions | `tap_segments_total{path,decision}` | human or repair share drifts |
| Engines | `tap_engine_calls_total{engine,outcome}` | error rate (every failed call is also logged with the engine name) |
| Gateway | `tap_gateway_calls_total{tool,status}` | bursts of `denied` (injection/scope) |
| Latency | `tap_segment_seconds`, `tap_http_request_seconds` | p95 per stage |
| Queues | `tap_review_queue_depth`, `tap_event_dead_letters` | depth over SLA; any dead letter |
| Cost | `GET /api/v1/metering/report` | cost per 1k words by path, reuse share, savings vs baseline |
| Quality | `GET /api/v1/feedback/stats` | post-edit distance trend; plus weekly human MQM sampling (process, not code) |
| Integrity | `GET /api/v1/audit/verify`, `tap verify-audit` | `ok: false` is an incident |

Logs are JSON with `job_id`/`request_id`; **source text is never logged**. LangSmith traces contain inputs/outputs unless
`LANGSMITH_HIDE_INPUTS/OUTPUTS=true` is set: set it for regulated content.

## Change management (how a change reaches production)

1. `make check` and `make eval` locally (the golden gate compares with `data/golden/baseline.json`).
2. PR → CI repeats lint, types, tests on 3.11–3.13, the gate and a container smoke test.
3. Prompt/model/threshold changes: extend the golden set with the failure you are fixing, bump the prompt version in
   `engines/prompts.py`, regenerate the baseline *in the same PR* and justify the diff.
4. Roll out behind an experiment: `POST /experiments` with variants of the route parameters (`{"medium_risk_route": "llm"}`);
   jobs are assigned deterministically; read `/experiments/{name}/analysis` (it refuses to call a winner below `min_samples`
   and flips to `guardrail_failed` if quality non-inferiority fails). Plan sizes with `/experiments/sample-size`.
5. Rollback = stop the experiment (`POST /experiments/{name}/stop`) or revert the release.

## Incident drills (from the guide) and what the platform does

| Scenario | Behaviour / first actions |
|---|---|
| NMT provider outage | retries (3, backoff+jitter) → breaker opens after 5 failures → segments fall back to the LLM; with no engine left they go to humans with reason `gateway_*`/`no_engine_available`. Check `tap_engine_calls_total`. |
| LLM/judge outage | repair is skipped (`can_repair` false) → human; scoring failure escalates the segment |
| Quality dip after a prompt/model change | stop the experiment / revert; run `make eval`; sample humans; compare `post_edits.distance` by engine |
| Glossary change | `POST /glossary/commit` returns the impacted TM entry count; new version ⇒ old cache keys unreachable; 100% matches from older versions are drafts that must pass the term check |
| Cost spike | `/metering/report` by path; look for repair loops (`nmt+repair` share), a cache backend that was cleared, a low `TAP_DEFAULT_JOB_BUDGET_UNITS` being hit (budget denials appear as `gateway_budget`) |
| Wrong tool call attempts | `GET /audit?action=tool.denied`; review prompts/tool descriptions; denials are expected and safe |
| Stuck job | `GET /jobs/{id}`: `needs_attention` → fix cause, `POST /jobs/{id}/retry`; `awaiting_review` → reviewer queue; restart is safe (recovery) |
| Dead-lettered events | `/readyz` and `tap_event_dead_letters` show them; fix the consumer, then `POST /api/v1/admin/events/requeue-dead` (admin) retries them; consumers are idempotent |

## Security checklist for production

- `TAP_ENV=prod` (refuses dev keys and demo backends); provide `TAP_API_KEYS_JSON` with long random keys per service/user,
  or front the API with an OIDC/JWT gateway and map claims to `Principal`.
- Terminate TLS in front; `/metrics` is unauthenticated: keep it on an internal network.
- Use vendor agreements with no-training/zero-retention terms for LLM providers; keep `LANGSMITH_HIDE_*` on.
- Run the container non-root (the image does) with `/data` as the only writable volume; rotate keys; restrict `TAP_CORS_ORIGINS`.
- Treat translated text, glossary files and TMX/XLIFF uploads as untrusted input (XML is parsed with defusedxml; HTML is never rendered by the console).
- Ship the audit head hash (`/audit/verify`) to WORM storage periodically: the chain is tamper-*evident* only.

## Scaling out (what changes)

| Today | For multi-node |
|---|---|
| SQLite WAL, one writer lock | Postgres per service (swap `db.py`; schema steps are plain SQL) |
| In-process thread pools | job queue + workers; `JobRunner.enqueue` is the seam |
| In-memory cache or Redis | Redis (already supported) |
| `SqliteSaver` checkpointer | LangGraph Postgres saver |
| In-process event relay | Kafka/SQS consumers; keep outbox + idempotent consumers |
| Hashing embedder scan | vector store (pgvector/Qdrant) behind `SemanticIndex` |
| Rate limiter per process | gateway/Redis token bucket |
