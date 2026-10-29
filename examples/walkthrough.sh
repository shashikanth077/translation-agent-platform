#!/usr/bin/env bash
# End-to-end walkthrough against a running server:   tap serve --seed-demo   (then, in another shell)   ./examples/walkthrough.sh
# Needs only curl and python3. Uses the dev keys (never in prod).
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000/api/v1}"
CLIENT=(-H "X-API-Key: dev-client")
LINGUIST=(-H "X-API-Key: dev-linguist")
JSON=(-H "Content-Type: application/json")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

echo "1) upload a document"
CONTENT=$(python3 -c "import json,sys; print(json.dumps(open(sys.argv[1]).read()))" "$HERE/leaflet.txt")
DOC=$(curl -fsS "${CLIENT[@]}" "${JSON[@]}" -d "{\"filename\":\"leaflet.txt\",\"content\":$CONTENT}" "$BASE/documents" | py "d['id']")
echo "   document: $DOC"

echo "2) create a job (idempotent: re-running with the same key returns the same job)"
JOB=$(curl -fsS "${CLIENT[@]}" "${JSON[@]}" -H "Idempotency-Key: walkthrough-$DOC" \
  -d "{\"document_id\":\"$DOC\",\"target_langs\":[\"de\"],\"domain\":\"clinical\",\"policy\":\"balanced\"}" "$BASE/jobs" | py "d['id']")
echo "   job: $JOB"

echo "3) wait for the workflow to reach the human-review gate"
for _ in $(seq 1 60); do
  STATUS=$(curl -fsS "${CLIENT[@]}" "$BASE/jobs/$JOB" | py "d['status']")
  [[ "$STATUS" == "awaiting_review" || "$STATUS" == "awaiting_approval" ]] && break
  sleep 0.5
done
echo "   status: $STATUS"

echo "4) review queue (as the linguist): resolve each item with the machine candidate, fixing the known meaning error"
for ITEM in $(curl -fsS "${LINGUIST[@]}" "$BASE/review/items?job_id=$JOB" | py "' '.join(i['id'] for i in d)"); do
  curl -fsS -X POST "${LINGUIST[@]}" "$BASE/review/items/$ITEM/claim" >/dev/null
  BODY=$(curl -fsS "${LINGUIST[@]}" "$BASE/review/items/$ITEM" | python3 -c "
import json,sys
p=json.load(sys.stdin)['payload']
print(json.dumps({'text': p['candidate'].replace('angepasst','titriert')}))")
  curl -fsS -X POST "${LINGUIST[@]}" "${JSON[@]}" -d "$BODY" "$BASE/review/items/$ITEM/submit" >/dev/null
  echo "   resolved $ITEM"
done
for _ in $(seq 1 60); do
  STATUS=$(curl -fsS "${CLIENT[@]}" "$BASE/jobs/$JOB" | py "d['status']")
  [[ "$STATUS" == "awaiting_approval" ]] && break
  sleep 0.5
done
echo "   status: $STATUS"

echo "5) approve & deliver (a client PM would get 403 here; linguists/admins can publish)"
curl -fsS -X POST "${LINGUIST[@]}" "$BASE/jobs/$JOB/approve" | py "'   status: ' + d['status']"

echo "6) result"
curl -fsS "${CLIENT[@]}" "$BASE/jobs/$JOB/output/de"; echo
echo "7) cost report (metering is event-driven: give the relay a moment to catch up)"
sleep 1
curl -fsS "${CLIENT[@]}" "$BASE/jobs/$JOB/report" | py "'   cost %s vs baseline %s units (savings %s%%)' % (d['cost_units'], d['baseline_cost_units'], d['savings_pct'])"
