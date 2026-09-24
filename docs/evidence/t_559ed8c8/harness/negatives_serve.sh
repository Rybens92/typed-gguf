#!/usr/bin/env bash
# Story 3, part 1: the serve wave's negative paths (SPEC A-E5-1/3 + the two §2.9 bounds).
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
unset PYTHONPATH
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
unset HTTPS_PROXY https_proxy
cd /workspace/e2e-t_559ed8c8
GG=venv/bin/typed-gguf
BASE=http://127.0.0.1:8088
SENTINEL=E2E-SENTINEL-AUTH-efe9

post() {  # post <logfile> <json>
  curl -sS -o "$1" -w '%{http_code}' -X POST "$BASE/v1/systemone" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer $SENTINEL" -d "$2"
  echo
}

$GG keep stop > /dev/null 2>&1
$GG serve --host 127.0.0.1 --port 8088 --keep-alive 0 > logs/serve_neg.log 2>&1 &
SERVE=$!
for _ in $(seq 1 40); do curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 0.25; done
echo "serve pid=$SERVE (--keep-alive 0: no host outlives the requests)"

echo
echo "### N1. the gate client against a *different* SDK (its own guard, exit 3)"
TYPESAFE_API_KEY=local sdkvenv/bin/python /workspace/ggufone/tools/host_gate_serve_client.py \
  --expect-sdk 9.9.9 --base-url "$BASE"; echo "EXIT=$?"

echo
echo "### N2. unknown top-level key -> 422 extra_forbidden"
echo -n "status: "
post logs/n2_extra_key.json '{"state": "x", "model": "jev-latest", "questions": {"a": {"type": "noul", "instructions": "?"}}, "extra_key": 1}'
cat logs/n2_extra_key.json; echo

echo
echo "### N3. malformed JSON -> 422 json_invalid"
echo -n "status: "
post logs/n3_malformed.json 'not json at all'
cat logs/n3_malformed.json; echo

echo
echo "### N3b. required key missing (no model) -> 422 missing"
echo -n "status: "
post logs/n3_missing.json '{"state": "x", "questions": {"a": {"type": "noul", "instructions": "?"}}}'
cat logs/n3_missing.json; echo

echo
echo "### N3c. unknown model -> 422 value_error (E_MODEL_NOT_FOUND)"
echo -n "status: "
post logs/n3_unknown_model.json '{"state": "x", "model": "no-such-alias", "questions": {"a": {"type": "noul", "instructions": "?"}}}'
cat logs/n3_unknown_model.json; echo

echo
echo "### N3d. engine validation (score with one level) -> 422 value_error"
echo -n "status: "
post logs/n3_score_levels.json '{"state": "x", "model": "jev-latest", "questions": {"a": {"type": "score", "instructions": "?", "criteria": ["only-one"]}}}'
cat logs/n3_score_levels.json; echo

echo
echo "### N3e. unknown route -> 404"
echo -n "status: "
curl -sS -o logs/n3_route.json -w '%{http_code}' "$BASE/v1/nope"; echo
cat logs/n3_route.json; echo

echo
echo "### N4. the 1 MiB body cap, answered from Content-Length alone"
env -u PYTHONPATH venv/bin/python src/http_bounds.py too_large --path /v1/systemone
env -u PYTHONPATH venv/bin/python src/http_bounds.py too_large --path /v1/decide

echo
echo "### N5. the idle-connection bound (30 s)"
env -u PYTHONPATH venv/bin/python src/http_bounds.py idle --wait 45

echo
echo "### N6. the sentinel Authorization value in the server's log (must be 0)"
grep -c "$SENTINEL" logs/serve_neg.log || true

echo
echo "### the server's own log for the whole negative pass"
cat logs/serve_neg.log

echo
echo "### teardown: SIGTERM, then the ledger"
kill -TERM "$SERVE" 2>/dev/null; wait "$SERVE" 2>/dev/null; echo "serve exit=$?"
sleep 1
$GG keep stop; echo "keep stop EXIT=$?"
ls -la "$TYPED_GGUF_HOME/keep"
ps -eo pid,args | grep -E "keep_host|serve --host" | grep -v grep || echo "(no keep/serve process)"
