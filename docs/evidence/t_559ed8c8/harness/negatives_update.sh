#!/usr/bin/env bash
# Story 3, part 2: the `runtime update|rollback` refusals and failure paths (SPEC A-E5-8/9).
# Every leg quotes the record's hash before/after and the fixture's own request log, so
# "nothing was changed" is a measurement, not a claim.
set -u
export TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home
export SSL_CERT_FILE=/workspace/e2e-t_559ed8c8/ca/ca.pem
export HTTPS_PROXY=http://127.0.0.1:8443 https_proxy=http://127.0.0.1:8443
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
cd /workspace/e2e-t_559ed8c8
GG="env -u PYTHONPATH venv/bin/typed-gguf"
REC="$TYPED_GGUF_HOME/runtime.json"

hash_now() { sha256sum "$REC" | awk '{print $1}'; }
scenario() { printf '{"releases": "%s", "asset": "%s", "throttle_ms": 0}\n' "$1" "$2" > fixtures/scenario.json; }

leg() {  # leg <title> <expected-code-fragment> <command...>
  local title="$1" expect="$2"; shift 2
  echo
  echo "======================================================================================="
  echo "### $title"
  echo "### scenario: $(cat fixtures/scenario.json)"
  : > logs/fixture.log
  local before after; before=$(hash_now)
  echo "### command: $(printf '%q ' "$@")"
  "$@" > logs/leg.out 2> logs/leg.err
  local code=$?
  echo "EXIT=$code"
  echo "--- stdout: $(cat logs/leg.out)"
  echo "--- stderr: $(cat logs/leg.err)"
  after=$(hash_now)
  echo "--- record before: $before"
  echo "--- record after:  $after"
  [ "$before" = "$after" ] && echo "--- runtime.json byte-identical: YES" \
                           || echo "--- runtime.json byte-identical: NO"
  echo "--- fixture saw:"; sed 's/^/      /' logs/fixture.log
  echo "--- runtime root:"; ls "$TYPED_GGUF_HOME/runtime" | sed 's/^/      /'
  if [ -n "$expect" ]; then
    grep -q "$expect" logs/leg.err && echo "--- expected code present: $expect" \
                                   || echo "--- EXPECTED CODE MISSING: $expect"
  fi
}

scenario ok ok
echo "### record at the start of the negative pass:"; env -u PYTHONPATH venv/bin/python src/record_summary.py "$REC"

# U1 — rung 1 is refused, not updated (SPEC 2.8 step 1)
scenario ok ok
leg "U1. TYPED_GGUF_RUNTIME_DIR is set -> E_UPDATE_UNAVAILABLE" "E_UPDATE_UNAVAILABLE" \
  env TYPED_GGUF_RUNTIME_DIR=/opt/managed-runtime "$GG" runtime update --check

# U2 — nothing installed at all
scenario ok ok
leg "U2. empty data home -> E_RUNTIME_MISSING" "E_RUNTIME_MISSING" \
  env TYPED_GGUF_HOME=/workspace/e2e-t_559ed8c8/home-clean "$GG" runtime update

# U3 — no network: the classic offline answer, naming the URL it could not reach
scenario ok ok
leg "U3. no network -> E_DOWNLOAD_FAILED naming the URL" "E_DOWNLOAD_FAILED" \
  env HTTPS_PROXY=http://127.0.0.1:1 https_proxy=http://127.0.0.1:1 "$GG" runtime update --check

# U4 — the release list has no asset under this host's pinned name
scenario no_target_asset ok
leg "U4. release carries no bundle for this host -> E_UPDATE_UNAVAILABLE" "E_UPDATE_UNAVAILABLE" \
  $GG runtime update --tag b99997

# U5 — the asset URL 404s
scenario ok missing
leg "U5. asset 404 -> E_DOWNLOAD_FAILED" "E_DOWNLOAD_FAILED" \
  $GG runtime update --tag b99997

# U6 — the transfer ends early (a truncated body)
scenario ok short
leg "U6. truncated asset -> E_DOWNLOAD_FAILED (size)" "E_DOWNLOAD_FAILED" \
  $GG runtime update --tag b99997

# U7 — the bytes do not match the advertised digest
scenario wrong_digest ok
leg "U7. digest mismatch -> E_SHA256_MISMATCH" "E_SHA256_MISMATCH" \
  $GG runtime update --tag b99997

# U8 — the t_ba767a2b fix: `init`'s own pre-flight runs BEFORE the download
scenario ok ok
leg "U8. --backend cuda on a host without the CUDA libs -> E_RUNTIME_SYMBOLS, no download" \
  "E_RUNTIME_SYMBOLS" \
  $GG runtime update --backend cuda

# U9 — the release API itself fails
scenario http_500 ok
leg "U9. release API 500 -> E_DOWNLOAD_FAILED" "E_DOWNLOAD_FAILED" \
  $GG runtime update

# U10 — rollback with no `previous` recorded
scenario ok ok
leg "U10. rollback with no previous -> E_UPDATE_UNAVAILABLE" "E_UPDATE_UNAVAILABLE" \
  $GG runtime rollback

scenario ok ok
echo
echo "### the record after the whole negative pass:"; env -u PYTHONPATH venv/bin/python src/record_summary.py "$REC"
echo
echo "### residue in the data home (downloads / staging)"
ls -la "$TYPED_GGUF_HOME/downloads"
ls -a "$TYPED_GGUF_HOME/runtime"
echo
echo "### the runtime is still the one that was active before the pass"
$GG version; echo "EXIT=$?"
