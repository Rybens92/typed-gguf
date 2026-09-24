#!/usr/bin/env bash
# E2E adversarial journeys for t_3831b7b3 (ggufone E1a host-gate card).
# Raw command + stdout/stderr/exit per journey under $OUT.
set -u
REPO=/work/e2e-t3831b7b3/repo
OUT=/work/e2e-t3831b7b3/logs/journeys
mkdir -p "$OUT"
export UV_CACHE_DIR=/work/.uv-cache
export PATH=/work/e2e-t3831b7b3/bin:$PATH   # fake nvidia-smi -> cuda expected backend
cd "$REPO"

run() {  # run <name> <home> <extra-env...> -- <cmd...>
    local name="$1" home="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "### $name :: HOME=$home ${envs[*]:-} :: $*"
    local start end
    start=$(date +%s)
    HOME="$home" GGUFONE_HOME="$home/.hermes" env ${envs[@]+"${envs[@]}"} "$@" \
        >"$OUT/$name.out" 2>"$OUT/$name.err"
    local code=$?
    end=$(date +%s)
    echo "$code" >"$OUT/$name.exit"
    echo "$((end - start))" >"$OUT/$name.elapsed"
    echo "-- $name exit=$code elapsed=$((end - start))s"
    echo "   stdout tail:"; tail -4 "$OUT/$name.out" | sed 's/^/     /'
    echo "   stderr: $(wc -c <"$OUT/$name.err") bytes"; tail -3 "$OUT/$name.err" | sed 's/^/     /'
    echo
}

H2=/work/e2e-t3831b7b3/home2          # runtime + model installed (host-like)
H4=/work/e2e-t3831b7b3/home4          # empty home
H5=/work/e2e-t3831b7b3/home5          # interruption test
mkdir -p "$H4" "$H5"

# J1 idempotent init re-run (finding 3: fallback reason must survive an already_installed run)
run j1_init_rerun "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone init --json

# J2 doctor --json on the installed home
run j2_doctor "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone doctor --json

# J3 dry-run plan (host fact: fake nvidia-smi present -> cuda-12.8)
run j3_init_dry_run "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone init --dry-run --json

# J4 garbage backend value
run j4_backend_garbage "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone init --backend bogus --json

# J5 registry reads + verify
run j5a_models_ls "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone models ls --json
run j5b_models_verify "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone models verify --json

# J6 doctor on an empty home (contract: exit 1 failures)
run j6_doctor_empty "$H4" GGUFONE_HOME="$H4/.hermes" -- uv run ggufone doctor --json

# J7 unwritable data home
run j7_unwritable_home "$H4" GGUFONE_HOME=/proc/ggufone-cannot-write -- uv run ggufone init --json

# J8 interruption: kill init mid-download, then re-run -> must recover
( HOME="$H5" GGUFONE_HOME="$H5/.hermes" timeout -s KILL 1 uv run ggufone init --json \
    >"$OUT/j8_kill.out" 2>"$OUT/j8_kill.err"; echo "$?" >"$OUT/j8_kill.exit" ) || true
echo "### j8_kill (SIGKILL after 1s) exit=$(cat "$OUT/j8_kill.exit" 2>/dev/null)"; tail -2 "$OUT/j8_kill.err" | sed 's/^/     /'; echo
run j8_init_after_kill "$H5" GGUFONE_HOME="$H5/.hermes" -- uv run ggufone init --json
run j8b_doctor_after_kill "$H5" GGUFONE_HOME="$H5/.hermes" -- uv run ggufone doctor --json

# J9 --backend cuda bypasses the pre-flight (offline cache holds the pinned CUDA archive)
H6=/work/e2e-t3831b7b3/home6; mkdir -p "$H6"
run j9_backend_cuda_bypass "$H6" GGUFONE_HOME="$H6/.hermes" GGUFONE_OFFLINE_CACHE=/work/e1a/dl \
    -- uv run ggufone init --backend cuda --json

# J10 rapid repeats: 3 x init in a row on the installed home (idempotency / state leak)
for i in 1 2 3; do
    HOME="$H2" GGUFONE_HOME="$H2/.hermes" uv run ggufone init --json >"$OUT/j10_run$i.out" 2>"$OUT/j10_run$i.err"
    echo "$i exit=$? stderr_bytes=$(wc -c <"$OUT/j10_run$i.err")"
done | tee "$OUT/j10.summary"

# J11 doctor determinism: two runs, byte-compare the JSON body
HOME="$H2" GGUFONE_HOME="$H2/.hermes" uv run ggufone doctor --json >"$OUT/j11_doctor_a.out" 2>&1
HOME="$H2" GGUFONE_HOME="$H2/.hermes" uv run ggufone doctor --json >"$OUT/j11_doctor_b.out" 2>&1
if cmp -s "$OUT/j11_doctor_a.out" "$OUT/j11_doctor_b.out"; then echo "j11 doctor deterministic: identical output"; else echo "j11 DIFFERS:"; diff "$OUT/j11_doctor_a.out" "$OUT/j11_doctor_b.out" | head; fi

# J12 unicode/garbage model reference
run j12_pull_garbage "$H2" GGUFONE_HOME="$H2/.hermes" -- uv run ggufone models pull '🔥not/a-repo:Q9_9'

echo "JOURNEYS-DONE"
