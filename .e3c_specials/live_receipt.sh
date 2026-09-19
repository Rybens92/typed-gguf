#!/bin/bash
# Card t_635124bf: the live re-run receipt — comparison + row identity, one file.
set -u
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
cd /workspace/ggufone || exit 126
OUT=.e3c_specials
{
  echo "# Card t_635124bf — Tiel, serving shape, live post-fix re-run vs the committed pre-fix run."
  echo "# $(date -u '+%Y-%m-%dT%H:%M:%SZ') · container: cgroup $(cat /sys/fs/cgroup/memory.max)"
  echo "#   before: .e3c_tiel/batch_response.json (host, pre-fix, 35.6 s)"
  echo "#   after:  /work/t635/tiel-post/batch_response.json (this box, post-fix, 1460.9 s)"
  echo
  uv run --frozen --extra dev python "$OUT/batch_verdicts.py" \
    --before .e3c_tiel/batch_response.json \
    --after /work/t635/tiel-post/batch_response.json \
    --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
    --label "Tiel serving shape · before = committed host run · after = this box" 2>&1 \
    | grep -vE "^(llama_|load_backend|ggml_|load:|print_info|init_|gguf_)"
  echo
  echo "## row identity (did the two boxes put the same row on the cue?)"
  uv run --frozen --extra dev python "$OUT/row_identity.py" 2>&1
} > "$OUT/tiel_serving_rerun.log"
grep -E "^(refusals|measured|silently|refusals the|argmax ids|cue masses|usage before|state id)" \
  "$OUT/tiel_serving_rerun.log"
