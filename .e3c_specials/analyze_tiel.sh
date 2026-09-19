#!/bin/bash
# Card t_635124bf: the live post-fix Tiel re-run vs the committed pre-fix run, item by item.
set -u
export HOME=/work/agent-home UV_CACHE_DIR=/work/.uv-cache TMPDIR=/tmp
cd /workspace/ggufone || exit 126
exec uv run --frozen --extra dev python .e3c_specials/batch_verdicts.py \
  --before .e3c_tiel/batch_response.json \
  --after /work/t635/tiel-post/batch_response.json \
  --model /var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
  --label "Tiel serving shape · before = committed host run (pre-fix) · after = this box (post-fix)" \
  --out .e3c_specials/tiel_serving_rerun.json
