#!/bin/bash
# E3e (card t_4c48f40a) stage 5: the table, the freeze and the re-score.
#
# Reads the eight `--backend vulkan` arms `.e3e/run_arms.sh` wrote, compares them item by item
# (paired), and checks the two claims the card rests on against the *committed* baseline
# (`.e3d/bench_templated_shipped.json`, measured with `--backend auto` by the E3d card):
#
#   * `--freeze-probe .e3e/probe_default.json` — the six-item arm run with the baseline's own recipe.
#     Contract: the prompt bytes (`prefix_tokens`) and the decisions are the baseline's. The numbers
#     are NOT expected to be bit-identical any more: card `t_55de5779` landed after the baseline was
#     published and a `--backend auto` row that claims `cpu` now really computes on the CPU, which is
#     exactly what the probe's log shows (`CPU compute buffer size`).
#   * `--placement-probe .e3e/bench_shipped_answer_sheet.json` — the table's own baseline cell (60
#     items, `--backend vulkan`), which is the same measurement under the table's instrument.
#
# Exit codes: 3 = the default moved (bytes or decisions), 4 = the table's own baseline cell answers
# differently from the committed one (then the table is not one measurement and must not be read).
#
# Reproduce (from the repo root, after `bash .e3e/run_arms.sh`):
#     bash .e3e/report.sh
#
# The recommendation lives in `.e3e/recommendation.md` (read verbatim, never re-quoted in the shell:
# the text carries backticks and apostrophes).
set -u
cd "$(dirname "$0")/.." || exit 126
export HOME="${HOME:-/work/agent-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/.uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"

RECOMMENDATION="$(cat .e3e/recommendation.md)"

exec uv run --frozen python tools/e3e_roles_decision.py \
    --report .e3e/bench_shipped_answer_sheet.json \
    --report .e3e/bench_shipped_role_split.json \
    --report .e3e/bench_two_step_answer_sheet.json \
    --report .e3e/bench_two_step_role_split.json \
    --report .e3e/bench_json_instructed_answer_sheet.json \
    --report .e3e/bench_json_instructed_role_split.json \
    --report .e3e/bench_json_instructed_answer_sheet_system.json \
    --report .e3e/bench_json_instructed_role_split_system.json \
    --baseline "shipped/answer_sheet" \
    --freeze-probe .e3e/probe_default.json \
    --freeze-against .e3d/bench_templated_shipped.json \
    --placement-probe .e3e/bench_shipped_answer_sheet.json \
    --json docs/evidence/e3e_roles_decision.json \
    --report-file docs/evidence/e3e_roles_decision.md \
    --recommendation "$RECOMMENDATION"
