#!/bin/bash
# E3c-Tiel deliverable 3 input: the E3b label-policy sweep run on Tiel, so the mass split can be
# read as "the model barely puts mass on our labels" vs "the cue shape suppressed the answer"
# (E3b's control), not just as one number.
#
# Same shape E3b ran on Occamy (tools/e3b_label_policy.py, card t_6952f0dd): every cue x every
# label variant, coverage only, 2 items per type, one model load. Policy recorded with the result:
# the shipped `bare` rendering (prompt.build_question is byte-identical — E3b's own gate), i.e.
# nothing in the engine's default label policy changed for this campaign.
#
# `--states-home` must be given on the host: the tool's default is a container path (`/work/e3b`).
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
export GGUFONE_RUNTIME_DIR=/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan
export VK_DRIVER_FILES=/home/rybens/.e3c_host/nvidia_egl_icd.json
export PYTHONUNBUFFERED=1
unset VK_INSTANCE_LAYERS
MODEL=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
echo "tiel label sweep start $(date -Is) pid $$"
python3 tools/e3b_label_policy.py run \
  --model "$MODEL" \
  --devset docs/evidence/tiel_chunks/devset_001.jsonl --per-type 2 \
  --gpu-layers 9 --threads 4 \
  --states-home .e3c_tiel/label_states \
  --out .e3c_tiel/tiel_label_sweep.json \
  --report docs/evidence/tiel_label_policy_tables.md \
  --report-dir .e3c_tiel/label_reports 2>&1 | tail -30
rc=$?
echo "tiel label sweep exit=$rc $(date -Is)"
exit "$rc"
