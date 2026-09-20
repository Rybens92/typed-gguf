#!/bin/bash
# t_9bcbecff — merge the per-chunk reports and regenerate the statistics + the evidence document.
#
# The evidence document is rendered from `.t9bcb/stats.json`, and the statistics are read from the
# merged reports by the committed tools (`tools/e3_reproduce.py` for the merge,
# `tools/e3e_roles_decision.py` for the Wilson / exact-McNemar / closed-form-interval work). No
# number is retyped anywhere in this chain.
set -u
cd /var/home/rybens/workspace/ggufone-wt-t9bcb || exit 126
CH=docs/evidence/t9bcbecff_tiel_chunks

python3 tools/e3_reproduce.py --suite merge --reports "$CH/report_*.json" \
  --out docs/evidence/t9bcbecff_tiel_challenger_quality.json \
  --label "Tiel-Coder-35B-A3B (t_9bcbecff, role_split + json_instructed)" || exit 1

python3 tools/e3_reproduce.py --suite merge --reports "$CH/aux_role_split_report_*.json" \
  --out docs/evidence/t9bcbecff_tiel_role_split_quality.json \
  --label "Tiel-Coder-35B-A3B (t_9bcbecff aux, role_split + shipped cue)" || exit 1

python3 tools/e3_reproduce.py --suite merge --reports "$CH/aux_two_step_report_*.json" \
  --out docs/evidence/t9bcbecff_tiel_two_step_quality.json \
  --label "Tiel-Coder-35B-A3B (t_9bcbecff aux, role_split + two_step)" || exit 1

# the optional Occamy pass: both cells are measured there, so both are merged (see run_occamy.sh)
OC=docs/evidence/t9bcbecff_occamy_chunks
if compgen -G "$OC/occ_base_report_*.json" > /dev/null; then
  python3 tools/e3_reproduce.py --suite merge --reports "$OC/occ_base_report_*.json" \
    --out docs/evidence/t9bcbecff_occamy_base_quality.json \
    --label "Occamy-1.0 (t_9bcbecff occamy, shipped placement + shipped cue)" || exit 1
  python3 tools/e3_reproduce.py --suite merge --reports "$OC/occ_e3e_report_*.json" \
    --out docs/evidence/t9bcbecff_occamy_e3e_quality.json \
    --label "Occamy-1.0 (t_9bcbecff occamy, role_split + json_instructed)" || exit 1
fi

python3 .t9bcb/analyse.py || exit 1
python3 .t9bcb/render_doc.py || exit 1
echo "merge + analysis + document done"
