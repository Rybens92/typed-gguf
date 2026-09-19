#!/bin/bash
# E3c-Tiel extras, second pass: the label-policy sweep (with the host's --states-home) and the
# 20-question batch (with src/ on the child's PYTHONPATH). Sequential — one model load at a time.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
LOG=/var/home/rybens/.e3c_tiel
{
  echo "=== tiel extras2 start $(date -Is)"
  bash .e3c_tiel/run_labels.sh
  echo "=== labels rc=$? $(date -Is)"
  bash .e3c_tiel/run_batch.sh
  echo "=== batch rc=$? $(date -Is)"
  echo "=== tiel extras2 done $(date -Is)"
} > "$LOG/extras2.log" 2>&1
