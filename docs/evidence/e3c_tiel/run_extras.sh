#!/bin/bash
# E3c-Tiel after the quality chunks: the label-policy sweep (deliverable 3's control), the
# 20-question batch (deliverable 4) and the threads probe (deliverable 5) — one systemd unit,
# sequentially, so the 21 GB model is never loaded twice at once.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
LOG=/var/home/rybens/.e3c_tiel
{
  echo "=== tiel extras start $(date -Is)"
  bash .e3c_tiel/run_labels.sh
  echo "=== labels rc=$? $(date -Is)"
  bash .e3c_tiel/run_batch.sh
  echo "=== batch rc=$? $(date -Is)"
  bash .e3c_tiel/run_threads.sh
  echo "=== threads rc=$? $(date -Is)"
  echo "=== tiel extras done $(date -Is)"
} > "$LOG/extras.log" 2>&1
