#!/bin/bash
# Wait for chunk 001's log to close (its driver appends EXIT=...), then run chunk 002 — one
# background chain so a single poll on this session covers both chunks. ~25 s poll, max 3 h.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
deadline=$(( $(date +%s) + 10800 ))
while ! grep -q '^EXIT=' .e3b/logs/remeasure_001.log 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "chunk 001 did not finish within 3 h"; exit 1
  fi
  sleep 25
done
echo "chunk 001 done ($(grep '^EXIT=' .e3b/logs/remeasure_001.log)) — starting chunk 002"
bash .e3b/remeasure_002.sh > .e3b/logs/remeasure_002.log 2>&1
echo "EXIT=$?" >> .e3b/logs/remeasure_002.log
