#!/bin/bash
# Watch the box until the 21 GB campaign can be run honestly: free RAM and free VRAM.
# Usage: watch_resources.sh <minutes> <tag>
set -u
MIN=${1:-10}
TAG=${2:-watch}
OUT=/var/home/rybens/workspace/ggufone/.e3c_tiel/resources_${TAG}.log
: > "$OUT"
end=$(( $(date +%s) + MIN * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  read -r mt mf mb ma rest < <(grep -E "^MemTotal|^MemFree|^MemAvailable" /proc/meminfo | tr -d ' kB' | tr '\n' ' ')
  big=$(ps -eo rss,args --sort=-rss | awk 'NR>1 && $1>2000000 {printf "%d:%s ", $1, $2}' | cut -c1-200)
  printf '%s vram_used=%sMiB memfree=%s memavail=%s | %s\n' "$(date -Is)" "$vram" "$(awk '/MemFree/{print $2}' /proc/meminfo)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" "$big" >> "$OUT"
  sleep 20
done
echo "wrote $OUT"
tail -5 "$OUT"
