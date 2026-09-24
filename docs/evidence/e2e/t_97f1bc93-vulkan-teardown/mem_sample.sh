#!/usr/bin/env bash
# Sample the *container's* memory accounting without forking (the pid cap is tight here):
#   mem_sample.sh <out-file> [interval-seconds]
# Each line: `<time> anon=<bytes> file=<bytes> current=<bytes> max=<bytes>`
OUT=${1:?out file}
INTERVAL=${2:-3}
: > "$OUT"
while :; do
  current=""; anon=""; filep=""; max=""
  read -r current < /sys/fs/cgroup/memory.current
  read -r max < /sys/fs/cgroup/memory.max
  while read -r key value _; do
    case "$key" in
      anon) anon=$value ;;
      file) filep=$value ;;
    esac
  done < /sys/fs/cgroup/memory.stat
  {
    printf '%(%H:%M:%S)T current=%s anon=%s file=%s max=%s\n' -1 "$current" "$anon" "$filep" "$max"
  } >> "$OUT"
  sleep "$INTERVAL"
done
