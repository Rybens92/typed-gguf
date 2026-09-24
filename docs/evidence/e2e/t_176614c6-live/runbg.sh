#!/usr/bin/env bash
# runbg.sh NAME -- CMD... : run in the background, capturing stdout/stderr/exit/wall_ms.
set -u
name="$1"; shift; shift  # drop NAME and --
start=$(date +%s.%N)
"$@" > "out/$name.stdout" 2> "out/$name.stderr"
rc=$?
end=$(date +%s.%N)
echo "$rc" > "out/$name.exit"
python3 -c "import sys; print(int((float(sys.argv[2]) - float(sys.argv[1])) * 1000))" "$start" "$end" > "out/$name.wall_ms"
exit $rc
