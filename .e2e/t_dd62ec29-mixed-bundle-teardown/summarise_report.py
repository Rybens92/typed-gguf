"""Summarise the rows of the bench reports inside `.raw` run logs (the evidence files)."""

import sys

from raw_report import report_at

for name in sys.argv[1:]:
    report = report_at(name)
    print(f"== {name.split('/')[-1]}: ok={report['ok']} wall_ms={report.get('wall_ms')}")
    print(f"   isolation={(report.get('isolation') or {}).get('one_bundle_per_process', False)}")
    for row in report["backends"]:
        if row.get("measured"):
            print(f"   row {row['backend']}: runtime_dir={row.get('runtime_dir')}")
            print(f"       devices={row.get('devices')} buffers={row.get('device_buffers')} "
                  f"effective={row.get('effective_backend')} warnings={row.get('warnings')}")
            print(f"       process={row.get('process')}")
            print(f"       prefill_tok_s={row.get('prefill_tok_per_s', {}).get('p50')} "
                  f"decision_ms={row.get('decision_ms', {}).get('p50')}")
        else:
            print(f"   row {row['backend']}: not measured: {str(row.get('reason'))[:220]}")
    for note in report.get("notes", []):
        if "one bundle per process" not in note:
            print(f"   note: {note[:220]}")
