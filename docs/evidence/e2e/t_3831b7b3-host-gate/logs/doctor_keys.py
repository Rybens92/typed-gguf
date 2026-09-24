#!/usr/bin/env python3
"""Extract the doctor/init contract keys this card's acceptance reads (gate logs)."""
import json
import pathlib
import sys


def last_json(p):
    t = pathlib.Path(p).read_text()
    i = t.find("{")
    while i != -1:
        try:
            return json.loads(t[i:])
        except json.JSONDecodeError:
            i = t.find("{", i + 1)
    return {}


def main(logdir):
    init = last_json(logdir + "/init.out")
    doc = last_json(logdir + "/doctor.out")
    print(f"== {logdir}/init.out (init --json)")
    for k in ("variant", "backend", "working_backend", "backends", "asset",
              "bytes_fetched", "asset_verified", "fallback_reason_code"):
        print(f"  init.{k} = {json.dumps(init.get(k))[:180]}")
    print(f"  init.fallback_reason = {json.dumps(init.get('fallback_reason'))[:220]}")
    print("  init.fallback_attempts =", json.dumps(init.get("fallback_attempts"))[:400])
    print(f"== {logdir}/doctor.out (doctor --json)")
    for k in ("status", "exit_code", "backend", "backends", "expected_backend"):
        print(f"  doctor.{k} = {json.dumps(doc.get(k))[:220]}")
    rt = doc.get("runtime", {})
    for k in ("working_backend", "backends", "backend_requested", "fallback_reason_code"):
        print(f"  doctor.runtime.{k} = {json.dumps(rt.get(k))[:220]}")
    print("  doctor.runtime.fallback_reason =", json.dumps(rt.get("fallback_reason"))[:300])
    print("  doctor.runtime.fallback_attempts =", json.dumps(rt.get("fallback_attempts"))[:400])
    chk = [c for c in doc.get("checks", [])
           if c.get("id") in ("runtime.fallback", "runtime.accelerator", "runtime.backends")]
    print("  checks:", json.dumps(chk)[:700])


if __name__ == "__main__":
    main(sys.argv[1])
