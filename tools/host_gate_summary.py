#!/usr/bin/env python3
"""Fold a `tools/host_gate_e1a.sh` log directory into one JSON evidence block.

    python3 tools/host_gate_summary.py <log-dir>      # writes <log-dir>/host_gate_e1a.json

Every number here comes from a file the gate wrote (exit codes, elapsed seconds, the raw
outputs); nothing is computed by hand. The result is the "host" / "sandbox_run" section of
docs/evidence/e1a_baseline.json: raw commands + outputs + measured timings.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

# step -> (command, what the exit code means)
STEPS: dict[str, tuple[str, str]] = {
    "pytest_before": ("uv run pytest -q", "0 = offline suite green on this host"),
    "init_dry_run": ("uv run ggufone init --dry-run --json", "0 = plan only, nothing written"),
    "init": ("uv run ggufone init --json", "0 = installed (see variant/fallback below)"),
    "doctor": ("uv run ggufone doctor --json", "0 ok / 2 warnings / 1 failures"),
    "version": ("uv run ggufone version --json", "0 = record read back"),
    "pytest_after": ("uv run pytest -q", "0 = suite green with the runtime installed"),
    "oracle": ("python3 docs/verify_runtime_contract.py", "0 = exit 0, no FAIL"),
    "pytest_network": ("uv run pytest -q --run-network", "0 = live tests green"),
    "init_poisoned_path": ("PATH=<poisoned> uv run ggufone init --json",
                           "0 = installed without a toolchain; <=180 s"),
}
TAIL = 1200


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def tail(text: str, limit: int = TAIL) -> str:
    return text[-limit:]


def step_block(log_dir: pathlib.Path, name: str) -> dict[str, object]:
    out = read(log_dir / f"{name}.out")
    err = read(log_dir / f"{name}.err")
    exit_code = read(log_dir / f"{name}.exit").strip()
    elapsed = read(log_dir / f"{name}.elapsed").strip()
    command, meaning = STEPS.get(name, ("?", ""))
    block: dict[str, object] = {
        "cmd": command,
        "exit": int(exit_code) if exit_code.lstrip("-").isdigit() else None,
        "elapsed_s": int(elapsed) if elapsed.isdigit() else None,
        "meaning": meaning,
        "stdout_tail": tail(out),
    }
    if err.strip():
        block["stderr_tail"] = tail(err)
    return block


def parse_json_output(text: str) -> dict:
    """The last complete JSON object in `text` (doctor/init/version print one)."""
    start = text.find("{")
    while start != -1:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return {}


def count_section_b_skips(oracle_out: str) -> int:
    section = oracle_out.split("[B] live runtime probes")[-1].split("[C] arithmetic mirror")[0]
    return len(re.findall(r"^\s*SKIP ", section, flags=re.MULTILINE))


def main(argv: list[str]) -> int:
    log_dir = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    facts = read(log_dir / "host_facts.txt")
    report: dict[str, object] = {
        "schema": "ggufone.evidence.e1a.host_gate/v1",
        "log_dir": str(log_dir),
        "host_facts": facts,
        "steps": {name: step_block(log_dir, name) for name in STEPS},
        "poison_shims_invoked": read(log_dir / "poison_calls.log").count("\n"),
    }

    init_payload = parse_json_output(read(log_dir / "init.out"))
    record = parse_json_output(read(log_dir / "fallback_evidence.txt"))
    record_reason = record.get("fallback_reason") if isinstance(record, dict) else None
    # `init` on an already-installed runtime reports the attempts it made plus the reason the
    # record carries; the fresh-install path reports its own. Either way the run must not read
    # as "no fallback happened" (E1a FIX finding 3).
    fallback_reason = init_payload.get("fallback_reason") or record_reason
    report["install"] = {
        "variant": init_payload.get("variant"),
        "backend": init_payload.get("backend"),
        "working_backend": init_payload.get("working_backend"),
        "backends": init_payload.get("backends"),
        "build": init_payload.get("build"),
        "symbols_ok": init_payload.get("symbols_ok"),
        "asset": init_payload.get("asset"),
        "bytes_fetched": init_payload.get("bytes_fetched"),
        "fallback_attempts": init_payload.get("fallback_attempts", []),
        "fallback_reason": fallback_reason,
        "fallback_reason_code": init_payload.get("fallback_reason_code")
        or (record.get("fallback_reason_code") if isinstance(record, dict) else None),
        "record": record or None,
    }

    doctor = parse_json_output(read(log_dir / "doctor.out"))
    report["doctor"] = {
        "status": doctor.get("status"),
        "exit_code": doctor.get("exit_code"),
        "runtime": doctor.get("runtime"),
        "checks": doctor.get("checks"),
        "expected_backend": doctor.get("expected_backend"),
    }

    oracle_out = read(log_dir / "oracle.out")
    report["oracle"] = {
        "failures": len(re.findall(r"^\s*FAIL ", oracle_out, flags=re.MULTILINE))
        if oracle_out
        else None,
        "section_b_skips": count_section_b_skips(oracle_out) if oracle_out else None,
        "tail": tail(oracle_out),
    }
    report["suite"] = {
        "pytest_before": tail(read(log_dir / "pytest_before.out"), 400).strip(),
        "pytest_after": tail(read(log_dir / "pytest_after.out"), 400).strip(),
        "pytest_network": tail(read(log_dir / "pytest_network.out"), 400).strip(),
    }

    target = log_dir / "host_gate_e1a.json"
    target.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    summary = {
        "init.exit": report["steps"]["init"]["exit"],  # type: ignore[index]
        "init.variant": report["install"]["variant"] if isinstance(report["install"], dict)
        else None,
        "init.fallback": report["install"].get("fallback_reason")  # type: ignore[union-attr]
        if isinstance(report["install"], dict) else None,
        "doctor.exit": report["steps"]["doctor"]["exit"],  # type: ignore[index]
        "oracle.exit": report["steps"]["oracle"]["exit"],  # type: ignore[index]
        "oracle.section_b_skips": report["oracle"]["section_b_skips"],  # type: ignore[index]
        "pytest_before.exit": report["steps"]["pytest_before"]["exit"],  # type: ignore[index]
        "pytest_after.exit": report["steps"]["pytest_after"]["exit"],  # type: ignore[index]
        "pytest_network.exit": report["steps"]["pytest_network"]["exit"],  # type: ignore[index]
        "poisoned_init.exit": report["steps"]["init_poisoned_path"]["exit"],  # type: ignore[index]
        "poison_shims_invoked": report["poison_shims_invoked"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
