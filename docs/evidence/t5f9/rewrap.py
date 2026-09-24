#!/usr/bin/env python3
"""Re-wrap the 24 lines the rename pushed past ruff's 100-char gate (E501), one exact edit each."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path("/workspace/ggufone")
BS = chr(92)  # a literal backslash, spelled out: the shell snippets below need two of them

EDITS: list[tuple[str, str, str]] = [
    ("src/typed_gguf/__main__.py",
     '"""`python -m typed_gguf` entry point -> `typed_gguf.cli.run` (the process entry point, card t_97f1bc93).',
     '"""`python -m typed_gguf` -> `typed_gguf.cli.run`, the process entry point (card t_97f1bc93).'),

    ("src/typed_gguf/bench/harness.py",
     '    """The device-evidence parser, imported lazily (A-E2-7: no `typed_gguf.runtime` at import time)."""',
     '    """The device-evidence parser, imported lazily (A-E2-7: not at import time)."""'),

    ("src/typed_gguf/bench/isolation.py",
     '"one that resolved `TYPED_GGUF_BENCH_MODEL` on its own could measure a model "',
     '"one that resolved `TYPED_GGUF_BENCH_MODEL` alone could measure a model "'),

    ("src/typed_gguf/bench/suites.py",
     "Every suite returns one JSON report (`typed_gguf.bench/v1`) whose tables `bench.harness.render_report`\n"
     "renders as the markdown published in `docs/BENCHMARKS.md`. The suites only talk to\n",
     "Every suite returns one JSON report (`typed_gguf.bench/v1`); its tables are rendered as the markdown\n"
     "published in `docs/BENCHMARKS.md` by `bench.harness.render_report`. The suites only talk to\n"),

    ("src/typed_gguf/cli.py",
     "    callers that also report it (`typed-gguf fit --json`), and re-read here otherwise — the plan must\n",
     "    callers that also report it (`typed-gguf fit --json`), re-read here otherwise — the plan must\n"),

    ("src/typed_gguf/cli.py",
     "    caller, `tests/` included. This is the other half — what `python -m typed_gguf` and the installed\n",
     "    caller, `tests/` included. This is the other half — what `python -m typed_gguf` and the\n"),

    ("src/typed_gguf/registry/store.py",
     '    """Base data dir: `$TYPED_GGUF_HOME` > `$XDG_DATA_HOME/typed-gguf` > `~/.local/share/typed-gguf`."""\n',
     '    """Base data dir: `$TYPED_GGUF_HOME` > `$XDG_DATA_HOME/typed-gguf` >\n'
     '    `~/.local/share/typed-gguf`."""\n'),

    ("src/typed_gguf/runtime/ctypes_binding.py",
     "    vocabulary's own answer to \"which of your tokens are not content\" — typed-gguf never decides that\n"
     "    from a string (card t_635124bf; measured cost and cross-check in\n",
     "    vocabulary's own answer to \"which of your tokens are not content\" — typed-gguf never decides\n"
     "    that from a string (card t_635124bf; measured cost and cross-check in\n"),

    ("src/typed_gguf/runtime/fit.py",
     "    Not every caller of `session.open_model` holds a fit plan: `typed-gguf bench` names its placement\n",
     "    Not every caller of `session.open_model` holds a fit plan: `typed-gguf bench` names it\n"),

    ("src/typed_gguf/runtime/teardown.py",
     "Measured on the operator's box (`typed-gguf bench --backend vulkan`, one bundle, a 4B model, a device\n"
     "that is nearly full): the whole report reaches stdout, and then the process dies with **SIGSEGV** —\n",
     "Measured on the operator's box (`typed-gguf bench --backend vulkan`, one bundle, a 4B model, a\n"
     "device that is nearly full): the whole report reaches stdout, then the process dies with\n"
     "**SIGSEGV** —\n"),

    ("tests/test_bench.py",
     '                assert not (node.module or "").startswith(("typed_gguf.registry", "typed_gguf.runtime"))\n'
     "            elif isinstance(node, ast.Import):\n"
     "                for alias in node.names:\n"
     '                    assert not alias.name.startswith(("typed_gguf.registry", "typed_gguf.runtime")), \\\n'
     '                        f"{module}.py imports {alias.name} at module level"\n',
     '                assert not (node.module or "").startswith(\n'
     '                    ("typed_gguf.registry", "typed_gguf.runtime"))\n'
     "            elif isinstance(node, ast.Import):\n"
     "                for alias in node.names:\n"
     "                    assert not alias.name.startswith(\n"
     '                        ("typed_gguf.registry", "typed_gguf.runtime")), \\\n'
     '                        f"{module}.py imports {alias.name} at module level"\n'),

    ("tests/test_bench_isolation.py",
     '    """A child resolves `TYPED_GGUF_BENCH_MODEL` on its own: never let it measure an unnamed model."""\n',
     '    """A child resolves `TYPED_GGUF_BENCH_MODEL` alone: never let it measure an unnamed model."""\n'),

    ("tests/test_bench_placement.py",
     "`typed-gguf bench` names its placement explicitly (``--gpu-layers``, the minimal ``harness.Placement``)\n"
     "instead of consuming a fit plan, and ``session.open_model`` builds the degradation ladder *before*\n"
     "its first load attempt. On the parent tree that combination died with\n",
     "`typed-gguf bench` names its placement explicitly (``--gpu-layers``, the minimal\n"
     "``harness.Placement``) instead of consuming a fit plan, and ``session.open_model`` builds the\n"
     "degradation ladder *before* its first load attempt. On the parent tree that combination died with\n"),

    ("tests/test_bench_quick.py",
     '    assert json.loads((tmp_path / "typed-gguf-bench-throughput_quick.json").read_text())["truncated"]\n',
     '    quick_report = tmp_path / "typed-gguf-bench-throughput_quick.json"\n'
     '    assert json.loads(quick_report.read_text())["truncated"]\n'),

    ("tests/test_bench_teardown_crash.py",
     "can die with **exit -11 (SIGSEGV)** *after* writing a complete `typed_gguf.bench/v1` report, while its\n",
     "can die with **exit -11 (SIGSEGV)** *after* writing a complete `typed_gguf.bench/v1` report, while\n"),

    ("tests/test_bench_vulkan_teardown_live.py",
     "    VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json " + BS + BS + "\n"
     "    TYPED_GGUF_RUNTIME_DIR=/var/home/rybens/.local/share/typed-gguf/runtime/b11026-linux-x64-vulkan "
     + BS + BS + "\n",
     "    RT=/var/home/rybens/.local/share/typed-gguf/runtime/b11026-linux-x64-vulkan\n"
     "    VK_DRIVER_FILES=/work/e3scratch/nvidia_egl_icd.json TYPED_GGUF_RUNTIME_DIR=$RT " + BS + BS + "\n"),

    ("tests/test_engine_fork.py",
     '    pytest.skip("no llama.cpp runtime on this box (set TYPED_GGUF_RUNTIME_DIR or run `typed-gguf init`)")\n',
     '    pytest.skip("no llama.cpp runtime on this box "\n'
     '                "(set TYPED_GGUF_RUNTIME_DIR or run `typed-gguf init`)")\n'),

    ("tests/test_fit.py",
     "`typed-gguf fit` CLI test here pins its host world through `pin_host_facts` (card t_e29734e6), so the\n",
     "`typed-gguf fit` CLI test here pins its host world through `pin_host_facts` (card t_e29734e6), so\n"),

    ("tests/test_probe_pressure.py",
     "    \"\"\"`TYPED_GGUF_TEST_PID_HEADROOM=250/256`: the fork gate skips *loudly*, the headroom gate fails,\n"
     "    and the run exits non-zero — a starved box can never be read as \"the product is fine\".\"\"\"\n",
     "    \"\"\"`TYPED_GGUF_TEST_PID_HEADROOM=250/256`: the fork gate skips *loudly*, the headroom gate\n"
     "    fails, and the run exits non-zero — a starved box can never read as \"the product is fine\".\"\"\"\n"),

    ("tests/test_typed_gguf_surface.py",
     '"""The public name is `typed-gguf` (card t_5f9c15fe): the living surface carries no trace of the old one.\n',
     '"""The public name is `typed-gguf` (card t_5f9c15fe): no trace of the old one in the living surface.\n'),

    ("tools/e2p5_reproduce.py",
     "`calibrate` and `route` drive the shipped CLI (`typed-gguf calibrate`, `typed-gguf run --route auto`) in\n"
     "process, so the evidence exercises the real code path; `escalate` measures the policy the CLI\n",
     "`calibrate` and `route` drive the shipped CLI (`typed-gguf calibrate`, `typed-gguf run --route auto`)\n"
     "in process, so the evidence exercises the real code path; `escalate` measures the policy the CLI\n"),

    ("tools/e3e_role_render.py",
     '                        help="hide the Vulkan ICD (`TYPED_GGUF_HIDDEN_ICD`) for a CPU-only live check")\n',
     '                        help="hide the Vulkan ICD (`TYPED_GGUF_HIDDEN_ICD`) for a CPU-only "\n'
     '                             "live check")\n'),

    ("tools/live_probe.py",
     '        raise SystemExit("no runtime installed (run `typed-gguf init` or set TYPED_GGUF_RUNTIME_DIR)")\n',
     '        raise SystemExit("no runtime installed "\n'
     '                         "(run `typed-gguf init` or set TYPED_GGUF_RUNTIME_DIR)")\n'),
]


def main() -> int:
    problems = []
    for name, old, new in EDITS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        seen = text.count(old)
        if seen != 1:
            problems.append(f"{name}: found {seen} of {old.splitlines()[0][:60]!r}")
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")
    print("\n".join(problems) if problems else f"applied {len(EDITS)} re-wraps")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
