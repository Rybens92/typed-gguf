"""Command-line surface (SPEC 2.8). Milestone: E1a (init/doctor/models), E1b (run/ask).

Implemented in E1a: `init`, `doctor`, `models {search,pull,use,ls,rm,verify,recommend-quant}`,
`version`. The remaining commands keep the frozen names from SPEC 2.8 and exit 3 with a
milestone pointer instead of pretending to work.

Exit codes (SPEC 2.5): 0 ok, 2 user error, 3 runtime/model error, 4 internal.
`doctor` additionally uses 2 for "works, but warnings" and 1 for "broken" (A-E1a-3).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time
from typing import Any

from ggufone import __version__, schema
from ggufone.bench import harness
from ggufone.bench import suites
from ggufone.engine import decide
from ggufone.engine import session as session_module
from ggufone.errors import GgufoneError, ModelNotFoundError, Sha256MismatchError, UserError
from ggufone.registry import gguf, hf, recommend, store
from ggufone.registry.gguf import sha256_file
from ggufone.runtime import capability, finder, fit, install, pins

COMMANDS = ("init", "doctor", "models", "run", "ask", "serve", "mcp", "bench",
            "fit", "calibrate", "version")
MODELS_SUBCOMMANDS = ("search", "pull", "use", "ls", "rm", "verify", "recommend-quant")
# command -> milestone that implements it (SPEC 5)
MILESTONES = {"version": "E0", "init": "E1a", "doctor": "E1a", "models": "E1a",
              "run": "E1b", "ask": "E1b", "fit": "E1c", "serve": "E1b", "mcp": "E1b",
              "bench": "E2", "calibrate": "E2.5"}
DOCTOR_SCHEMA = "ggufone.doctor/v1"
MODELS_SCHEMA = "ggufone.models/v1"

#: per-command help (SPEC 2.8 surface). `ggufone <cmd> --help` prints the flags that command
#: accepts — carried finding #1 of the E1c card: `--help` used to be an E_UNKNOWN_KEY error.
COMMAND_HELP: dict[str, tuple[str, ...]] = {
    "init": ("--backend auto|cpu|vulkan|cuda|metal", "--force", "--dry-run",
             "--offline-cache DIR", "--json"),
    "doctor": ("--json",),
    "models": ("search <query>", "pull <repo[:quant]> [--file NAME] [--no-verify] [--jobs N]",
               "use <alias>", "ls [--json]", "rm <alias>", "verify [alias]",
               "recommend-quant [--vram GIB] [--json]"),
    "run": ("--questions FILE", "--state TEXT|@FILE", "--state-json FILE", "--model REF",
            "--format native|typesafe", "--out FILE", "--template auto|plain|NAME|PATH",
            "--thinking", "--no-fit", "--fit-target MIB", "--fit-ctx N", "--no-fit-cache",
            "--threads N", "--n-ctx N", "--n-seq-max N", "--kv-type auto|f16|q8_0|q4_0",
            "--readout sequence|single_token", "--confidence-mode MODE", "--temperature F",
            "--length-norm F", "--coverage-floor F", "--state-id ID", "--save-state",
            "--no-state-cache", "--strict", "--max-waves N"),
    "ask": ("--state TEXT|@FILE", "--state-json FILE", "--choice 'id=instr:opt1|opt2'",
            "--score 'id=instr:l0|l1'", "--noul 'id=instr'", "… plus every `run` flag"),
    "fit": ("[<model>]", "--print", "--no-cache", "--json", "--fit-target MIB", "--fit-ctx N",
            "--n-ctx N", "--n-seq-max N", "--kv-type auto|f16|q8_0|q4_0", "--timeout S"),
    "serve": ("--host IP", "--port N", "--format native|typesafe"),
    "mcp": (),
    "bench": ("--suite latency|throughput|quality|calibration|determinism", "--model PATH.GGUF",
              "--backend auto|cpu|vulkan|cuda|all", "--runs N", "--threads N", "--devset FILE",
              "--items N", "--n-seq-max N", "--kv-type auto|f16|q8_0|q4_0", "--gpu-layers N",
              "--sizes 256,2048,8192", "--out FILE", "--json"),
    "calibrate": ("--model REF", "--dry-run"),
    "version": ("--json",),
}


def _command_usage(command: str) -> str:
    lines = [f"usage: ggufone {command} " + (" ".join(COMMAND_HELP.get(command, ()) ) or ""),
             "", f"milestone: {MILESTONES.get(command, 'E1')}",
             "run `ggufone --help` for the command list"]
    return "\n".join(lines)


# --------------------------------------------------------------------- plumbing
def _usage() -> str:
    lines = [f"ggufone {__version__}", "usage: ggufone <command> [options]", "", "commands:"]
    for cmd in COMMANDS:
        lines.append(f"  {cmd:12s} (implemented in {MILESTONES.get(cmd, 'E1')})")
    lines.append("")
    lines.append("models: " + ", ".join(MODELS_SUBCOMMANDS))
    return "\n".join(lines)


def _parse_args(args: list[str], *, value_flags: tuple[str, ...] = (),
                bool_flags: tuple[str, ...] = (),
                multi_flags: tuple[str, ...] = ()) -> tuple[list[str], dict[str, Any]]:
    values = {name.replace("-", "_") for name in value_flags}
    flags = {name.replace("-", "_") for name in bool_flags}
    repeats = {name.replace("-", "_") for name in multi_flags}
    positionals: list[str] = []
    options: dict[str, Any] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("--"):
            name, _, inline = arg[2:].partition("=")
            key = name.replace("-", "_")
            if inline:
                options[key] = inline
            elif key in flags:
                options[key] = True
            elif key in values or key in repeats:
                if index + 1 >= len(args):
                    raise UserError(f"--{name} needs a value", code="E_UNKNOWN_KEY")
                value = args[index + 1]
                if key in repeats:
                    options.setdefault(key, []).append(value)
                else:
                    options[key] = value
                index += 1
            else:
                raise UserError(f"unknown option --{name}", code="E_UNKNOWN_KEY")
        else:
            positionals.append(arg)
        index += 1
    return positionals, options


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(_render(payload))


def _render(payload: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(payload, dict):
        return "\n".join(f"{pad}{key}: {_render(value, indent + 1)}"
                         for key, value in payload.items())
    if isinstance(payload, list):
        return "\n".join(f"{pad}- {_render(item, indent + 1)}" for item in payload) or f"{pad}-"
    if isinstance(payload, float):
        return f"{payload:.4g}"
    return f"{payload}"


def _progress_printer(label: str):
    state = {"last": 0.0}

    def report(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.5 and (total is None or done < total):
            return
        state["last"] = now
        if total:
            pct = 100.0 * done / total
            print(f"\r{label}: {pct:5.1f}%  {hf.human_bytes(done)} / {hf.human_bytes(total)}",
                  end="", file=sys.stderr, flush=True)
        else:
            print(f"\r{label}: {hf.human_bytes(done)}", end="", file=sys.stderr, flush=True)
        if total and done >= total:
            print("", file=sys.stderr, flush=True)

    return report


# --------------------------------------------------------------------- init
def _cmd_init(args: list[str]) -> int:
    _, options = _parse_args(args, value_flags=("backend", "offline-cache"),
                             bool_flags=("force", "dry-run", "json"))
    backend = options.get("backend", "auto")
    result = install.install(backend, force=bool(options.get("force")),
                             dry_run=bool(options.get("dry_run")),
                             offline_cache=options.get("offline_cache"),
                             progress=None if options.get("json")
                             else _progress_printer("downloading runtime"))
    if result.get("dry_run"):
        plan = result["plan"]
        payload = {"dry_run": True, "rung": plan["rung"], "backend": plan["backend"],
                   "variant": plan["variant"], "asset": plan["asset"], "url": plan["url"],
                   "size": plan["size"], "size_human": hf.human_bytes(plan["size"]),
                   "sha256": plan["sha256"], "destination": plan["dest"],
                   "required_bytes": plan["required_bytes"],
                   "cached": plan["cached"], "host": plan["host"]}
        _emit(payload, bool(options.get("json")))
        if not options.get("json"):
            print("\n(dry run: nothing downloaded, nothing written)")
        return 0
    if result.get("already_installed"):
        record = result.get("record") or {}
        payload = {"already_installed": True, "variant": result["variant"],
                   "backend": result.get("backend"), "dir": result["dir"],
                   "working_backend": result.get("working_backend"),
                   "backend_requested": (result.get("backend_requested")
                                         or record.get("backend_requested")),
                   "fallback_attempts": result.get("fallback_attempts", []),
                   "fallback_reason": result.get("fallback_reason"),
                   "fallback_reason_code": result.get("fallback_reason_code"),
                   "hint": result["hint"]}
        _emit(payload, bool(options.get("json")))
        if result.get("fallback_reason"):
            print(f"warning: fell back from {payload['backend_requested']} to "
                  f"{result['variant']}: {result['fallback_reason']}", file=sys.stderr)
        for warning in record.get("probe_warnings", []):
            print(f"warning: {warning}", file=sys.stderr)
        return 0
    record = result["record"]
    payload = {"installed": True, "variant": result["variant"], "dir": result["dir"],
               "backend": result.get("backend"),
               "working_backend": result.get("working_backend"),
               "source": result["source"], "asset": record["asset"],
               "asset_sha256": record["asset_sha256"],
               "asset_verified": record["asset_verified"],
               "libllama_sha256": record["libllama_sha256"],
               "bytes_fetched": result.get("bytes_fetched"),
               "resumed_from": result.get("resumed_from"),
               "build": record["build"], "backends": record["backends"],
               "backend_errors": record.get("backend_errors", {}),
               "fallback_attempts": record.get("fallback_attempts", []),
               "fallback_reason": record.get("fallback_reason"),
               "fallback_reason_code": record.get("fallback_reason_code"),
               "symbols_ok": record["symbols_ok"], "warmup_ms": record["warmup_ms"],
               "rung": record["rung"]}
    _emit(payload, bool(options.get("json")))
    if record.get("fallback_reason"):
        print(f"warning: fell back from {record.get('backend_requested')} to "
              f"{result['variant']}: {record['fallback_reason']}", file=sys.stderr)
    for warning in record.get("probe_warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    for failure in record.get("probe_failures", []):
        print(f"warning: {failure}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------- doctor
def _recorded_backend_reason(record: dict | None, backend: str) -> str | None:
    """Why the install record says `backend` is not the one in use (or `None` if it does not).

    `init` records the probe result that made it fall back, so `doctor` can repeat the *real*
    reason instead of suggesting a retry that would fail the same way (finding 4).
    """
    if not record:
        return None
    for attempt in record.get("fallback_attempts") or []:
        if attempt.get("backend") == backend and attempt.get("reason"):
            return str(attempt["reason"])
    if record.get("backend_requested") == backend and record.get("fallback_reason"):
        return str(record["fallback_reason"])
    return None


def doctor_checks(home: pathlib.Path | None = None,
                  lock: pins.RuntimeLock | None = None) -> dict[str, Any]:
    """Build the doctor report (checks + runtime + model). Pure: no printing."""
    home = home or store.data_home()
    lock = lock or pins.load_lock()
    checks: list[dict[str, str]] = []

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    try:
        runtime_dir = finder.find_runtime(home=home)
        runtime_error = None
    except GgufoneError as exc:
        runtime_dir, runtime_error = None, str(exc)
    probe = None
    if runtime_dir is None:
        add("runtime.present", "fail",
            runtime_error or "no runtime installed under "
            f"{store.runtime_root(home)}; run `ggufone init` (no compiler needed)")
    else:
        probe = capability.probe_runtime(runtime_dir, deep=capability.deep_probe_enabled(),
                                         lock=lock, expect_backend=capability.host_expectation(),
                                         run_tools=True)
        add("runtime.present", "ok", str(runtime_dir))
        if probe.error:
            add("runtime.loadable", "fail", probe.error)
        add("runtime.files", "ok" if not probe.missing_files else "fail",
            "all required libraries present: " + ", ".join(lock.required_files)
            if not probe.missing_files else "missing " + ", ".join(probe.missing_files))
        add("runtime.symbols", "ok" if probe.symbols_checked and not probe.missing_symbols
            else ("warn" if not probe.symbols_checked else "fail"),
            f"resolved {len(lock.all_symbols) - len(probe.missing_symbols)}"
            f"/{len(lock.all_symbols)} required symbols" if probe.symbols_checked
            else "symbol probe skipped (GGUFONE_DEEP_PROBE=0)")
        build_state = "ok"
        build_detail = f"build {probe.tag}"
        if probe.build is None:
            build_state, build_detail = "fail", "cannot determine the build number"
        elif probe.min_build and probe.build < probe.min_build:
            build_state = "fail"
            build_detail += f" < b{probe.min_build} (spark2_5 needs b{probe.min_build})"
        elif probe.tag != lock.tag:
            build_state = "warn"
            build_detail += f" != pinned {lock.tag}"
        add("runtime.build", build_state, build_detail)
        if "llama-fit-params" in probe.tools:
            state = "ok" if probe.fit_params_help_exit == 0 else "fail"
            add("runtime.fit_params", state,
                f"llama-fit-params --help exit {probe.fit_params_help_exit}")
        else:
            add("runtime.fit_params", "warn", "llama-fit-params not bundled (auto-fit limited)")
        accel = [b for b in probe.backends if b not in ("cpu", "rpc", "base")]
        expected_backend = capability.host_expectation()
        working_backend = probe.accelerator()
        add("runtime.backends", "ok" if probe.backends else "warn",
            "backends: " + (", ".join(probe.backends) or "none")
            + (f" (driveable here: {working_backend})" if probe.backends else ""))
        for name, load_error in sorted(probe.backend_errors.items()):
            add(f"runtime.backend.{name}", "warn",
                f"the {name} backend does not load on this host: {load_error}")
        record = finder.runtime_record(home)
        if expected_backend == "cpu":
            add("runtime.accelerator", "ok",
                "no GPU on this host; CPU placement is the expected backend")
        elif probe.usable(expected_backend):
            add("runtime.accelerator", "ok", f"{expected_backend} present and loadable")
        elif expected_backend in probe.backend_errors:
            add("runtime.accelerator", "warn",
                f"expected accelerator {expected_backend!r} does not load here "
                f"({probe.backend_errors[expected_backend]}); using {working_backend!r}")
        elif accel:
            reason = _recorded_backend_reason(record, expected_backend)
            if reason:
                add("runtime.accelerator", "warn",
                    f"expected accelerator {expected_backend!r} is not in this bundle "
                    f"(backends: {', '.join(probe.backends) or 'none'}): {reason}; install the "
                    f"{expected_backend} runtime it needs and re-run `ggufone init`, or keep "
                    f"{working_backend!r} (driveable here)")
            else:
                add("runtime.accelerator", "warn",
                    f"expected accelerator {expected_backend!r} is not in this bundle "
                    f"(backends: {', '.join(probe.backends) or 'none'}); re-run `ggufone init "
                    f"--backend {expected_backend}`")
        else:
            add("runtime.accelerator", "warn",
                f"no accelerator in this bundle (expected {expected_backend!r}); CPU works "
                f"but decoding is slower")
        if record and record.get("fallback_reason"):
            code = record.get("fallback_reason_code") or "unknown"
            add("runtime.fallback", "warn",
                f"installed {record.get('variant')} after {record.get('backend_requested')} "
                f"was unusable here [{code}]: {record['fallback_reason']}")
        if not record or not record.get("libllama_sha256"):
            add("runtime.sha_recorded", "warn",
                "runtime.json has no libllama.so SHA-256 (re-run `ggufone init --force`)")
        elif record.get("libllama_sha256") != sha256_file(
                runtime_dir / finder.library_names()["llama"]):
            add("runtime.sha_recorded", "fail",
                "libllama.so SHA-256 differs from the recorded value (re-install)")
        else:
            digest = record["libllama_sha256"][:16]
            add("runtime.sha_recorded", "ok", f"libllama.so sha256 {digest}…")

    registry, registry_warnings = store.load_registry(store.registry_path(home))
    for warning in registry_warnings:
        add("registry.corrupt", "warn", warning)
    entry = store.resolve(registry, None, use_current=True)
    model_payload: dict[str, Any] = {"alias": None}
    if entry is None:
        add("model.present", "warn",
            "no model in the registry; run `ggufone models pull` (default model is pinned)")
    else:
        model_payload = {"alias": entry.alias, "path": entry.path, "size": entry.size,
                         "sha256": entry.sha256, "arch": entry.arch, "quant": entry.quant,
                         "license": entry.license, "source": entry.source}
        path = pathlib.Path(entry.path)
        add("model.file", "ok" if path.exists() else "warn",
            entry.path if path.exists() else f"{entry.path} is missing (re-pull it)")
        if path.exists() and entry.sha256:
            actual = sha256_file(path)
            if actual != entry.sha256:
                add("model.sha256", "warn",
                    f"{path.name}: sha256 {actual[:16]}… != recorded {entry.sha256[:16]}…")
            else:
                add("model.sha256", "ok", f"{path.name}: sha256 matches the registry")
        elif not entry.sha256:
            add("model.sha256", "warn", "registry entry has no SHA-256 recorded")
        if entry.arch and runtime_dir is not None:
            try:
                capability.require_arch(runtime_dir, entry.arch, lock=lock)
                add("model.arch", "ok", f"{entry.arch} supported by the installed runtime")
            except GgufoneError as exc:
                add("model.arch", "fail", str(exc))

    if any(c["status"] == "fail" for c in checks):
        status, exit_code = "failures", 1
    elif any(c["status"] == "warn" for c in checks):
        status, exit_code = "warnings", 2
    else:
        status, exit_code = "ok", 0
    return {
        "schema": DOCTOR_SCHEMA,
        "ggufone": __version__,
        "status": status,
        "exit_code": exit_code,
        "checks": checks,
        "runtime": {
            "dir": str(runtime_dir) if runtime_dir else None,
            "installed": runtime_dir is not None,
            "tag": probe.tag if probe else None,
            "pinned_tag": lock.tag,
            "min_build": lock.min_build_for_spark2_5,
            "variant": (finder.runtime_record(home) or {}).get("variant"),
            "build": probe.build if probe else None,
            "backends": list(probe.backends) if probe else [],
            "working_backend": probe.accelerator() if probe else None,
            "backend_errors": dict(probe.backend_errors) if probe else {},
            "backend_requested": (finder.runtime_record(home) or {}).get("backend_requested"),
            "fallback_reason": (finder.runtime_record(home) or {}).get("fallback_reason"),
            "fallback_reason_code": (finder.runtime_record(home) or {}).get(
                "fallback_reason_code"),
            "fallback_attempts": (finder.runtime_record(home) or {}).get("fallback_attempts", []),
            "symbols_required": len(lock.all_symbols),
            "symbols_missing": list(probe.missing_symbols) if probe else [],
            "symbols_probed": bool(probe and probe.symbols_checked),
            "fit_params_help_exit": probe.fit_params_help_exit if probe else None,
            "libllama_sha256": (finder.runtime_record(home) or {}).get("libllama_sha256"),
            "warmup_ms": (finder.runtime_record(home) or {}).get("warmup_ms"),
        },
        "model": model_payload,
        # The requirement-4 acceptance reads `doctor --json` for the backend that actually works
        # here and the full list the probe found (aliases of `runtime.working_backend` /
        # `runtime.backends`, kept top-level so a caller does not have to know the layout).
        "backend": probe.accelerator() if probe else None,
        "backends": list(probe.backends) if probe else [],
        "expected_backend": capability.host_expectation(),
    }


def _cmd_doctor(args: list[str]) -> int:
    _, options = _parse_args(args, bool_flags=("json",))
    report = doctor_checks()
    if options.get("json"):
        print(json.dumps(report, indent=2))
    else:
        print(f"ggufone doctor ({report['status']})")
        for check in report["checks"]:
            mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[check["status"]]
            print(f"  {mark} {check['id']:22s} {check['detail']}")
        runtime = report["runtime"]
        print(f"  runtime: {runtime['dir'] or '<none>'}  build={runtime['tag']}  "
              f"backend={report['backend'] or 'none'}  "
              f"backends={','.join(runtime['backends']) or 'none'}")
        if runtime.get("fallback_reason"):
            print(f"  fallback: {runtime['backend_requested']} -> "
                  f"{runtime['backend_working']} [{runtime.get('fallback_reason_code')}]")
        print(f"  model:   {report['model'].get('alias') or '<none>'}")
    return int(report["exit_code"])


# --------------------------------------------------------------------- models
def _models_payload(registry: store.Registry) -> dict[str, Any]:
    return {
        "schema": MODELS_SCHEMA,
        "current": registry.current,
        "models": [
            {"alias": entry.alias, "path": entry.path, "size": entry.size,
             "sha256": entry.sha256, "arch": entry.arch, "quant": entry.quant,
             "license": entry.license, "source": entry.source, "repo": entry.repo,
             "added_at": entry.added_at, "current": entry.alias == registry.current}
            for entry in registry.aliases.values()
        ],
    }


def _cmd_models(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print("usage: ggufone models " + "|".join(MODELS_SUBCOMMANDS))
        return 0
    sub, rest = args[0], args[1:]
    if sub not in MODELS_SUBCOMMANDS:
        raise UserError(f"unknown models subcommand {sub!r}; known: "
                        f"{', '.join(MODELS_SUBCOMMANDS)}", code="E_UNKNOWN_KEY")
    handler = {
        "search": _models_search, "pull": _models_pull, "use": _models_use,
        "ls": _models_ls, "rm": _models_rm, "verify": _models_verify,
        "recommend-quant": _models_recommend_quant,
    }[sub]
    return handler(rest)


def _models_search(args: list[str]) -> int:
    positionals, options = _parse_args(args, value_flags=("limit",), bool_flags=("json",))
    if not positionals:
        raise UserError("usage: ggufone models search <query>", code="E_UNKNOWN_KEY")
    limit = int(options.get("limit", 20))
    results = hf.search(positionals[0], limit=limit)
    if options.get("json"):
        print(json.dumps({"schema": MODELS_SCHEMA, "query": positionals[0],
                          "results": results}, indent=2))
        return 0
    if not results:
        print(f"no GGUF repos matched {positionals[0]!r}")
        return 0
    for entry in results:
        likes = entry.get("likes")
        downloads = entry.get("downloads")
        print(f"  {entry['id']:52s} downloads={downloads} likes={likes}")
    return 0


def _split_repo_quant(spec: str | None, lock: pins.RuntimeLock) -> tuple[str, str | None]:
    if not spec:
        return lock.default_model.repo, lock.default_model.quant
    if ":" in spec:
        repo, _, quant = spec.rpartition(":")
        return (repo or lock.default_model.repo), (quant or None)
    return spec, None


def _models_pull(args: list[str]) -> int:
    positionals, options = _parse_args(
        args, value_flags=("file", "jobs", "alias", "revision"),
        bool_flags=("no-verify", "json", "offline"))
    lock = pins.load_lock()
    repo, quant = _split_repo_quant(positionals[0] if positionals else None, lock)
    offline = bool(options.get("offline")) or hf.offline_enabled()
    info = hf.model_info(repo, revision=options.get("revision"), offline=offline)
    revision = options.get("revision") or (
        lock.default_model.repo_sha if repo == lock.default_model.repo else (info.sha or "main"))
    files = [f.to_dict() for f in info.files]
    budget = recommend.host_budget()
    choice = recommend.select_file(files, quant=quant, explicit_file=options.get("file"),
                                   repo=repo, lock=lock, vram_bytes=budget.vram_bytes,
                                   ram_bytes=budget.ram_bytes)
    file_info = choice.file
    name = pathlib.PurePosixPath(file_info["path"]).name
    size = int(file_info.get("size") or 0)
    oid = file_info.get("oid")
    destination = store.models_dir() / name
    store.models_dir().mkdir(parents=True, exist_ok=True)
    hf.check_disk_space(store.models_dir(), size)

    progress = None if options.get("json") else _progress_printer(f"pulling {name}")
    result = hf.download_file(repo, file_info["path"], destination, revision=revision,
                              size=size or None, sha256=oid,
                              no_verify=bool(options.get("no_verify")),
                              progress=progress)
    metadata = gguf.parse_gguf_metadata(destination)["kv"]
    arch = gguf.arch_of(metadata)
    file_type = gguf.file_type_of(metadata)
    quant_label = choice.quant or (gguf.quant_label(file_type) if file_type is not None else None)
    registry, warnings = store.load_registry()
    entry = store.Entry(alias=options.get("alias") or store.slugify(name),
                        path=str(destination), sha256=result.sha256, arch=arch,
                        quant=quant_label, size=destination.stat().st_size,
                        license=info.license, source=repo, fit_plan=None,
                        file_type=file_type, repo=repo)
    entry = store.add_entry(registry, entry, alias=options.get("alias"))
    store.save_registry(registry)

    payload = {
        "schema": MODELS_SCHEMA,
        "alias": entry.alias,
        "path": str(destination),
        "repo": repo,
        "revision": revision,
        "file": file_info["path"],
        "size": destination.stat().st_size,
        "size_human": hf.human_bytes(destination.stat().st_size),
        "sha256": result.sha256,
        "sha256_verified": result.verified,
        "license": info.license,
        "arch": arch,
        "quant": quant_label,
        "quant_reason": choice.reason,
        "bytes_fetched": result.bytes_fetched,
        "resumed_from": result.resumed_from,
    }
    if options.get("json"):
        print(json.dumps(payload, indent=2))
    else:
        print(f"pulled {entry.alias} -> {destination}")
        print(f"  repo      {repo}@{revision[:12]}")
        print(f"  file      {file_info['path']} ({hf.human_bytes(destination.stat().st_size)})")
        print(f"  sha256    {result.sha256}" + (" (verified against lfs.oid)"
                                                if result.verified else " (not verified)"))
        print(f"  license   {info.license or 'unknown'}")
        print(f"  arch      {arch}  quant {quant_label}  [{choice.reason}]")
        if result.resumed_from:
            print(f"  resumed   from {hf.human_bytes(result.resumed_from)}")
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def _models_use(args: list[str]) -> int:
    positionals, options = _parse_args(args, bool_flags=("json",))
    if not positionals:
        raise UserError("usage: ggufone models use <alias>", code="E_UNKNOWN_KEY")
    registry, _ = store.load_registry()
    entry = store.resolve(registry, positionals[0])
    if entry is None:
        raise UserError(f"unknown alias {positionals[0]!r}; known: "
                        f"{', '.join(sorted(registry.aliases)) or '<none>'}",
                        code="E_MODEL_NOT_FOUND")
    registry.current = entry.alias
    store.save_registry(registry)
    payload = {"schema": MODELS_SCHEMA, "current": entry.alias, "path": entry.path}
    _emit(payload, bool(options.get("json")))
    return 0


def _models_ls(args: list[str]) -> int:
    _, options = _parse_args(args, bool_flags=("json",))
    registry, warnings = store.load_registry()
    payload = _models_payload(registry)
    if options.get("json"):
        print(json.dumps(payload, indent=2))
    elif not payload["models"]:
        print("no models in the registry; run `ggufone models pull`")
    else:
        for model in payload["models"]:
            marker = "*" if model["current"] else " "
            exists = "ok" if pathlib.Path(model["path"]).exists() else "MISSING"
            print(f" {marker} {model['alias']:26s} {model['quant'] or '?':10s} "
                  f"{model['arch'] or '?':12s} {hf.human_bytes(model['size'] or 0):>10s} "
                  f"{model['license'] or '?':12s} [{exists}] {model['path']}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _models_rm(args: list[str]) -> int:
    positionals, options = _parse_args(args, bool_flags=("json", "keep-file"))
    if not positionals:
        raise UserError("usage: ggufone models rm <alias>", code="E_UNKNOWN_KEY")
    registry, _ = store.load_registry()
    entry = store.remove_entry(registry, positionals[0])
    deleted = False
    path = pathlib.Path(entry.path)
    if not options.get("keep_file") and path.exists():
        path.unlink()
        deleted = True
    store.save_registry(registry)
    payload = {"schema": MODELS_SCHEMA, "removed": entry.alias, "path": entry.path,
               "file_deleted": deleted, "current": registry.current}
    _emit(payload, bool(options.get("json")))
    return 0


def _models_verify(args: list[str]) -> int:
    positionals, options = _parse_args(args, bool_flags=("json",))
    registry, warnings = store.load_registry()
    if positionals:
        entries = [store.resolve(registry, positionals[0])]
        if entries[0] is None:
            raise UserError(f"unknown alias {positionals[0]!r}", code="E_MODEL_NOT_FOUND")
    else:
        entries = list(registry.aliases.values())
    results = []
    for entry in entries:
        path = pathlib.Path(entry.path)
        if not path.exists():
            results.append({"alias": entry.alias, "status": "missing", "path": entry.path})
            continue
        actual = sha256_file(path)
        ok = entry.sha256 is None or actual == entry.sha256
        results.append({"alias": entry.alias, "status": "ok" if ok else "sha256_mismatch",
                        "path": entry.path, "size": path.stat().st_size,
                        "sha256": actual, "expected": entry.sha256})
    mismatches = [r for r in results if r["status"] == "sha256_mismatch"]
    missing = [r for r in results if r["status"] == "missing"]
    payload = {"schema": MODELS_SCHEMA, "verified": len(results) - len(mismatches) - len(missing),
               "failed": len(mismatches) + len(missing), "results": results}
    if options.get("json"):
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print(f"  {result['status']:14s} {result['alias']}  {result['path']}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if mismatches:
        raise Sha256MismatchError(
            f"E_SHA256_MISMATCH: {len(mismatches)} file(s) failed verification "
            f"({', '.join(r['alias'] for r in mismatches)}); re-run `ggufone models pull`")
    if missing:
        raise UserError(
            f"E_MODEL_NOT_FOUND: {len(missing)} registry file(s) are missing on disk "
            f"({', '.join(r['alias'] for r in missing)}); re-pull or `ggufone models rm`")
    return 0


def _pinned_candidates(lock: pins.RuntimeLock) -> list[tuple[str, int]]:
    candidates = [(lock.default_model.file, lock.default_model.size)]
    for spec in lock.default_model.alternates.values():
        candidates.append((spec["file"], int(spec["size"])))
    return candidates


def _models_recommend_quant(args: list[str]) -> int:
    _, options = _parse_args(args, value_flags=("vram", "ram", "n-ctx", "n-seq-max"),
                             bool_flags=("json", "table"))
    lock = pins.load_lock()
    candidates = _pinned_candidates(lock)
    if options.get("table"):
        rows = []
        for scenario in recommend.PINNED_SCENARIOS:
            plan = recommend.recommend_quant(
                candidates, vram_bytes=int(scenario["vram_gib"] * 1024 ** 3),
                ram_bytes=int(scenario["ram_gib"] * 1024 ** 3),
                kv_per_token_f16=recommend.DEFAULT_KV_PER_TOKEN_F16,
                n_ctx=scenario["n_ctx"], n_seq_max=scenario["n_seq_max"])
            rows.append({"scenario": scenario["label"], "n_ctx": scenario["n_ctx"],
                         "n_seq_max": scenario["n_seq_max"],
                         "quant": plan["quant"], "kv_type": plan["kv_type"],
                         "placement": plan["placement"],
                         "total_bytes": plan.get("total"),
                         # SPEC 2.7's executed table is decimal GB
                         "total_gb": round(plan["total"] / 1e9, 2)
                         if plan.get("total") else None})
        payload = {"schema": "ggufone.recommend-quant/v1", "source": "pinned-table",
                   "scenarios": rows}
        if options.get("json"):
            print(json.dumps(payload, indent=2))
        else:
            for row in rows:
                total = f"{row['total_gb']:.2f} GB" if row["total_gb"] else "-"
                print(f"  {row['scenario']:44s} -> {row['quant'] or 'insufficient'} "
                      f"kv={row['kv_type']} {row['placement']} {total}")
        return 0
    budget = recommend.host_budget()
    vram = int(float(options["vram"]) * 1024 ** 3) if options.get("vram") else budget.vram_bytes
    ram = int(float(options["ram"]) * 1024 ** 3) if options.get("ram") else budget.ram_bytes
    n_ctx = int(options.get("n_ctx", recommend.DEFAULT_N_CTX))
    n_seq_max = int(options.get("n_seq_max", recommend.DEFAULT_N_SEQ_MAX))
    plan = recommend.recommend_quant(candidates, vram_bytes=vram, ram_bytes=ram,
                                     kv_per_token_f16=recommend.DEFAULT_KV_PER_TOKEN_F16,
                                     n_ctx=n_ctx, n_seq_max=n_seq_max)
    payload = {"schema": "ggufone.recommend-quant/v1", "source": "host",
               "vram_bytes": vram, "ram_bytes": ram, "n_ctx": n_ctx, "n_seq_max": n_seq_max,
               "quant": plan["quant"], "kv_type": plan["kv_type"],
               "placement": plan["placement"],
               "est_weights_bytes": plan.get("weights"), "est_kv_bytes": plan.get("kv"),
               "est_total_bytes": plan.get("total"),
               "est_total_gib": round(plan["total"] / 1024 ** 3, 2) if plan.get("total") else None,
               "warning": plan.get("warning")}
    _emit(payload, bool(options.get("json")))
    return 0


# --------------------------------------------------------------------- run / ask
ENGINE_VALUE_FLAGS = ("model", "format", "state-id", "temperature", "length-norm", "readout",
                      "confidence-mode", "coverage-floor", "n-ctx", "n-seq-max", "kv-type",
                      "threads", "backend", "seed", "max-waves", "out", "questions", "state",
                      "state-json", "template")
ENGINE_BOOL_FLAGS = ("strict", "save-state", "no-state-cache", "thinking", "no-fit",
                     "no-fit-cache")
FIT_VALUE_FLAGS = ("fit-target", "fit-ctx")


def _engine_options(options: dict[str, Any]) -> dict[str, Any]:
    """Map CLI flags onto the frozen `options` keys (SPEC 2.5)."""
    mapping = {
        "temperature": float, "length_norm": float, "coverage_floor": float,
        "n_ctx": int, "n_seq_max": int, "threads": int, "seed": int, "max_waves": int,
        "readout": str, "confidence_mode": str, "kv_type": str, "backend": str,
        "state_id": str, "strict": bool, "save_state": bool, "template": str,
        "thinking": bool,
    }
    engine: dict[str, Any] = {}
    for key, caster in mapping.items():
        if key not in options:
            continue
        value = options[key]
        try:
            if caster is bool:
                engine[key] = bool(value) and value is not False
            else:
                engine[key] = caster(value)
        except (TypeError, ValueError) as exc:
            raise UserError(f"--{key.replace('_', '-')} got an invalid value {value!r}",
                            code="E_UNKNOWN_KEY") from exc
    if options.get("no_state_cache"):
        engine["state_cache"] = False
    return engine


def _load_state_argument(value: str | None, json_path: str | None) -> Any:
    """`--state <text|@file>` / `--state-json <file>` -> the request's `state`."""
    if json_path:
        path = pathlib.Path(json_path)
        if not path.exists():
            raise UserError(f"--state-json {path} does not exist", code="E_UNKNOWN_KEY")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UserError(f"--state-json {path} is not valid JSON ({exc})",
                            code="E_UNKNOWN_KEY") from exc
    if value is None:
        return None
    if value.startswith("@"):
        path = pathlib.Path(value[1:])
        if not path.exists():
            raise UserError(f"--state @{path} does not exist", code="E_UNKNOWN_KEY")
        return path.read_text(encoding="utf-8")
    return value


def _load_questions(path: str) -> dict[str, Any]:
    questions_path = pathlib.Path(path)
    if not questions_path.exists():
        raise UserError(f"--questions {questions_path} does not exist", code="E_UNKNOWN_KEY")
    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UserError(f"--questions {questions_path} is not valid JSON ({exc})",
                        code="E_UNKNOWN_KEY") from exc
    if not isinstance(payload, dict):
        raise UserError("--questions must contain a JSON object", code="E_UNKNOWN_KEY")
    return payload


def engine_request_payload(payload: dict[str, Any], *, state: Any = None,
                           model: str | None = None, fmt: str | None = None,
                           engine_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Overlay CLI flags on a `run`/`ask` request body (schema validation does the rest)."""
    body = dict(payload)
    if "questions" not in body:                       # a bare {qid: {...}} map is accepted
        body = {"questions": body}
    if state is not None:
        body["state"] = state
    if model is not None:
        body["model"] = model
    if fmt is not None:
        body["format"] = fmt
    options = dict(body.get("options") or {})
    options.update(engine_options or {})
    if options:
        body["options"] = options
    return body


def _resolve_model(request: schema.Request, *, home: pathlib.Path | None = None) -> tuple[str, str]:
    """`alias | path | repo[:quant]` -> (alias, path) through the E1a registry."""
    registry, _warnings = store.load_registry(store.registry_path(home))
    known = tuple(registry.aliases)
    ref = request.model
    if request.format == "typesafe":
        ref = schema.adapter_model_ref(ref, known_aliases=known, default_alias=registry.current)
    if ref is None:
        entry = store.resolve(registry, None, use_current=True)
    else:
        candidate = pathlib.Path(ref)
        if candidate.exists() and candidate.is_file():
            return candidate.name.removesuffix(".gguf"), str(candidate)
        entry = store.resolve(registry, ref)
    if entry is None:
        raise ModelNotFoundError(
            f"E_MODEL_NOT_FOUND: {ref!r} is not a registry alias, a path or a pulled model; "
            f"known aliases: {', '.join(known) or '<none>'} (use `ggufone models pull` or "
            f"`ggufone models use`)")
    return entry.alias, entry.path


def _resolve_model_ref(ref: str | None, *, home: pathlib.Path | None = None) -> tuple[str, str]:
    """`alias | path | repo[:quant]` -> (alias, path) without a full request (used by `fit`)."""
    if ref:
        candidate = pathlib.Path(ref)
        if candidate.exists() and candidate.is_file():
            return candidate.name.removesuffix(".gguf"), str(candidate)
    registry, _warnings = store.load_registry(store.registry_path(home))
    entry = store.resolve(registry, ref, use_current=ref is None)
    if entry is None:
        known = ", ".join(registry.aliases) or "<none>"
        raise ModelNotFoundError(
            f"E_MODEL_NOT_FOUND: {ref!r} is not a registry alias or an existing file; known "
            f"aliases: {known} (use `ggufone models pull` / `ggufone models use`)")
    return entry.alias, entry.path


def fit_plan_for(model_path: str, *, home: pathlib.Path | None = None, use_cache: bool = True,
                 fit_target_mb: int | None = None, min_ctx: int | None = None,
                 n_ctx: int | None = None, n_seq_max: int | None = None,
                 kv_type: str = "auto", host: fit.HostFacts | None = None) -> fit.FitPlan:
    """The A-E1c-4 plan for one model on this host (cached, binary when available).

    `host` is the fresh reading of the box (nominal + free device memory). It is passed in by
    callers that also report it (`ggufone fit --json`), and re-read here otherwise — the plan must
    never be built from facts an earlier call captured (card t_8cb0a05e).
    """
    model = fit.ModelFacts.read(model_path)
    host = host if host is not None else fit.host_facts()
    runtime_dir = finder.find_runtime(home=home)
    kwargs: dict[str, Any] = {"use_cache": use_cache, "kv_type": kv_type}
    if fit_target_mb is not None:
        kwargs["fit_target_mb"] = int(fit_target_mb)
    if min_ctx is not None:
        kwargs["min_ctx"] = int(min_ctx)
    if n_ctx is not None:
        kwargs["n_ctx"] = int(n_ctx)
    if n_seq_max is not None:
        kwargs["n_seq_max"] = int(n_seq_max)
    return fit.plan_for_model(model, host, home=home, runtime_dir=runtime_dir, **kwargs)


def decide_payload(payload: dict[str, Any], *, home: pathlib.Path | None = None,
                   fit_enabled: bool = True, fit_target_mb: int | None = None,
                   fit_ctx: int | None = None, fit_cache: bool = True) -> dict[str, Any]:
    """Validate -> resolve -> fit -> load -> prefilled fork-decide -> rendered response body."""
    request = schema.parse_request(payload)
    alias, model_path = _resolve_model(request, home=home)
    plan: fit.FitPlan | None = None
    n_ctx_cap: int | None = None
    if fit_enabled:
        plan = fit_plan_for(model_path, home=home, use_cache=fit_cache,
                            fit_target_mb=fit_target_mb, min_ctx=fit_ctx,
                            kv_type=request.options.kv_type)
        n_ctx_cap = plan.n_ctx
        request = _with_fit_options(request, plan)
    with session_module.open_model(model_path, home=home, fit_plan=plan,
                                   fit_disabled=not fit_enabled) as handle:
        # the plan the load actually used: a degraded retry (allocation failure) may offload less
        # than the plan that was requested (card t_8cb0a05e)
        effective = getattr(handle, "fit_plan", None) or plan
        context_plan = decide.plan_context(request, handle, n_ctx_cap=n_ctx_cap)
        with session_module.ModelSession(handle, context_plan,
                                         backend=session_module.runtime_backend(home),
                                         states_home=store.states_dir(home)) as live:
            result = decide.DecisionEngine(live).decide(request, plan=context_plan,
                                                        model_alias=alias)
        body = schema.render_response(result.payload(), format=request.format)
    if effective is not None and isinstance(body.get("engine"), dict):
        # surface the plan the load really used, not the one that was asked for (card t_8cb0a05e):
        # a degraded retry offloads fewer layers, and the response has to say so.
        body["engine"]["fit"] = effective.to_dict()
    return body


def _with_fit_options(request: schema.Request, plan: fit.FitPlan) -> schema.Request:
    """Fill what the request left open with the plan (explicit words always win)."""
    options = request.options
    needed_sequences = max(3, 1 + max((len(q.options) for q in request.questions), default=1))
    return dataclasses.replace(request, options=dataclasses.replace(
        options,
        kv_type=plan.kv_type if options.kv_type in ("auto", None) else options.kv_type,
        n_seq_max=max(options.n_seq_max or 0, plan.n_seq_max, needed_sequences)
        if options.n_seq_max is None else options.n_seq_max,
    ))


def _cmd_run(args: list[str]) -> int:
    positionals, options = _parse_args(args, value_flags=ENGINE_VALUE_FLAGS + FIT_VALUE_FLAGS,
                                        bool_flags=ENGINE_BOOL_FLAGS)
    if positionals:
        raise UserError(f"unexpected argument {positionals[0]!r}", code="E_UNKNOWN_KEY")
    if "questions" not in options:
        raise UserError("run needs --questions <file.json> (SPEC 2.8)", code="E_UNKNOWN_KEY")
    body = _load_questions(options["questions"])
    state = _load_state_argument(options.get("state"), options.get("state_json"))
    payload = engine_request_payload(body, state=state, model=options.get("model"),
                                     fmt=options.get("format"),
                                     engine_options=_engine_options(options))
    response = decide_payload(payload, **_fit_arguments(options))
    text = json.dumps(response, indent=2, sort_keys=False)
    if options.get("out"):
        pathlib.Path(options["out"]).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _fit_arguments(options: dict[str, Any]) -> dict[str, Any]:
    """`--no-fit` / `--fit-target` / `--fit-ctx` / `--no-fit-cache` -> `decide_payload` kwargs."""
    kwargs: dict[str, Any] = {"fit_enabled": not options.get("no_fit"),
                              "fit_cache": not options.get("no_fit_cache")}
    if options.get("fit_target") is not None:
        kwargs["fit_target_mb"] = int(options["fit_target"])
    if options.get("fit_ctx") is not None:
        kwargs["fit_ctx"] = int(options["fit_ctx"])
    return kwargs


def _ask_question(spec: str, qtype: str) -> tuple[str, dict[str, Any]]:
    """`id=instr:opt1|opt2` -> (id, question body). SPEC 2.8."""
    qid, sep, rest = spec.partition("=")
    if not sep or not qid.strip():
        raise UserError(f"--{qtype} needs 'id=instructions:labels' (got {spec!r})",
                        code="E_QID_INVALID")
    if qtype == "noul":
        body: dict[str, Any] = {"type": "noul"}
        if rest:
            body["instructions"] = rest
        return qid, body
    instructions, sep, labels = rest.partition(":")
    if not sep or not labels.strip():
        raise UserError(f"--{qtype} needs 'id=instructions:labels' (got {spec!r})",
                        code="E_CHOICE_CRITERIA" if qtype == "choice" else "E_SCORE_LEVELS")
    parts = [part.strip() for part in labels.split("|") if part.strip()]
    criteria: Any = {part: None for part in parts} if qtype == "choice" else parts
    body = {"type": qtype, "criteria": criteria}
    if instructions:
        body["instructions"] = instructions
    return qid, body


def _cmd_ask(args: list[str]) -> int:
    positionals, options = _parse_args(args, value_flags=ENGINE_VALUE_FLAGS + FIT_VALUE_FLAGS,
                                        bool_flags=ENGINE_BOOL_FLAGS,
                                        multi_flags=("choice", "score", "noul"))
    if positionals:
        raise UserError(f"unexpected argument {positionals[0]!r}", code="E_UNKNOWN_KEY")
    if "state" not in options and "state_json" not in options:
        raise UserError("ask needs --state <text|@file> (SPEC 2.8)", code="E_STATE_EMPTY")
    questions: dict[str, Any] = {}
    for qtype in ("choice", "score", "noul"):
        for spec in options.get(qtype, []):
            qid, body = _ask_question(spec, qtype)
            if qid in questions:
                raise UserError(f"duplicate question id {qid!r}", code="E_QID_INVALID")
            questions[qid] = body
    if not questions:
        raise UserError("ask needs at least one --choice/--score/--noul question",
                        code="E_QID_INVALID")
    state = _load_state_argument(options.get("state"), options.get("state_json"))
    payload = engine_request_payload({"questions": questions}, state=state,
                                     model=options.get("model"), fmt=options.get("format"),
                                     engine_options=_engine_options(options))
    response = decide_payload(payload, **_fit_arguments(options))
    text = json.dumps(response, indent=2, sort_keys=False)
    if options.get("out"):
        pathlib.Path(options["out"]).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------- bench (E2)
BENCH_VALUE_FLAGS = ("suite", "model", "backend", "runs", "threads", "devset", "items",
                     "n-seq-max", "kv-type", "gpu-layers", "out", "sizes")
BENCH_BOOL_FLAGS = ("json",)
BENCH_DEFAULTS = {"backend": "auto", "runs": harness.DEFAULT_RUNS, "kv-type": "auto"}


def _bench_sizes(value: str | None) -> tuple[int, ...]:
    """`--sizes 256,2048` -> the prefill sizes of this run (default: the three pinned ones)."""
    if not value:
        return harness.PREFILL_SIZES
    sizes: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or int(part) <= 0:
            raise UserError(f"--sizes takes positive token counts (got {part!r})",
                            code="E_BENCH_USAGE")
        sizes.append(int(part))
    if not sizes:
        raise UserError("--sizes needs at least one token count", code="E_BENCH_USAGE")
    return tuple(sizes)


def _cmd_bench(args: list[str]) -> int:
    """`ggufone bench --suite latency|throughput|quality|calibration|determinism` (SPEC 2.8).

    The report goes to stdout (`--json` or the rendered tables) and, with `--out FILE`, to a JSON
    file whose bytes are the published artifact. Exit 0 = the suite ran, 1 = a suite gate failed
    (determinism bytes differ / nothing was measured), 2 = user error, 3 = runtime or model error.
    """
    positionals, options = _parse_args(args, value_flags=BENCH_VALUE_FLAGS,
                                       bool_flags=BENCH_BOOL_FLAGS)
    if positionals:
        raise UserError(f"unexpected argument {positionals[0]!r}", code="E_UNKNOWN_KEY")
    suite = options.get("suite")
    if not suite:
        raise UserError("bench needs --suite latency|throughput|quality|calibration|determinism "
                        "(SPEC 2.8)", code="E_BENCH_SUITE")
    harness.valid_suite(suite)
    model_path = harness.resolve_model_path(options.get("model"))
    config = harness.BenchConfig(
        suite=suite, model_path=model_path,
        backend=options.get("backend", BENCH_DEFAULTS["backend"]),
        runs=int(options.get("runs", BENCH_DEFAULTS["runs"])),
        threads=int(options["threads"]) if "threads" in options else None,
        devset=options.get("devset"),
        items=int(options["items"]) if "items" in options else None,
        n_seq_max=int(options["n_seq_max"]) if "n_seq_max" in options else None,
        kv_type=options.get("kv_type", BENCH_DEFAULTS["kv-type"]),
        gpu_layers=int(options["gpu_layers"]) if "gpu_layers" in options else None,
        prefill_sizes=_bench_sizes(options.get("sizes")))
    report = suites.run_suite(config, factory=suites.live_factory)
    if options.get("out"):
        harness.write_report(report, options["out"])
    if options.get("json"):
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(harness.render_report(report))
        if options.get("out"):
            print(f"report: {options['out']}")
    return 0 if report.get("ok", True) else 1


# --------------------------------------------------------------------- fit (E1c)
def _cmd_fit(args: list[str]) -> int:
    """`ggufone fit [<model>] [--print] [--no-cache]` (SPEC 2.8/2.10, A-E1c-4)."""
    positionals, options = _parse_args(
        args, value_flags=("model", "fit-target", "fit-ctx", "n-ctx", "n-seq-max", "kv-type",
                           "timeout"),
        bool_flags=("print", "no-cache", "json"))
    if len(positionals) > 1:
        raise UserError(f"unexpected argument {positionals[1]!r}", code="E_UNKNOWN_KEY")
    home = store.data_home()
    ref = positionals[0] if positionals else options.get("model")
    alias, model_path = _resolve_model_ref(ref, home=home)
    host = fit.host_facts()
    plan = fit_plan_for(model_path, home=home, use_cache=not options.get("no_cache"),
                        fit_target_mb=int(options["fit_target"]) if "fit_target" in options
                        else None,
                        min_ctx=int(options["fit_ctx"]) if "fit_ctx" in options else None,
                        n_ctx=int(options["n_ctx"]) if "n_ctx" in options else None,
                        n_seq_max=int(options["n_seq_max"]) if "n_seq_max" in options else None,
                        kv_type=options.get("kv_type", "auto"), host=host)
    payload = {"model": alias, "path": model_path, **plan.to_dict(),
               # the free number the plan was bounded by (card t_8cb0a05e: any fit-touching
               # evidence must carry it; a plan without it cannot be audited later)
               "host": host.to_dict()}
    cache = fit.cache_path(plan.model_sha256, plan.host_fingerprint, home)
    payload["cache"] = str(cache) if cache.exists() else None
    if options.get("json"):
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    for key, value in payload.items():
        print(f"{key}: {_render(value) if not isinstance(value, (bool, type(None))) else value}")
    for note in plan.notes:
        print(f"note: {note}")
    if plan.insufficient:
        print("hint: the plan exceeds this host's memory — use a smaller quant or "
              "raise --fit-target", file=sys.stderr)
    return 0


# --------------------------------------------------------------------- version
def _cmd_version(args: list[str]) -> int:
    _, options = _parse_args(args, bool_flags=("json",))
    lock = pins.load_lock()
    record = finder.runtime_record() or {}
    payload = {"name": "ggufone", "version": __version__, "python": sys.version.split()[0],
               "lock": {"tag": lock.tag, "min_build_for_spark2_5": lock.min_build_for_spark2_5},
               "runtime": {"installed": bool(record.get("dir")), "dir": record.get("dir"),
                           "variant": record.get("variant"), "build": record.get("build"),
                           "backends": record.get("backends", [])},
               "home": str(store.data_home())}
    if options.get("json"):
        print(json.dumps(payload, indent=2))
    else:
        print(f"ggufone {__version__}")
        if record.get("dir"):
            print(f"  runtime  {lock.tag} ({record.get('variant')}) at {record['dir']}")
            print(f"  backends {', '.join(record.get('backends', [])) or 'unknown'}")
        else:
            print("  runtime  not installed (`ggufone init`)")
        print(f"  home     {store.data_home()}")
    return 0


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    cmd, rest = args[0], args[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    # carried finding #1 (E1c card): `ggufone <cmd> --help` used to be an E_UNKNOWN_KEY error
    if any(arg in ("-h", "--help") for arg in rest):
        if cmd == "models" and rest and rest[0] in MODELS_SUBCOMMANDS:
            wanted = rest[0]
            flags = " ".join(flag for flag in COMMAND_HELP["models"] if flag.startswith(wanted))
            print(f"usage: ggufone models {wanted} {flags}")
            return 0
        print(_command_usage(cmd))
        return 0
    try:
        if cmd == "version":
            return _cmd_version(rest)
        if cmd == "init":
            return _cmd_init(rest)
        if cmd == "doctor":
            return _cmd_doctor(rest)
        if cmd == "models":
            return _cmd_models(rest)
        if cmd == "run":
            return _cmd_run(rest)
        if cmd == "ask":
            return _cmd_ask(rest)
        if cmd == "bench":
            return _cmd_bench(rest)
        if cmd == "fit":
            return _cmd_fit(rest)
    except GgufoneError as exc:
        return _fail(exc, command=cmd)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - never show a traceback for an ordinary run
        print(f"error: E_INTERNAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 4
    print(f"'{cmd}' is not implemented yet (milestone {MILESTONES.get(cmd, 'E1')}); "
          f"see SPEC.md 5", file=sys.stderr)
    return 3


def _fail(exc: GgufoneError, *, command: str | None = None) -> int:
    message = str(exc)
    if not message.startswith(exc.code):
        message = f"{exc.code}: {message}"
    print(f"error: {message}", file=sys.stderr)
    if exc.code == "E_UNKNOWN_KEY" and command:
        print(f"hint: run `ggufone {command} --help` for the flags this command accepts",
              file=sys.stderr)
    return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
