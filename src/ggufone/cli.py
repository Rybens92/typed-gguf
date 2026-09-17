"""Command-line surface (SPEC 2.8). Milestone: E1a (init/doctor/models), E1b (run/ask).

The full command set is frozen by SPEC 2.8; unimplemented commands exit 3 with a
milestone pointer instead of pretending to work.
"""

from __future__ import annotations

import sys

from ggufone import __version__

COMMANDS = ("init", "doctor", "models", "run", "ask", "serve", "mcp", "bench",
            "fit", "calibrate", "version")
MODELS_SUBCOMMANDS = ("search", "pull", "use", "ls", "rm", "verify", "recommend-quant")
# command -> milestone that implements it (SPEC 5)
MILESTONES = {"version": "E0", "init": "E1a", "doctor": "E1a", "models": "E1a",
              "run": "E1b", "ask": "E1b", "fit": "E1c", "serve": "E1b", "mcp": "E1b",
              "bench": "E2", "calibrate": "E2.5"}


def _usage() -> str:
    lines = [f"ggufone {__version__}", "usage: ggufone <command> [options]", "", "commands:"]
    for cmd in COMMANDS:
        lines.append(f"  {cmd:12s} (implemented in {MILESTONES.get(cmd, 'E1')})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    cmd = args[0]
    if cmd in ("-V", "--version", "version"):
        print(f"ggufone {__version__}")
        return 0
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    print(f"'{cmd}' is not implemented yet (milestone {MILESTONES.get(cmd, 'E1')}); "
          f"see SPEC.md 5", file=sys.stderr)
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
