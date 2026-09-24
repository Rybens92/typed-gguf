"""Triage a mutmut 3.8 tree without trusting the survivor list (E1c card t_c8e36cad).

Reads the run artifacts directly:
  mutants/<rel>.meta    -> exit_code_by_key (0 survived, 1/3 killed, 33 no-tests, 37 type-check,
                           -24 timeout, 5 no-tests, None not checked)
  mutants/<rel>.spans   -> mutant key -> [start, end] line span inside the inlined mutant file
  mutants/mutmut-stats.json -> tests_by_mangled_function_name / duration_by_test (the test selection)

Usage:
  python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py status [repo]
  python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py tests <mangled_name> [repo]
  python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py classify <n> [repo]
  python .e2e/t_c8e36cad-e1c/logs/mutation_triage.py diff <mutant_key> [repo]
"""
from __future__ import annotations

import collections
import difflib
import json
import pathlib
import re
import sys

STATUS = collections.defaultdict(
    lambda: "suspicious",
    {
        1: "killed",
        3: "killed",
        0: "survived",
        5: "no-tests",
        2: "interrupted",
        None: "not-checked",
        33: "no-tests",
        34: "skipped",
        35: "suspicious",
        36: "timeout",
        37: "type-check",
        -24: "timeout",
        24: "timeout",
        152: "timeout",
        255: "timeout",
        -11: "segfault",
        -9: "segfault",
    },
)

FILES = ["src/ggufone/engine/template.py", "src/ggufone/runtime/fit.py"]


def iso(s: str) -> tuple[str, str, int, int, int, int, int, int, int, int, int, int]:
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)\.(\d+)", s)
    assert m, s
    return m.groups()  # type: ignore[return-value]


def load(repo: pathlib.Path, rel: str):
    base = repo / "mutants" / rel
    meta = json.loads((base.with_name(base.name + ".meta")).read_text())
    spans = json.loads((base.with_name(base.name + ".spans")).read_text())["spans"]
    return meta, spans


def cmd_status(repo: pathlib.Path) -> None:
    for rel in FILES:
        meta, spans = load(repo, rel)
        codes = meta["exit_code_by_key"]
        c = collections.Counter(STATUS[v] for v in codes.values())
        total = len(codes)
        ran = c["killed"] + c["survived"] + c["timeout"] + c["segfault"] + c["suspicious"]
        denom = total - c["no-tests"] - c["not-checked"] - c["skipped"] - c["type-check"]
        print(f"== {rel}")
        print(f"   total={total}  {dict(c)}")
        print(f"   score(killed/ran) = {100.0 * c['killed'] / denom:.1f}%   "
              f"score(killed/total) = {100.0 * c['killed'] / total:.1f}%   (ran={ran})")
    # stats-file summary
    stats = json.loads((repo / "mutants" / "mutmut-stats.json").read_text())
    tbm = stats["tests_by_mangled_function_name"]
    print(f"\n== mutmut-stats.json: {len(tbm)} functions with recorded coverage, "
          f"{len(stats['duration_by_test'])} tests timed")
    print(f"   stats_time={stats['stats_time']}  git_commit={stats['git_commit']}")
    durs = sorted(stats["duration_by_test"].items(), key=lambda kv: -kv[1])
    print("   slowest tests:")
    for name, d in durs[:5]:
        print(f"     {d:8.2f}s  {name}")
    empty = [k for k, v in tbm.items() if not v]
    print(f"   functions with an empty test set: {len(empty)}")


def cmd_tests(repo: pathlib.Path, name: str) -> None:
    stats = json.loads((repo / "mutants" / "mutmut-stats.json").read_text())
    tbm = stats["tests_by_mangled_function_name"]
    hits = {k: v for k, v in tbm.items() if name in k}
    print(json.dumps(hits, indent=2)[:4000])


def short(key: str) -> str:
    """Spans keys carry no module prefix: 'ggufone.engine.template.x_fn__mutmut_3' -> 'x_fn__mutmut_3'."""
    return key.split(".", 3)[3] if key.count(".") >= 3 else key


def body(lines: list[str], spans, key: str) -> str:
    start, end = spans[key]
    return "".join(lines[start - 1:end])


def cmd_classify(repo: pathlib.Path, n: int) -> None:
    for rel in FILES:
        meta, spans = load(repo, rel)
        lines = (repo / "mutants" / rel).read_text().splitlines(keepends=True)
        codes = meta["exit_code_by_key"]
        survivors = {k: v for k, v in codes.items() if v == 0}
        buckets: dict[str, list[str]] = collections.defaultdict(list)
        for key in survivors:
            parts = key.split("__mutmut_")[0].split("ǁ")
            fn = ".".join(parts[-2:]) if len(parts) > 1 else parts[0]
            buckets[fn].append(key)
        print(f"\n== {rel}: {len(survivors)} survivors in {len(buckets)} functions")
        for fn, keys in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            print(f"   {len(keys):4d}  {fn}")
        # diff digest for the biggest buckets
        for fn, keys in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"\n--- {fn}: {len(keys)} survivors, digest of first {n} ---")
            for key in keys[:n]:
                skey = short(key)
                orig_key = skey.split("__mutmut_")[0] + "__mutmut_orig"
                if orig_key not in spans:
                    print(f"   [{skey}] no orig span")
                    continue
                a = body(lines, spans, orig_key).splitlines()
                b = body(lines, spans, skey).splitlines()
                diff = [
                    ln for ln in difflib.unified_diff(a, b, lineterm="", n=0)
                    if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
                    and ln[1:].strip()
                ]
                diff = diff[:4]
                print(f"   [{skey}]")
                for ln in diff:
                    print(f"      {ln.strip()[:150]}")


def changed_lines(lines, spans, skey) -> list[str]:
    orig_key = skey.split("__mutmut_")[0] + "__mutmut_orig"
    if orig_key not in spans or skey not in spans:
        return []
    a = body(lines, spans, orig_key).splitlines()
    b = body(lines, spans, skey).splitlines()
    return [
        ln for ln in difflib.unified_diff(a, b, lineterm="", n=0)
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")) and ln[1:].strip()
    ]


def kind_of(changed: list[str]) -> str:
    """Classify a survivor by the mutation operator that produced it (see mutmut/mutation/mutators.py)."""
    plus = " ".join(ln[1:] for ln in changed if ln.startswith("+"))
    minus = " ".join(ln[1:] for ln in changed if ln.startswith("-"))
    if "XX" in plus and "XX" not in minus:
        return "string-XX-wrap"
    if (".upper()" in plus and ".lower()" in minus) or (".lower()" in plus and ".upper()" in minus):
        return "string-case-swap"
    if re.search(r"\b\w+\.(split|rsplit|find|rfind|lstrip|rstrip|ljust|rjust|partition|rpartition|removeprefix|removesuffix|index|rindex)\b", plus) \
            and re.search(r"\b\w+\.(split|rsplit|find|rfind|lstrip|rstrip|ljust|rjust|partition|rpartition|removeprefix|removesuffix|index|rindex)\b", minus):
        return "string-method-swap"
    if "and False" in plus or "or True" in plus:
        return "ifexp-forced"
    if "= None" in plus and "= None" not in minus:
        return "assign->None"
    if re.search(r"\bNone\b", plus) and not re.search(r"\bNone\b", minus):
        return "arg->None/removal"
    if re.search(r"(?<![\w.])not\s+(\w|\()", minus) and len(plus.split()) < len(minus.split()):
        return "unary-removal"
    if re.search(r"\bTrue\b", plus) != re.search(r"\bTrue\b", minus) and "XX" not in plus:
        return "bool-flip"
    if re.search(r"\bdeepcopy\b", minus) or re.search(r"\bcopy\b", plus):
        return "name-deepcopy->copy"
    if "lambda" in plus or "lambda" in minus:
        return "lambda-body"
    # operator / operand changes: compare the operator tokens
    ops = ["<=", ">=", "==", "!=", "<<", ">>", "//", "**", " is not ", " is ", " not in ", " in ", " and ", " or ", "+", "-", "*", "/", "%", "|", "&", "^", "<", ">"]
    def op_multiset(text: str) -> dict:
        found = {}
        for op in ops:
            n = text.count(op)
            if n:
                found[op] = n
        return found
    if op_multiset(plus) != op_multiset(minus):
        return "operator-swap"
    if re.search(r"\d", plus) and re.search(r"\d", minus):
        return "number+1"
    if "dict(" in plus and re.search(r"=[^,)]+", plus):
        return "kwarg-rename"
    return "other"


def cmd_kinds(repo: pathlib.Path, n: int) -> None:
    for rel in FILES:
        meta, spans = load(repo, rel)
        lines = (repo / "mutants" / rel).read_text().splitlines(keepends=True)
        survivors = [k for k, v in meta["exit_code_by_key"].items() if v == 0]
        kinds: dict[str, list[str]] = collections.defaultdict(list)
        nochg = 0
        for key in survivors:
            changed = changed_lines(lines, spans, short(key))
            if not changed:
                nochg += 1
            kinds[kind_of(changed)].append(short(key))
        print(f"\n== {rel}: {len(survivors)} survivors, class -> count (no-diff: {nochg})")
        for kind, keys in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            print(f"   {len(keys):5d}  {kind}")
            for k in keys[:n]:
                print(f"            {k}")


def cmd_digests(repo: pathlib.Path, want: str, max_show: int) -> None:
    """Group one function's survivors by unique changed-line signature (most informative view)."""
    total_shown = 0
    for rel in FILES:
        meta, spans = load(repo, rel)
        lines = (repo / "mutants" / rel).read_text().splitlines(keepends=True)
        survivors = [k for k, v in meta["exit_code_by_key"].items() if v == 0 and want in k]
        if not survivors:
            continue
        groups: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
        for key in survivors:
            changed = changed_lines(lines, spans, short(key))
            sig = tuple(sorted({ln[0] + ln[1:].strip()[:110] for ln in changed}))
            groups[sig].append(short(key))
        print(f"== {rel} :: {want}: {len(survivors)} survivors, "
              f"{len(groups)} distinct mutation signatures")
        for sig, keys in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if total_shown >= max_show:
                print(f"   ... ({len(groups) - total_shown} more signatures)")
                break
            total_shown += 1
            print(f"\n   x{len(keys)}  [{', '.join(k.split('__mutmut_')[1] for k in keys[:6])}]")
            for ln in sig:
                print(f"        {ln}")
        return


def cmd_source(repo: pathlib.Path, name: str, context: int) -> None:
    """Print the function's source with mutant-name -> line map, to plan tests against."""
    for rel in FILES:
        path = repo / rel
        text = path.read_text().splitlines()
        for i, line in enumerate(text, 1):
            if re.match(rf"\s*def {re.escape(name)}\(", line):
                start = i
                indent = len(line) - len(line.lstrip())
                end = len(text)
                for j in range(i, len(text)):
                    ln = text[j]
                    if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent and j + 1 > start:
                        end = j
                        break
                print(f"== {rel}:{start}-{end}  def {name}")
                lo = max(0, start - 1 - context)
                for k in range(lo, min(len(text), end + context)):
                    print(f"{k + 1:5d}  {text[k]}")
                return
    print("function not found", name)


def cmd_diff(repo: pathlib.Path, key: str) -> None:
    for rel in FILES:
        meta, spans = load(repo, rel)
        skey = short(key)
        if skey not in spans:
            continue
        lines = (repo / "mutants" / rel).read_text().splitlines(keepends=True)
        orig_key = skey.split("__mutmut_")[0] + "__mutmut_orig"
        print(f"== {rel}  key={skey}  status={STATUS[meta['exit_code_by_key'].get(key, 'x')]}")
        a = body(lines, spans, orig_key).splitlines()
        b = body(lines, spans, skey).splitlines()
        for ln in difflib.unified_diff(a, b, lineterm="", n=1):
            print(ln)
        return
    print("key not found", key)


def cmd_pick(repo: pathlib.Path, want: str, n: int) -> None:
    for rel in FILES:
        meta, _spans = load(repo, rel)
        codes = meta["exit_code_by_key"]
        hits = [k for k, v in codes.items() if STATUS[v] == want]
        print(f"== {rel}: {len(hits)} {want}; first {n}:")
        for k in hits[:n]:
            print(f"   {k}")


def cmd_affected(repo: pathlib.Path, test_suffix: str) -> None:
    """List the mutant keys whose recorded test selection contains a given test node id."""
    stats = json.loads((repo / "mutants" / "mutmut-stats.json").read_text())
    tbm = stats["tests_by_mangled_function_name"]
    for rel in FILES:
        meta, _spans = load(repo, rel)
        codes = meta["exit_code_by_key"]
        prefix = "ggufone.engine.template." if "template.py" in rel else "ggufone.runtime.fit."
        hits = 0
        for fn, tests in sorted(tbm.items()):
            if not any(t.endswith(test_suffix) for t in tests):
                continue
            for key, code in codes.items():
                if key.startswith(prefix + "x") and fn.endswith(key.split(".")[-1]
                                                                .split("__mutmut_")[0]):
                    print(f"  {STATUS[code]:10s} {key}")
                    hits += 1
        if hits:
            print(f"-- {rel}: {hits} keys run the test '{test_suffix}'")


def cmd_verdicts(repo: pathlib.Path, path: str) -> None:
    """Print the sweep's verdict for each key listed in a file (cross-check for the replays)."""
    wanted = [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    seen: dict[str, str] = {}
    for rel in FILES:
        meta, _spans = load(repo, rel)
        for key, code in meta["exit_code_by_key"].items():
            if key in wanted:
                seen[key] = STATUS[code]
    for key in wanted:
        print(f"  {seen.get(key, 'ABSENT'):10s} {key}")


if __name__ == "__main__":
    repo = pathlib.Path(sys.argv[-1] if sys.argv[-1].startswith("/") else "/workspace/ggufone")
    mode = sys.argv[1]
    if mode == "status":
        cmd_status(repo)
    elif mode == "tests":
        cmd_tests(repo, sys.argv[2])
    elif mode == "classify":
        cmd_classify(repo, int(sys.argv[2]))
    elif mode == "diff":
        cmd_diff(repo, sys.argv[2])
    elif mode == "pick":
        cmd_pick(repo, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    elif mode == "kinds":
        cmd_kinds(repo, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif mode == "digests":
        cmd_digests(repo, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif mode == "source":
        cmd_source(repo, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif mode == "verdicts":
        cmd_verdicts(repo, sys.argv[2])
    elif mode == "affected":
        cmd_affected(repo, sys.argv[2])
    else:
        raise SystemExit(__doc__)
