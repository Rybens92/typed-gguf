"""E3d: does the *serving* path reproduce the probe's cue-shape rows? (card t_d90404ac)

The E3d decision rests on a probe (`tools/e3c_cue_shapes.py`) whose three shapes were measured on
the committed 60-item dev set. A probe is an instrument, and an instrument that disagrees with the
thing it claims to measure is worse than no measurement — so this tool runs the *serving* path
(`decide.plan_context(request, handle)` -> `ModelSession` -> `DecisionEngine`, exactly what
`ggufone ask`/`run` do, chat-template framing included) over the same items and the same cues and
prints both rows side by side.

What it compares, per item and per cue:

* the winning label (`choice` / the probability argmax) — the probe's `ranked[<cue>=bare].got`;
* the reliability verdict — the engine's own word against the probe's label mass at the 0.10 floor.

The two paths are *not* byte-equal and this tool does not pretend otherwise. The probe decodes every
shape of an item in **one batch** (three sequences, one graph) while the serving path decodes one
request at a time, and on this box that moves the row's tail: item c01's `bare` label mass is
0.01144 through the probe and 0.00696 through the serving path, with the same winner (`billing`)
and the same verdict (`low_mass`). The compared cells are therefore the *decisions* — the label and
the reliability word — which is what a published row is read from; the coverage is printed next to
them instead of asserted.

The model is opened on the CPU by default — the probe's placement — with `--gpu-layers` to say
otherwise. Offline by construction: no network.

    python3 tools/e3d_engine_check.py --record /work/e3d/full.json --items 12 --threads 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone import schema  # noqa: E402
from ggufone.bench import devset, harness  # noqa: E402
from ggufone.engine import decide, readout  # noqa: E402
from ggufone.engine import session as session_module  # noqa: E402

#: engine cue -> (the probe's ranked policy key, the probe's shape key)
CUES = {"shipped": ("shipped=bare", "shipped"),
        "two_step": ("two_step_shipped=bare", "two_step_shipped")}
FLOOR = float(schema.OPTION_DEFAULTS["coverage_floor"])
DEFAULT_MODEL = "/var/home/rybens/.hermes/models/Spark-X2.5-4B-Q8_0.gguf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True, help="the probe's JSON (e3c_cue_shapes run)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runtime", default=os.environ.get("GGUFONE_RUNTIME_DIR"))
    parser.add_argument("--devset", default=None)
    parser.add_argument("--items", type=int, default=12, help="items to re-run (stratified)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=0,
                        help="0 (default) keeps the model on the CPU, the probe's placement")
    return parser.parse_args(argv)


def hide_devices() -> None:
    """The probe's placement: no Vulkan driver is enumerated, so every tensor stays on the CPU."""
    missing = "/nonexistent/no-vulkan-icd.json"
    for key in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        os.environ[key] = missing


def stratified(items: list[devset.DevItem], count: int) -> list[devset.DevItem]:
    """A per-type mixture, so a shortened run cannot measure one question type and call it a run."""
    if count >= len(items):
        return list(items)
    by_type: dict[str, list[devset.DevItem]] = {}
    for item in items:
        by_type.setdefault(item.type, []).append(item)
    picked: list[devset.DevItem] = []
    index = 0
    while len(picked) < count:
        for qtype in sorted(by_type):
            bucket = by_type[qtype]
            if index < len(bucket) and len(picked) < count:
                picked.append(bucket[index])
        index += 1
    return picked


def serving_answer(item: devset.DevItem, handle, *, cue: str, threads: int) -> dict:
    """One item through the serving path: the handle's plan (chat template), one session."""
    payload = devset.request_for(item, model="e3d-engine-check", threads=threads, cue=cue)
    request = schema.parse_request(payload)
    plan = decide.plan_context(request, handle)      # the *serving* plan: the handle, not a session
    with session_module.ModelSession(handle, plan, backend="cpu", log=[]) as live:
        result = decide.DecisionEngine(live).decide(request, plan=plan)
    return result.answers[item.id]


def winner_of(answer: Mapping[str, Any]) -> str:
    """The label the engine answered with, for every question type.

    `choice` answers carry it directly; `score`/`noul` answers carry `probabilities` and a derived
    scalar (`score` is a weighted mean, `noul` a probability), so the compared cell is the frozen
    tie-break over the probability map — the same read the bench's own rows use
    (`readout.argmax_first` over `probabilities.values()`), which makes this comparable to the
    probe's `ranked[<policy>].got`.
    """
    if "choice" in answer:
        return str(answer["choice"])
    keys = list(answer["probabilities"])
    return str(keys[readout.argmax_first([float(answer["probabilities"][key]) for key in keys])])


def probe_cell(row: Mapping[str, Any], policy: str, shape: str) -> tuple[str, float]:
    """The probe's cell for one item/policy: the ranked winner and the `bare` label mass.

    The record nests it (`shapes[<shape>].readout.labels.bare.coverage`); `readout.row` says which
    row the readout sat on (`cue` or `advanced`). Read defensively so a record from another probe
    version fails with a sentence, not a `KeyError`.
    """
    try:
        got = row["ranked"][policy]["got"]
        coverage = float(row["shapes"][shape]["readout"]["labels"]["bare"]["coverage"])
    except (KeyError, TypeError) as exc:  # pragma: no cover - a mis-shaped record
        raise SystemExit(f"record item {row.get('id')!r}: no {policy!r}/{shape!r} cell "
                         f"({type(exc).__name__}: {exc})") from exc
    return got, coverage


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu_layers == 0:
        hide_devices()
    record = json.loads(pathlib.Path(args.record).read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in record["items"]}
    chosen = stratified([item for item in devset.load(args.devset) if item.id in rows],
                        args.items)

    log: list[str] = []
    handle = session_module.open_model(args.model, runtime_dir=args.runtime,
                                       fit_plan=harness.Placement(args.gpu_layers), log=log)
    print(f"model {pathlib.Path(args.model).name} · n_gpu_layers={args.gpu_layers} · "
          f"threads {args.threads} · cues {', '.join(CUES)}")
    print(f"record {args.record} · items {len(chosen)} stratified · floor {FLOOR}")
    print()
    print(f"{'item':5s} {'type':6s} {'cue':10s} | {'probe':>27s} | {'engine':>27s} | verdict")
    print(f"{'':5s} {'':6s} {'':10s} | {'got':>10s} {'cov':>9s} {'low':>5s} | "
          f"{'got':>10s} {'cov':>9s} {'rel':>5s} |")
    differences: list[str] = []
    for item in chosen:
        for cue, (policy, shape) in CUES.items():
            probe_got, probe_cov = probe_cell(rows[item.id], policy, shape)
            probe_low = probe_cov < FLOOR
            answer = serving_answer(item, handle, cue=cue, threads=args.threads)
            engine_got = winner_of(answer)
            same_label = engine_got == probe_got
            same_low = (answer["reliability"] == "low_mass") == probe_low
            verdict = "ok" if (same_label and same_low) else "DIFF"
            if verdict != "ok":
                differences.append(f"{item.id}/{cue}: probe {probe_got!r} {probe_cov:.4g} "
                                   f"low={probe_low} vs engine {engine_got!r} "
                                   f"{answer['coverage']:.4g} rel={answer['reliability']}")
            print(f"{item.id:5s} {item.type:6s} {cue:10s} | {probe_got:>10s} {probe_cov:9.4g} "
                  f"{str(probe_low):>5s} | {engine_got:>10s} {answer['coverage']:9.4g} "
                  f"{answer['reliability']:>5s} | {verdict}")
    handle.close()
    print()
    cells = len(chosen) * len(CUES)
    print(f"engine/probe verdicts: {cells - len(differences)} ok, {len(differences)} different")
    for line in differences:
        print(f"  DIFF {line}")
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
