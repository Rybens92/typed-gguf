#!/usr/bin/env python3
"""E3b: the `qwen35moe` label policy — candidate-mass coverage per label/cue variant (t_6952f0dd).

E3 (`t_a431be85`) published a paired comparison in which **every** Occamy answer was `low_mass`
(20/20) while the 4B was mostly `measured` on the same 20 items: the engine's coverage — the
full-vocabulary mass of the candidate label's first token at the cue — sat below the 0.10 floor on
all of them. This tool measures whether the *rendering* of the label is part of that, by scoring
several documented renderings of the same question inside one run on the same box:

* five **label variants** (`bench/labels.py`): `bare` (what the engine ships), `space`, `caps`,
  `newline` (the two-step readout), `long`;
* three **cue variants**: `shipped`, `blank` (the empty line the 4B's own probe measured to triple
  the label mass, `docs/TEMPLATES.md` §4), `explicit` (the cue names the labels);
* optionally the **ranked** readout (restricted softmax over sequence scores — the engine's own
  decision) for chosen `cue=label` policies;
* optionally a second **prefix variant** (`--extra-prefix kept`, or `--only-prefix kept` for the
  re-measure path): the same prompt with the template's own empty `<think></think>` block left in
  place, i.e. without the family policy's `strip_empty_think_block`. That is not a label policy —
  it is the control that tells a *distribution* finding ("the model barely puts mass on our
  labels") apart from a *prompt shape* finding ("the model closes the assistant turn before
  answering"). It costs one prefill per item, so `--extra-prefix-items N` caps how many items pay
  for it (and `--only-prefix kept` skips the shipped prefix entirely — the shipped side of a
  before/after is E3's published report, already on disk).

Design constraints that make this affordable on a box where one weight sweep costs minutes:

* **one model load, one context per item-prefix** (the production pattern: `LiveModel` loads once
  and hands every `decide()` a fresh session, because the recurrent state of a hybrid model cannot
  be re-prefilled in place);
* **all cue variants of one prefix in ONE decode batch** — each cue is a sequence forked from the
  prefix, exactly the engine's candidate protocol, so the cue variants share the sweep;
* **coverage needs no further decode**: it is read from the cue row by `labels.label_coverage`;
* **ranked scoring shares prefixes** (a trie over the candidate sequences): the engine's step
  protocol without decoding a shared prefix once per candidate.

Every number printed here comes from `readout` (the engine's own arithmetic) and every cue/label
variant is a pure transformation of what `prompt.build_question` rendered (`bench/labels.py`).
The shipped policy can be cross-checked against `DecisionEngine` itself (`--cross-check N`: those
items pay one extra prefill each), so a probe bug cannot be published as a model finding.

    # the sweep (default: every cue x every label variant, coverage only)
    GGUFONE_RUNTIME_DIR=<bundle> VK_DRIVER_FILES=<icd> python3 tools/e3b_label_policy.py run \\
        --model ~/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf \\
        --devset docs/evidence/e3_chunks/devset_001.jsonl --per-type 2 \\
        --rank shipped=newline --rank shipped=bare --cross-check 1 \\
        --out .e3b/sweep.json --report .e3b/sweep.md --report-dir .e3b/reports

    # offline: the markdown tables from a stored run
    python3 tools/e3b_label_policy.py report --run .e3b/sweep.json --out .e3b/sweep.md
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggufone import schema  # noqa: E402
from ggufone.bench import devset, harness, labels, suites  # noqa: E402
from ggufone.engine import decide, prompt, readout  # noqa: E402
from ggufone.engine import session as session_module  # noqa: E402
from ggufone.engine import template as template_module  # noqa: E402

SCHEMA = "ggufone.e3b.label-policy/v1"
SHIPPED_PREFIX = "shipped"
KEPT_PREFIX = "kept"
PREFIX_VARIANTS = (SHIPPED_PREFIX, KEPT_PREFIX)
DEFAULT_THREADS = 4
DEFAULT_VK_DRIVER_FILES = "/work/e3scratch/nvidia_egl_icd.json"
DEFAULT_STATES_HOME = "/work/e3b/states"
DEFAULT_TOP_TOKENS = 8
#: ceiling for the context's `n_seq_max` (cue heads + ranked tries). The KV cache is *unified*
#: (n_ctx cells, not n_ctx x n_seq_max), so this only bounds how many branches may coexist.
MAX_SEQUENCES = 64


# --------------------------------------------------------------------------- item selection
def select_items(items: Sequence[devset.DevItem], *, per_type: int | None,
                 ids: Sequence[str] | None, limit: int | None) -> list[devset.DevItem]:
    """A small fixed subset: `--ids` wins, else `per_type` per question type, then `--limit`."""
    chosen = list(items)
    if ids:
        wanted = {str(value) for value in ids}
        found = [item for item in chosen if item.id in wanted]
        missing = sorted(wanted - {item.id for item in found})
        if missing:
            raise SystemExit(f"--ids: no such dev item(s): {', '.join(missing)}")
        return found
    if per_type:
        picked: list[devset.DevItem] = []
        for qtype in devset.QUESTION_TYPES:
            picked.extend([item for item in chosen if item.type == qtype][:per_type])
        chosen = picked
    return chosen[:limit] if limit else chosen


def request_for_item(item: devset.DevItem, *, threads: int, n_seq_max: int) -> schema.Request:
    return schema.parse_request(devset.request_for(item, model="e3b", threads=threads,
                                                   n_seq_max=n_seq_max))


# --------------------------------------------------------------------------- prompt variants
def prefix_text(request: schema.Request, handle: session_module.ModelHandle, *,
                variant: str) -> str:
    """The bytes every question of this item shares, under a prefix variant.

    `shipped` is what `decide.plan_context` builds (`prompt.build_prefix` with the resolved
    template and the family's thinking policy applied). `kept` renders the model's own template
    with `enable_thinking=false` but *skips* the family policy's empty-block strip, so the
    assistant turn keeps the `<think>…</think>` pair the Qwen3.5 templates emit — the shape the
    model was trained against, and the only difference between the two variants.
    """
    if variant not in PREFIX_VARIANTS:
        raise SystemExit(f"unknown prefix variant {variant!r}; known: {', '.join(PREFIX_VARIANTS)}")
    resolution = decide.resolve_template(request, handle)
    if variant == SHIPPED_PREFIX:
        return prompt.build_prefix(request.state, resolution=resolution,
                                   enable_thinking=request.options.thinking)
    if resolution is None or resolution.template is None:
        raise SystemExit(f"--extra-prefix {variant} needs a resolvable chat template")
    return template_module.render(resolution.template, messages=prompt.chat_messages(request.state),
                                  add_generation_prompt=True,
                                  kwargs={"enable_thinking": request.options.thinking})


def cue_suffixes(question: schema.Question, cues: Sequence[str]) -> dict[str, str]:
    """`{cue variant: suffix}` — the rendered suffix with only its cue line replaced."""
    shipped = prompt.build_question(question).suffix
    return {cue: labels.suffix_with_cue(shipped, question.type, question.options, cue)
            for cue in cues}


def label_sets(question: schema.Question, variants: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return {variant: labels.label_texts(question.type, question.options, question.descriptions,
                                        variant)
            for variant in variants}


# --------------------------------------------------------------------------- batching
def batch_for(wanted: Sequence[tuple[int, Sequence[int]]],
              positions: Sequence[Sequence[int]]) -> decide.Batch:
    """One decode: every `(seq, tokens)` pair in one batch, `logits=1` at each last token."""
    tokens: list[int] = []
    seq_ids: list[int] = []
    spots: list[int] = []
    flags: list[bool] = []
    for (seq, sequence), places in zip(wanted, positions, strict=True):
        last = len(sequence) - 1
        for index, token in enumerate(sequence):
            tokens.append(int(token))
            seq_ids.append(int(seq))
            spots.append(int(places[index]))
            flags.append(index == last)
    return decide.Batch(tokens=tuple(tokens), seq_ids=tuple(seq_ids), positions=tuple(spots),
                        logits=tuple(flags))


def top_tokens(row: Sequence[float], scale: float, handle: session_module.ModelHandle,
               count: int) -> list[dict[str, Any]]:
    """The `count` most likely next tokens at one row — the evidence behind a coverage number."""
    order = sorted(range(len(row)), key=lambda token: -float(row[token]))[:max(0, int(count))]
    return [{"token": int(token),
             "piece": session_module.ctypes_binding.token_piece(handle.runtime, handle.vocab,
                                                                token),
             "p_full": readout.coverage_from_scale(row, (token,), scale)} for token in order]


def rank_policy(live: session_module.ModelSession, *, head_seq: int, base: int,
                decision_row: Sequence[float], decision_scale: float, pool: Sequence[int],
                sequences: Sequence[tuple[int, ...]], options: Sequence[str], length_norm: float,
                temperature: float) -> dict[str, Any]:
    """The engine's decision for one policy: sequence scores -> restricted softmax -> argmax."""
    logprobs = labels.score_paths(live, head_seq=head_seq, base=base, pool=pool,
                                  paths=sequences)
    first = [readout.logprob_from_scale(decision_row, path[0], decision_scale)
             for path in sequences]
    z = [readout.candidate_sequence_score([first[index], *logprobs[path]], length_norm)
         for index, path in enumerate(sequences)]
    probabilities = readout.restricted_softmax(z, temperature)
    winner = readout.argmax_first(probabilities)
    scores = sorted(zip(sequences, z, strict=True), key=lambda item: list(item[0]))
    return {"probabilities": dict(zip(options, probabilities, strict=True)),
            "sequence_scores": {str(list(path)): value for path, value in scores},
            "got": options[winner], "confidence": readout.confidence(probabilities)}


# --------------------------------------------------------------------------- one item x prefix
def measure_prefix(handle: session_module.ModelHandle, item: devset.DevItem, *,
                   prefix_variant: str, cues: Sequence[str], variants: Sequence[str],
                   rank: Sequence[tuple[str, str]], threads: int, states_home: pathlib.Path,
                   length_norm: float, temperature: float, max_sequences: int,
                   top_tokens_count: int) -> dict[str, Any]:
    """One dev item under one prefix variant: prefill, one batched cue decode, all variants."""
    n_seq_max = max(1 + len(cues) + 4 * len(item.criteria) + 4, 8)
    request = request_for_item(item, threads=threads, n_seq_max=n_seq_max)
    question = request.questions[0]
    plan = decide.plan_context(request, handle)
    rendered = prefix_text(request, handle, variant=prefix_variant)
    fixed = dataclasses.replace(plan, prefix_tokens=tuple(handle.tokenize(rendered)),
                                n_ctx=plan.n_ctx + 4096)
    suffixes = cue_suffixes(question, cues)
    sets = label_sets(question, variants)
    tokens = {cue: handle.tokenize(suffix) for cue, suffix in suffixes.items()}
    tokenized = {variant: {text: handle.tokenize(text) for text in texts}
                 for variant, texts in sets.items()}
    record: dict[str, Any] = {
        "prefix_variant": prefix_variant, "prefix_tokens": fixed.n_prefix,
        "n_ctx": fixed.n_ctx, "n_seq_max": fixed.n_seq_max,
        "prefix_tail": rendered[-60:], "suffix_tokens": {cue: len(value)
                                                         for cue, value in tokens.items()},
        "tokenized": {variant: dict(mapping) for variant, mapping in tokenized.items()},
        "cues": {}, "ranked": {}}
    with session_module.ModelSession(handle, fixed, backend="vulkan", states_home=states_home,
                                     log=[]) as live:
        info = live.prefill(list(fixed.prefix_tokens), state_cache=False)
        record["prefill_ms"] = round(info.prefill_ms, 1)
        wanted: list[tuple[int, Sequence[int]]] = []
        heads: dict[str, int] = {}
        for index, cue in enumerate(cues, start=1):
            heads[cue] = index
            live.fork(0, index, fixed.n_prefix)
            wanted.append((index, tokens[cue]))
        places = [list(range(fixed.n_prefix, fixed.n_prefix + len(tokens[cue]))) for cue in cues]
        started = time.perf_counter()
        rows = live.decode(batch_for(wanted, places))
        record["cue_decode_s"] = round(time.perf_counter() - started, 1)
        decision_rows: dict[str, list[float]] = {}
        scales: dict[str, float] = {}
        for cue, row in zip(cues, rows, strict=True):
            scale = readout.logsumexp(row)
            decision_rows[cue] = row
            scales[cue] = scale
            entry: dict[str, Any] = {"scale": scale, "labels": {},
                                     "top_tokens": top_tokens(row, scale, handle,
                                                              top_tokens_count)}
            for variant, texts in sets.items():
                texts_list = list(texts)
                first = [tokenized[variant][text][0] for text in texts_list]
                coverage = labels.label_coverage(texts_list, handle.tokenize, row, scale)
                entry["labels"][variant] = {
                    "texts": texts_list,
                    "first_tokens": first,
                    "pieces": [session_module.ctypes_binding.token_piece(
                        handle.runtime, handle.vocab, token) for token in first],
                    "coverage": coverage,
                    "reliability": labels.reliability_of(coverage),
                    "shared_first_tokens": [list(group) for group in
                                            labels.shared_first_tokens(texts_list,
                                                                       handle.tokenize)],
                }
            record["cues"][cue] = entry
        # ---- ranked policies: one disjoint sequence pool per policy, sized to its trie
        cursor = 1 + len(cues)
        for cue, variant in rank:
            sequences = [tuple(tokenized[variant][text]) for text in sets[variant]]
            nodes = labels.trie_nodes(sequences)
            if cursor + nodes > max_sequences:
                raise SystemExit(
                    f"--max-sequences {max_sequences} is too small: the ranked policy "
                    f"{cue}={variant} needs {cursor + nodes} sequences (cue heads + trie)")
            pool = list(range(cursor, cursor + nodes))
            cursor += nodes
            ranked = rank_policy(live, head_seq=heads[cue], base=fixed.n_prefix + len(tokens[cue]),
                                 decision_row=decision_rows[cue], decision_scale=scales[cue],
                                 pool=pool, sequences=sequences, options=question.options,
                                 length_norm=length_norm, temperature=temperature)
            ranked["label"] = variant
            ranked["cue"] = cue
            ranked["coverage"] = record["cues"][cue]["labels"][variant]["coverage"]
            ranked["reliability"] = labels.reliability_of(ranked["coverage"])
            ranked["correct"] = ranked["got"] == devset.gold_key(item)
            ranked["sequences"] = [list(path) for path in sequences]
            record["ranked"][f"{cue}={variant}"] = ranked
    return record


def cross_check_item(handle: session_module.ModelHandle, item: devset.DevItem, *,
                     threads: int, states_home: pathlib.Path) -> dict[str, Any]:
    """The shipped policy through `DecisionEngine` itself — the probe's own control.

    A *fresh* session on the same item through the production engine: its winner, coverage and
    reliability must agree with the probe's `shipped`/`bare` numbers within float noise (the only
    difference is the batch shape). A probe that disagrees here cannot publish a variant finding.
    """
    request = request_for_item(item, threads=threads, n_seq_max=8)
    plan = decide.plan_context(request, handle)
    with session_module.ModelSession(handle, plan, backend="vulkan", states_home=states_home,
                                     log=[]) as live:
        result = decide.DecisionEngine(live).decide(request, plan=plan)
    answer = dict(result.answers[item.id])
    probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
    winner = list(probabilities)[readout.argmax_first(list(probabilities.values()))]
    return {"got": winner, "coverage": float(answer.get("coverage") or 0.0),
            "reliability": answer.get("reliability"), "probabilities": probabilities}


# --------------------------------------------------------------------------- offline report
def piece_of(item: dict[str, Any], prefix: str | None = None) -> dict[str, Any]:
    """One item's measurement under a prefix variant (`None` = the first one measured there).

    A run with `--only-prefix kept` has no `shipped` piece at all — every report function reads
    this helper, so "the prefix this run is about" is the first key the item carries.
    """
    prefixes = item.get("prefixes") or {}
    if not prefixes:
        raise KeyError(f"item {item.get('id')} carries no prefix measurement")
    key = SHIPPED_PREFIX if prefix is None and SHIPPED_PREFIX in prefixes else prefix
    if key is None:
        return next(iter(prefixes.values()))
    if key in prefixes:
        return prefixes[key]
    raise KeyError(f"item {item.get('id')} has no `{key}` prefix measurement")


def shipped_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [piece_of(item) for item in record["items"]]


def top_token_block(record: dict[str, Any], *, cues: Sequence[str], count: int = 4) -> list[str]:
    """What the model actually wants to emit at each cue — the row the coverage is read from."""
    lines = ["| item | cue | top tokens at the cue (full-vocab p) |", "|---|---|---|"]
    for item in record["items"]:
        piece = piece_of(item)
        for cue in cues:
            entry = piece["cues"][cue]
            rendered = " · ".join(
                f"`{token['piece']}` {token['p_full']:.4f}"
                for token in entry.get("top_tokens", [])[:count])
            lines.append(f"| {item['id']} | `{cue}` | {rendered or '—'} |")
    return lines


def value_format(values: Sequence[float]) -> str:
    """The format string that prints `values` readably — the report's own number policy.

    These masses run from ~1e-8 to ~1e-6, and a fixed 4-decimal rendering turns the whole card
    into a column of `0.0000` — precisely the numbers the reader must tell apart. So a table whose
    smallest value fixed-point cannot express (it would print as `0.0000`) switches to 3-decimal
    scientific notation; a table whose numbers are at a readable scale keeps the familiar form.
    """
    smallest = min((abs(float(value)) for value in values), default=0.0)
    return ".3e" if 0.0 < smallest < 5e-5 else ".4f"


def coverage_table(record: dict[str, Any]) -> list[str]:
    """One row per (cue, label) variant: per-item coverage, the floor verdict, and the aggregate."""
    pieces = shipped_items(record)
    lines = ["| cue | label | " + " | ".join(item["id"] for item in record["items"])
             + " | above floor | median |",
             "|---|---|" + "---|" * (len(pieces) + 2)]
    for cue, variant in itertools.product(record["cues"], record["label_variants"]):
        values = [piece["cues"][cue]["labels"][variant]["coverage"] for piece in pieces]
        fmt = value_format(values)
        above = sum(1 for value in values if value >= record["mass_floor"])
        ordered = sorted(values)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        cells = [f"{value:{fmt}}{'' if value >= record['mass_floor'] else '*'}"
                 for value in values]
        lines.append(f"| `{cue}` | `{variant}` | " + " | ".join(cells)
                     + f" | {above}/{len(values)} | {median:{fmt}} |")
    lines.append("")
    lines.append(f"`*` = below the engine's floor ({record['mass_floor']:.2f} ⇒ `low_mass`). "
                 f"Coverage is the full-vocabulary mass of the label's first token at the cue "
                 f"(`readout.coverage_from_scale`), read from the cue row — so every label variant "
                 f"of one cue costs no extra forward pass.")
    return lines


def variant_summary(record: dict[str, Any]) -> list[str]:
    pieces = shipped_items(record)
    lines = ["| cue | label | mean coverage | median | above floor | low_mass share | best item "
             "| worst item |", "|---|---|---|---|---|---|---|---|"]
    for cue in record["cues"]:
        for variant in record["label_variants"]:
            values = [piece["cues"][cue]["labels"][variant]["coverage"] for piece in pieces]
            fmt = value_format(values)
            above = sum(1 for value in values if value >= record["mass_floor"])
            order = sorted(range(len(values)), key=lambda index: -values[index])
            lines.append(
                f"| `{cue}` | `{variant}` | {sum(values) / len(values):{fmt}} "
                f"| {sorted(values)[len(values) // 2]:{fmt}} | {above}/{len(values)} "
                f"| {1 - above / len(values):.2f} "
                f"| {record['items'][order[0]]['id']} {values[order[0]]:{fmt}} "
                f"| {record['items'][order[-1]]['id']} {values[order[-1]]:{fmt}} |")
    return lines


def prefix_block(record: dict[str, Any]) -> list[str]:
    """The shipped prefix against the `kept` control, on the items that measured both."""
    paired = [item for item in record["items"] if len(item.get("prefixes") or {}) > 1]
    if not paired:
        return ["No `--extra-prefix` in this run: only the shipped prompt shape was measured."]
    lines = ["| item | prefix | prefix tokens | cue | top token (p) | `bare` coverage "
             "| `newline` coverage |", "|---|---|---|---|---|---|---|"]
    for item in paired:
        for variant in PREFIX_VARIANTS:
            piece = item["prefixes"].get(variant)
            if piece is None:
                continue
            for cue in [record["cues"][0]]:
                entry = piece["cues"][cue]
                top = entry.get("top_tokens") or [{"piece": "—", "p_full": 0.0}]
                lines.append(f"| {item['id']} | `{variant}` | {piece['prefix_tokens']} "
                             f"| `{cue}` | `{top[0]['piece']}` {top[0]['p_full']:.4f} "
                             f"| {entry['labels']['bare']['coverage']:.3e} "
                             f"| {entry['labels']['newline']['coverage']:.3e} |")
    return lines


def ranked_block(record: dict[str, Any]) -> list[str]:
    keys = list(record.get("ranked_keys", []))
    if not keys:
        return ["No policy was ranked in this run: coverage was measured alone "
                "(`--rank cue=label` runs the full readout)."]
    lines = ["| policy | n | correct | agreement | 95% CI | low_mass | median coverage |",
             "|---|---|---|---|---|---|---|"]
    for key in keys:
        rows = [piece_of(item)["ranked"][key] for item in record["items"]
                if key in piece_of(item).get("ranked", {})]
        if not rows:
            continue
        correct = sum(1 for row in rows if row["correct"])
        low, high = harness.wilson_interval(correct, len(rows))
        low_mass = sum(1 for row in rows if row["reliability"] == "low_mass")
        coverages = sorted(row["coverage"] for row in rows)
        lines.append(f"| `{key}` | {len(rows)} | {correct} | {correct / len(rows):.3f} "
                     f"| {low:.3f}–{high:.3f} | {low_mass}/{len(rows)} "
                     f"| {coverages[len(coverages) // 2]:.4f} |")
    return lines


def cross_check_block(record: dict[str, Any]) -> list[str]:
    rows = [item for item in record["items"] if item.get("cross_check")]
    if not rows:
        return ["No `--cross-check N` in this run: the probe's shipped-policy numbers were not "
                "confirmed against `DecisionEngine`."]
    lines = ["| item | probe coverage | engine coverage | |Δ coverage| | probe winner "
             "| engine winner |", "|---|---|---|---|---|---|"]
    worst = 0.0
    for item in rows:
        probe = piece_of(item)["cues"]["shipped"]["labels"]["bare"]
        engine = item["cross_check"]
        delta = abs(probe["coverage"] - engine["coverage"])
        worst = max(worst, delta)
        lines.append(f"| {item['id']} | {probe['coverage']:.4e} | {engine['coverage']:.4e} "
                     f"| {delta:.2e} | "
                     f"{piece_of(item)['ranked'].get('shipped=bare', {}).get('got', '—')} "
                     f"| {engine['got']} |")
    lines.append("")
    lines.append(f"Worst |Δ coverage| over the cross-checked items: **{worst:.2e}** — the probe "
                 f"reads the same row with the same function; only the batch shape differs.")
    return lines


def render_report(record: dict[str, Any]) -> str:
    model = record.get("model") or {}
    prefixes = record.get("prefix_variants") or [SHIPPED_PREFIX]
    lines = [
        f"# E3b — `{model.get('arch')}` label policy: candidate-mass coverage per variant",
        "",
        f"- card `t_6952f0dd` · model `{model.get('name')}` "
        f"({model.get('bytes') or 0:,} bytes, sha256 `{str(record.get('model_sha256'))[:16]}…`)",
        f"- runtime `{record['runtime']}` · backend `vulkan` · threads {record['threads']} "
        f"· `--gpu-layers` requested {record['gpu_layers']}",
        f"- placement used: `{json.dumps(record['placement'])}`",
        f"- dev items: {', '.join(item['id'] for item in record['items'])} "
        f"({', '.join(f'{key} {value}' for key, value in sorted(record['counts'].items()))}) "
        f"· prefix variants: {', '.join(f'`{name}`' for name in prefixes)} "
        f"· generated {record['generated_at']}",
        f"- engine floor: coverage < **{record['mass_floor']:.2f}** ⇒ `low_mass` "
        f"(`OPTION_DEFAULTS['coverage_floor']`)",
        f"- wall: {record['wall_s']:.1f} s (one model load {record['load_ms']:.0f} ms, one context "
        f"per item-prefix, one decode batch per prefix for all cue variants)",
        "",
        "## 1. What the model puts at the cue (the row coverage is read from)",
        "",
        *top_token_block(record, cues=record["cues"]),
        "",
        "## 2. Coverage per item, cue × label variant",
        "",
        *coverage_table(record),
        "",
        "## 3. Variant summary",
        "",
        *variant_summary(record),
        "",
        "## 4. The prefix control (shipped vs the template's own empty think block)",
        "",
        *prefix_block(record),
        "",
        "## 5. The ranked readout (the engine's decision under a policy)",
        "",
        *ranked_block(record),
        "",
        "## 6. Cross-check against `DecisionEngine` (shipped policy, `bare`)",
        "",
        *cross_check_block(record),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_policy_reports(record: dict[str, Any], directory: pathlib.Path) -> list[pathlib.Path]:
    """One `ggufone.bench/v1` quality report per ranked policy (`compare.py` can read them)."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for key in record.get("ranked_keys", []):
        rows: list[dict[str, Any]] = []
        for item in record["items"]:
            ranked = piece_of(item).get("ranked", {}).get(key)
            if ranked is None:
                continue
            rows.append({"id": item["id"], "type": item["type"], "expected": item["expected"],
                         "got": ranked["got"], "correct": bool(ranked["correct"]),
                         "confidence": ranked["confidence"], "coverage": ranked["coverage"],
                         "reliability": ranked["reliability"],
                         "probabilities": ranked["probabilities"]})
        cue, _, variant = key.partition("=")
        report = {
            "schema": harness.SCHEMA, "suite": "quality",
            "generated_at": record["generated_at"],
            "host": {"platform": "container", "cpu_count": 24},
            "model": dict(record.get("model") or {}, name=f"{record['model']['name']} [{key}]"),
            "items": rows,
            "per_type": suites.agreement_by_type(rows),
            "overall": suites.agreement(rows),
            "counts": record["counts"],
            "label_policy": {"cue": cue, "label": variant, "mass_floor": record["mass_floor"],
                             "engine": "readout.candidate_sequence_score + restricted_softmax"},
            "notes": [f"E3b label policy {key} (card t_6952f0dd): coverage is the mass of the "
                      f"label's first token under the `{variant}` rendering with the `{cue}` cue; "
                      f"the ranked readout is the engine's own arithmetic, driven by "
                      f"tools/e3b_label_policy.py."],
            "ok": True,
            "commands": {"reproduce": f"python3 tools/e3b_label_policy.py run --rank {key} "
                                      f"--model <path> --devset <devset.jsonl>"},
        }
        path = directory / f"policy_{key.replace('=', '-')}.json"
        harness.write_report(report, path)
        written.append(path)
        print(f"policy report: {path}")
    return written


# --------------------------------------------------------------------------- live driver
def live_run(args: argparse.Namespace) -> int:
    os.environ.setdefault("VK_DRIVER_FILES", args.vk_driver_files)
    model_path = str(pathlib.Path(os.path.expanduser(args.model)))
    runtime = args.runtime or os.environ.get("GGUFONE_RUNTIME_DIR")
    items = select_items(devset.load(args.devset), per_type=args.per_type, ids=args.ids,
                         limit=args.limit)
    if not items:
        raise SystemExit("no dev item selected")
    rank: list[tuple[str, str]] = []
    for pair in args.rank or []:
        cue, _, variant = pair.partition("=")
        if not variant:
            raise SystemExit(f"--rank takes cue=label, got {pair!r}")
        labels.cue_line("choice", ("a",), cue)               # reject an unknown cue early
        labels.label_texts("choice", ("a",), (None,), variant)
        rank.append((cue, variant))
    sha = ""
    try:
        from ggufone.runtime import fit
        sha = fit.ModelFacts.read(model_path, want_sha256=True).sha256
    except Exception as exc:                                # noqa: BLE001 - a pin, not a gate
        print(f"warning: could not pin the model sha256 ({exc})", file=sys.stderr)
    log: list[str] = []
    started = time.perf_counter()
    handle = session_module.open_model(model_path, runtime_dir=runtime,
                                       fit_plan=harness.Placement(args.gpu_layers), log=log)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": harness.model_facts(model_path),
        "model_sha256": sha, "runtime": str(handle.runtime.directory),
        "threads": args.threads, "gpu_layers": args.gpu_layers,
        "placement": handle.placement.to_dict(), "load_ms": round(handle.load_ms, 1),
        "cues": list(args.cues), "label_variants": list(args.label_variants),
        "prefix_variants": [SHIPPED_PREFIX] + ([args.extra_prefix] if args.extra_prefix else []),
        "mass_floor": labels.MASS_FLOOR,
        "ranked_keys": [f"{cue}={variant}" for cue, variant in rank],
        "counts": devset.counts(items), "devset": args.devset or str(devset.devset_path()),
        "items": [],
    }
    states = pathlib.Path(args.states_home)
    states.mkdir(parents=True, exist_ok=True)
    try:
        for number, item in enumerate(items):
            variants = [args.only_prefix] if args.only_prefix else [SHIPPED_PREFIX]
            if not args.only_prefix and args.extra_prefix and number < int(args.extra_prefix_items):
                variants.append(args.extra_prefix)
            piece: dict[str, Any] = {"id": item.id, "type": item.type,
                                     "expected": devset.gold_key(item), "prefixes": {}}
            for prefix_variant in variants:
                piece["prefixes"][prefix_variant] = measure_prefix(
                    handle, item, prefix_variant=prefix_variant, cues=args.cues,
                    variants=args.label_variants, rank=rank, threads=args.threads,
                    states_home=states, length_norm=args.length_norm,
                    temperature=args.temperature, max_sequences=args.max_sequences,
                    top_tokens_count=args.top_tokens)
            if number < int(args.cross_check):
                piece["cross_check"] = cross_check_item(handle, item, threads=args.threads,
                                                        states_home=states)
            record["items"].append(piece)
            print(json.dumps(
                {"item": item.id, "prefixes": {
                    variant: {"prefill_ms": value["prefill_ms"],
                              "cue_decode_s": value["cue_decode_s"],
                              "top_token": [(token["piece"], round(token["p_full"], 4))
                                            for token in value["cues"][args.cues[0]]
                                            ["top_tokens"][:2]],
                              "coverage": {cue: {label: round(
                                  value["cues"][cue]["labels"][label]["coverage"], 8)
                                  for label in args.label_variants} for cue in args.cues},
                              "ranked": {key: {"got": row["got"], "correct": row["correct"],
                                               "confidence": round(row["confidence"], 4)}
                                         for key, row in value["ranked"].items()}}
                    for variant, value in piece["prefixes"].items()},
                 "cross_check": ({"coverage": piece["cross_check"]["coverage"],
                                  "got": piece["cross_check"]["got"]}
                                 if piece.get("cross_check") else None)},
                ensure_ascii=False), flush=True)
    finally:
        handle.close()
    record["wall_s"] = round(time.perf_counter() - started, 1)
    record["device_log_tail"] = "\n".join(log[-6:])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"run: {out}", flush=True)
    markdown = render_report(record)
    if args.report:
        pathlib.Path(args.report).write_text(markdown, encoding="utf-8")
        print(f"report: {args.report}")
    if args.report_dir:
        write_policy_reports(record, pathlib.Path(args.report_dir))
    print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the live sweep (loads the model once)")
    run.add_argument("--model", required=True)
    run.add_argument("--runtime", default=None, help="GGUFONE_RUNTIME_DIR; default: the env")
    run.add_argument("--devset", default=None, help="devset jsonl (default: the committed set)")
    run.add_argument("--ids", nargs="+", default=None, help="exact dev item ids")
    run.add_argument("--per-type", dest="per_type", type=int, default=None,
                     help="at most N items per question type")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--cues", nargs="+", default=list(labels.CUE_VARIANTS))
    run.add_argument("--label-variants", dest="label_variants", nargs="+",
                     default=list(labels.LABEL_VARIANTS))
    run.add_argument("--rank", action="append", default=None,
                     help="cue=label policies to score fully (repeatable)")
    run.add_argument("--extra-prefix", dest="extra_prefix", default=None,
                     choices=[KEPT_PREFIX],
                     help="a second prefix variant for the first N items (the strip control)")
    run.add_argument("--only-prefix", dest="only_prefix", default=None,
                     choices=[KEPT_PREFIX],
                     help="measure only this prefix variant (the re-measure path: the shipped "
                          "side is E3's published report, already on disk)")
    run.add_argument("--extra-prefix-items", dest="extra_prefix_items", type=int, default=2)
    run.add_argument("--cross-check", dest="cross_check", type=int, default=0,
                     help="cross-check the first N items against DecisionEngine (one extra "
                          "prefill each)")
    run.add_argument("--top-tokens", dest="top_tokens", type=int, default=DEFAULT_TOP_TOKENS)
    run.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    run.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=7)
    run.add_argument("--length-norm", dest="length_norm", type=float, default=1.0)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--max-sequences", dest="max_sequences", type=int, default=MAX_SEQUENCES)
    run.add_argument("--states-home", dest="states_home", default=DEFAULT_STATES_HOME)
    run.add_argument("--vk-driver-files", dest="vk_driver_files",
                     default=os.environ.get("GGUFONE_VK_DRIVER_FILES", DEFAULT_VK_DRIVER_FILES))
    run.add_argument("--out", required=True)
    run.add_argument("--report", default=None)
    run.add_argument("--report-dir", dest="report_dir", default=None)
    report = sub.add_parser("report", help="offline: markdown tables from a stored run")
    report.add_argument("--run", required=True)
    report.add_argument("--out", default=None)
    report.add_argument("--report-dir", dest="report_dir", default=None)
    args = parser.parse_args(argv)
    if args.command == "report":
        record = json.loads(pathlib.Path(args.run).read_text(encoding="utf-8"))
        markdown = render_report(record)
        if args.out:
            pathlib.Path(args.out).write_text(markdown, encoding="utf-8")
        if args.report_dir:
            write_policy_reports(record, pathlib.Path(args.report_dir))
        print(markdown)
        return 0
    return live_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
