"""The cue verdict: what the model wants to emit at the decision position (card t_6c119626).

`coverage` answers "how much mass does the candidate label carry at the cue". It cannot answer
"what does the model want to say instead" — and E3b (`t_6952f0dd`) showed that *is* the question
for the `qwen35moe` family on the shipped prompt shape: `<|im_end|>` holds 0.9976…1.00000 of the
row's mass on all 6 probe items and all 15 (cue × label) combinations, so every answer is
`low_mass` no matter how the label is rendered. A caller reading only `low_mass` sees a number to
tune; the honest verdict is a **prompt-shape refusal** — the model closes the assistant turn
before it answers.

`cue_verdict()` reads the row the engine already decoded (no extra forward pass) and reports

    {"refused": bool, "token": int, "closer": str | None, "mass": float[, "hint": str]}

`mass` is the top token's full-vocabulary mass (`readout.coverage_from_scale` — the same function
the engine's `coverage` uses). `hint` is present only on a refusal: it is the actionable pointer
that the flat `warnings` list cannot carry, because `warnings` is a list of codes frozen by
SPEC 2.5.

Matching rule: a closer counts only when the **session's own tokenizer** encodes that exact string
as **one** token and that token is the row's argmax. Nothing is matched by a hard-coded token id,
so a family whose vocabulary carries no `<|im_end|>` can never fire this warning, and a closer the
vocabulary splits into several tokens (some GGUFs carry the literal text instead of a special
token) is not reported as a refusal — the probe measures its first-token mass instead.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ggufone.engine import readout

#: The turn-closer strings a cue row may put its mass on — `<|im_end|>`, `</s>`, `<|endoftext|>`,
#: `eos` and the family variants of the same idea (card t_6c119626 §2). Not a preference list:
#: each entry is a documented end-of-turn token of a family ggufone can be pointed at.
TURN_CLOSERS: tuple[str, ...] = (
    "<|im_end|>",        # ChatML — Qwen and the qwen35moe family this card measured
    "<|eot_id|>",        # Llama 3/3.1
    "<|eom_id|>",        # Llama 4
    "<|end_of_text|>",   # Llama 3 base
    "<|endoftext|>",     # GPT-2 lineage
    "</s>",              # Llama 1/2, Mistral, most sentencepiece EOS
    "<end_of_turn>",     # Gemma 2/3
    "<|end|>",           # Phi-3/Phi-4
    "eos",               # the literal token some vocabularies carry
)

#: What a refused cue means and where the fix is documented. `--cue` does not exist as a CLI flag
#: yet, so the pointer names the section instead of a flag a reader would search for in vain.
CUE_REFUSED_HINT = (
    "docs/TEMPLATES.md §4 (the label policy, measured) — the cue decides where the readout sits; "
    "this prompt shape ends at the start of the assistant turn, so the model can close it instead "
    "of answering"
)


def single_token_closers(tokenize: Callable[[str], list[int]],
                         closers: Sequence[str] = TURN_CLOSERS) -> dict[int, str]:
    """`{token id: closer}` for every closer this vocabulary encodes as exactly one token.

    The vocabulary decides: a closer the model's tokenizer splits into several tokens is not in
    the map, and a caller reporting it would be reporting a prefix of a token as if it were the
    token. (Two catalogue entries may collapse onto one id in some vocabularies — `</s>` and
    `<|endoftext|>` share an id in the GPT-2 lineage — so the first catalogue entry wins and the
    report names a string the vocabulary really carries.)
    """
    mapping: dict[int, str] = {}
    for text in closers:
        tokens = tokenize(text)
        if len(tokens) == 1:
            mapping.setdefault(int(tokens[0]), text)
    return mapping


def cue_verdict(row: Sequence[float], scale: float, closers: Mapping[int, str]) -> dict[str, Any]:
    """The verdict for one decision row — pure arithmetic over the row and a closer map.

    `row` is the full-vocabulary logit row at the cue and `scale` its logsumexp (the engine holds
    both); `closers` comes from `single_token_closers()` for the session's own vocabulary. The
    argmax tie-break is `readout.argmax_first` — the frozen lowest-index rule, so this verdict can
    never disagree with `choice`/`score`/`noul` about which token the row put first.
    """
    top = readout.argmax_first(row)
    closer = closers.get(top)
    verdict: dict[str, Any] = {
        "refused": closer is not None,
        "token": int(top),
        "closer": closer,
        "mass": readout.coverage_from_scale(row, (top,), scale),
    }
    if closer is not None:
        verdict["hint"] = CUE_REFUSED_HINT
    return verdict
