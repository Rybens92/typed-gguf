"""The cue verdict: what the model wants to emit at the decision position (cards t_6c119626,
t_635124bf).

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
SPEC 2.5. `closer` names the token that refused the cue and is `None` on every other row — the
block's E3c invariant, unchanged.

Two sources name the class, and one rule decides:

* the **catalogue** (`TURN_CLOSERS`): the documented turn-closer strings of the families ggufone
  can be pointed at, matched only when the **session's own tokenizer** encodes that exact string as
  **one** token. Nothing is matched by a hard-coded token id, so a family whose vocabulary carries
  no `<|im_end|>` can never fire on it, and a closer the vocabulary splits into several tokens
  (some GGUFs carry the literal text instead of a special token) is not reported as a refusal —
  the probe measures its first-token mass instead;
* the **vocabulary's own attribute table** (`special`, card `t_635124bf`): every token the model
  marks `CONTROL` or `USER_DEFINED` is a turn-shaping/special token, never content. This is the
  fallback that catches a family special the catalogue has never heard of — Tiel-Coder's serving
  shape puts 0.68…0.99 of the cue row's mass on `</think>` (token 248069), which the catalogue
  cannot name, so the row read `low_mass` with no reason at all. llama.cpp classifies the token;
  `ModelHandle.special_tokens()` hands the table over; ggufone only reports it. A vocabulary that
  carries no text for its token is named by its id (`<special 248069>`) — never by nothing.

**Dominating** is the rule, and it is one number: the token is the row's argmax (`readout.
argmax_first`, the frozen tie-break, so this verdict can never disagree with `choice`/`score`/
`noul`) **and** holds at least `REFUSAL_FLOOR` of the row's mass. The floor is the engine's own
coverage floor (`OPTION_DEFAULTS["coverage_floor"]`, 0.10) — the same threshold `readout.
reliability` calls a row's mass adequate with, so "dominating" and "measured" cannot drift apart —
and the engine passes the request's effective floor in. A control token that wins a flat row by a
hair is a `low_mass` row, not a refusal: the fix a refusal points at (change the prompt shape) is
not what such a row is telling the caller.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ggufone.engine import readout
from ggufone.schema import OPTION_DEFAULTS

#: The turn-closer strings a cue row may put its mass on — `<|im_end|>`, `</s>`, `<|endoftext|>`,
#: `eos` and the family variants of the same idea (card t_6c119626 §2). Not a preference list:
#: each entry is a documented end-of-turn token of a family ggufone can be pointed at. It is a
#: *name* catalogue: the vocabulary's attribute table (`special`) is what decides the class.
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

#: The mass a turn-closing/special token must hold at the cue to count as **dominating**: the
#: engine's own coverage floor (`OPTION_DEFAULTS["coverage_floor"]`, 0.10). One threshold, not
#: two: a row the engine calls `measured` is a row whose candidate mass is at least this, so the
#: verdict and the reliability word are read off the same number (card t_635124bf).
REFUSAL_FLOOR = float(OPTION_DEFAULTS["coverage_floor"])

#: What a refused cue means and where the fix is documented. Since E3d the fix has a name a
#: reader can run: `--cue two_step` reads the label one row past the cue (see the shape catalogue
#: below and `docs/TEMPLATES.md` §5) — and a refused cue still refuses there, which is why the
#: pointer leads with the measured table rather than the flag.
CUE_REFUSED_HINT = (
    "docs/TEMPLATES.md §4 (the label policy, measured) and §5 (the cue shapes) — the cue decides "
    "where the readout sits; this prompt shape ends at the start of the assistant turn, so the "
    "model can close it (or emit another turn-shaping special token) instead of answering. "
    "`--cue two_step` moves the readout one token in without changing these bytes"
)

#: The rule that chooses the token `two_step` decodes: the row's argmax among tokens that are not
#: turn-closers ("if the model cannot close the turn, what does it start to say?"). A row whose
#: argmax *is* a closer is a refusal, and a refusal never advances (card t_d90404ac, `CUE_SHAPES`
#: in `ggufone.schema`).
ADVANCE_RULE = "content"


def single_token_closers(tokenize: Callable[[str], list[int]],
                         closers: Sequence[str] = TURN_CLOSERS,
                         special: Iterable[tuple[int, str]] | None = None) -> dict[int, str]:
    """`{token id: name}` for the catalogue and the vocabulary's own special tokens.

    The vocabulary decides which catalogue entries count: a closer this tokenizer splits into
    several tokens is not in the map, and a caller reporting it would be reporting a prefix of a
    token as if it were the token. (Two catalogue entries may collapse onto one id in some
    vocabularies — `</s>` and `<|endoftext|>` share an id in the GPT-2 lineage — so the first
    catalogue entry wins and the report names a string the vocabulary really carries.)

    `special` is the session's own `[(token id, the vocabulary's text)]` table
    (`ModelHandle.special_tokens()`, card t_635124bf): the tokens the model marks CONTROL or
    USER_DEFINED. They are the class a string catalogue cannot enumerate, and the vocabulary names
    them. The catalogue wins a collision — `<|im_end|>` *is* a CONTROL token in most vocabularies,
    and the documented name is the better report — and a token the vocabulary does not name is
    named by its id, because "some special token" is not a report a caller can act on.
    """
    mapping: dict[int, str] = {}
    for text in closers:
        tokens = tokenize(text)
        if len(tokens) == 1:
            mapping.setdefault(int(tokens[0]), text)
    for token, text in special or ():
        mapping.setdefault(int(token), str(text) or f"<special {int(token)}>")
    return mapping


def closer_map(session: Any) -> dict[int, str]:
    """The `{token id: name}` map of ONE session: the catalogue plus its vocabulary's specials.

    Duck-typed on the engine seam (`session.tokenize`, `session.special_tokens`): a session that
    cannot enumerate its vocabulary — a test double, or a bundle without llama.cpp's attribute API
    — gets the catalogue alone, which is the pre-fix behaviour, byte for byte.
    """
    table = getattr(session, "special_tokens", None)
    return single_token_closers(session.tokenize,
                                special=table() if callable(table) else ())


def cue_verdict(row: Sequence[float], scale: float, closers: Mapping[int, str], *,
                floor: float = REFUSAL_FLOOR) -> dict[str, Any]:
    """The verdict for one decision row — pure arithmetic over the row and the closer map.

    `row` is the full-vocabulary logit row at the cue and `scale` its logsumexp (the engine holds
    both); `closers` comes from `closer_map()` for the session's own vocabulary and `floor` is the
    effective coverage floor of the request (`REFUSAL_FLOOR` when the caller has none). The argmax
    tie-break is `readout.argmax_first` — the frozen lowest-index rule, so this verdict can never
    disagree with `choice`/`score`/`noul` about which token the row put first.
    """
    top = readout.argmax_first(row)
    mass = readout.coverage_from_scale(row, (top,), scale)
    closer = closers.get(top)
    refused = closer is not None and mass >= floor
    verdict: dict[str, Any] = {
        "refused": refused,
        "token": int(top),
        "closer": closer if refused else None,
        "mass": mass,
    }
    if refused:
        verdict["hint"] = CUE_REFUSED_HINT
    return verdict
