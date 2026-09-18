"""Label rendering policies for the readout (E3b — card t_6952f0dd; `docs/TEMPLATES.md` §4).

A label policy answers one question: which *string* does the engine score for each candidate?
`prompt.build_question` renders the candidate labels as bare wire keys (option names, the level
numbers `"0".."K-1"`, `yes`/`no`), and `coverage` — the engine's honesty check — is the
full-vocabulary mass of that string's **first token** at the cue row (SPEC 2.3 step 6,
`readout.coverage_from_scale`). So a family can decide exactly as asked and still report
`low_mass`, simply because its answer opens with a newline or with a space-prefixed word instead
of the bare key. E3 measured precisely that: all 20 Occamy answers below the 0.10 floor against
3/20 for the 4B on the same items.

This module is the measurement side of the per-family label policy: the variants a probe may ask
for, as deterministic transformations of the shipped rendering, plus the two aggregates the probe
reports (`label_coverage`, `shared_first_tokens`).

**Nothing here changes the engine's defaults.** `bare` *is* the shipped rendering — the gate for
that is byte identity with `prompt.build_question(...).texts`, because every published number was
measured with it. A variant only ever exists inside a probe run, and a policy that is adopted
later is a change to `prompt.py` with its own card.

The engine's collision rule is not re-implemented here: `decide._check_collisions` compares the
*token sequence* of every candidate, and a variant whose labels only share a first token is
legal — it just means the readout cannot separate them at step one (`shared_first_tokens` reports
that instead of hiding it).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ggufone.bench import harness
from ggufone.engine import prompt, readout
from ggufone.schema import OPTION_DEFAULTS

#: how the candidate label is rendered before it is tokenized and scored
LABEL_VARIANTS = ("bare", "space", "caps", "newline", "long")
#: what the prompt says at the decision position (the line the readout reads at)
CUE_VARIANTS = ("shipped", "blank", "explicit")
#: the engine's own floor (`OPTION_DEFAULTS["coverage_floor"]`, SPEC 2.5): below it -> `low_mass`
MASS_FLOOR = float(OPTION_DEFAULTS["coverage_floor"])
_SEPARATOR = ": "


class LabelsError(harness.BenchError):
    """A variant problem the caller can fix (exit code 2, like every other bench error).

    The message carries its own code (the repository's error style: `E_TEMPLATE_UNRESOLVED: …`),
    because a probe prints these lines next to the numbers they invalidate.
    """

    def __init__(self, message: str, *, code: str = "E_LABEL_VARIANT") -> None:
        super().__init__(message, code=code)
        self.message = f"{code}: {message}"

    def __str__(self) -> str:                    # pragma: no cover - trivial
        return self.message


def _check_variant(value: str, known: Sequence[str], *, code: str) -> str:
    if value not in known:
        raise LabelsError(f"unknown variant {value!r}; known: {', '.join(known)}", code=code)
    return value


def label_texts(question_type: str, options: Sequence[str],
                descriptions: Sequence[str | None], variant: str = "bare") -> tuple[str, ...]:
    """The label string scored for each candidate under `variant` (wire order preserved).

    | variant | rendering | what it measures |
    |---|---|---|
    | `bare` | the wire key itself | the shipped policy (E3's `low_mass` 20/20) |
    | `space` | `" " + bare` | BPE vocabularies usually carry the space as part of the word token |
    | `caps` | first character upper-cased | a sentence-initial answer rather than a bare key |
    | `newline` | `"\\n" + bare` | the two-step readout: the model's own newline, then the label |
    | `long` | the description (`name: description` for `choice`) | the label the prompt *shows* |
    """
    _check_variant(variant, LABEL_VARIANTS, code="E_LABEL_VARIANT")
    descs = [str(item) if item else None for item in list(descriptions)[:len(options)]]
    descs += [None] * (len(options) - len(descs))
    texts: list[str] = []
    for option, description in zip(options, descs, strict=True):
        base = str(option)
        if variant == "bare":
            texts.append(base)
        elif variant == "space":
            texts.append(" " + base)
        elif variant == "caps":
            texts.append(base[:1].upper() + base[1:])
        elif variant == "newline":
            texts.append("\n" + base)
        else:                                   # long
            if not description:
                texts.append(base)
            elif question_type == "choice":
                texts.append(f"{base}{_SEPARATOR}{description}")
            else:
                texts.append(str(description))
    return tuple(texts)


def cue_line(question_type: str, options: Sequence[str], variant: str = "shipped") -> str:
    """The last line of the question suffix under `variant` (the line before the answer).

    `shipped` and `blank` are the shipped cue — `blank` differs by the *empty line* the suffix
    gets after it, which is the knob the 4B's own probe measured to triple the label mass
    (`docs/TEMPLATES.md` §4). `explicit` names the labels in the cue itself, so the format the
    readout wants is stated rather than implied.
    """
    _check_variant(variant, CUE_VARIANTS, code="E_LABEL_CUE")
    if variant != "explicit":
        return prompt.CANDIDATE_CUE[question_type]
    listing = ", ".join(str(option) for option in options)
    noun = {"choice": "candidate names", "score": "level numbers",
            "noul": "words"}.get(question_type, "candidate labels")
    return f"Answer with exactly one of these {noun}: {listing}"


def suffix_with_cue(suffix: str, question_type: str, options: Sequence[str],
                    variant: str = "shipped") -> str:
    """`suffix` with its cue line replaced by `variant`'s (`shipped` is byte-identical).

    Only the cue line moves: every byte before it — the instructions, the rendered criteria — is
    the one `prompt.build_question` produced, so a variant can never smuggle in a different
    question. A suffix that does not end at the shipped cue is rejected instead of silently
    rewritten (`E_LABEL_CUE`).
    """
    _check_variant(variant, CUE_VARIANTS, code="E_LABEL_CUE")
    shipped = prompt.CANDIDATE_CUE[question_type] + "\n"
    if not suffix.endswith(shipped):
        raise LabelsError(
            f"the rendered suffix does not end at the shipped {question_type} cue "
            f"({shipped!r}); the last 40 characters are {suffix[-40:]!r}", code="E_LABEL_CUE")
    head = suffix[: -len(shipped)]
    return head + cue_line(question_type, options, variant) + ("\n\n" if variant == "blank"
                                                               else "\n")


def label_coverage(texts: Sequence[str], tokenize: Callable[[str], list[int]],
                   row: Sequence[float], scale: float) -> float:
    """The engine's own coverage for these labels against one decision row (never re-derived).

    `row` is the full-vocabulary logit row at the cue and `scale` its logsumexp; the arithmetic
    is `readout.coverage_from_scale`, the same function the engine calls — a probe that
    recomputed it could disagree with the engine it is measuring. A label that tokenizes to
    nothing is an error here too (the engine raises `E_CHOICE_CRITERIA`/`E_SCORE_LEVELS`/
    `E_NOUL_CRITERIA` for that), because a zero-token candidate has no first token to read.
    """
    first_tokens: list[int] = []
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            raise LabelsError(
                f"candidate label {text!r} tokenizes to zero tokens", code="E_LABEL_EMPTY")
        first_tokens.append(int(tokens[0]))
    return readout.coverage_from_scale(row, first_tokens, scale)


def shared_first_tokens(texts: Sequence[str],
                        tokenize: Callable[[str], list[int]]) -> list[tuple[str, ...]]:
    """Groups of labels whose *first* token is identical (the readout cannot separate them there).

    Not an error — the engine scores whole sequences — but a variant that puts every candidate
    on one first token hands the decision to the later steps, and that is worth reporting next to
    the coverage number it inflates.
    """
    groups: dict[tuple[int, ...], list[str]] = {}
    for text in texts:
        tokens = tokenize(text)
        groups.setdefault(tuple(tokens[:1]), []).append(text)
    return [tuple(group) for _, group in sorted(groups.items()) if len(group) > 1]


def reliability_of(coverage_value: float, *, floor: float = MASS_FLOOR) -> str:
    """`ok | low_mass` from coverage alone — the engine's verdict when no confidence floor is set.

    `readout.reliability` is called with the request's own floor in the engine; the bench and the
    CLI never set `confidence_floor`, so coverage is the only input there (`low_confidence` is an
    API-only mode). Kept as one call so the probe cannot drift from the engine's rule.
    """
    return readout.reliability(coverage_value, coverage_floor=floor)
