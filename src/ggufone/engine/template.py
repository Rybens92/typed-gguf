"""Template resolution chain + thinking suppression (SPEC 5 A-E1c-1..3, SPEC 2.3.1).

Milestone: E1c.

E1b assembled prompts with a fixed, model-agnostic framing. E1c renders through the model's
**own chat template** instead, because the label a family expects at the decision position
(`<|Bot|></think>`, `<|im_start|>assistant\\n`) is part of the template, not of our framing.

The chain (A-E1c-1), in order — `CHAIN` below is the machine-readable copy:

  1. ``gguf-renderer``  the GGUF's `tokenizer.chat_template`, rendered by the *internal*
     renderer (`render()` below, a documented Jinja subset). Most faithful: it is the template
     the model was trained against and the only step that can honour `enable_thinking`-style
     keyword arguments.
  2. ``builtin``        `llama_chat_apply_template` (the runtime's built-in family table), used
     when the internal renderer rejects the template; emits `W_TEMPLATE_FALLBACK`.
  3. ``user``           an explicit override: `--template plain | <builtin name> | <path> |
     <template text>`, the documented fix for a model we cannot render automatically.
  4. ``error``          `E_TEMPLATE_UNRESOLVED`, whose message names the unsupported construct
     and the fix.

An *explicit* override (step 3 supplied by the operator) short-circuits steps 1-2 — the operator
asked for that template by name, so it wins. Automatic resolution never consults step 3 before
steps 1-2.

Thinking suppression (A-E1c-2). `no_open_think()` is the predicate the tests assert: after the
last rendered byte the model is *not* inside a thinking block. Hard-switch families (Spark,
Qwen3.5) close the block through `enable_thinking=false`; a template that ignores the switch and
ends with an open `<think>` gets the trailing opener stripped (recorded as `thinking="stripped"`);
soft-switch families get their documented marker appended to the user turn (`thinking="marker"`).
Every path is verified on the rendered bytes, never assumed.

The readout is unaffected by anything the model would emit *after* the cue (A-E1c-3): the engine
decodes no generated token — `decide.py` reads candidate rows at the cue position only, and the
candidate sequences are the ones the request named.
"""
from __future__ import annotations

import ctypes as C  # noqa: N812
import dataclasses
import json
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ggufone.errors import RuntimeError_
from ggufone.runtime import ctypes_binding

# ---------------------------------------------------------------------- the chain
CHAIN: tuple[str, ...] = ("gguf-renderer", "builtin", "user", "error")
CHAIN_LABELS: tuple[str, ...] = (
    "1. tokenizer.chat_template rendered by the internal renderer (supported subset)",
    "2. llama_chat_apply_template built-ins (W_TEMPLATE_FALLBACK)",
    "3. explicit user override (--template plain|<name>|<path>)",
    "4. E_TEMPLATE_UNRESOLVED (the message names the construct and the fix)",
)
PLAIN = "plain"
TEMPLATE_KEY = "tokenizer.chat_template"

#: `llama_chat_builtin_templates()` at the pinned b11026 build. The live list is read from the
#: runtime when one is loaded (`runtime_builtin_names`); this tuple is the offline mirror used to
#: validate `--template <name>` and by docs/TEMPLATES.md.
BUILTIN_TEMPLATES: tuple[str, ...] = (
    "chatml", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip", "mistral-v1",
    "mistral-v3", "mistral-v3-tekken", "mistral-v7", "mistral-v7-tekken", "phi3", "phi4",
    "falcon3", "zephyr", "monarch", "gemma", "orion", "openchat", "vicuna", "vicuna-orca",
    "deepseek", "deepseek2", "deepseek3", "deepseek-ocr", "command-r", "llama3", "chatglm3",
    "chatglm4", "glmedge", "minicpm", "exaone3", "exaone4", "exaone-moe", "rwkv-world",
    "granite", "granite-4.0", "granite-4.1", "gigachat", "megrez", "yandex", "bailing",
    "bailing-think", "bailing2", "llama4", "smolvlm", "hunyuan-moe", "gpt-oss", "hunyuan-dense",
    "hunyuan-vl", "kimi-k2", "seed_oss", "grok-2", "pangu-embedded", "solar-open",
)

THINK_OPENERS: tuple[str, ...] = ("<think>", "<|think|>")
THINK_CLOSERS: tuple[str, ...] = ("</think>", "<|/think|>")
_THINK_RE = re.compile("|".join(re.escape(marker) for marker in (*THINK_OPENERS, *THINK_CLOSERS)))
_EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\s*\Z")


class TemplateError(Exception):
    """Rendering failed for a reason other than 'unsupported syntax'."""


class UnsupportedTemplate(TemplateError):
    """The template uses a construct outside the documented subset (step 1 -> step 2)."""

    def __init__(self, construct: str, line: int, detail: str = "") -> None:
        self.construct = construct
        self.line = line
        super().__init__(f"unsupported template construct {construct!r} at line {line}"
                         + (f" ({detail})" if detail else ""))


class TemplateUnresolvedError(RuntimeError_):
    """Step 4: nothing could render the prompt (SPEC 2.5 `E_TEMPLATE_UNRESOLVED`)."""

    code = "E_TEMPLATE_UNRESOLVED"

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ------------------------------------------------------------------ family policy
@dataclass(frozen=True, slots=True)
class FamilyPolicy:
    """One model family's rendering facts — the table `docs/TEMPLATES.md` documents."""

    arch: str
    template_key: str
    thinking: str               # "hard" | "soft" | "none"
    soft_marker: str | None     # appended to the user turn for soft-switch families
    strip_empty_think_block: bool
    label_policy: str           # how the candidate label is asked for and scored
    notes: str
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"arch": self.arch, "template_key": self.template_key, "thinking": self.thinking,
                "soft_marker": self.soft_marker,
                "strip_empty_think_block": self.strip_empty_think_block,
                "label_policy": self.label_policy, "notes": self.notes,
                "aliases": list(self.aliases)}


FAMILIES: dict[str, FamilyPolicy] = {
    "spark2_5": FamilyPolicy(
        arch="spark2_5", template_key=TEMPLATE_KEY, thinking="hard", soft_marker=None,
        strip_empty_think_block=False,
        label_policy="option name / level number as plain text right after the assistant header",
        notes="Hybrid reasoning family. With enable_thinking=false the GGUF template emits the "
              "assistant header followed by a bare `</think>`, i.e. a *closed* thinking block, so "
              "the prompt never opens one. Special tokens (<｜start▁of▁sentence｜>, <|Bot|>) are "
              "single tokens in this vocabulary — verified in docs/TEMPLATES.md.",
        aliases=("spark", "spark2.5", "spark-x2.5")),
    "qwen35": FamilyPolicy(
        arch="qwen35", template_key=TEMPLATE_KEY, thinking="hard", soft_marker=None,
        strip_empty_think_block=True,
        label_policy="option name / level number as plain text after the assistant header",
        notes="Hybrid SSM+attention (recurrent state) family. enable_thinking=false renders an "
              "*empty closed* block (`<think>\\n\\n</think>\\n\\n`); ggufone strips that block "
              "from the generation prompt so no think-opener is present at all.",
        aliases=("qwen3.5", "qwen3_5", "qwen35-dense")),
    "qwen35moe": FamilyPolicy(
        arch="qwen35moe", template_key=TEMPLATE_KEY, thinking="hard", soft_marker=None,
        strip_empty_think_block=True,
        label_policy="option name / level number as plain text after the assistant header",
        notes="MoE sibling of qwen35: same template shape and the same empty-block marker, so "
              "the same suppression path. The expert layout only changes the fit plan.",
        aliases=("qwen3.5-moe", "qwen35-moe", "qwen3moe")),
    "k2-horizon": FamilyPolicy(
        arch="k2-horizon", template_key=TEMPLATE_KEY, thinking="soft", soft_marker="/no_think",
        strip_empty_think_block=False,
        label_policy="option name / level number after the `<|im_assistant|>assistant"
                     "<|im_middle|>` header",
        notes="Kimi-K2 lineage (`<|im_system|>...<|im_middle|>` roles, no enable_thinking in the "
              "template). Thinking in this family is controlled by the *serving stack* "
              "(`thinking.type` on Moonshot's API), not by the prompt: ggufone appends the "
              "documented `/no_think` soft marker AND keeps the trailing-opener strip, but the "
              "marker is advisory here — the guarantee is the `no_open_think` predicate. The "
              "llama.cpp bundle also ships a `kimi-k2` built-in, so chain step 2 covers variants "
              "our renderer rejects.",
        aliases=("kimi-k2", "kimi_k2", "kimi-k2-horizon", "k2")),
}
FAMILY_ALIASES: dict[str, str] = {alias: arch for arch, policy in FAMILIES.items()
                                  for alias in (policy.arch, *policy.aliases)}


def policy_for(arch: str | None) -> FamilyPolicy | None:
    """The policy row for an architecture string (alias-tolerant, `None` when unknown)."""
    if not arch:
        return None
    key = arch.strip().lower()
    return FAMILIES.get(FAMILY_ALIASES.get(key, key))


def detect_family(arch: str | None, template: str | None) -> str | None:
    """The architecture when we know it, else the family the template *shape* suggests."""
    policy = policy_for(arch)
    if policy is not None:
        return policy.arch
    if not template:
        return None
    if "<|im_start|>" in template:
        return "qwen-unknown" if "enable_thinking" in template else "chatml-unknown"
    if "<｜start▁of▁sentence｜>" in template or "<|Bot|>" in template:
        return "spark-unknown"
    return None


# ------------------------------------------------------------------------- scopes
class _Undefined:
    """Jinja's `Undefined`: falsy, prints empty, `is defined` is False."""

    __slots__ = ("name",)

    def __init__(self, name: str = "") -> None:
        self.name = name

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "Undefined"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Undefined)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, _Undefined)

    def __iter__(self) -> Any:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        raise TemplateError(f"{self.name or 'value'} is undefined")


UNDEFINED = _Undefined()


class _Namespace:
    """`{% set ns = namespace(x='') %}` — Jinja's mutable binding for loop scopes."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def __repr__(self) -> str:                                  # pragma: no cover
        return f"namespace({self.__dict__})"


class Scope:
    """A lexical scope chain: `for`/`macro` push a child, `if` does not (Jinja semantics)."""

    __slots__ = ("values", "parent")

    def __init__(self, values: Mapping[str, Any] | None = None, parent: Scope | None = None
                 ) -> None:
        self.values: dict[str, Any] = dict(values or {})
        self.parent = parent

    def get(self, name: str) -> Any:
        scope: Scope | None = self
        while scope is not None:
            if name in scope.values:
                return scope.values[name]
            scope = scope.parent
        return UNDEFINED

    def set(self, name: str, value: Any) -> None:
        self.values[name] = value

    def child(self) -> Scope:
        return Scope(parent=self)


class LoopState:
    """`loop.first` / `loop.index` / `loop.previtem` inside a `for` body."""

    __slots__ = ("index0", "length", "items")

    def __init__(self, items: Sequence[Any]) -> None:
        self.items = list(items)
        self.length = len(self.items)
        self.index0 = 0

    def attr(self, name: str) -> Any:
        index = self.index0
        if name == "index":
            return index + 1
        if name == "index0":
            return index
        if name == "revindex":
            return self.length - index
        if name == "revindex0":
            return self.length - index - 1
        if name == "first":
            return index == 0
        if name == "last":
            return index == self.length - 1
        if name == "length":
            return self.length
        if name == "previtem":
            return self.items[index - 1] if index > 0 else UNDEFINED
        if name == "nextitem":
            return self.items[index + 1] if index + 1 < self.length else UNDEFINED
        if name == "cycle":
            return lambda *args: args[index % len(args)] if args else ""
        return UNDEFINED


# ------------------------------------------------------------ expression lexer
OPERATORS = ("==", "!=", "<=", ">=", "//", "**", "~", "+", "-", "*", "/", "%", "<", ">",
             "|", ".", ",", ":", "(", ")", "[", "]", "{", "}", "=")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
STRING_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"")


@dataclass(frozen=True, slots=True)
class Token:
    kind: str          # "name" | "number" | "string" | "op" | "eof"
    value: Any
    pos: int


def _unescape(raw: str) -> str:
    escapes = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"'}
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            out.append(escapes.get(raw[index + 1], raw[index + 1]))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def tokenize_expression(text: str, line: int) -> list[Token]:
    out: list[Token] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
            continue
        match = STRING_RE.match(text, pos)
        if match:
            out.append(Token("string", _unescape(match.group(0)[1:-1]), pos))
            pos = match.end()
            continue
        match = NUMBER_RE.match(text, pos)
        if match:
            raw = match.group(0)
            out.append(Token("number", float(raw) if "." in raw else int(raw), pos))
            pos = match.end()
            continue
        match = NAME_RE.match(text, pos)
        if match:
            out.append(Token("name", match.group(0), pos))
            pos = match.end()
            continue
        for operator in OPERATORS:
            if text.startswith(operator, pos):
                out.append(Token("op", operator, pos))
                pos += len(operator)
                break
        else:
            raise UnsupportedTemplate(f"expression character {char!r}", line, text)
    out.append(Token("eof", None, pos))
    return out


# --------------------------------------------------------------- expression AST
@dataclass(frozen=True, slots=True)
class Literal:
    value: Any


@dataclass(frozen=True, slots=True)
class Name:
    name: str
    line: int


@dataclass(frozen=True, slots=True)
class Attr:
    base: Any
    name: str
    line: int


@dataclass(frozen=True, slots=True)
class Index:
    base: Any
    item: Any
    line: int


@dataclass(frozen=True, slots=True)
class SliceExpr:
    base: Any
    start: Any
    stop: Any
    step: Any
    line: int


@dataclass(frozen=True, slots=True)
class Call:
    func: Any
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]
    line: int


@dataclass(frozen=True, slots=True)
class Unary:
    op: str
    operand: Any
    line: int


@dataclass(frozen=True, slots=True)
class BoolOp:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Binary:
    op: str
    left: Any
    right: Any
    line: int


@dataclass(frozen=True, slots=True)
class Compare:
    op: str
    left: Any
    right: Any
    line: int


@dataclass(frozen=True, slots=True)
class Test:
    name: str
    negated: bool
    operand: Any
    line: int


@dataclass(frozen=True, slots=True)
class Filter:
    name: str
    operand: Any
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]
    line: int


@dataclass(frozen=True, slots=True)
class Ternary:
    condition: Any
    then: Any
    otherwise: Any
    line: int


@dataclass(frozen=True, slots=True)
class ListLiteral:
    items: tuple[Any, ...]


#: Jinja globals we deliberately do NOT implement: naming one is an unsupported construct, not a
#: silently undefined name (the caller gets a message that names the construct — A-E1c-1 step 4).
UNSUPPORTED_GLOBALS = ("range", "dict", "lipsum", "cycler", "joiner", "urlencode", "csrf_token")

_RESERVED_IN_PRIMARY = ("and", "or", "not", "in", "is", "if", "else")
_KEYWORD_LITERALS = {"true": True, "True": True, "false": False, "False": False,
                     "none": None, "None": None}


class ExprParser:
    """Recursive descent over the Jinja *supported subset* of the expression grammar."""

    def __init__(self, tokens: list[Token], line: int, source: str) -> None:
        self.tokens = tokens
        self.line = line
        self.source = source
        self.index = 0

    # ---- plumbing
    def peek(self) -> Token:
        return self.tokens[self.index]

    def next(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def at_op(self, *ops: str) -> bool:
        token = self.peek()
        return token.kind == "op" and token.value in ops

    def at_name(self, *names: str) -> bool:
        token = self.peek()
        return token.kind == "name" and token.value in names

    def expect_op(self, op: str) -> Token:
        if not self.at_op(op):
            raise UnsupportedTemplate(f"expected {op!r}, got {self.peek().value!r}", self.line,
                                      self.source)
        return self.next()

    # ---- grammar
    def parse(self) -> Any:
        node = self.parse_or()
        if self.peek().kind != "eof":
            raise UnsupportedTemplate(f"trailing token {self.peek().value!r}", self.line,
                                      self.source)
        return node

    def parse_or(self) -> Any:
        node = self.parse_and()
        while self.at_name("or"):
            self.next()
            node = BoolOp("or", node, self.parse_and())
        if self.at_name("if"):
            self.next()
            condition = self.parse_or()
            if not self.at_name("else"):
                raise UnsupportedTemplate("inline conditional without `else`", self.line,
                                          self.source)
            self.next()
            node = Ternary(condition, node, self.parse_or(), self.line)
        return node

    def parse_and(self) -> Any:
        node = self.parse_not()
        while self.at_name("and"):
            self.next()
            node = BoolOp("and", node, self.parse_not())
        return node

    def parse_not(self) -> Any:
        if self.at_name("not"):
            self.next()
            return Unary("not", self.parse_not(), self.line)
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        node = self.parse_concat()
        while True:
            if self.at_op("==", "!=", "<=", ">=", "<", ">"):
                op = self.next().value
                node = Compare(op, node, self.parse_concat(), self.line)
                continue
            if self.at_name("in"):
                self.next()
                node = Compare("in", node, self.parse_concat(), self.line)
                continue
            if self.at_name("not") and self.index + 1 < len(self.tokens) \
                    and self.tokens[self.index + 1].kind == "name" \
                    and self.tokens[self.index + 1].value == "in":
                self.next()
                self.next()
                node = Unary("not", Compare("in", node, self.parse_concat(), self.line),
                             self.line)
                continue
            if self.at_name("is"):
                self.next()
                negated = False
                if self.at_name("not"):
                    self.next()
                    negated = True
                if self.peek().kind != "name":
                    raise UnsupportedTemplate("`is` without a test name", self.line, self.source)
                name = self.next().value
                if name not in TESTS:
                    raise UnsupportedTemplate(f"test {name!r}", self.line,
                                              f"supported: {', '.join(sorted(TESTS))}")
                node = Test(name, negated, node, self.line)
                continue
            return node

    def parse_concat(self) -> Any:
        node = self.parse_additive()
        while self.at_op("~"):
            self.next()
            node = Binary("~", node, self.parse_additive(), self.line)
        return node

    def parse_additive(self) -> Any:
        node = self.parse_multiplicative()
        while self.at_op("+", "-"):
            op = self.next().value
            node = Binary(op, node, self.parse_multiplicative(), self.line)
        return node

    def parse_multiplicative(self) -> Any:
        node = self.parse_unary()
        while self.at_op("*", "/", "//", "%"):
            op = self.next().value
            node = Binary(op, node, self.parse_unary(), self.line)
        return node

    def parse_unary(self) -> Any:
        if self.at_op("-"):
            self.next()
            return Unary("-", self.parse_unary(), self.line)
        if self.at_op("+"):
            self.next()
            return self.parse_unary()
        return self.parse_filter()

    def parse_filter(self) -> Any:
        node = self.parse_postfix()
        while self.at_op("|"):
            self.next()
            if self.peek().kind != "name":
                raise UnsupportedTemplate("filter without a name", self.line, self.source)
            name = self.next().value
            if name not in FILTERS:
                raise UnsupportedTemplate(f"filter {name!r}", self.line,
                                          f"supported: {', '.join(sorted(FILTERS))}")
            args, kwargs = self.parse_call_args() if self.at_op("(") else ((), ())
            node = Filter(name, node, args, kwargs, self.line)
        return node

    def parse_postfix(self) -> Any:
        node = self.parse_primary()
        while True:
            if self.at_op("."):
                self.next()
                if self.peek().kind != "name":
                    raise UnsupportedTemplate("attribute access without a name", self.line,
                                              self.source)
                node = Attr(node, self.next().value, self.line)
            elif self.at_op("["):
                self.next()
                node = self.parse_subscript(node)
            elif self.at_op("("):
                args, kwargs = self.parse_call_args()
                node = Call(node, args, kwargs, self.line)
            else:
                return node

    def parse_subscript(self, base: Any) -> Any:
        start = None if self.at_op(":") else self.parse_or()
        if self.at_op(":"):
            self.next()
            stop = None if self.at_op(":", "]") else self.parse_or()
            step = None
            if self.at_op(":"):
                self.next()
                step = None if self.at_op("]") else self.parse_or()
            self.expect_op("]")
            return SliceExpr(base, start, stop, step, self.line)
        self.expect_op("]")
        return Index(base, start, self.line)

    def parse_call_args(self) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]:
        self.expect_op("(")
        args: list[Any] = []
        kwargs: list[tuple[str, Any]] = []
        while not self.at_op(")"):
            if self.peek().kind == "name" and self.index + 1 < len(self.tokens) \
                    and self.tokens[self.index + 1].kind == "op" \
                    and self.tokens[self.index + 1].value == "=":
                name = self.next().value
                self.next()
                kwargs.append((name, self.parse_or()))
            else:
                args.append(self.parse_or())
            if self.at_op(","):
                self.next()
            elif not self.at_op(")"):
                raise UnsupportedTemplate("call arguments must be comma separated", self.line,
                                          self.source)
        self.expect_op(")")
        return tuple(args), tuple(kwargs)

    def parse_primary(self) -> Any:
        token = self.peek()
        if token.kind == "number":
            self.next()
            return Literal(token.value)
        if token.kind == "string":
            self.next()
            return Literal(token.value)
        if token.kind == "name":
            self.next()
            if token.value in _KEYWORD_LITERALS:
                return Literal(_KEYWORD_LITERALS[token.value])
            if token.value in _RESERVED_IN_PRIMARY:
                raise UnsupportedTemplate(f"expression keyword {token.value!r}", self.line,
                                          self.source)
            if token.value in UNSUPPORTED_GLOBALS:
                raise UnsupportedTemplate(f"global {token.value!r}", self.line,
                                          "not in the supported subset")
            return Name(token.value, self.line)
        if self.at_op("("):
            self.next()
            node = self.parse_or()
            while self.at_op(","):
                self.next()
                if not self.at_op(")"):
                    node = ListLiteral((node, self.parse_or()))
            self.expect_op(")")
            return node
        if self.at_op("["):
            self.next()
            items: list[Any] = []
            while not self.at_op("]"):
                items.append(self.parse_or())
                if self.at_op(","):
                    self.next()
            self.expect_op("]")
            return ListLiteral(tuple(items))
        raise UnsupportedTemplate(f"expression token {token.value!r}", self.line, self.source)


def parse_expression(source: str, line: int) -> Any:
    return ExprParser(tokenize_expression(source, line), line, source).parse()


# ------------------------------------------------------------------ template AST
@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Output:
    expr: Any
    line: int


@dataclass(frozen=True, slots=True)
class If:
    branches: tuple[tuple[Any, tuple[Any, ...]], ...]
    otherwise: tuple[Any, ...]
    line: int


@dataclass(frozen=True, slots=True)
class For:
    targets: tuple[str, ...]
    iterable: Any
    condition: Any
    body: tuple[Any, ...]
    line: int


@dataclass(frozen=True, slots=True)
class Set:
    name: str
    target_attr: str | None
    expr: Any
    line: int


@dataclass(frozen=True, slots=True)
class Macro:
    name: str
    params: tuple[tuple[str, Any], ...]         # (name, default-expression or None)
    body: tuple[Any, ...]
    line: int


_TAG_RE = re.compile(r"{{(.*?)}}|{%(.*?)%}|{#(.*?)#}", re.DOTALL)
_STOP_TAGS = ("elif", "else", "endif", "endfor", "endmacro")


@dataclass(frozen=True, slots=True)
class _Item:
    kind: str        # "text" | "output" | "tag"
    payload: str
    line: int


def _scan(source: str) -> list[_Item]:
    """Split the template into text/output/tag items, applying Jinja whitespace control."""
    items: list[_Item] = []
    index = 0
    line = 1
    pending_trim = False                    # a previous tag ended with `-%}`
    while index < len(source):
        match = _TAG_RE.search(source, index)
        if match is None:
            tail = source[index:]
            items.append(_Item("text", tail.lstrip() if pending_trim else tail, line))
            break
        text = source[index:match.start()]
        items.append(_Item("text", text.lstrip() if pending_trim else text, line))
        pending_trim = False
        start_line = line + source.count("\n", index, match.start())
        raw = match.group(0)
        inner = raw[2:-2]
        if inner.startswith("-"):
            inner = inner[1:]
            if items and items[-1].kind == "text":
                items[-1] = _Item("text", items[-1].payload.rstrip(), items[-1].line)
        if inner.endswith("-"):
            inner = inner[:-1]
            pending_trim = True
        if raw.startswith("{{"):
            items.append(_Item("output", inner.strip(), start_line))
        elif raw.startswith("{%"):
            items.append(_Item("tag", inner.strip(), start_line))
        # `{# ... #}` comments are dropped
        line = start_line + raw.count("\n")
        index = match.end()
    return [item for item in items if item.kind != "text" or item.payload]


class TemplateParser:
    """Turn the scanned items into the node tree (blocks nest recursively)."""

    def __init__(self, items: list[_Item]) -> None:
        self.items = items
        self.index = 0

    def parse(self) -> tuple[Any, ...]:
        nodes, stop = self.parse_block()
        if stop is not None:
            raise UnsupportedTemplate(f"unexpected {stop!r}", self.items[self.index - 1].line)
        return tuple(nodes)

    def parse_block(self) -> tuple[list[Any], str | None]:
        nodes: list[Any] = []
        while self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            if item.kind == "text":
                nodes.append(Text(item.payload))
                continue
            if item.kind == "output":
                nodes.append(Output(parse_expression(item.payload, item.line), item.line))
                continue
            head = item.payload.split(" ", 1)[0].rstrip("-").split(" ", 1)[0] \
                if item.payload else ""
            tail = item.payload[len(head):].strip()
            if head in _STOP_TAGS:
                return nodes, head
            nodes.append(self.parse_tag(head, tail, item.line))
        return nodes, None

    def parse_tag(self, head: str, tail: str, line: int) -> Any:
        if head == "if":
            return self.parse_if(parse_expression(tail, line), line)
        if head == "for":
            return self.parse_for(tail, line)
        if head == "macro":
            return self.parse_macro(tail, line)
        if head == "set":
            target, _, value = tail.partition("=")
            if not value:
                raise UnsupportedTemplate("`set` without a value", line, tail)
            name, _, attr = target.strip().partition(".")
            return Set(name, attr or None, parse_expression(value.strip(), line), line)
        raise UnsupportedTemplate(f"tag {head or '<empty>'!r}", line, tail)

    def parse_if(self, condition: Any, line: int) -> If:
        branches: list[tuple[Any, tuple[Any, ...]]] = []
        otherwise: tuple[Any, ...] = ()
        current_condition: Any | None = condition
        while True:
            body, stop = self.parse_block()
            if current_condition is not None:
                branches.append((current_condition, tuple(body)))
            else:
                otherwise = tuple(body)
            if stop is None:
                raise UnsupportedTemplate("if without endif", line, "missing end tag")
            if stop == "elif":
                item = self.items[self.index - 1]
                current_condition = parse_expression(item.payload[len("elif"):].strip(),
                                                     item.line)
                continue
            if stop == "else":
                current_condition = None
                continue
            return If(tuple(branches), otherwise, line)

    def parse_for(self, tail: str, line: int) -> For:
        iterable_source, _, condition_source = tail.partition(" if ")
        targets_raw, sep, iterable_raw = iterable_source.partition(" in ")
        if not sep:
            raise UnsupportedTemplate("`for` without `in`", line, tail)
        targets = tuple(part.strip() for part in targets_raw.split(","))
        body, stop = self.parse_block()
        if stop != "endfor":
            raise UnsupportedTemplate("for without endfor", line, "missing end tag")
        return For(targets, parse_expression(iterable_raw.strip(), line),
                   parse_expression(condition_source.strip(), line) if condition_source else None,
                   tuple(body), line)

    def parse_macro(self, tail: str, line: int) -> Macro:
        name, _, params_source = tail.partition("(")
        params: list[tuple[str, Any]] = []
        for chunk in _split_commas(params_source.rstrip(")")):
            param, _, default = chunk.partition("=")
            params.append((param.strip(), parse_expression(default.strip(), line)
                           if default else None))
        body, stop = self.parse_block()
        if stop != "endmacro":
            raise UnsupportedTemplate("macro without endmacro", line, "missing end tag")
        return Macro(name.strip(), tuple(params), tuple(body), line)


def _split_commas(source: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for char in source:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current)
    return [part.strip() for part in parts]


def parse(source: str) -> tuple[Any, ...]:
    """Parse a chat template; raises `UnsupportedTemplate` (chain step 1 -> step 2)."""
    if not isinstance(source, str) or not source.strip():
        raise UnsupportedTemplate("empty template", 0, "no template text")
    return TemplateParser(_scan(source)).parse()


def supports(template: str | None) -> bool:
    """True when the internal renderer can parse (and therefore render) `template`."""
    try:
        parse(template or "")
    except UnsupportedTemplate:
        return False
    return True


# ------------------------------------------------------------------- evaluation
def _lookup(obj: Any, name: str) -> Any:
    if isinstance(obj, _Namespace):
        return getattr(obj, name, UNDEFINED)
    if isinstance(obj, LoopState):
        return obj.attr(name)
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        return getattr(obj, name, UNDEFINED)
    return getattr(obj, name, UNDEFINED)


def _stringify(value: Any) -> str:
    if value is None or isinstance(value, _Undefined):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    return str(value)


def _filter_default(value: Any, *args: Any, **kwargs: Any) -> Any:
    fallback = args[0] if args else ""
    if kwargs.get("boolean"):
        return value if value else fallback
    return fallback if isinstance(value, _Undefined) else value


def _filter_join(value: Any, *args: Any, **kwargs: Any) -> str:
    separator = kwargs.get("d") or (args[0] if args else "")
    return str(separator).join(_stringify(item) for item in value)


def _filter_replace(value: Any, *args: Any, **kwargs: Any) -> str:
    old = args[0] if args else kwargs.get("old", "")
    new = args[1] if len(args) > 1 else kwargs.get("new", "")
    return _stringify(value).replace(_stringify(old), _stringify(new))


def _filter_round(value: Any, *args: Any, **kwargs: Any) -> Any:
    precision = int(args[0]) if args else int(kwargs.get("precision", 0))
    return round(float(value), precision)


FILTERS: dict[str, Callable[..., Any]] = {
    "abs": lambda value: abs(value),
    "count": lambda value: len(value),
    "default": _filter_default,
    "d": _filter_default,
    "first": lambda value: next(iter(value), UNDEFINED),
    "float": lambda value: float(value),
    "int": lambda value: int(value),
    "items": lambda value: list(value.items()),
    "join": _filter_join,
    "keys": lambda value: list(value.keys()),
    "last": lambda value: list(value)[-1] if len(list(value)) else UNDEFINED,
    "length": lambda value: len(value),
    "list": lambda value: list(value),
    "lower": lambda value: _stringify(value).lower(),
    "lstrip": lambda value, *a: _stringify(value).lstrip(a[0] if a else None),
    "replace": _filter_replace,
    "reverse": lambda value: list(value)[::-1],
    "round": _filter_round,
    "rstrip": lambda value, *a: _stringify(value).rstrip(a[0] if a else None),
    "safe": lambda value: value,
    "sort": lambda value: sorted(value),
    "split": lambda value, *a, **k: _stringify(value).split(k.get("sep") or (a[0] if a else None)),
    "string": _stringify,
    "tojson": lambda value, *a, **k: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")),
    "trim": lambda value: _stringify(value).strip(),
    "upper": lambda value: _stringify(value).upper(),
    "values": lambda value: list(value.values()),
}

TESTS: dict[str, Callable[[Any], bool]] = {
    "defined": lambda value: not isinstance(value, _Undefined),
    "undefined": lambda value: isinstance(value, _Undefined),
    "none": lambda value: value is None,
    "boolean": lambda value: isinstance(value, bool),
    "callable": callable,
    "false": lambda value: value is False,
    "iterable": lambda value: hasattr(value, "__iter__"),
    "mapping": lambda value: isinstance(value, Mapping),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "sequence": lambda value: isinstance(value, Sequence) and not isinstance(value, str),
    "string": lambda value: isinstance(value, str),
    "true": lambda value: value is True,
    "even": lambda value: int(value) % 2 == 0,
    "odd": lambda value: int(value) % 2 == 1,
}


class Renderer:
    """Evaluates the parsed template against a scope (one instance per render call)."""

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    # ---- expressions
    def eval(self, node: Any) -> Any:
        handler = getattr(self, f"_eval_{type(node).__name__.lower()}", None)
        if handler is None:                                   # pragma: no cover - defensive
            raise TemplateError(f"cannot evaluate {type(node).__name__}")
        return handler(node)

    def _eval_literal(self, node: Literal) -> Any:
        return node.value

    def _eval_listliteral(self, node: ListLiteral) -> Any:
        return [self.eval(item) for item in node.items]

    def _eval_name(self, node: Name) -> Any:
        if node.name == "namespace":
            return lambda **kwargs: _Namespace(**kwargs)
        if node.name == "raise_exception":
            def raise_exception(*args: Any) -> Any:
                raise TemplateError(_stringify(args[0]) if args else "template raised")
            return raise_exception
        return self.scope.get(node.name)

    def _eval_attr(self, node: Attr) -> Any:
        base = self.eval(node.base)
        if isinstance(base, _Undefined):
            raise TemplateError(f"{node.name} is not defined")
        return _lookup(base, node.name)

    def _eval_index(self, node: Index) -> Any:
        return self.eval(node.base)[self.eval(node.item)]

    def _eval_sliceexpr(self, node: SliceExpr) -> Any:
        return self.eval(node.base)[slice(self._maybe(node.start), self._maybe(node.stop),
                                          self._maybe(node.step))]

    def _maybe(self, node: Any) -> Any:
        return None if node is None else self.eval(node)

    def _eval_call(self, node: Call) -> Any:
        func = self.eval(node.func)
        args = [self.eval(arg) for arg in node.args]
        kwargs = {name: self.eval(value) for name, value in node.kwargs}
        if isinstance(func, _Undefined):
            raise TemplateError("call of an undefined name")
        return func(*args, **kwargs)

    def _eval_unary(self, node: Unary) -> Any:
        value = self.eval(node.operand)
        if node.op == "not":
            return not value
        if node.op == "-":
            return -value
        raise TemplateError(f"unary {node.op}")                # pragma: no cover

    def _eval_boolop(self, node: BoolOp) -> Any:
        left = self.eval(node.left)
        if node.op == "and":
            return self.eval(node.right) if left else left
        return left if left else self.eval(node.right)

    def _eval_binary(self, node: Binary) -> Any:
        left, right = self.eval(node.left), self.eval(node.right)
        if node.op == "+":
            if isinstance(left, str) or isinstance(right, str):
                return _stringify(left) + _stringify(right)
            return left + right
        if node.op == "~":
            return _stringify(left) + _stringify(right)
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "//":
            return left // right
        if node.op == "%":
            return left % right
        raise TemplateError(f"binary {node.op}")               # pragma: no cover

    def _eval_compare(self, node: Compare) -> Any:
        left, right = self.eval(node.left), self.eval(node.right)
        op = node.op
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "in":
            return left in right
        raise TemplateError(f"compare {op}")                   # pragma: no cover

    def _eval_test(self, node: Test) -> Any:
        verdict = TESTS[node.name](self.eval(node.operand))
        return not verdict if node.negated else verdict

    def _eval_ternary(self, node: Ternary) -> Any:
        return self.eval(node.then) if self.eval(node.condition) else self.eval(node.otherwise)

    def _eval_filter(self, node: Filter) -> Any:
        value = self.eval(node.operand)
        args = [self.eval(arg) for arg in node.args]
        kwargs = {name: self.eval(item) for name, item in node.kwargs}
        return FILTERS[node.name](value, *args, **kwargs)

    # ---- statements
    def render(self, nodes: Sequence[Any]) -> str:
        out: list[str] = []
        for node in nodes:
            kind = type(node).__name__
            if kind == "Text":
                out.append(node.text)
            elif kind == "Output":
                out.append(_stringify(self.eval(node.expr)))
            elif kind == "If":
                out.append(self._render_if(node))
            elif kind == "For":
                out.append(self._render_for(node))
            elif kind == "Set":
                self._render_set(node)
            elif kind == "Macro":
                self.scope.set(node.name, self._make_macro(node))
            else:                                              # pragma: no cover - defensive
                raise TemplateError(f"cannot render {kind}")
        return "".join(out)

    def _render_if(self, node: If) -> str:
        for condition, body in node.branches:
            if self.eval(condition):
                return self.render(body)
        return self.render(node.otherwise)

    def _render_for(self, node: For) -> str:
        items = list(self.eval(node.iterable))
        state = LoopState(items)
        out: list[str] = []
        for index, item in enumerate(items):
            state.index0 = index
            values = item if len(node.targets) > 1 else (item,)
            if len(node.targets) > 1 and not isinstance(values, (tuple, list)):
                raise TemplateError("cannot unpack this value in a for loop")
            child = self.scope.child()
            for name, value in zip(node.targets, values, strict=False):
                child.set(name, value)
            child.set("loop", state)
            if node.condition is not None:
                previous = self.scope
                self.scope = child
                try:
                    keep = self.eval(node.condition)
                finally:
                    self.scope = previous
                if not keep:
                    continue
            out.append(self._render_in(child, node.body))
        return "".join(out)

    def _render_in(self, scope: Scope, nodes: Sequence[Any]) -> str:
        previous = self.scope
        self.scope = scope
        try:
            return self.render(nodes)
        finally:
            self.scope = previous

    def _render_set(self, node: Set) -> None:
        value = self.eval(node.expr)
        if node.target_attr is None:
            self.scope.set(node.name, value)
            return
        target = self.scope.get(node.name)
        if isinstance(target, _Undefined):
            raise TemplateError(f"cannot set attribute on undefined {node.name!r}")
        if isinstance(target, Mapping):
            target[node.target_attr] = value                     # type: ignore[index]
        else:
            setattr(target, node.target_attr, value)

    def _make_macro(self, node: Macro) -> Callable[..., str]:
        def macro(*args: Any, **kwargs: Any) -> str:
            child = self.scope.child()
            for index, (name, default) in enumerate(node.params):
                if index < len(args):
                    child.set(name, args[index])
                elif name in kwargs:
                    child.set(name, kwargs.pop(name))
                elif default is not None:
                    previous = self.scope
                    self.scope = child
                    try:
                        child.set(name, self.eval(default))
                    finally:
                        self.scope = previous
                else:
                    child.set(name, UNDEFINED)
            return self._render_in(child, node.body)
        return macro


def render(template: str, *, messages: Sequence[Mapping[str, Any]],
           add_generation_prompt: bool = False, kwargs: Mapping[str, Any] | None = None,
           extra: Mapping[str, Any] | None = None) -> str:
    """Render one chat template with the internal renderer (raises `UnsupportedTemplate`)."""
    nodes = parse(template)
    scope = Scope({"messages": list(messages),
                   "add_generation_prompt": bool(add_generation_prompt)})
    for key, value in (kwargs or {}).items():
        scope.set(key, value)
    for key, value in (extra or {}).items():
        scope.set(key, value)
    return _render_nodes(nodes, scope)


def _render_nodes(nodes: Sequence[Any], scope: Scope) -> str:
    return Renderer(scope).render(nodes)


# ------------------------------------------------------------- thinking policy
def no_open_think(text: str) -> bool:
    """True when the prompt does not leave the model inside a thinking block (A-E1c-2)."""
    depth = 0
    for match in _THINK_RE.finditer(text):
        depth = depth + 1 if match.group(0) in THINK_OPENERS else max(0, depth - 1)
    return depth == 0


def strip_trailing_empty_think_block(text: str) -> str:
    """Remove an *empty* closed block at the end (Qwen3.5's `enable_thinking=false` marker)."""
    updated = _EMPTY_THINK_RE.sub("", text)
    while updated != text:
        text, updated = updated, _EMPTY_THINK_RE.sub("", updated)
    return updated


def strip_trailing_open_think(text: str) -> str:
    """Remove a trailing *unclosed* opener — the provable fallback of A-E1c-2."""
    updated = text.rstrip()
    for opener in THINK_OPENERS:
        if updated.endswith(opener):
            return updated[: -len(opener)]
    return text


def suppress_thinking(text: str, *, policy: FamilyPolicy | None,
                      think_mode: str = "auto") -> tuple[str, str]:
    """Return `(text, mode)` with the prompt provably outside a thinking block."""
    if think_mode == "on":
        return text, "on"
    if policy is not None and policy.thinking == "soft" and policy.soft_marker:
        return text, "marker"
    if no_open_think(text):
        if policy is not None and policy.strip_empty_think_block:
            return strip_trailing_empty_think_block(text), "suppressed"
        return text, "suppressed"
    return strip_trailing_open_think(text), "stripped"


# ------------------------------------------------------------------ resolution
@dataclass(frozen=True, slots=True)
class Resolution:
    """Which template won, what it rendered with, and how thinking was handled."""

    kind: str                 # "gguf-renderer" | "builtin" | "user" | "error"
    renderer: str             # "internal" | "builtin" | "plain"
    source: str
    template: str | None
    family: str | None
    thinking: str             # "suppressed" | "stripped" | "marker" | "on" | "n/a" | "unknown"
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    explicit: bool = False
    policy: FamilyPolicy | None = None
    builtin: Callable[..., str | None] | None = None      # llama_chat_apply_template bridge

    @property
    def is_plain(self) -> bool:
        return self.renderer == "plain"

    def __call__(self, template: str, messages: Sequence[Mapping[str, Any]], add_ass: bool
                 ) -> str | None:
        """The resolution is itself the builtin renderer (so callers can just pass it along)."""
        if self.builtin is None:
            return None
        return self.builtin(template, messages, add_ass)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "renderer": self.renderer, "source": self.source,
                "family": self.family, "thinking": self.thinking,
                "warnings": list(self.warnings), "notes": list(self.notes),
                "explicit": self.explicit}


def _probe_thinking(resolution: Resolution, messages: Sequence[Mapping[str, Any]],
                    policy: FamilyPolicy | None, builtin_renderer: Callable[..., str] | None
                    ) -> str:
    """Render the raw template with thinking off and report how suppression had to happen."""
    try:
        raw = _render_raw(messages, resolution, add_generation_prompt=True,
                          enable_thinking=False, builtin_renderer=builtin_renderer)
    except (TemplateError, TemplateUnresolvedError):
        return "unknown"
    if policy is not None and policy.thinking == "soft":
        return "marker"
    return "suppressed" if no_open_think(raw) else "stripped"


def _unresolved_reason(model_template: str | None, detail: str) -> str:
    return (
        "E_TEMPLATE_UNRESOLVED: no template could render this prompt. "
        f"The GGUF {TEMPLATE_KEY} {detail}; libllama's built-in templates "
        "(llama_chat_apply_template) do not match it either. Fix: pass "
        "--template <builtin-name|path-to-jinja|plain> (options.template in a request), or use a "
        "model whose template is inside the supported subset "
        f"({len(FILTERS)} filters, if/for/set/macro, `is` tests)."
    )


def resolve(*, messages: Sequence[Mapping[str, Any]], model_template: str | None,
            arch: str | None = None, user_template: str | None = None,
            builtin_renderer: Callable[..., str] | None = None,
            explicit_user: bool = False,
            think_mode: str = "auto") -> Resolution:
    """The A-E1c-1 chain. Returns the winning `Resolution` (never silently "no template")."""
    family = detect_family(arch, model_template)
    policy = policy_for(family)

    if explicit_user and user_template:
        return _user_resolution(user_template, family, policy, builtin_renderer,
                               messages, think_mode)

    if model_template and supports(model_template):
        resolution = Resolution(kind="gguf-renderer", renderer="internal",
                                source="gguf:" + TEMPLATE_KEY, template=model_template,
                                family=family, thinking="suppressed", policy=policy)
        return _with_thinking(resolution, messages, builtin_renderer, policy, think_mode)

    detail = "is not present" if not model_template else ""
    if model_template and builtin_renderer is not None:
        try:
            probe = builtin_renderer(model_template, list(messages), True)
        except Exception:                                     # noqa: BLE001 - step 2 is best effort
            probe = None
        if probe is not None:
            resolution = Resolution(kind="builtin", renderer="builtin",
                                    source="llama_chat_apply_template",
                                    template=model_template, family=family,
                                    thinking="n/a", warnings=("W_TEMPLATE_FALLBACK",),
                                    notes=("the internal renderer rejected this template; "
                                           "the runtime's built-in family table rendered it",),
                                    policy=policy, builtin=builtin_renderer)
            return _with_thinking(resolution, messages, builtin_renderer, policy, think_mode)

    if user_template:
        return _user_resolution(user_template, family, policy, builtin_renderer, messages,
                                think_mode)

    if model_template and not supports(model_template):
        try:
            parse(model_template)
        except UnsupportedTemplate as exc:
            detail = f"uses {exc}"
    raise TemplateUnresolvedError(_unresolved_reason(model_template, detail))


def _with_thinking(resolution: Resolution, messages: Sequence[Mapping[str, Any]],
                   builtin_renderer: Callable[..., str] | None, policy: FamilyPolicy | None,
                   think_mode: str) -> Resolution:
    if think_mode == "on":
        thinking = "on"
    elif policy is not None and policy.thinking == "soft":
        thinking = "marker"
    elif policy is not None and policy.thinking == "none":
        thinking = "n/a"
    else:
        thinking = _probe_thinking(resolution, messages, policy, builtin_renderer)
    return dataclasses.replace(resolution, thinking=thinking)


def _user_resolution(user_template: str, family: str | None, policy: FamilyPolicy | None,
                     builtin_renderer: Callable[..., str] | None,
                     messages: Sequence[Mapping[str, Any]], think_mode: str) -> Resolution:
    text = user_template.strip()
    if text == PLAIN:
        return Resolution(kind="user", renderer="plain", source="user:plain", template=None,
                          family=family, thinking="n/a", explicit=True, policy=policy,
                          notes=("--template plain keeps the model-agnostic framing of "
                                 "engine/prompt.py",))
    if text in BUILTIN_TEMPLATES:
        if builtin_renderer is None:
            raise TemplateUnresolvedError(
                f"E_TEMPLATE_UNRESOLVED: --template {text} needs a loaded llama.cpp runtime "
                f"(llama_chat_apply_template); none is available in this process. Fix: run "
                f"`ggufone init` or pass GGUFONE_RUNTIME_DIR, or use --template <path>")
        resolution = Resolution(kind="user", renderer="builtin", source=f"user:{text}",
                                template=text, family=family, thinking="n/a", explicit=True,
                                policy=policy, builtin=builtin_renderer)
        return _with_thinking(resolution, messages, builtin_renderer, policy, think_mode)
    path = pathlib.Path(text)
    if path.exists() and path.is_file():
        try:
            template_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TemplateUnresolvedError(
                f"E_TEMPLATE_UNRESOLVED: cannot read --template {path} ({exc})") from exc
        if not supports(template_text):
            raise TemplateUnresolvedError(
                f"E_TEMPLATE_UNRESOLVED: --template {path} is outside the supported subset; "
                f"fix the template or pass --template plain (see docs/TEMPLATES.md)")
        resolution = Resolution(kind="user", renderer="internal", source=f"user:{path}",
                                template=template_text, family=family, thinking="suppressed",
                                explicit=True, policy=policy)
        return _with_thinking(resolution, messages, builtin_renderer, policy, think_mode)
    if "{{" in text or "{%" in text:
        if not supports(text):
            raise TemplateUnresolvedError(
                "E_TEMPLATE_UNRESOLVED: the inline --template text is outside the supported "
                "subset (docs/TEMPLATES.md)")
        resolution = Resolution(kind="user", renderer="internal", source="user:inline",
                                template=text, family=family, thinking="suppressed",
                                explicit=True, policy=policy)
        return _with_thinking(resolution, messages, builtin_renderer, policy, think_mode)
    raise TemplateUnresolvedError(
        f"E_TEMPLATE_UNRESOLVED: --template {text!r} is neither a builtin template name "
        f"({', '.join(BUILTIN_TEMPLATES[:6])}, ...), a readable file, nor inline template text. "
        f"Fix: --template plain, --template chatml, or --template /path/to/family.jinja")


def render_prompt(messages: Sequence[Mapping[str, Any]], resolution: Resolution, *,
                  add_generation_prompt: bool = True, enable_thinking: bool = False,
                  builtin_renderer: Callable[..., str] | None = None) -> str:
    """Render the resolved template; thinking is suppressed unless `enable_thinking`."""
    if resolution.renderer == "plain":
        return ""
    text = _render_raw(messages, resolution, add_generation_prompt=add_generation_prompt,
                       enable_thinking=enable_thinking, builtin_renderer=builtin_renderer)
    return text if enable_thinking else _apply_suppression(text, resolution.policy)


def _render_raw(messages: Sequence[Mapping[str, Any]], resolution: Resolution, *,
                add_generation_prompt: bool = True, enable_thinking: bool = False,
                builtin_renderer: Callable[..., str | None] | None = None) -> str:
    render_messages: list[Mapping[str, Any]] = list(messages)
    policy = resolution.policy
    if not enable_thinking and policy is not None and policy.thinking == "soft" \
            and policy.soft_marker:
        render_messages = _append_soft_marker(render_messages, policy.soft_marker)
    if resolution.renderer == "builtin":
        renderer = builtin_renderer or resolution.builtin
        if renderer is None:
            raise TemplateUnresolvedError(
                "E_TEMPLATE_UNRESOLVED: the resolved template needs llama_chat_apply_template "
                "but no runtime is loaded")
        text = renderer(resolution.template or "", render_messages, bool(add_generation_prompt))
        if text is None:
            raise TemplateUnresolvedError(_unresolved_reason(
                resolution.template, "is not a built-in llama.cpp template"))
        return text
    if resolution.template is None:
        raise TemplateUnresolvedError(
            "E_TEMPLATE_UNRESOLVED: the resolution carries no template text")
    return render(resolution.template, messages=render_messages,
                  add_generation_prompt=add_generation_prompt,
                  kwargs={"enable_thinking": bool(enable_thinking)})


def _apply_suppression(text: str, policy: FamilyPolicy | None) -> str:
    suppressed, _mode = suppress_thinking(text, policy=policy)
    return suppressed


def _append_soft_marker(messages: Sequence[Mapping[str, Any]], marker: str
                        ) -> list[Mapping[str, Any]]:
    updated: list[Mapping[str, Any]] = []
    index = max((i for i, message in enumerate(messages)
                 if str(message.get("role")) == "user"), default=None)
    for position, message in enumerate(messages):
        if position == index:
            updated.append({**message, "content": f"{message.get('content', '')}\n{marker}"})
        else:
            updated.append(message)
    return updated


# --------------------------------------------------- the runtime side of step 2
#: `struct llama_chat_message` lives in the ABI module (`runtime/ctypes_binding.py`) so the
#: signature pin covers it; re-exported here for callers that only import the resolver.
llama_chat_message = ctypes_binding.llama_chat_message


def runtime_builtin_renderer(runtime: Any) -> Callable[..., str | None]:
    """A renderer backed by `llama_chat_apply_template` (returns None when unsupported)."""
    apply_template = runtime.llama.llama_chat_apply_template
    message_struct = ctypes_binding.llama_chat_message

    def render_builtin(template: str, messages: Sequence[Mapping[str, Any]],
                       add_ass: bool) -> str | None:
        entries = [message_struct(str(message.get("role", "")).encode(),
                                  str(message.get("content", "")).encode())
                   for message in messages]
        array = (message_struct * len(entries))(*entries) if entries else None
        buffer = C.create_string_buffer(1 << 20)
        written = apply_template(str(template).encode(), array, len(entries), bool(add_ass),
                                 buffer, len(buffer))
        if written < 0:
            return None
        return buffer.value.decode("utf-8", errors="replace")

    return render_builtin


def runtime_builtin_names(runtime: Any) -> tuple[str, ...]:
    """`llama_chat_builtin_templates()` — the names the installed runtime knows."""
    builtin = runtime.llama.llama_chat_builtin_templates
    count = int(builtin(None, 0))
    if count <= 0:
        return BUILTIN_TEMPLATES
    array = (C.c_char_p * count)()
    builtin(array, count)
    return tuple(name.decode() for name in array if name)


def model_template_text(handle: Any) -> str | None:
    """The model's `tokenizer.chat_template` through `llama_model_chat_template`."""
    if handle is None or getattr(handle, "model", None) is None:
        return None
    raw = handle.runtime.llama.llama_model_chat_template(handle.model, None)
    if not raw:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def resolve_for_handle(handle: Any, *, messages: Sequence[Mapping[str, Any]],
                       user_template: str | None = None, explicit_user: bool = False,
                       think_mode: str = "auto") -> Resolution:
    """Convenience: the full chain for a loaded `ModelHandle` (its own template + runtime)."""
    template = model_template_text(handle)
    builtin = runtime_builtin_renderer(handle.runtime) if getattr(handle, "runtime", None) else None
    return resolve(messages=messages, model_template=template,
                   arch=getattr(handle, "arch", None), user_template=user_template,
                   builtin_renderer=builtin, explicit_user=explicit_user, think_mode=think_mode)
