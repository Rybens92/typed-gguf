"""Budget-aware routing + bounded escalation (SPEC 2.10, A-E2p5-4/5/7/8).

Milestone: E2.5.

**Routing.** `route()` turns a list of candidates (registry alias + file + quant, in practice)
into a `RoutePlan`: which alias to load, at which `kv_type`/`n_ctx`/`n_seq_max`, on which
backend, *and why* — every candidate gets a verdict step, so the response and the audit log can
explain a decision that went against the user's first guess.

The budget arithmetic is the SPEC's own conservative plan (`weights + kv_per_token · ctx · seq +
512 MiB`, SPEC 2.4), not the shared-KV reading `fit` uses for a single request: a route has to be
safe for the sequence count it picks. Consequences that are deliberate:

* the placement the router finds is never *bigger* than `fit.estimate_plan` would produce for the
  same request — a test asserts it, because `A-E2p5-4` wants the route checked against the plan;
* a device placement always outranks a CPU placement (same box, faster), and among equal
  placements the larger model wins — ties break alphabetically so the same inputs route the same
  way twice;
* a model that fits neither the device nor the system budget is rejected outright instead of
  being squeezed to an unusable context.

**Capability (A-E2p5-7).** When runtime directories are known, the arch pre-flight runs *before*
the budget: a model whose arch no installed runtime implements is rejected with its arch named.
The load-time guard (`capability.require_arch`) still stands; the router just refuses to spend a
user's time on a load that cannot work.

**Escalation (A-E2p5-5).** `escalation_candidates()` is the policy (which answers deserve a second
opinion, how many, in what order), `apply_escalation()` is the merge + the log. Both are pure:
the caller owns the second model, the threshold and the `max_escalations` bound (default 1), and
"off" is the default state of the whole path.
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ggufone.errors import BackendOomError, ModelArchUnsupportedError, ModelNotFoundError
from ggufone.runtime import fit

#: the engine's own floor (SPEC 2.2): prefix + question + one candidate
MIN_SEQ_MAX = 3
#: the context a route aims for when the request does not name one
DEFAULT_ROUTE_CTX = 4096
#: how much of the system RAM a CPU plan may spend (matches `registry.recommend`)
RAM_MARGIN = 0.20
#: an answer at or above this confidence is never escalated without a complaint from the engine
DEFAULT_ESCALATION_THRESHOLD = 0.5
MIB = 1024 * 1024


# --------------------------------------------------------------------- inputs
@dataclass(frozen=True, slots=True)
class Needs:
    """What the request needs from a context: a desired `n_ctx` and a sequence ceiling."""

    n_ctx: int = DEFAULT_ROUTE_CTX
    n_seq_max: int = MIN_SEQ_MAX


def needs_for(request: Any) -> Needs:
    """`Needs` from a parsed request: the request's own ceilings, or the documented defaults."""
    options = request.options
    candidates = [len(question.options) for question in request.questions]
    n_seq_max = options.n_seq_max or max(MIN_SEQ_MAX, 1 + max(candidates, default=1))
    return Needs(n_ctx=int(options.n_ctx or DEFAULT_ROUTE_CTX), n_seq_max=int(n_seq_max))


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model the router may pick: an alias, a GGUF path and what the registry knows about it."""

    alias: str
    path: str
    quant: str | None = None
    arch: str | None = None
    size: int | None = None
    available: bool = True

    @classmethod
    def from_entry(cls, entry: Any) -> Candidate:
        """From a registry entry (duck-typed: alias/path/quant/arch/size)."""
        path = str(entry.path)
        return cls(alias=str(entry.alias), path=path, quant=entry.quant, arch=entry.arch,
                   size=entry.size, available=pathlib.Path(path).is_file())


# -------------------------------------------------------------------- outputs
@dataclass(frozen=True, slots=True)
class RouteStep:
    """One candidate's verdict, kept in evaluation order for the response and the audit log."""

    alias: str
    quant: str | None
    verdict: str                      # "chosen" | "rejected"
    reason: str
    n_ctx: int = 0
    n_seq_max: int = 0
    kv_type: str = ""
    device_bytes: int = 0
    tier: str = ""                    # "device" | "cpu" | ""

    def to_dict(self) -> dict[str, Any]:
        return {"alias": self.alias, "quant": self.quant, "verdict": self.verdict,
                "reason": self.reason, "n_ctx": self.n_ctx, "n_seq_max": self.n_seq_max,
                "kv_type": self.kv_type, "device_bytes": self.device_bytes,
                "placement": self.tier}


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """The decision: what to load, how to size it, and the receipts for every alternative."""

    mode: str
    alias: str
    path: str
    quant: str | None
    kv_type: str
    n_ctx: int
    n_seq_max: int
    n_gpu_layers: int
    backend: str
    device_bytes: int
    reason: str
    steps: tuple[RouteStep, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    capability: dict[str, Any] = field(default_factory=dict)
    needs: Needs = field(default_factory=Needs)

    @property
    def placement(self) -> str:
        return "device" if self.n_gpu_layers > 0 else "cpu"

    def to_dict(self) -> dict[str, Any]:
        """The `engine.route` object (A-E2p5-8): the winner, the reason and the alternatives."""
        return {
            "mode": self.mode,
            "alias": self.alias,
            "path": self.path,
            "quant": self.quant,
            "kv_type": self.kv_type,
            "n_ctx": self.n_ctx,
            "n_seq_max": self.n_seq_max,
            "n_gpu_layers": self.n_gpu_layers,
            "backend": self.backend,
            "placement": self.placement,
            "device_bytes": self.device_bytes,
            "reason": self.reason,
            "budget": dict(self.budget),
            "capability": dict(self.capability),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class _Placement:                     # internal: what one candidate could actually run as
    alias: str
    quant: str | None
    path: str
    kv_type: str
    n_ctx: int
    n_seq_max: int
    n_gpu_layers: int
    device_bytes: int
    weights_bytes: int
    tier: int                        # 0 = device, 1 = cpu (ranking key)
    note: str                        # why this placement and not a better one


def _per_token(facts: fit.ModelFacts, kv_type: str) -> int:
    return fit.kv_bytes_per_token(facts.n_layer, facts.n_kv_head, facts.key_len, facts.value_len,
                                  fit.KV_BYTES_PER_ELEMENT[kv_type])


def _fit_context(facts: fit.ModelFacts, kv_type: str, weights_bytes: int, overhead_bytes: int,
                 budget: int, needs: Needs) -> tuple[int, int] | None:
    """The largest (n_ctx, n_seq_max) this budget can hold — SPEC 2.4's conservative formula.

    Prefers the requested sequence count and shrinks the context first (a context that is too
    small fails the engine's own guard loudly, while silently dropping sequences would change what
    the request can do); only when even the floor context does not fit one sequence fewer is
    tried. `None` means the budget cannot hold the weights + overhead at all.
    """
    room = budget - weights_bytes - overhead_bytes
    if room <= 0:
        return None
    per_token = _per_token(facts, kv_type)
    seq = max(MIN_SEQ_MAX, int(needs.n_seq_max))
    wanted = max(fit.MIN_CTX_FLOOR, int(needs.n_ctx))
    if per_token <= 0:
        return wanted, seq
    ctx = min(wanted, room // (per_token * seq))
    if ctx < fit.MIN_CTX_FLOOR:
        seq = MIN_SEQ_MAX
        ctx = min(wanted, room // (per_token * seq))
    if ctx < fit.MIN_CTX_FLOOR:
        return None
    return max(fit.MIN_CTX_FLOOR, int(ctx)), seq


def _plan_candidate(candidate: Candidate, facts: fit.ModelFacts, *, host: fit.HostFacts,
                    needs: Needs, kv_type: str, fit_target_mb: int
                    ) -> _Placement | None:
    """What this candidate can be on this host: a device placement, a CPU one, or nothing."""
    device_budget = fit.fit_budget(host, fit_target_mb)
    system_budget = int(host.ram_bytes * (1.0 - RAM_MARGIN))
    baseline = fit.estimate_plan(facts, host, n_ctx=max(fit.MIN_CTX_FLOOR, needs.n_ctx),
                                 n_seq_max=max(MIN_SEQ_MAX, needs.n_seq_max), kv_type=kv_type,
                                 fit_target_mb=fit_target_mb, min_ctx=fit.MIN_CTX_FLOOR)
    chosen_kv = baseline.kv_type
    if baseline.n_gpu_layers > 0 and facts.n_layer > 0:
        share = min(1.0, baseline.n_gpu_layers / facts.n_layer)
        weights = int(facts.weights_bytes * share)
        fitted = _fit_context(facts, chosen_kv, weights, fit.OVERHEAD_BYTES, device_budget, needs)
        if fitted is not None:
            n_ctx, n_seq_max = fitted
            device_bytes = weights + _per_token(facts, chosen_kv) * n_ctx * n_seq_max \
                + fit.OVERHEAD_BYTES
            full = baseline.n_gpu_layers >= facts.n_layer
            tier = 0 if full else 1
            note = (f"device placement: {baseline.n_gpu_layers} of {facts.n_layer} layer(s) "
                    f"offloaded" + ("" if full else " (partial offload: slower than a full one)"))
            return _Placement(candidate.alias, candidate.quant, candidate.path, chosen_kv, n_ctx,
                              n_seq_max, baseline.n_gpu_layers, device_bytes,
                              facts.weights_bytes, tier, note)
        device_reason = (f"device budget {device_budget / MIB:.0f} MiB cannot hold "
                         f"{weights / MIB:.0f} MiB of weights + the KV cache")
    else:
        device_reason = (f"no device budget ({device_budget / MIB:.0f} MiB free after the "
                         f"fit-target margin)")
    fitted = _fit_context(facts, chosen_kv, facts.weights_bytes, fit.OVERHEAD_BYTES,
                          system_budget, needs)
    if fitted is None:
        return None
    n_ctx, n_seq_max = fitted
    return _Placement(candidate.alias, candidate.quant, candidate.path, chosen_kv, n_ctx,
                      n_seq_max, 0, 0, facts.weights_bytes, 2,
                      f"cpu placement ({device_reason})")


# ------------------------------------------------------------------- the router
def default_supports_arch(runtime_dirs: Sequence[str]) -> Callable[[str, str], bool]:
    """The default arch probe: ask each runtime directory whether it carries the arch symbols.

    Imported lazily so a router with injected probes (and the offline tests) never touches the
    runtime finder or a bundle on disk.
    """
    from ggufone.runtime import capability

    def probe(runtime_dir: str, arch: str) -> bool:
        try:
            return bool(capability.supports_arch(runtime_dir, arch))
        except Exception:                     # noqa: BLE001 - an unreadable bundle is a "no"
            return False

    return probe


def route(candidates: Sequence[Candidate], *, needs: Needs, host: fit.HostFacts,
          facts_for: Callable[[str], fit.ModelFacts] | None = None,
          runtime_dirs: Sequence[str] = (), fit_target_mb: int = fit.DEFAULT_FIT_TARGET_MB,
          kv_type: str = "auto", supports_arch: Callable[[str, str], bool] | None = None,
          mode: str = "auto") -> RoutePlan:
    """Pick (alias, quant, kv_type, n_ctx, n_seq_max) inside the device budget (A-E2p5-4/7)."""
    if not candidates:
        raise ModelNotFoundError(
            "E_MODEL_NOT_FOUND: no model to route — the registry is empty and no --model was "
            "given (use `ggufone models pull` / `ggufone models use`, or pass a GGUF path)")
    reader = facts_for or fit.ModelFacts.read
    probe = supports_arch or default_supports_arch(runtime_dirs)
    runtimes = [str(entry) for entry in runtime_dirs]
    device_budget = fit.fit_budget(host, fit_target_mb)
    system_budget = int(host.ram_bytes * (1.0 - RAM_MARGIN))
    capability_report: dict[str, Any] = {"checked": bool(runtimes), "runtimes": runtimes,
                                         "unsupported": {}}

    placements: list[_Placement] = []
    rejected: list[RouteStep] = []
    capability_rejections: list[str] = []
    budget_rejections: list[str] = []
    for candidate in candidates:
        if not candidate.available:
            reason = f"availability: {candidate.path} is not on disk"
            rejected.append(RouteStep(candidate.alias, candidate.quant, "rejected", reason))
            continue
        try:
            facts = reader(candidate.path)
        except Exception as exc:              # noqa: BLE001 - an unreadable model is "unavailable"
            reason = f"availability: {candidate.path} cannot be read ({exc})"
            rejected.append(RouteStep(candidate.alias, candidate.quant, "rejected", reason))
            continue
        arch = facts.arch or candidate.arch
        if runtimes and arch and not any(probe(runtime_dir, arch) for runtime_dir in runtimes):
            reason = (f"arch capability: {arch} is not supported by any installed runtime "
                      f"({', '.join(runtimes)})")
            capability_report["unsupported"][candidate.alias] = arch
            capability_rejections.append(reason)
            rejected.append(RouteStep(candidate.alias, candidate.quant, "rejected", reason,
                                      kv_type="", tier=""))
            continue
        placement = _plan_candidate(candidate, facts, host=host, needs=needs, kv_type=kv_type,
                                    fit_target_mb=fit_target_mb)
        if placement is None:
            reason = (f"budget: {facts.weights_bytes / MIB:.0f} MiB of weights do not fit "
                      f"{max(device_budget, system_budget) / MIB:.0f} MiB of usable memory on "
                      f"{host.backend}")
            budget_rejections.append(reason)
            rejected.append(RouteStep(candidate.alias, candidate.quant, "rejected", reason))
            continue
        placements.append(placement)

    if not placements:
        if capability_rejections:
            raise ModelArchUnsupportedError(
                "E_MODEL_ARCH_UNSUPPORTED: " + "; ".join(capability_rejections))
        if budget_rejections:
            raise BackendOomError(
                "E_BACKEND_OOM: no candidate fits this host (" + "; ".join(budget_rejections)
                + f"); device budget {device_budget / MIB:.0f} MiB, system budget "
                  f"{system_budget / MIB:.0f} MiB")
        raise ModelNotFoundError(
            "E_MODEL_NOT_FOUND: no candidate is available (" + "; ".join(
                step.reason for step in rejected) + ")")

    placements.sort(key=lambda entry: (entry.tier, -entry.weights_bytes, -entry.device_bytes,
                                       entry.alias))
    chosen = placements[0]
    chosen_step = RouteStep(chosen.alias, chosen.quant, "chosen", chosen.note,
                            n_ctx=chosen.n_ctx, n_seq_max=chosen.n_seq_max,
                            kv_type=chosen.kv_type, device_bytes=chosen.device_bytes,
                            tier="device" if chosen.tier < 2 else "cpu")
    losers = [RouteStep(entry.alias, entry.quant, "rejected",
                        f"budget fit: ranked below {chosen.alias} — {entry.note} "
                        f"({entry.weights_bytes / MIB:.0f} MiB of weights; device budget "
                        f"{device_budget / MIB:.0f} MiB)", n_ctx=entry.n_ctx,
                        n_seq_max=entry.n_seq_max, kv_type=entry.kv_type,
                        device_bytes=entry.device_bytes,
                        tier="device" if entry.tier < 2 else "cpu")
              for entry in placements[1:]]
    steps = (chosen_step, *rejected, *losers)
    backend = host.backend if chosen.tier < 2 else "cpu"
    reason = (f"budget fit: {chosen.alias}"
              f"{' (' + chosen.quant + ')' if chosen.quant else ''} kv={chosen.kv_type} "
              f"n_ctx={chosen.n_ctx} n_seq_max={chosen.n_seq_max} on {backend}"
              f" — {chosen.note}"
              f"; {len(steps) - 1} alternative(s) rejected")
    return RoutePlan(
        mode=mode, alias=chosen.alias, path=chosen.path, quant=chosen.quant,
        kv_type=chosen.kv_type, n_ctx=chosen.n_ctx, n_seq_max=chosen.n_seq_max,
        n_gpu_layers=chosen.n_gpu_layers, backend=backend, device_bytes=chosen.device_bytes,
        reason=reason, steps=steps,
        budget={"device_bytes": device_budget, "system_bytes": system_budget,
                "free_bytes": host.vram_free_bytes or host.vram_bytes,
                "total_bytes": host.vram_bytes or host.ram_bytes, "ram_bytes": host.ram_bytes,
                "backend": host.backend, "fit_target_mb": int(fit_target_mb),
                "ram_margin": RAM_MARGIN},
        capability=capability_report, needs=needs)


# --------------------------------------------------------------------- escalation
@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """One answer that deserves a second opinion, and why."""

    question: str
    reason: str
    confidence: float
    reliability: str

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "reason": self.reason,
                "confidence": self.confidence, "reliability": self.reliability}


@dataclass(frozen=True, slots=True)
class EscalationLog:
    """The escalation record (A-E2p5-5/8): opt-in, bounded, and never silent."""

    enabled: bool = False
    limit: int = 0
    threshold: float = DEFAULT_ESCALATION_THRESHOLD
    target: dict[str, Any] | None = None
    decisions: tuple[dict[str, Any], ...] = ()
    skipped: tuple[dict[str, Any], ...] = ()

    @classmethod
    def disabled(cls) -> EscalationLog:
        return cls(enabled=False, limit=0)

    @property
    def count(self) -> int:
        return sum(1 for entry in self.decisions if entry.get("replaced"))

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "limit": self.limit, "threshold": self.threshold,
                "target": self.target, "count": self.count,
                "decisions": [dict(entry) for entry in self.decisions],
                "skipped": [dict(entry) for entry in self.skipped]}


def _confidence_of(answer: Mapping[str, Any]) -> float:
    """The answer's own confidence; `noul` carries none, so its peak probability stands in."""
    if answer.get("confidence") is not None:
        return float(answer["confidence"])
    probabilities = answer.get("probabilities") or {}
    return max((float(value) for value in probabilities.values()), default=0.0)


def _decision_of(answer: Mapping[str, Any]) -> str:
    """The answer's discrete decision, for the before/after line of the log."""
    if answer.get("type") == "choice":
        return str(answer.get("choice", ""))
    if answer.get("type") == "score":
        probabilities = list((answer.get("probabilities") or {}).values())
        if probabilities:
            best = 0
            for index in range(1, len(probabilities)):
                if float(probabilities[index]) > float(probabilities[best]):
                    best = index
            return list(answer.get("probabilities") or {})[best]
        return str(answer.get("score", ""))
    return "yes" if float(answer.get("noul", 0.0)) >= 0.5 else "no"


def escalation_candidates(answers: Mapping[str, Mapping[str, Any]], *,
                          threshold: float = DEFAULT_ESCALATION_THRESHOLD,
                          max_escalations: int = 1) -> tuple[EscalationDecision, ...]:
    """The policy: which answers to re-ask, lowest confidence first, at most `max_escalations`.

    An answer qualifies when the engine called it low (`low_mass` / `low_confidence`) or when its
    confidence is under `threshold`. `max_escalations <= 0` means "off" and is the default of the
    whole path: escalation only ever happens because a caller asked for it.
    """
    if max_escalations <= 0:
        return ()
    decisions: list[EscalationDecision] = []
    for question, answer in answers.items():
        reliability = str(answer.get("reliability") or "ok")
        confidence = _confidence_of(answer)
        if reliability in ("ok", "none", "") and confidence >= threshold:
            continue
        decisions.append(EscalationDecision(
            question=str(question),
            reason=reliability if reliability not in ("ok", "", "none") else "low_confidence",
            confidence=confidence, reliability=reliability))
    decisions.sort(key=lambda entry: (entry.confidence, entry.question))
    return tuple(decisions[:max_escalations])


def apply_escalation(answers: Mapping[str, Mapping[str, Any]],
                     replacement: Mapping[str, Mapping[str, Any]],
                     decisions: Iterable[EscalationDecision], *, target: Mapping[str, Any] | None,
                     limit: int = 1, threshold: float = DEFAULT_ESCALATION_THRESHOLD
                     ) -> tuple[dict[str, Any], EscalationLog]:
    """Replace the escalated answers with the second opinion and write the log.

    A decision whose question the target did not answer keeps the original answer
    (`replaced: false`) — the log says what happened instead of pretending the escalation
    succeeded. With no target at all, every decision is `skipped` and nothing changes.
    """
    merged: dict[str, Any] = {key: dict(value) for key, value in answers.items()}
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for decision in decisions:
        if target is None:
            skipped.append({**decision.to_dict(), "reason": "no escalation target available"})
            continue
        original = merged.get(decision.question, {})
        before = _decision_of(original)
        confidence_before = _confidence_of(original)
        if decision.question in replacement:
            escalated = dict(replacement[decision.question])
            merged[decision.question] = escalated
            entries.append({"question": decision.question, "reason": decision.reason,
                            "replaced": True, "was": before, "now": _decision_of(escalated),
                            "confidence_before": confidence_before,
                            "confidence_after": _confidence_of(escalated)})
        else:
            entries.append({"question": decision.question, "reason": decision.reason,
                            "replaced": False, "was": before, "now": before,
                            "confidence_before": confidence_before,
                            "confidence_after": confidence_before})
    log = EscalationLog(enabled=True, limit=int(limit), threshold=float(threshold),
                        target=dict(target) if target is not None else None,
                        decisions=tuple(entries), skipped=tuple(skipped))
    return merged, log
