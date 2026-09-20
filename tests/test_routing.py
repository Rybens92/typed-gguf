"""A-E2p5-4/5/7/8: budget-aware routing, the arch pre-flight and bounded escalation.

Offline: the device budget, the model facts and the runtime capability check are all injected, so
every routing decision is reproducible without a GPU, a runtime bundle or a GGUF on disk. The
live counterpart (a real registry + a real box) lives in `tools/e2p5_reproduce.py`.

The rules this file pins:
  * a candidate is rejected — never squeezed — when its weights and the 512 MiB overhead do not
    fit the budget, and routing then picks the *next* candidate;
  * the router never returns a model whose arch no installed runtime supports (A-E2p5-7), and it
    says so in the plan instead of failing later inside `llama_load_model_from_file`;
  * the chosen (n_ctx, n_seq_max, kv_type) fit the budget under the conservative reading (KV is
    budgeted per *sequence*, not shared) and never exceed the fit plan's own ceiling;
  * escalation is off unless asked for, bounded by `max_escalations` (default 1), only ever
    touches answers that are actually low-confidence, and leaves a log entry either way.
"""
from __future__ import annotations

import pytest

from typed_gguf.calibration import routing
from typed_gguf.errors import BackendOomError, ModelArchUnsupportedError, ModelNotFoundError
from typed_gguf.registry import store
from typed_gguf.runtime import fit

MIB = 1024 * 1024


def _candidate(alias: str, path: str, *, quant: str | None = None, arch: str | None = None,
               available: bool = True) -> routing.Candidate:
    """A registry-shaped candidate; `facts` come from the injected reader, not from disk."""
    return routing.Candidate(alias=alias, path=path, quant=quant, arch=arch, available=available)


def _facts(path: str, *, weights_mib: int, arch: str | None = "spark2_5",
           layers: int = 36, kv_heads: int = 4, dim: int = 256) -> fit.ModelFacts:
    return fit.ModelFacts(path=path, sha256="0" * 64, arch=arch, n_layer=layers,
                          n_kv_head=kv_heads, key_len=dim, value_len=dim, n_ctx_train=32768,
                          weights_bytes=weights_mib * MIB, file_size=weights_mib * MIB)


def _host(*, vram_mib: int, free_mib: int | None = None, backend: str = "vulkan"
          ) -> fit.HostFacts:
    free = vram_mib if free_mib is None else free_mib
    return fit.HostFacts(backend=backend, ram_bytes=31 * 1024 * MIB,
                         vram_bytes=vram_mib * MIB, n_cpu=8, fingerprint="test-host",
                         vram_free_bytes=free * MIB)


VULKAN = "/runtime/b11026-linux-x64-vulkan"
CPU = "/runtime/b11026-linux-x64-cpu"


def _supports(arch: str) -> object:
    """A capability probe: the Vulkan bundle supports every arch, the CPU bundle only qwen35."""
    def probe(runtime_dir: str, want: str) -> bool:
        if runtime_dir == VULKAN:
            return True
        return want == "qwen35"
    return probe


# ------------------------------------------------------------------ the router
def test_the_router_picks_the_biggest_model_that_fits_the_budget() -> None:
    facts = {"/m/small.gguf": _facts("/m/small.gguf", weights_mib=1024),
             "/m/mid.gguf": _facts("/m/mid.gguf", weights_mib=3 * 1024),
             "/m/big.gguf": _facts("/m/big.gguf", weights_mib=9 * 1024)}
    plan = routing.route(
        [routing.Candidate(alias="small", path="/m/small.gguf"),
         routing.Candidate(alias="mid", path="/m/mid.gguf"),
         routing.Candidate(alias="big", path="/m/big.gguf")],
        needs=routing.Needs(n_ctx=2048, n_seq_max=4),
        host=_host(vram_mib=8 * 1024),
        facts_for=lambda path: facts[path],
        runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    assert plan.alias == "mid"
    assert plan.mode == "auto"
    verdicts = {step.alias: step for step in plan.steps}
    assert verdicts["big"].verdict == "rejected"
    assert "budget" in verdicts["big"].reason
    assert "budget fit" in plan.reason and "mid" in plan.reason


def test_the_router_never_picks_a_model_no_installed_runtime_supports() -> None:
    """A-E2p5-7: the arch pre-flight runs before the budget, and it can veto a whole candidate."""
    facts = {"/m/spark.gguf": _facts("/m/spark.gguf", weights_mib=1024, arch="spark2_5"),
             "/m/qwen.gguf": _facts("/m/qwen.gguf", weights_mib=1024, arch="qwen35")}
    plan = routing.route(
        [routing.Candidate(alias="spark", path="/m/spark.gguf"),
         routing.Candidate(alias="qwen", path="/m/qwen.gguf")],
        needs=routing.Needs(n_ctx=1024, n_seq_max=3),
        host=_host(vram_mib=8 * 1024),
        facts_for=lambda path: facts[path],
        runtime_dirs=[CPU], supports_arch=_supports("spark2_5"))
    assert plan.alias == "qwen"
    spark = next(step for step in plan.steps if step.alias == "spark")
    assert spark.verdict == "rejected" and "arch" in spark.reason
    assert plan.capability["checked"] is True
    assert plan.capability["runtimes"] == [CPU]


def test_a_router_with_no_supported_model_names_the_arch_it_could_not_serve() -> None:
    facts = {"/m/spark.gguf": _facts("/m/spark.gguf", weights_mib=1024, arch="spark2_5")}
    with pytest.raises(ModelArchUnsupportedError) as excinfo:
        routing.route([routing.Candidate(alias="spark", path="/m/spark.gguf")],
                      needs=routing.Needs(n_ctx=1024, n_seq_max=3),
                      host=_host(vram_mib=8 * 1024),
                      facts_for=lambda path: facts[path],
                      runtime_dirs=[CPU], supports_arch=_supports("spark2_5"))
    assert excinfo.value.code == "E_MODEL_ARCH_UNSUPPORTED"
    assert "spark2_5" in str(excinfo.value)


def test_a_model_that_does_not_fit_anywhere_is_an_oom_with_the_numbers() -> None:
    """A CPU-only box (or a device with nothing free) cannot hold a model that beats the RAM too."""
    facts = {"/m/huge.gguf": _facts("/m/huge.gguf", weights_mib=40 * 1024)}
    with pytest.raises(BackendOomError) as excinfo:
        routing.route([routing.Candidate(alias="huge", path="/m/huge.gguf")],
                      needs=routing.Needs(n_ctx=1024, n_seq_max=3),
                      host=_host(vram_mib=0, backend="cpu"),
                      facts_for=lambda path: facts[path],
                      runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    assert excinfo.value.code == "E_BACKEND_OOM"
    assert "MiB" in str(excinfo.value)


def test_a_big_model_that_only_fits_partially_is_still_a_candidate() -> None:
    """The E3 case: a model larger than the device runs with a partial offload, and says so."""
    facts = {"/m/big.gguf": _facts("/m/big.gguf", weights_mib=9 * 1024)}
    host = _host(vram_mib=8 * 1024)
    plan = routing.route([routing.Candidate(alias="big", path="/m/big.gguf")],
                         needs=routing.Needs(n_ctx=2048, n_seq_max=4), host=host,
                         facts_for=lambda path: facts[path], runtime_dirs=[VULKAN],
                         supports_arch=_supports("spark2_5"))
    assert plan.alias == "big"
    assert 0 < plan.n_gpu_layers < 36
    assert plan.placement == "device"
    assert "partial" in plan.steps[0].reason
    assert plan.device_bytes <= fit.fit_budget(host, fit.DEFAULT_FIT_TARGET_MB)


def test_an_unavailable_model_is_rejected_before_anything_is_read() -> None:
    def boom(path: str) -> fit.ModelFacts:
        raise AssertionError("facts must not be read for a model that is not on disk")

    facts = {"/m/ok.gguf": _facts("/m/ok.gguf", weights_mib=1024)}
    plan = routing.route(
        [routing.Candidate(alias="gone", path="/m/gone.gguf", available=False),
         routing.Candidate(alias="ok", path="/m/ok.gguf")],
        needs=routing.Needs(n_ctx=1024, n_seq_max=3),
        host=_host(vram_mib=8 * 1024),
        facts_for=lambda path: facts[path] if path in facts else boom(path),
        runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    assert plan.alias == "ok"
    gone = next(step for step in plan.steps if step.alias == "gone")
    assert gone.verdict == "rejected" and "availability" in gone.reason


def test_the_route_keeps_the_context_and_the_sequences_inside_the_budget() -> None:
    """The conservative reading: KV is budgeted per sequence, so the plan cannot lie about it."""
    facts = {"/m/mid.gguf": _facts("/m/mid.gguf", weights_mib=3 * 1024)}
    host = _host(vram_mib=8 * 1024)
    plan = routing.route([routing.Candidate(alias="mid", path="/m/mid.gguf")],
                         needs=routing.Needs(n_ctx=32768, n_seq_max=16),
                         host=host, facts_for=lambda path: facts[path],
                         runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    budget = fit.fit_budget(host, fit.DEFAULT_FIT_TARGET_MB)
    assert plan.device_bytes <= budget
    assert plan.n_ctx >= fit.MIN_CTX_FLOOR
    assert plan.n_ctx <= 32768 and plan.n_seq_max <= 16
    assert plan.n_seq_max >= routing.MIN_SEQ_MAX


def test_the_route_never_exceeds_the_fit_plan_it_was_asserted_against() -> None:
    """A-E2p5-4: the picked (kv_type, n_ctx) is inside the plan `typed-gguf fit` would produce."""
    facts = {"/m/mid.gguf": _facts("/m/mid.gguf", weights_mib=3 * 1024)}
    host = _host(vram_mib=8 * 1024)
    plan = routing.route([routing.Candidate(alias="mid", path="/m/mid.gguf")],
                         needs=routing.Needs(n_ctx=8192, n_seq_max=4), host=host,
                         facts_for=lambda path: facts[path], runtime_dirs=[VULKAN],
                         supports_arch=_supports("spark2_5"))
    reference = fit.estimate_plan(facts["/m/mid.gguf"], host, n_ctx=8192,
                                  n_seq_max=4, kv_type="auto")
    assert plan.kv_type == reference.kv_type
    assert plan.n_ctx <= reference.n_ctx
    assert plan.kv_type in fit.KV_DOWNGRADE_ORDER


def test_a_free_vram_report_smaller_than_the_card_shrinks_the_route() -> None:
    """E1c FIX (card t_8cb0a05e) applies to routing too: the *free* number is the budget."""
    facts = {"/m/mid.gguf": _facts("/m/mid.gguf", weights_mib=6 * 1024)}
    busy = _host(vram_mib=8 * 1024, free_mib=1024)
    plan = routing.route([routing.Candidate(alias="mid", path="/m/mid.gguf")],
                         needs=routing.Needs(n_ctx=4096, n_seq_max=4), host=busy,
                         facts_for=lambda path: facts[path], runtime_dirs=[VULKAN],
                         supports_arch=_supports("spark2_5"))
    assert plan.backend == "cpu" or plan.n_gpu_layers == 0
    assert plan.budget["free_bytes"] == 1024 * MIB
    assert "free" in plan.reason or "cpu" in plan.reason


def test_the_plan_is_json_ready_for_the_response() -> None:
    """A-E2p5-8: `engine.route` carries the decision and the reason, not just the winner."""
    facts = {"/m/mid.gguf": _facts("/m/mid.gguf", weights_mib=3 * 1024)}
    plan = routing.route([routing.Candidate(alias="mid", path="/m/mid.gguf", quant="Q8_0")],
                         needs=routing.Needs(n_ctx=4096, n_seq_max=4),
                         host=_host(vram_mib=8 * 1024), facts_for=lambda path: facts[path],
                         runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    payload = plan.to_dict()
    assert set(payload) >= {"mode", "alias", "path", "quant", "kv_type", "n_ctx", "n_seq_max",
                            "reason", "steps", "budget", "capability"}
    assert payload["alias"] == "mid" and payload["quant"] == "Q8_0"
    assert isinstance(payload["steps"], list) and payload["steps"]
    assert payload["steps"][0]["verdict"] == "chosen"


def test_the_same_inputs_route_the_same_way() -> None:
    facts = {"/m/a.gguf": _facts("/m/a.gguf", weights_mib=1024),
             "/m/b.gguf": _facts("/m/b.gguf", weights_mib=1024)}
    kwargs = dict(needs=routing.Needs(n_ctx=1024, n_seq_max=3), host=_host(vram_mib=8 * 1024),
                  facts_for=lambda path: facts[path], runtime_dirs=[VULKAN],
                  supports_arch=_supports("spark2_5"))
    first = routing.route([routing.Candidate(alias="b", path="/m/b.gguf"),
                           routing.Candidate(alias="a", path="/m/a.gguf")], **kwargs)
    second = routing.route([routing.Candidate(alias="a", path="/m/a.gguf"),
                            routing.Candidate(alias="b", path="/m/b.gguf")], **kwargs)
    assert first.alias == second.alias == "a"          # alphabetical tie-break, not input order


def test_candidates_can_be_built_from_registry_entries(tmp_path) -> None:
    entry = store.Entry(alias="spark", path="/m/spark.gguf", quant="Q8_0", arch="spark2_5",
                        size=4 * 1024 * MIB)
    candidate = routing.Candidate.from_entry(entry)
    assert candidate.alias == "spark" and candidate.quant == "Q8_0"
    assert candidate.arch == "spark2_5" and candidate.size == 4 * 1024 * MIB


def test_needs_are_derived_from_the_request() -> None:
    """The ceilings a route must respect come from the request: ctx and the candidate count."""
    from typed_gguf import schema
    request = schema.parse_request({
        "state": "blank dashboard",
        "questions": {"a": {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}},
    })
    needs = routing.needs_for(request)
    assert needs.n_seq_max == max(routing.MIN_SEQ_MAX, 1 + 3)
    assert needs.n_ctx == routing.DEFAULT_ROUTE_CTX
    explicit = schema.parse_request({
        "state": "blank dashboard",
        "options": {"n_ctx": 2048, "n_seq_max": 6},
        "questions": {"a": {"type": "noul"}},
    })
    assert routing.needs_for(explicit) == routing.Needs(n_ctx=2048, n_seq_max=6)


# --------------------------------------------------------------- escalation
def _answers(**overrides) -> dict:
    answers = {
        "sure": {"type": "choice", "choice": "billing", "confidence": 0.91, "reliability": "ok",
                 "probabilities": {"billing": 0.91, "api": 0.09}},
        "shaky": {"type": "choice", "choice": "api", "confidence": 0.31, "reliability": "ok",
                  "probabilities": {"api": 0.51, "billing": 0.49}},
        "empty": {"type": "noul", "noul": 0.55, "reliability": "low_mass",
                  "probabilities": {"yes": 0.55, "no": 0.45}},
    }
    answers.update(overrides)
    return answers


def test_escalation_has_to_be_asked_for() -> None:
    decisions = routing.escalation_candidates(_answers(), max_escalations=0)
    assert decisions == ()
    assert routing.EscalationLog.disabled().to_dict()["enabled"] is False


def test_only_low_confidence_answers_are_escalated() -> None:
    decisions = routing.escalation_candidates(_answers(), threshold=0.5, max_escalations=2)
    assert [decision.question for decision in decisions] == ["shaky", "empty"]
    assert decisions[0].reason == "low_confidence"
    assert decisions[1].reason == "low_mass"
    assert decisions[0].confidence == pytest.approx(0.31)


def test_escalation_is_bounded_by_the_limit_and_ordered_by_confidence() -> None:
    bounded = routing.escalation_candidates(_answers(), threshold=0.5, max_escalations=1)
    assert [decision.question for decision in bounded] == ["shaky"]
    # a lower threshold spares the confident answer, but `low_mass` is the engine's own
    # complaint: it is escalated whatever the threshold says
    assert [decision.question for decision in
            routing.escalation_candidates(_answers(), threshold=0.1, max_escalations=3)] == \
        ["empty"]


def test_a_confident_answer_is_never_escalated() -> None:
    answers = _answers(sure={"type": "choice", "choice": "billing", "confidence": 0.99,
                             "reliability": "ok", "probabilities": {"billing": 0.99}})
    decisions = routing.escalation_candidates(answers, threshold=0.5, max_escalations=3)
    assert "sure" not in {decision.question for decision in decisions}


def test_the_escalation_log_records_the_trigger_and_the_replacement() -> None:
    answers = _answers()
    decisions = routing.escalation_candidates(answers, threshold=0.5, max_escalations=1)
    replacement = {"shaky": {"type": "choice", "choice": "billing", "confidence": 0.88,
                             "reliability": "ok",
                             "probabilities": {"billing": 0.88, "api": 0.12}}}
    merged, log = routing.apply_escalation(answers, replacement, decisions,
                                           target={"alias": "spark-4b", "model": "spark-4b"})
    assert merged["shaky"]["choice"] == "billing" and merged["shaky"]["confidence"] == 0.88
    assert merged["sure"] == answers["sure"] and merged["empty"] == answers["empty"]
    payload = log.to_dict()
    assert payload["enabled"] is True and payload["limit"] == 1 and payload["count"] == 1
    assert payload["target"] == {"alias": "spark-4b", "model": "spark-4b"}
    entry = payload["decisions"][0]
    assert entry["question"] == "shaky" and entry["reason"] == "low_confidence"
    assert entry["was"] == "api" and entry["now"] == "billing"
    assert entry["confidence_before"] == pytest.approx(0.31)
    assert entry["confidence_after"] == pytest.approx(0.88)


def test_an_answer_the_target_did_not_return_is_left_alone_and_logged() -> None:
    decisions = routing.escalation_candidates(_answers(), threshold=0.5, max_escalations=2)
    merged, log = routing.apply_escalation(_answers(), {}, decisions, target={"alias": "spark"})
    assert merged == _answers()
    payload = log.to_dict()
    assert all(entry["replaced"] is False for entry in payload["decisions"])
    assert all(entry["was"] == entry["now"] for entry in payload["decisions"])


def test_a_device_budget_that_cannot_hold_the_kv_floor_falls_back_to_the_cpu() -> None:
    """The KV floor is a hard limit: a device that cannot hold it gets no layers at all."""
    facts = {"/m/lean.gguf": _facts("/m/lean.gguf", weights_mib=200)}
    needs = routing.Needs(n_ctx=2048, n_seq_max=4)
    plan = routing.route([_candidate("lean", "/m/lean.gguf")], needs=needs,
                         host=_host(vram_mib=8 * 1024, free_mib=1792),
                         facts_for=lambda path: facts[path], runtime_dirs=[VULKAN],
                         supports_arch=_supports("spark2_5"))
    assert plan.n_gpu_layers == 0
    assert plan.backend == "cpu"
    assert "device budget" in plan.reason


def test_a_candidate_whose_facts_reader_raises_is_merely_unavailable() -> None:
    def reader(path: str):
        if path == "/m/broken.gguf":
            raise ValueError("truncated header")
        return _facts(path, weights_mib=1024)

    candidates = [_candidate("broken", "/m/broken.gguf"), _candidate("good", "/m/good.gguf")]
    plan = routing.route(candidates, needs=routing.Needs(n_ctx=2048, n_seq_max=4),
                         host=_host(vram_mib=8 * 1024), facts_for=reader,
                         runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    assert plan.alias == "good"
    broken = next(step for step in plan.steps if step.alias == "broken")
    assert broken.verdict == "rejected" and "cannot be read" in broken.reason


def test_nothing_available_at_all_is_a_model_not_found(tmp_path) -> None:
    with pytest.raises(ModelNotFoundError) as excinfo:
        routing.route([_candidate("gone", str(tmp_path / "gone.gguf"), available=False)],
                      needs=routing.Needs(n_ctx=1024, n_seq_max=3),
                      host=_host(vram_mib=8 * 1024), facts_for=lambda path: None,
                      runtime_dirs=[VULKAN], supports_arch=_supports("spark2_5"))
    assert "E_MODEL_NOT_FOUND" in str(excinfo.value)


def test_the_default_capability_probe_reads_the_bundle_and_says_no_when_it_cannot(tmp_path) -> None:
    """Without an injected probe the router asks the real runtime bundle (`capability`)."""
    with pytest.raises(ModelArchUnsupportedError):
        routing.route([_candidate("plain", "/m/plain.gguf")],
                      needs=routing.Needs(n_ctx=1024, n_seq_max=3),
                      host=_host(vram_mib=8 * 1024),
                      facts_for=lambda path: _facts(path, weights_mib=1024),
                      runtime_dirs=[str(tmp_path)])    # an empty dir cannot support anything


def test_an_escalation_candidate_can_be_a_score_answer() -> None:
    """A score answer has no `confidence` field — its peak probability stands in for one."""
    answers = {"sev": {"type": "score", "score": 0.6, "reliability": "ok",
                       "probabilities": {"0": 0.3, "1": 0.3, "2": 0.2, "3": 0.2}}}
    decisions = routing.escalation_candidates(answers, threshold=0.5, max_escalations=1)
    assert [decision.question for decision in decisions] == ["sev"]
    assert decisions[0].confidence == pytest.approx(0.3)
    assert decisions[0].reason == "low_confidence"
    merged, log = routing.apply_escalation(answers, {}, decisions, target=None, limit=1)
    assert merged == answers and log.to_dict()["count"] == 0
    assert routing.decision_of(answers["sev"]) == "0"


def test_escalating_with_no_target_logs_the_skip_and_changes_nothing() -> None:
    answers = _answers()
    decisions = routing.escalation_candidates(answers, threshold=0.5, max_escalations=1)
    merged, log = routing.apply_escalation(answers, {}, decisions, target=None)
    payload = log.to_dict()
    assert merged == answers
    assert payload["enabled"] is True and payload["count"] == 0
    assert payload["skipped"][0]["reason"] == "no escalation target available"
    assert payload["skipped"][0]["question"] == "shaky"


# ------------------------------------------------- the CLI surface (A-E2p5-4/5/8)
def _registry_home(tmp_path, entries: dict) -> object:
    from typed_gguf.registry import store
    home = tmp_path / "home"
    aliases = {}
    for alias, weights_mib in entries.items():
        path = tmp_path / f"{alias}.gguf"
        path.write_bytes(b"GGUF" + b"\0" * 16)
        aliases[alias] = store.Entry(alias=alias, path=str(path), quant="Q8_0", arch="spark2_5",
                                     size=weights_mib * MIB)
    store.save_registry(store.Registry(aliases=aliases, current=next(iter(aliases))),
                        path=store.registry_path(home))
    return home


def test_route_request_picks_from_the_registry_and_explains_itself(tmp_path) -> None:
    from typed_gguf import cli, schema
    home = _registry_home(tmp_path, {"small": 1024, "mid": 3 * 1024})
    facts = {str(tmp_path / "small.gguf"): _facts(str(tmp_path / "small.gguf"),
                                                  weights_mib=1024),
             str(tmp_path / "mid.gguf"): _facts(str(tmp_path / "mid.gguf"),
                                                weights_mib=3 * 1024)}
    request = schema.parse_request({"state": "blank dashboard", "options": {"route": "auto"},
                                    "questions": {"a": {"type": "noul"}}})
    plan = cli.route_request(request, home=home, host=_host(vram_mib=8 * 1024),
                             runtime_dirs=[VULKAN], facts_for=lambda path: facts[path],
                             supports_arch=_supports("spark2_5"))
    assert plan is not None
    assert plan.mode == "auto" and plan.alias == "mid"
    assert plan.reason.startswith("budget fit")
    assert plan.to_dict()["steps"][0]["alias"] == "mid"


def test_route_request_is_off_by_default(tmp_path) -> None:
    from typed_gguf import cli, schema
    home = _registry_home(tmp_path, {"small": 1024})
    request = schema.parse_request({"state": "blank dashboard",
                                    "questions": {"a": {"type": "noul"}}})
    assert cli.route_request(request, home=home) is None


def test_the_route_fills_the_options_it_owns_and_leaves_the_rest_alone() -> None:
    from typed_gguf import cli, schema
    plan = routing.RoutePlan(mode="auto", alias="mid", path="/m/mid.gguf", quant="Q8_0",
                             kv_type="q8_0", n_ctx=2048, n_seq_max=6, n_gpu_layers=36,
                             backend="vulkan", device_bytes=1024, reason="budget fit: mid")
    request = schema.parse_request({"state": "x", "options": {"route": "auto"},
                                    "questions": {"a": {"type": "noul"}}})
    updated = cli._with_route_options(request, plan)
    assert updated.options.kv_type == "q8_0"
    assert updated.options.n_ctx == 2048 and updated.options.n_seq_max == 6
    explicit = schema.parse_request({
        "state": "x", "options": {"route": "auto", "kv_type": "f16", "n_ctx": 8192,
                                  "n_seq_max": 3},
        "questions": {"a": {"type": "noul"}}})
    kept = cli._with_route_options(explicit, plan)
    assert kept.options.kv_type == "f16"
    assert kept.options.n_ctx == 2048                  # the router's ceiling still applies
    assert kept.options.n_seq_max == 3


def test_escalation_replaces_only_the_flagged_answers(tmp_path) -> None:
    from typed_gguf import cli
    target = tmp_path / "big.gguf"
    target.write_bytes(b"GGUF")
    payload = {"state": "blank dashboard",
               "options": {"escalate": True, "max_escalations": 1},
               "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}}
    response = {"model": "small",
                "engine": {"runtime": "llama.cpp b11026"},
                "answers": {"a": {"type": "noul", "noul": 0.52, "reliability": "ok",
                                  "probabilities": {"yes": 0.52, "no": 0.48}},
                            "b": {"type": "noul", "noul": 0.55, "reliability": "low_mass",
                                  "probabilities": {"yes": 0.55, "no": 0.45}}},
                "usage": {}, "timings": {}, "warnings": []}
    calls: list[dict] = []

    def fake_decide(sub_payload: dict, **kwargs) -> dict:
        calls.append(sub_payload)
        return {"answers": {"b": {"type": "noul", "noul": 0.93, "reliability": "ok",
                                  "probabilities": {"yes": 0.93, "no": 0.07}}}}

    out = cli.escalate_if_requested(payload, response,
                                    target={"alias": "big", "path": str(target),
                                            "model": "big"},
                                    decide_fn=fake_decide)
    assert list(calls[0]["questions"]) == ["b"]           # only the low_mass answer
    assert calls[0]["model"] == str(target)
    assert out["answers"]["b"]["noul"] == 0.93
    assert out["answers"]["a"]["noul"] == 0.52            # untouched
    escalations = out["engine"]["escalations"]
    assert escalations["enabled"] is True and escalations["count"] == 1
    assert escalations["target"]["alias"] == "big"
    assert escalations["decisions"][0]["question"] == "b"
    assert escalations["decisions"][0]["reason"] == "low_mass"
    assert "W_ESCALATED" in out["warnings"]


def test_escalation_is_off_unless_the_request_asks_for_it() -> None:
    from typed_gguf import cli
    payload = {"state": "x", "questions": {"a": {"type": "noul"}}}
    response = {"model": "small", "engine": {},
                "answers": {"a": {"type": "noul", "noul": 0.52, "reliability": "low_mass",
                                  "probabilities": {"yes": 0.52, "no": 0.48}}},
                "usage": {}, "timings": {}, "warnings": []}
    out = cli.escalate_if_requested(payload, response, target=None, decide_fn=None)
    assert out["engine"]["escalations"]["enabled"] is False
    assert out["engine"]["escalations"]["count"] == 0
    assert out["warnings"] == []


def test_escalation_never_exceeds_the_bound(monkeypatch) -> None:
    from typed_gguf import cli
    payload = {"state": "x", "options": {"escalate": True, "max_escalations": 2},
               "questions": {key: {"type": "noul"} for key in "abcd"}}
    answers = {key: {"type": "noul", "noul": 0.51, "confidence": 0.4, "reliability": "ok",
                     "probabilities": {"yes": 0.51, "no": 0.49}} for key in "abcd"}
    response = {"model": "small", "engine": {}, "answers": answers, "usage": {}, "timings": {},
                "warnings": []}

    def fake_decide(sub_payload: dict, **kwargs) -> dict:
        return {"answers": {key: {"type": "noul", "noul": 0.8, "confidence": 0.8,
                                  "reliability": "ok",
                                  "probabilities": {"yes": 0.8, "no": 0.2}}
                            for key in sub_payload["questions"]}}

    out = cli.escalate_if_requested(payload, response, target={"alias": "big", "path": "/m/big"},
                                    decide_fn=fake_decide)
    assert out["engine"]["escalations"]["count"] == 2
    assert out["engine"]["escalations"]["limit"] == 2


def test_the_audit_log_records_the_route_the_calibration_and_the_escalations(tmp_path) -> None:
    import json

    from typed_gguf import cli
    payload = {"state": "x", "questions": {"a": {"type": "noul"}}}
    response = {"model": "mid", "engine": {"route": {"reason": "budget fit: mid", "alias": "mid"},
                                           "escalations": {"enabled": False, "count": 0}},
                "calibration": {"source": "/home/calibration.json", "applied": True},
                "answers": {}, "usage": {}, "timings": {}, "warnings": ["W_LOW_MASS"]}
    path = cli.write_audit(tmp_path / "audit", command="run", payload=payload,
                           response=response)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["route"]["reason"] == "budget fit: mid"
    assert records[-1]["calibration"]["source"] == "/home/calibration.json"
    assert records[-1]["escalations"]["enabled"] is False
    assert records[-1]["model"] == "mid"
    assert records[-1]["command"] == "run"
    assert records[-1]["ts"]
