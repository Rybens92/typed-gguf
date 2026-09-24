"""Differential-fuzz the 'equivalent' survivors: mutant vs original on random inputs.

Artifact-based triage (the run tree holds every variant as a callable); asserting equality over
thousands of random inputs is stronger evidence than reading the diff.
"""
import importlib.util
import pathlib
import random
import sys

REPO = pathlib.Path("/var/home/rybens/workspace/ggufone")
sys.path.insert(0, str(REPO / "src"))

from ggufone.engine import readout as orig  # noqa: E402

MUTANTS = REPO / "mutants" / "src" / "ggufone" / "engine" / "readout.py"
spec = importlib.util.spec_from_file_location("mutant_readout", MUTANTS)
mutant = importlib.util.module_from_spec(spec)
sys.modules["mutant_readout"] = mutant
spec.loader.exec_module(mutant)

rng = random.Random(20260917)
cases = [[rng.uniform(-30, 30) for _ in range(rng.randint(1, 40))] for _ in range(4000)]
probs = [[abs(v) / 40 for v in case] for case in cases]

checks = []


def worst_diff(a, b) -> float:
    """Recursive comparison of floats / lists of floats."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((worst_diff(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b)
    return 0.0 if a == b else float("inf")


def compare(label, name_mutant, name_orig, call, data):
    worst = 0.0
    for item in data:
        a = call(getattr(mutant, name_mutant), item)
        b = call(getattr(orig, name_orig), item)
        worst = max(worst, worst_diff(a, b))
    checks.append((label, name_mutant, worst))


compare("softmax shift-invariance (T=1)", "x_softmax__mutmut_14", "softmax",
        lambda fn, case: fn(case, temperature=1.0), cases)
compare("softmax shift-invariance (T=0.25)", "x_softmax__mutmut_14", "softmax",
        lambda fn, case: fn(case, temperature=0.25), cases)
compare("softmax default temperature", "x_restricted_softmax__mutmut_5", "restricted_softmax",
        lambda fn, case: fn(case), cases)
compare("argmax redundant self-comparison", "x_argmax_first__mutmut_10", "argmax_first",
        lambda fn, case: float(fn(case)), cases)
compare("round_sig default digits", "x_score_weighted_mean__mutmut_4", "score_weighted_mean",
        lambda fn, case: fn([abs(v) / 40 for v in case]), probs)
compare("clamp upper bound (valid probs)", "x__clamp01__mutmut_10", "_clamp01",
        lambda fn, case: fn(min(1.0, abs(sum(case)) / 1200.0)), cases)

for label, name, worst in checks:
    verdict = "EQUIVALENT on 4000 random cases" if worst == 0.0 else f"DIFFERS (max {worst:.3e})"
    print(f"{name:38s} {label:34s} {verdict}")
