"""typed-gguf — GGUF-native typed decision engine (Jev-like, no fine-tuning).

state + typed questions -> typed answers with probabilities/confidence.
One prefill over a shared prefix, one fork per question, no text generation,
no fine-tuning. See SPEC.md.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
