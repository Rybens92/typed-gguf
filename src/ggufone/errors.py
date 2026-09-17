"""Exception hierarchy + the frozen error/warning catalog (SPEC 2.5).

Milestone: E1a (codes used by registry/runtime) and E1b (schema codes).
"""

from __future__ import annotations


class GgufoneError(Exception):
    """Base class. `code` is one of the E_* constants from SPEC 2.5."""

    code = "E_INTERNAL"
    exit_code = 4

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class UserError(GgufoneError):
    """Bad request / bad input (exit code 2)."""

    exit_code = 2


class RuntimeError_(GgufoneError):
    """Runtime or model problem (exit code 3)."""

    exit_code = 3


# Codes are frozen by SPEC 2.5; implementations must use exactly these strings.
ERROR_CODES = (
    "E_UNKNOWN_KEY", "E_STATE_EMPTY", "E_QID_INVALID", "E_Q_TYPE_UNKNOWN",
    "E_CHOICE_CRITERIA", "E_CHOICE_TOO_MANY", "E_SCORE_LEVELS", "E_NOUL_CRITERIA",
    "E_CANDIDATE_COLLISION", "E_MODEL_NOT_FOUND", "E_MODEL_ARCH_UNSUPPORTED",
    "E_RUNTIME_MISSING", "E_RUNTIME_SYMBOLS", "E_RUNTIME_BUILD_OLD", "E_CTX_TOO_SMALL",
    "E_SEQ_MAX_EXCEEDED", "E_PREFILL_FAILED", "E_DECODE_FAILED", "E_GGUF_CORRUPT",
    "E_SHA256_MISMATCH", "E_DOWNLOAD_FAILED", "E_AMBIGUOUS_QUANT", "E_TEMPLATE_UNRESOLVED",
)
WARNING_CODES = (
    "W_LOW_MASS", "W_LOW_CONFIDENCE", "W_UNKNOWN_OPTION", "W_TRUNCATED_STATE",
    "W_KV_TYPE_DOWNGRADE", "W_VULKAN_WARMUP", "W_TEMPLATE_FALLBACK", "W_FIT_ESTIMATED",
)
