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


# --- named errors for the codes this milestone uses (SPEC 2.5 catalog + E1a additions) ---
class ModelNotFoundError(UserError):
    code = "E_MODEL_NOT_FOUND"


class AmbiguousQuantError(UserError):
    code = "E_AMBIGUOUS_QUANT"


class GgufCorruptError(UserError):
    code = "E_GGUF_CORRUPT"


class HfAuthError(UserError):
    code = "E_HF_AUTH_REQUIRED"


class InsufficientDiskError(UserError):
    code = "E_INSUFFICIENT_DISK"


class RegistryCorruptError(UserError):
    code = "E_REGISTRY_CORRUPT"


class RuntimeMissingError(RuntimeError_):
    code = "E_RUNTIME_MISSING"


class RuntimeSymbolsError(RuntimeError_):
    code = "E_RUNTIME_SYMBOLS"


class RuntimeBuildOldError(RuntimeError_):
    code = "E_RUNTIME_BUILD_OLD"


class ModelArchUnsupportedError(RuntimeError_):
    code = "E_MODEL_ARCH_UNSUPPORTED"


class DownloadError(RuntimeError_):
    code = "E_DOWNLOAD_FAILED"


class Sha256MismatchError(RuntimeError_):
    code = "E_SHA256_MISMATCH"


# --- E1b: engine-side codes (SPEC 2.5 catalog + the E_STATE_LOAD_FAILED addition) ---
class CandidateCollisionError(UserError):
    """Two candidates of one question score the same token sequence (exit 2)."""

    code = "E_CANDIDATE_COLLISION"


class ContextTooSmallError(RuntimeError_):
    code = "E_CTX_TOO_SMALL"


class SeqMaxExceededError(RuntimeError_):
    code = "E_SEQ_MAX_EXCEEDED"


class PrefillFailedError(RuntimeError_):
    code = "E_PREFILL_FAILED"


class DecodeFailedError(RuntimeError_):
    code = "E_DECODE_FAILED"


class StateLoadFailedError(RuntimeError_):
    """A saved prefix state could not be loaded (corrupt/truncated) — A-E1b-8.

    Not in the SPEC 2.5 catalog: added in E1b because the acceptance criterion requires a
    pinned code, and the cache must be invalidated instead of crashing.
    """

    code = "E_STATE_LOAD_FAILED"


# Codes are frozen by SPEC 2.5; implementations must use exactly these strings.
ERROR_CODES = (
    "E_UNKNOWN_KEY", "E_STATE_EMPTY", "E_QID_INVALID", "E_Q_TYPE_UNKNOWN",
    "E_CHOICE_CRITERIA", "E_CHOICE_TOO_MANY", "E_SCORE_LEVELS", "E_NOUL_CRITERIA",
    "E_CANDIDATE_COLLISION", "E_MODEL_NOT_FOUND", "E_MODEL_ARCH_UNSUPPORTED",
    "E_RUNTIME_MISSING", "E_RUNTIME_SYMBOLS", "E_RUNTIME_BUILD_OLD", "E_CTX_TOO_SMALL",
    "E_SEQ_MAX_EXCEEDED", "E_PREFILL_FAILED", "E_DECODE_FAILED", "E_GGUF_CORRUPT",
    "E_SHA256_MISMATCH", "E_DOWNLOAD_FAILED", "E_AMBIGUOUS_QUANT", "E_TEMPLATE_UNRESOLVED",
    # E1a additions (coordinator completeness pass, operator-approved):
    "E_HF_AUTH_REQUIRED",   # gated/private HF repo without a usable token
    "E_INSUFFICIENT_DISK",  # download precheck: required bytes > free bytes
    "E_REGISTRY_CORRUPT",   # registry.json unreadable (quarantined, never silently lost)
    # E1b addition (A-E1b-8 requires a pinned code for a corrupt/truncated state file):
    "E_STATE_LOAD_FAILED",
)
WARNING_CODES = (
    "W_LOW_MASS", "W_LOW_CONFIDENCE", "W_UNKNOWN_OPTION", "W_TRUNCATED_STATE",
    "W_KV_TYPE_DOWNGRADE", "W_VULKAN_WARMUP", "W_TEMPLATE_FALLBACK", "W_FIT_ESTIMATED",
)
