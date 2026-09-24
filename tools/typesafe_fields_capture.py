"""Capture the TypeSafe SDK's wire field lists into a committed fixture (SPEC 2.9, A-E5-1).

"no invented fields is a test, not a promise": `tests/test_serve.py` validates every served key set
against `tests/fixtures/typesafe_sdk_0_7_1_fields.json`, and this script is where that fixture comes
from — the *installed* SDK, read through `importlib`/pydantic, never typed in by hand.

It must run under a Python that has the SDK, e.g. the host gate's venv::

    python3 -m venv /tmp/ts-venv && /tmp/ts-venv/bin/pip install 'typesafe-sdk==0.7.1'
    /tmp/ts-venv/bin/python tools/typesafe_fields_capture.py            # prints the JSON
    /tmp/ts-venv/bin/python tools/typesafe_fields_capture.py --out tests/fixtures/<name>.json

The tool imports only the SDK (never `typed_gguf`), so the fixture can be regenerated on any box
with a different SDK version. `--require-version` makes a version mismatch a loud failure instead of
a silently regenerated fixture: the committed pin is for exactly one SDK release.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import typing
from typing import Any

#: The release the committed fixture is pinned to (SPEC 2.9 is written against it [sdk-0.7.1]).
PINNED_VERSION = "0.7.1"
#: The generated wire models (`typesafe_sdk._schemas.models`) and their order in the OpenAPI file.
WIRE_MODELS = (
    "ChoiceAnswer", "ChoiceQuestion", "ModelMetadata", "ModelMetadataList", "NoulAnswer",
    "NoulCriteria", "NoulQuestion", "ScoreAnswer", "ScoreQuestion", "Usage", "ValidationError",
    "HTTPValidationError", "Question", "Answer", "SystemOneRequest", "SystemOneResponse",
)
#: The per-answer-type key sets the SDK *validates against* (the dedicated pydantic subclasses in
#: `typesafe_sdk._core.response_types`, which are what `SystemOneResponse.answers` really parses).
ANSWER_MODELS = {"noul": "NoulAnswer", "choice": "ChoiceAnswer", "score": "ScoreAnswer"}
#: The constants a server has to agree with (paths, defaults, header names, secret redaction).
CONSTANTS = (
    ("typesafe_sdk._core.constants", ("SYSTEM_ONE_PATH", "MODELS_PATH", "JSON_CONTENT_TYPE",
                                      "AUTHORIZATION_HEADER", "ACCEPT_HEADER",
                                      "CONTENT_TYPE_HEADER", "USER_AGENT_HEADER", "SDK_HEADER",
                                      "RUNTIME_HEADER", "RETRY_COUNT_HEADER", "REQUEST_ID_HEADER",
                                      "SDK_NAME")),
    ("typesafe_sdk.constants", ("API_KEY_ENV", "BASE_URL_ENV", "DEFAULT_MODEL_ENV",
                                "DEFAULT_BASE_URL", "DEFAULT_MODEL", "DEFAULT_TIMEOUT")),
)


def _fields(model: type) -> list[str]:
    return list(model.model_fields)


def _union_members(annotated: Any) -> tuple[type, ...]:
    """The member classes of an `Annotated[Union[...], Field(...)]` alias (in declaration order)."""
    inner = typing.get_args(annotated)[0]
    return tuple(arg for arg in typing.get_args(inner) if isinstance(arg, type))


def capture() -> dict[str, Any]:
    """The fixture: every field list, path and default the served wire stands on."""
    sdk = importlib.import_module("typesafe_sdk")
    wire = importlib.import_module("typesafe_sdk._schemas.models")
    answers = importlib.import_module("typesafe_sdk._core.response_types")
    retry = importlib.import_module("typesafe_sdk._core.retry")
    errors = importlib.import_module("typesafe_sdk._core.errors")

    constants: dict[str, dict[str, Any]] = {}
    for module_name, names in CONSTANTS:
        module = importlib.import_module(module_name)
        constants[module_name] = {name: getattr(module, name) for name in names}

    policy = retry.RetryPolicy()
    return {
        "sdk": {"name": "typesafe-sdk", "version": str(sdk.__version__),
                "captured_by": "tools/typesafe_fields_capture.py"},
        "_provenance": {
            "note": ("Field lists read from the installed SDK's pydantic models "
                     "(`typesafe_sdk._schemas.models` and the `_core.response_types` answer "
                     "classes), the endpoint builders' constants, `RetryPolicy`'s defaults and "
                     "`STATUS_ERROR_TYPES`. Regenerate with this script; never hand-edit."),
            "spec": "SPEC.md 2.6 (adapter) + 2.9 (HTTP surface, tagged [sdk-0.7.1])",
            "fixture_pin": PINNED_VERSION,
        },
        "constants": constants,
        "wire_models": {name: _fields(getattr(wire, name)) for name in WIRE_MODELS},
        "answers": {kind: _fields(getattr(answers, model))
                    for kind, model in ANSWER_MODELS.items()},
        "answer_discriminator": {
            # `Answer` is an `Annotated[... , Field(discriminator="type")]` union, not a pydantic
            # model: record the union's member names and the tag the SDK dispatches on.
            "field": "type",
            "members": [member.__name__ for member in _union_members(answers.Answer)],
        },
        "retry_policy": {
            "max_retries": policy.max_retries,
            "timeout_s": policy.timeout,
            "http_statuses": sorted(policy.http_statuses),
            "backoff_initial": policy.backoff_initial,
            "backoff_max": policy.backoff_max,
        },
        "status_error_types": {str(status): kind.__name__
                               for status, kind in errors.STATUS_ERROR_TYPES.items()},
        "internal_server_status": "any other 5xx -> TypeSafeInternalServerError",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="write the fixture here (default: stdout)")
    parser.add_argument("--require-version", default=PINNED_VERSION,
                        help="fail unless the installed SDK is this version ('' disables)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        payload = capture()
    except ModuleNotFoundError as exc:      # the SDK is not importable in this interpreter
        print(f"error: {exc.name} is not installed — run this with the SDK's own interpreter "
              f"(see the module docstring)", file=sys.stderr)
        return 2
    version = payload["sdk"]["version"]
    if args.require_version and version != args.require_version:
        print(f"error: the installed typesafe-sdk is {version}, the committed fixture pins "
              f"{args.require_version}: install the pinned release, or regenerate deliberately "
              f"with --require-version ''", file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} (typesafe-sdk {version})")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
