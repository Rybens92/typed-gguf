"""Registry store: XDG paths, aliases, atomic writes, corrupt-file recovery (A-E1a-6)."""
from __future__ import annotations

import json
import pathlib

import pytest

from ggufone.errors import GgufoneError
from ggufone.registry import store


@pytest.fixture(autouse=True)
def _home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    home = tmp_path / "ggufone-home"
    monkeypatch.setenv("GGUFONE_HOME", str(home))
    return home


def entry(alias: str = "spark", **over: object) -> store.Entry:
    base: dict[str, object] = {
        "alias": alias,
        "path": "/models/Spark-X2.5-4B-Q8_0.gguf",
        "sha256": "5c2c3c19" + "0" * 56,
        "arch": "spark2_5",
        "quant": "Q8_0",
        "size": 4_375_021_152,
        "license": "apache-2.0",
        "source": "XHToken/Spark-X2.5-4B-GGUF",
        "added_at": "2026-09-17T12:00:00Z",
        "fit_plan": None,
    }
    base.update(over)
    return store.Entry(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ paths
def test_data_home_honours_ggufone_home(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("GGUFONE_HOME", str(tmp_path / "custom"))
    assert store.data_home() == tmp_path / "custom"
    assert store.models_dir() == tmp_path / "custom" / "models"
    assert store.registry_path() == tmp_path / "custom" / "registry.json"
    assert store.runtime_root() == tmp_path / "custom" / "runtime"
    assert store.runtime_record_path() == tmp_path / "custom" / "runtime.json"
    assert store.states_dir() == tmp_path / "custom" / "states"
    assert store.downloads_dir() == tmp_path / "custom" / "downloads"


def test_data_home_falls_back_to_xdg(monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: pathlib.Path) -> None:
    monkeypatch.delenv("GGUFONE_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert store.data_home() == tmp_path / "xdg" / "ggufone"


def test_data_home_expands_tilde(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GGUFONE_HOME", "~/ggufone-data")
    assert store.data_home() == tmp_path / "ggufone-data"


# ------------------------------------------------------------------ roundtrip
def test_save_then_load_roundtrip() -> None:
    reg = store.Registry(current="spark")
    reg.aliases["spark"] = entry()
    path = store.save_registry(reg)
    assert path == store.registry_path() and path.exists()
    got, warnings = store.load_registry()
    assert warnings == []
    assert got.current == "spark"
    assert got.aliases["spark"] == entry()
    assert got.aliases["spark"].license == "apache-2.0"


def test_save_is_atomic_and_leaves_no_temp_files() -> None:
    store.save_registry(store.Registry())
    leftovers = [p.name for p in store.data_home().iterdir()
                 if "tmp" in p.name or p.name.endswith(".part")]
    assert leftovers == []
    payload = json.loads(store.registry_path().read_text())
    assert payload["schema"] == store.SCHEMA
    assert payload["aliases"] == {}


def test_load_missing_registry_is_empty_not_an_error() -> None:
    got, warnings = store.load_registry()
    assert got.aliases == {} and got.current is None and warnings == []


def test_add_and_remove_entry() -> None:
    reg = store.Registry()
    store.add_entry(reg, entry("spark"))
    store.add_entry(reg, entry("qwen", path="/models/q.gguf", quant="Q4_K_M"))
    store.save_registry(reg)
    reg2, _ = store.load_registry()
    assert set(reg2.aliases) == {"spark", "qwen"}
    assert reg2.current == "spark"  # first entry becomes the current alias
    removed = store.remove_entry(reg2, "spark")
    store.save_registry(reg2)
    assert removed.alias == "spark"
    assert set(store.load_registry()[0].aliases) == {"qwen"}


def test_add_entry_alias_collision_gets_a_suffix() -> None:
    reg = store.Registry()
    first = store.add_entry(reg, entry("spark"))
    second = store.add_entry(reg, entry("spark", path="/models/other.gguf", quant="Q4_K_M"))
    assert first.alias == "spark" and second.alias == "spark-2"


def test_remove_unknown_alias_is_user_error() -> None:
    reg = store.Registry()
    with pytest.raises(GgufoneError) as exc:
        store.remove_entry(reg, "nope")
    assert exc.value.code == "E_MODEL_NOT_FOUND"


def test_slugify_makes_a_stable_alias() -> None:
    assert store.slugify("Spark-X2.5-4B-Q8_0.gguf") == "spark-x2.5-4b-q8_0"
    assert store.slugify("Model GGUF (v2).gguf") == "model-gguf-v2"
    assert store.slugify("!!!") == "model"


def test_resolve_by_alias_or_path() -> None:
    reg = store.Registry()
    store.add_entry(reg, entry("spark"))
    assert store.resolve(reg, "spark").alias == "spark"
    assert store.resolve(reg, "/models/Spark-X2.5-4B-Q8_0.gguf").alias == "spark"
    assert store.resolve(reg, "missing") is None
    assert store.resolve(reg, None) is None


def test_current_alias_is_used_when_no_model_is_given() -> None:
    reg = store.Registry()
    store.add_entry(reg, entry("spark"))
    store.add_entry(reg, entry("qwen", path="/models/q.gguf"))
    reg.current = "qwen"
    assert store.resolve(reg, None, use_current=True).alias == "qwen"


# ------------------------------------------------------------------ corruption
def test_corrupt_registry_is_quarantined_without_data_loss() -> None:
    path = store.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")
    reg, warnings = store.load_registry()
    assert reg.aliases == {}
    assert len(warnings) == 1
    assert "E_REGISTRY_CORRUPT" in warnings[0]
    quarantined = list(path.parent.glob("registry.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == "{ this is not json"
    assert not path.exists()  # next write starts clean


def test_corrupt_registry_strict_mode_raises() -> None:
    path = store.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")
    with pytest.raises(GgufoneError) as exc:
        store.load_registry(recover=False)
    assert exc.value.code == "E_REGISTRY_CORRUPT"
    assert path.exists()  # strict mode must not touch the file


def test_registry_with_wrong_shape_is_treated_as_corrupt() -> None:
    path = store.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "other", "aliases": "nope"}))
    reg, warnings = store.load_registry()
    assert reg.aliases == {} and warnings


def test_registry_entry_with_missing_fields_is_dropped_with_warning() -> None:
    path = store.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": store.SCHEMA,
        "current": "ok",
        "aliases": {"ok": entry("ok").to_dict(), "bad": {"alias": "bad"}},
    }))
    reg, warnings = store.load_registry()
    assert set(reg.aliases) == {"ok"}
    assert any("bad" in w for w in warnings)
