"""The keep home's socket path: short where the gates build it, refused by name where it is not.

Card t_c3195a5c. `keep` binds `<data-home>/keep/<digest>.sock`, and the product refuses a path that
cannot fit in `sun_path` (108 bytes including the NUL) — `typed_gguf.keep.state.socket_path`, with
the "set a shorter TYPED_GGUF_HOME" hint. pytest's `tmp_path` inherits the *ambient* `TMPDIR`, and
agent/session environments point that at a long profile-scratch path, so a home built from it
pushed the socket over the limit and turned **every** keep gate red on a box where the product was
right (measured on this host: 35 red of 93, every one of them `E_UNKNOWN_KEY`, all in
`tests/test_keep*.py`).

Both halves are pinned here: the gate fixture's home carries a socket the kernel accepts whatever
`TMPDIR` the session runs under, and the product's refusal — hint included — still fires for a
genuinely long `TYPED_GGUF_HOME`. The fixture machinery itself (`tests/conftest.py`,
`keep_home_that_binds`) is pinned with the long base the card measured, with the fallback that
still runs the gates on a short ambient base, and with the named skip when there is no short base
at all.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import socket
import tempfile

import pytest

from tests import conftest as suite_environment
from tests.conftest import keep_home_that_binds, socket_fits
from typed_gguf.errors import UserError
from typed_gguf.keep import identity, state
from typed_gguf.registry import store


def _key() -> identity.KeepKey:
    return identity.KeepKey.of(None, model_path="/models/a.gguf", model_sha="deadbeef")


def _bind(path: pathlib.Path) -> None:
    """Bind it for real: the kernel, not a length formula, is the `sun_path` authority."""
    with contextlib.closing(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)) as probe:
        probe.bind(str(path))


def _long_base(tmp_path: pathlib.Path, marker: str) -> pathlib.Path:
    """A Hermes profile-scratch `TMPDIR`'s shape: too long for a socket under it to survive."""
    base = tmp_path / (f"{marker}-" + "x" * 90)
    assert not socket_fits(base / "home"), "the pin needs a base that genuinely cannot bind"
    return base


def _home_whose_socket_is(base: pathlib.Path, socket_bytes: int) -> pathlib.Path:
    """A home under `base` whose `keep/<digest>.sock` path is exactly `socket_bytes` long.

    What a home adds to its own socket path is the `keep/` segment plus the socket's file name —
    both constant for a given digest — so the home's own length is the free variable.
    """
    fixed = len("/keep/") + len(f"{_key().digest}.sock")
    room = socket_bytes - fixed - len(os.fsencode(str(base))) - 1
    if room <= 0:
        pytest.skip(f"{base} is too long to build a {socket_bytes}-byte home under it")
    return base / ("b" * room)


def test_the_gate_home_carries_a_socket_the_kernel_accepts(keep_home: pathlib.Path,
                                                           tmp_path: pathlib.Path) -> None:
    """RED-first: under a long `TMPDIR` this raised the product's own `sun_path` refusal.

    The gate home is not allowed to inherit the ambient base's length — whatever `tmp_path` is,
    `socket_path` has to hand back a path the kernel accepts for the gate that owns it.
    """
    path = state.socket_path(keep_home, _key().digest)
    assert len(os.fsencode(path)) + 1 <= state.SUN_PATH_MAX
    state.ensure_dir(keep_home)                     # the ledger's own dir, as a spawn would make it
    _bind(path)
    if not socket_fits(tmp_path / "home"):
        # …and this is the world the card fixes: pytest's own base cannot host the socket, so a
        # home under it false-failed every keep gate. The gate home is deliberately not under it.
        assert keep_home != tmp_path / "home"


def _base_to_offer() -> pathlib.Path | None:
    """A fresh short base for these pins to offer the machinery — or None when this box has none.

    Deliberately *not* `short_socket_base()`: the code under test must not be able to turn a pin
    into a skip, because "the helper cannot find a perfectly good base" is precisely the failure
    the pin exists to catch.
    """
    for candidate in suite_environment.SHORT_SOCKET_BASES:
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return pathlib.Path(tempfile.mkdtemp(prefix="tg-base-", dir=str(candidate)))
    return None


def test_the_home_sits_under_the_configured_base_not_the_ambient_one(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A short base the machinery is told to use is the one it uses — and it can bind there.

    The sweep's own hole: when `short_socket_base()` (or `socket_fits()`) is broken, the machinery
    falls back to the ambient base and *skips* — a silent degradation from "90 gates ran" to "90
    gates skipped", which no negative-case pin can see. This one asserts the positive branch.
    """
    offered = _base_to_offer()
    if offered is None:
        pytest.skip("no writable short base on this box: this pin has to offer one")
    monkeypatch.setattr(suite_environment, "SHORT_SOCKET_BASES", (offered,))
    try:
        with keep_home_that_binds(tmp_path) as home:
            assert offered in home.parents and home.is_dir()
            assert socket_fits(home)                 # the offered base really is short enough
    finally:
        shutil.rmtree(offered, ignore_errors=True)


def test_a_candidate_that_cannot_be_a_base_is_stepped_over(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A candidate that cannot be a base is stepped over, and the next one is used.

    `tmp_path / "not-a-dir"` is a regular file: `mkdir` on it raises for any user, so no box can
    call it a base. The helper has to `continue` past it — `break`ing there would hand the gate to
    the ambient base, which is the very thing this card is about.
    """
    good = _base_to_offer()
    if good is None:
        pytest.skip("no writable short base on this box: this pin has to offer one")
    unusable = tmp_path / "not-a-dir"
    unusable.write_text("not a base\n", encoding="utf-8")
    monkeypatch.setattr(suite_environment, "SHORT_SOCKET_BASES", (unusable, good))
    try:
        with keep_home_that_binds(tmp_path) as home:
            assert good in home.parents and home.is_dir()
            assert socket_fits(home)
    finally:
        shutil.rmtree(good, ignore_errors=True)


def test_a_home_under_a_deliberately_long_base_still_carries_a_socket(
        tmp_path: pathlib.Path) -> None:
    """The machinery, driven with the base that used to decide it: the socket still binds.

    A gate home is only as good as the base it is built on, so this runs `keep_home_that_binds`
    with the profile-scratch shape the card measured the refusal under. With no short base at all
    `/tmp` it skips by name instead — the documented skip, never the product's refusal.
    """
    long_base = _long_base(tmp_path, "ambient-scratch")
    with keep_home_that_binds(long_base) as home:
        path = state.socket_path(home, _key().digest)
        assert len(os.fsencode(path)) + 1 <= state.SUN_PATH_MAX
        state.ensure_dir(home)
        _bind(path)


def test_without_a_short_base_a_short_ambient_base_is_still_used(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """No `/tmp` is not a reason to skip: a short ambient base hosts the gate exactly as before."""
    monkeypatch.setattr(suite_environment, "SHORT_SOCKET_BASES", ())
    if not socket_fits(tmp_path / "home"):
        pytest.skip("this pin drives the no-short-base branch, so its ambient base has to be short "
                    f"of itself — this session's own ({tmp_path}) cannot carry the socket")
    with keep_home_that_binds(tmp_path) as home:
        assert home == tmp_path / "home" and home.is_dir()


def test_without_a_short_base_a_long_ambient_base_skips_by_name(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The box cannot host the gate: a skip that says why, never the product's refusal as a fail."""
    monkeypatch.setattr(suite_environment, "SHORT_SOCKET_BASES", ())
    long_base = _long_base(tmp_path, "no-short-base")
    with pytest.raises(pytest.skip.Exception) as caught, keep_home_that_binds(long_base):
        raise AssertionError("the machinery yielded a home it cannot bind a socket in")
    reason = str(caught.value)
    assert ("no short writable base" in reason and "sun_path" in reason
            and "shorter TMPDIR" in reason)


def test_the_refusal_starts_exactly_one_byte_past_the_kernel_limit(
        keep_home: pathlib.Path) -> None:
    """`sun_path` is 108 bytes *including* the NUL: 107 fits (and binds), 108 is refused.

    The product's rule is a boundary, so both sides of it are pinned: a home one byte under the
    limit gets a path the kernel accepts, and one byte over gets the refusal — plus the hint —
    instead of a path truncated into someone else's socket (card t_c3195a5c; the two hand mutants
    of this boundary, `>=` and the dropped `+ 1`, survived the gates before this pin).
    """
    base = keep_home.parent                       # the short base the gate home was built under
    digest = _key().digest
    fits = _home_whose_socket_is(base, state.SUN_PATH_MAX - 1)
    path = state.socket_path(fits, digest)
    assert len(os.fsencode(path)) == state.SUN_PATH_MAX - 1
    state.ensure_dir(fits)
    _bind(path)
    over = _home_whose_socket_is(base, state.SUN_PATH_MAX)
    with pytest.raises(UserError) as caught:
        state.socket_path(over, digest)
    assert "set a shorter TYPED_GGUF_HOME" in str(caught.value)


def test_a_genuinely_long_data_home_is_still_refused_by_name(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: pathlib.Path) -> None:
    """The fix must not touch the product: a deep home is still refused, hint and all."""
    deep = tmp_path / ("deep-" + "x" * 96) / "data-home"
    monkeypatch.setenv("TYPED_GGUF_HOME", str(deep))
    home = store.data_home()                            # the env is the home the product reads
    assert home == deep and len(os.fsencode(home)) > state.SUN_PATH_MAX
    with pytest.raises(UserError) as caught:
        state.socket_path(home, _key().digest)
    message = str(caught.value)
    assert "sun_path" in message and str(state.SUN_PATH_MAX) in message
    assert "set a shorter TYPED_GGUF_HOME" in message   # the hint is part of the contract
