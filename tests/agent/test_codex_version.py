"""Tests for the codex CLI version resolver (agent.codex_version).

The resolution precedence is: fresh weekly cache, else a live npm fetch
(cached on success), else the installed codex CLI *if* it meets the floor,
else the hard-coded floor. Every tier fails soft to the next and the public
entry point never raises. These tests mock the npm fetch and the installed
CLI query, they never touch the real network.
"""

from __future__ import annotations

import json
import time

import pytest

import agent.codex_version as cv


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    """Isolate each test: clear the in-process memo and point the cache at
    a fresh tmp ``HERMES_HOME`` so tests never share disk state."""
    monkeypatch.setattr(cv, "_memo", None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


def _write_cache(tmp_path, version, age_seconds):
    path = tmp_path / ".cache" / "codex_version.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": version, "fetched_at": time.time() - age_seconds}
        ),
        encoding="utf-8",
    )
    return path


def test_npm_success_returns_fetched(monkeypatch):
    """A live npm fetch wins and is returned verbatim."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: "0.149.1")
    monkeypatch.setattr(
        cv,
        "_query_installed_version",
        lambda _bin: pytest.fail("installed CLI must not be queried"),
    )
    assert cv.get_codex_cli_version() == "0.149.1"


def test_npm_success_is_cached_to_disk(monkeypatch, tmp_path):
    """A successful fetch is persisted so the next call skips the network."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: "0.149.1")
    assert cv.get_codex_cli_version() == "0.149.1"
    cached = json.loads(
        (tmp_path / ".cache" / "codex_version.json").read_text("utf-8")
    )
    assert cached["version"] == "0.149.1"
    assert isinstance(cached["fetched_at"], (int, float))


def test_network_fail_uses_fresh_cache(monkeypatch, tmp_path):
    """Network down but a fresh cache present, return the cached value
    without ever calling the fetch."""
    _write_cache(tmp_path, "0.149.1", age_seconds=60)
    monkeypatch.setattr(
        cv,
        "_fetch_latest_from_npm",
        lambda: pytest.fail("must not fetch when cache is fresh"),
    )
    assert cv.get_codex_cli_version() == "0.149.1"


def test_stale_cache_is_ignored(monkeypatch, tmp_path):
    """A cache older than the TTL is bypassed and the fetch runs."""
    _write_cache(tmp_path, "0.148.0", age_seconds=cv._CACHE_TTL_SECONDS + 1)
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: "0.149.1")
    assert cv.get_codex_cli_version() == "0.149.1"


def test_network_fail_no_cache_uses_installed_at_or_above_floor(monkeypatch):
    """No cache, network down, installed CLI >= floor, use installed."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: None)
    monkeypatch.setattr(
        cv, "_query_installed_version", lambda _bin: "0.150.0"
    )
    assert cv.get_codex_cli_version() == "0.150.0"


def test_installed_below_floor_is_ignored(monkeypatch):
    """An installed CLI older than the floor is ignored, fall to floor."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: None)
    monkeypatch.setattr(
        cv, "_query_installed_version", lambda _bin: "0.148.9"
    )
    assert cv.get_codex_cli_version() == cv._FLOOR_CODEX_CLI_VERSION


def test_installed_equal_to_floor_is_used(monkeypatch):
    """Installed exactly at the floor counts as meeting it."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: None)
    monkeypatch.setattr(
        cv,
        "_query_installed_version",
        lambda _bin: cv._FLOOR_CODEX_CLI_VERSION,
    )
    assert cv.get_codex_cli_version() == cv._FLOOR_CODEX_CLI_VERSION


def test_all_fail_returns_floor(monkeypatch):
    """No cache, no network, no installed CLI, return the floor."""
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: None)
    monkeypatch.setattr(cv, "_query_installed_version", lambda _bin: None)
    assert cv.get_codex_cli_version() == cv._FLOOR_CODEX_CLI_VERSION


def test_result_is_memoized(monkeypatch):
    """Repeated calls in one process resolve once, later calls hit memo."""
    calls = {"n": 0}

    def _one_shot():
        calls["n"] += 1
        return "0.149.1"

    monkeypatch.setattr(cv, "_fetch_latest_from_npm", _one_shot)
    assert cv.get_codex_cli_version() == "0.149.1"
    assert cv.get_codex_cli_version() == "0.149.1"
    assert calls["n"] == 1


def test_never_raises_on_internal_error(monkeypatch):
    """A resolver failure is swallowed and falls back to the floor."""

    def _boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cv, "_resolve", _boom)
    assert cv.get_codex_cli_version() == cv._FLOOR_CODEX_CLI_VERSION


def test_fetch_parses_npm_version_field(monkeypatch):
    """_fetch_latest_from_npm reads the ``version`` field from the JSON
    payload and normalizes to MAJOR.MINOR.PATCH."""

    class _Resp:
        status_code = 200

        def json(self):
            return {"version": "0.149.1"}

    class _FakeHttpx:
        @staticmethod
        def get(url, timeout):
            assert "registry.npmjs.org" in url
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "httpx", _FakeHttpx)
    assert cv._fetch_latest_from_npm() == "0.149.1"


def test_fetch_non_200_returns_none(monkeypatch):
    """A non-200 registry response fails soft to None."""

    class _Resp:
        status_code = 503

        def json(self):  # pragma: no cover - not reached
            return {}

    class _FakeHttpx:
        @staticmethod
        def get(url, timeout):
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "httpx", _FakeHttpx)
    assert cv._fetch_latest_from_npm() is None


def test_corrupt_cache_is_ignored(monkeypatch, tmp_path):
    """A malformed cache file is treated as absent."""
    path = tmp_path / ".cache" / "codex_version.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json {{{", encoding="utf-8")
    monkeypatch.setattr(cv, "_fetch_latest_from_npm", lambda: "0.149.1")
    assert cv.get_codex_cli_version() == "0.149.1"
