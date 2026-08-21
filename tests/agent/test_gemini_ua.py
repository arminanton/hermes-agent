"""Direct coverage for the surviving native Gemini user-agent helpers."""

from __future__ import annotations

import json

import pytest


def test_environment_overrides_shape_both_public_headers(monkeypatch):
    import agent.gemini_ua as ua

    monkeypatch.setenv("HERMES_GEMINI_CLI_VERSION", "9.8.7")
    monkeypatch.setenv("HERMES_GEMINI_NODE_VERSION", "22.14.0")
    monkeypatch.setenv("HERMES_GEMINI_CLI_SURFACE", "test-surface")
    monkeypatch.setattr(ua, "_process_platform", lambda: "linux")
    monkeypatch.setattr(ua, "_process_arch", lambda: "arm64")

    assert ua.gemini_cli_user_agent("gemini-test") == (
        "GeminiCLI/9.8.7/gemini-test (linux; arm64; test-surface)"
    )
    assert ua.gemini_cli_x_goog_api_client() == "gl-node/22.14.0 gccl/9.8.7"


def test_fresh_cache_is_used_without_network(monkeypatch, tmp_path):
    import agent.gemini_ua as ua

    cache = tmp_path / "gemini_cli_version.json"
    cache.write_text(json.dumps({"version": "1.2.3", "fetched_at": 1000.0}))
    monkeypatch.delenv("HERMES_GEMINI_CLI_VERSION", raising=False)
    monkeypatch.setattr(ua, "_VERSION_CACHE_PATH", cache)
    monkeypatch.setattr(ua, "_version_memo", None)
    monkeypatch.setattr(ua.time, "time", lambda: 1001.0)

    assert ua._gemini_cli_version() == "1.2.3"


def test_malformed_cache_and_offline_registry_use_fallback(monkeypatch, tmp_path):
    import agent.gemini_ua as ua

    cache = tmp_path / "gemini_cli_version.json"
    cache.write_text("not json")
    monkeypatch.delenv("HERMES_GEMINI_CLI_VERSION", raising=False)
    monkeypatch.setattr(ua, "_VERSION_CACHE_PATH", cache)
    monkeypatch.setattr(ua, "_version_memo", None)
    monkeypatch.setattr(ua.time, "time", lambda: 2000.0)

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    assert ua._gemini_cli_version() == ua._GEMINI_CLI_VERSION_FALLBACK


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "x64"),
        ("amd64", "x64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("i686", "ia32"),
        ("riscv64", "riscv64"),
        ("", "x64"),
    ],
)
def test_process_arch_matches_node_names(monkeypatch, machine, expected):
    import agent.gemini_ua as ua

    monkeypatch.setattr(ua.platform, "machine", lambda: machine)
    assert ua._process_arch() == expected


def test_default_public_helpers_return_complete_values(monkeypatch):
    import agent.gemini_ua as ua

    monkeypatch.setattr(ua, "_gemini_cli_version", lambda: "1.0.0")
    monkeypatch.setattr(ua, "_node_version", lambda: "20.0.0")
    monkeypatch.setattr(ua, "_process_platform", lambda: "darwin")
    monkeypatch.setattr(ua, "_process_arch", lambda: "x64")
    monkeypatch.setattr(ua, "_surface", lambda: "hermes")

    assert ua.gemini_cli_user_agent() == "GeminiCLI/1.0.0 (darwin; x64; hermes)"
    assert ua.gemini_cli_x_goog_api_client() == "gl-node/20.0.0 gccl/1.0.0"
