"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import pytest
from unittest.mock import patch


@pytest.fixture
def isolated_copilot_base_url_state(tmp_path, monkeypatch):
    import hermes_cli.copilot_auth as copilot_auth

    monkeypatch.setattr(copilot_auth, "_copilot_base_url_memo", None)
    monkeypatch.setattr(copilot_auth, "_COPILOT_BASE_URL_CACHE_PATH", None)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    for name in (
        "HERMES_COPILOT_API_URL",
        "COPILOT_API_URL",
        "COPILOT_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return copilot_auth


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg

    def test_oauth_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("gho_abcdefghijklmnop1234")
        assert valid is True

    def test_fine_grained_pat_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("github_pat_abcdefghijklmnop1234")
        assert valid is True

    def test_github_app_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghu_abcdefghijklmnop1234")
        assert valid is True

    def test_empty_token_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("")
        assert valid is False


class TestCopilotBaseUrlResolution:
    def test_cache_is_scoped_to_active_profile(
        self, isolated_copilot_base_url_state, tmp_path
    ):
        copilot_auth = isolated_copilot_base_url_state
        assert copilot_auth._copilot_base_url_cache_path() == (
            tmp_path / "cache" / "copilot_base_url.json"
        )

    def test_process_memo_does_not_cross_profile_scope(
        self, isolated_copilot_base_url_state, tmp_path, monkeypatch
    ):
        copilot_auth = isolated_copilot_base_url_state
        active_home = tmp_path / "profile-a"
        calls = []
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home", lambda: active_home
        )
        monkeypatch.setattr(
            copilot_auth,
            "_resolve_copilot_user_info",
            lambda *args, **kwargs: calls.append(active_home) or {
                "endpoints": {"api": "https://api.githubcopilot.com"}
            },
        )

        copilot_auth.resolve_copilot_base_url("same-token")
        active_home = tmp_path / "profile-b"
        copilot_auth.resolve_copilot_base_url("same-token")

        assert calls == [tmp_path / "profile-a", tmp_path / "profile-b"]

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://api.githubcopilot.com",
            "https://api.business.githubcopilot.com/",
            "https://api.enterprise.githubcopilot.com/inference",
        ],
    )
    def test_accepts_strict_provisioned_endpoints(
        self, isolated_copilot_base_url_state, monkeypatch, endpoint
    ):
        copilot_auth = isolated_copilot_base_url_state
        monkeypatch.setattr(
            copilot_auth,
            "_resolve_copilot_user_info",
            lambda *args, **kwargs: {
                "copilot_plan": "business",
                "endpoints": {"api": endpoint},
            },
        )
        base, plan = copilot_auth.resolve_copilot_plan_and_base_url("token")
        assert base == endpoint.rstrip("/")
        assert plan == "business"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://api.githubcopilot.com",
            "https://githubcopilot.com.evil.example",
            "https://user@api.githubcopilot.com",
            "https://api.githubcopilot.com#fragment",
            "file:///tmp/copilot",
        ],
    )
    def test_rejects_invalid_provisioned_endpoints(
        self, isolated_copilot_base_url_state, monkeypatch, endpoint
    ):
        copilot_auth = isolated_copilot_base_url_state
        monkeypatch.setattr(
            copilot_auth,
            "_resolve_copilot_user_info",
            lambda *args, **kwargs: {
                "copilot_plan": "business",
                "endpoints": {"api": endpoint},
            },
        )
        base, _ = copilot_auth.resolve_copilot_plan_and_base_url("token")
        assert base == "https://api.githubcopilot.com"

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ("https://copilot.corp.example/api/", "https://copilot.corp.example/api"),
            ("http://localhost:8642/v1", "http://localhost:8642/v1"),
            ("http://127.0.0.1:8642/v1", "http://127.0.0.1:8642/v1"),
        ],
    )
    def test_operator_override_allows_enterprise_https_and_loopback_http(
        self, isolated_copilot_base_url_state, monkeypatch, override, expected
    ):
        copilot_auth = isolated_copilot_base_url_state
        monkeypatch.setenv("HERMES_COPILOT_API_URL", override)
        assert copilot_auth.resolve_copilot_base_url("token") == expected

    @pytest.mark.parametrize(
        "override",
        [
            "http://copilot.corp.example",
            "http://localhost.evil.example",
            "https://user:pass@copilot.corp.example",
            "https://copilot.corp.example?endpoint=other",
        ],
    )
    def test_operator_override_rejects_unsafe_urls(
        self, isolated_copilot_base_url_state, monkeypatch, override
    ):
        copilot_auth = isolated_copilot_base_url_state
        monkeypatch.setenv("HERMES_COPILOT_API_URL", override)
        with pytest.raises(ValueError):
            copilot_auth.resolve_copilot_base_url("token")


def test_fallback_identity_matches_current_copilot_contract():
    from hermes_cli.copilot_auth import copilot_fallback_request_headers

    headers = copilot_fallback_request_headers()
    assert headers["User-Agent"] == headers["Editor-Version"]
    assert headers["Copilot-Integration-Id"] == "copilot-developer-cli"
    assert headers["Openai-Intent"] == "conversation-agent"
    assert headers["X-Interaction-Type"] == "conversation-user"
    assert headers["Copilot-Harness-Id"] == "copilot-sdk"


def test_auto_router_degraded_path_keeps_canonical_fallback_identity(monkeypatch):
    from agent.auto_router import AutoRouter

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.copilot_request_headers",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("discovery unavailable")),
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        lambda token: (_ for _ in ()).throw(RuntimeError("exchange unavailable")),
    )

    headers = AutoRouter()._base_headers("raw-token")

    assert headers["Authorization"] == "Bearer raw-token"
    assert headers["Copilot-Integration-Id"] == "copilot-developer-cli"
    assert headers["Openai-Intent"] == "conversation-agent"
    assert headers["X-Interaction-Type"] == "conversation-user"
    assert headers["Copilot-Harness-Id"] == "copilot-sdk"


class TestIdentityAudit:
    """Structured Copilot identity resolution audit."""

    def test_identity_precedence_records_skipped_classic_pat(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        audit = resolve_copilot_identity_audit()

        assert audit.token == "gho_gh_second"
        assert audit.source == "GH_TOKEN"
        assert audit.source_kind == "env"
        assert len(audit.skipped_sources) == 1
        assert audit.skipped_sources[0].source == "COPILOT_GITHUB_TOKEN"
        assert "Classic Personal Access Tokens" in audit.skipped_sources[0].reason

    def test_pool_audit_records_skipped_invalid_entries_and_gh_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                "not-a-dict",
                {"label": "no-token-here"},
                {"access_token": ""},
                {"access_token": "ghp_classic_pat"},
            ],
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ):
            audit = resolve_copilot_identity_audit(include_credential_pool=True)

        assert audit.token == "gho_from_cli"
        assert audit.source == "gh auth token"
        assert audit.source_kind == "gh_auth"
        assert [skip.source for skip in audit.skipped_sources] == [
            "credential_pool:copilot[0]",
            "credential_pool:copilot[1]",
            "credential_pool:copilot[2]",
            "credential_pool:copilot[3]",
        ]
        assert any(
            "Non-dict credential pool entry" in skip.reason
            for skip in audit.skipped_sources
        )
        assert any("Missing access_token" in skip.reason for skip in audit.skipped_sources)
        assert any("Classic Personal Access Tokens" in skip.reason for skip in audit.skipped_sources)

    def test_pool_token_wins_before_gh_auth(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_pool_token"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_from_pool", 1234567890.0),
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ):
            audit = resolve_copilot_identity_audit(
                include_credential_pool=True,
                exchange_pool_tokens=True,
            )

        assert audit.token == "tid_from_pool"
        assert audit.source == "credential_pool:copilot[0]"
        assert audit.source_kind == "credential_pool"


class TestResolveToken:
    """Token resolution with env var priority."""

    def test_copilot_github_token_first_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_copilot_first"
        assert source == "COPILOT_GITHUB_TOKEN"

    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"

    def test_github_token_third_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_github_third"
        assert source == "GITHUB_TOKEN"

    def test_classic_pat_in_env_skipped(self, monkeypatch):
        """Classic PATs in env vars should be skipped, not returned."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_valid_oauth")
        token, source = resolve_copilot_token()
        # Should skip the ghp_ token and find the gho_ one
        assert token == "gho_valid_oauth"
        assert source == "GITHUB_TOKEN"

    def test_gh_cli_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_from_cli"):
            token, source = resolve_copilot_token()
        assert token == "gho_from_cli"
        assert source == "gh auth token"

    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_no_token_returns_empty(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None):
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self, monkeypatch):
        from hermes_cli.copilot_auth import copilot_request_headers
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._latest_copilot_cli_version",
            lambda: "1.0.63",
        )
        headers = copilot_request_headers()
        # CLI 1.0.81-6 sends Openai-Intent=conversation-agent (MITM-captured).
        assert headers["Openai-Intent"] == "conversation-agent"
        # Presents as the @github/copilot CLI: UA is copilot/<ver> (short form
        # or full "copilot/<ver> (<platform> <node>) term/<term>" when node is
        # resolvable). CLI 1.0.81-6 DOES send Editor-Version (value copilot/<ver>);
        # it does NOT send the Editor-Plugin-Version pair or Runtime-Client-Version.
        assert headers["User-Agent"].startswith("copilot/1.0.63")
        assert headers["Editor-Version"] == "copilot/1.0.63"
        assert "Editor-Plugin-Version" not in headers
        assert "Runtime-Client-Version" not in headers
        # Fixed identity headers the CLI carries on every inference call.
        assert headers["Copilot-Harness-Id"] == "copilot-sdk"
        assert headers["X-Interaction-Type"] == "conversation-user"

    def test_user_agent_full_cli_form_when_node_present(self, monkeypatch):
        """When a Node runtime + TERM_PROGRAM are resolvable, the UA matches the
        real CLI ``FG()`` builder: copilot/<ver> (<platform> <node>) term/<term>.
        """
        from hermes_cli import copilot_auth
        monkeypatch.setattr(copilot_auth, "_latest_copilot_cli_version", lambda: "1.0.63")
        monkeypatch.setattr(copilot_auth, "_copilot_node_version", lambda: "v22.22.3")
        monkeypatch.setattr(copilot_auth.sys, "platform", "linux")
        monkeypatch.setenv("HERMES_COPILOT_TERM_PROGRAM", "vscode")
        ua = copilot_auth._copilot_user_agent()
        assert ua == "copilot/1.0.63 (linux v22.22.3) term/vscode"

    def test_user_agent_short_form_when_no_node(self, monkeypatch):
        """No resolvable Node runtime → honest short core, no fabricated runtime."""
        from hermes_cli import copilot_auth
        monkeypatch.setattr(copilot_auth, "_latest_copilot_cli_version", lambda: "1.0.63")
        monkeypatch.setattr(copilot_auth, "_copilot_node_version", lambda: "")
        ua = copilot_auth._copilot_user_agent()
        assert ua == "copilot/1.0.63"

    def test_term_program_defaults_to_vscode_not_unknown(self, monkeypatch):
        """Unset TERM_PROGRAM resolves to a valid default (vscode), never the
        bot-signalling literal ``unknown`` the raw CLI builder would emit."""
        from hermes_cli import copilot_auth
        monkeypatch.delenv("HERMES_COPILOT_TERM_PROGRAM", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert copilot_auth._copilot_term_program() == "vscode"

    def test_agent_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_agent_turn=True)
        assert headers["X-Initiator"] == "agent"

    def test_user_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_agent_turn=False)
        assert headers["X-Initiator"] == "user"

    def test_vision_header(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers(is_vision=True)
        assert headers["Copilot-Vision-Request"] == "true"

    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""

    def test_includes_openai_intent(self, monkeypatch):
        from hermes_cli.models import copilot_default_headers
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._latest_copilot_cli_version",
            lambda: "1.0.63",
        )
        headers = copilot_default_headers()
        assert "Openai-Intent" in headers
        assert headers["Openai-Intent"] == "conversation-agent"
        assert headers["User-Agent"].startswith("copilot/1.0.63")

    def test_includes_x_initiator(self):
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers()
        assert "X-Initiator" in headers


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5-mini") is False

    def test_gpt4_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-4.1") is False
        assert _should_use_copilot_responses_api("gpt-4o") is False
        assert _should_use_copilot_responses_api("gpt-4o-mini") is False

    def test_non_gpt_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("claude-sonnet-4.6") is False
        assert _should_use_copilot_responses_api("claude-opus-4.6") is False
        assert _should_use_copilot_responses_api("gemini-2.5-pro") is False
        assert _should_use_copilot_responses_api("grok-code-fast-1") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

    def test_copilot_env_vars_order_matches_docs(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert copilot.api_key_env_vars == (
            "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"
        )


class TestApiVersionExtraction:
    """CAPI X-GitHub-Api-Version extraction from CLI bundles.

    Regression coverage for the 2026-07-13 fix: as of Copilot CLI 1.0.71 the
    api-version literal moved out of the JS bundle and into the Rust core binary
    (prebuilds/*/runtime.node), sitting in the CAPI header cluster next to
    Openai-Intent / Editor-Version with no JS assignment syntax. The old regex
    only matched the JS-constant form and silently fell back to a stale version,
    which served the old context tier (gpt-5.6 capped at the 922k default-tier
    wall instead of the long_context tier).
    """

    def test_extracts_js_constant_form(self, tmp_path):
        from hermes_cli.copilot_auth import _extract_api_version_from_bundle
        js = tmp_path / "index.js"
        js.write_text('a="X-GitHub-Api-Version",b="2026-06-01";other="2022-11-28"')
        assert _extract_api_version_from_bundle(js) == "2026-06-01"

    def test_extracts_binary_cluster_form(self, tmp_path):
        from hermes_cli.copilot_auth import _extract_api_version_from_bundle
        # Mimic the Rust binary's CAPI header cluster: date adjacent to
        # Openai-Intent / Editor-Version with NO assignment syntax.
        node = tmp_path / "runtime.node"
        node.write_bytes(
            b"...Openai-Intentconversation-agent2026-07-01Editor-VersionX-Copilot-Traceparent..."
        )
        assert _extract_api_version_from_bundle(node) == "2026-07-01"

    def test_ignores_rest_api_version_and_unanchored_dates(self, tmp_path):
        from hermes_cli.copilot_auth import _extract_api_version_from_bundle
        # 2022-11-28 is the github.com REST/gist version (must be dropped); a
        # far-off date with no CAPI-header anchor nearby must NOT be picked.
        f = tmp_path / "app.js"
        f.write_text(
            'headers={"X-GitHub-Api-Version":"2022-11-28"};'
            'someCertExpiry="2027-12-31";' + ("x" * 200) + "unrelated"
        )
        assert _extract_api_version_from_bundle(f) is None

    def test_resolver_picks_newest_across_bundles(self, tmp_path, monkeypatch):
        # Two bundles: an older JS one (2026-08-01) discovered first and a newer
        # binary one (2026-09-01). The resolver must return the NEWEST, not the
        # first hit. Values are kept at/above the hard fallback so the fallback
        # (which also participates in the max()) can't mask the ordering being
        # tested — this asserts the newest-wins invariant, not a literal.
        import hermes_cli.copilot_auth as C
        old = tmp_path / "old_index.js"
        old.write_text('a="X-GitHub-Api-Version",b="2026-08-01"')
        new = tmp_path / "new_runtime.node"
        new.write_bytes(b"Openai-Intentconversation-agent2026-09-01Editor-Version")

        monkeypatch.setattr(C, "_discover_copilot_cli_bundles", lambda: [old, new])
        monkeypatch.setattr(C, "_copilot_api_version_memo", None)
        monkeypatch.setenv("HERMES_COPILOT_API_VERSION", "")
        # Point the on-disk cache at an empty temp path so it doesn't short-circuit.
        monkeypatch.setattr(C, "_COPILOT_API_VERSION_CACHE_PATH", tmp_path / "nonexistent.json")
        assert C._latest_copilot_api_version() == "2026-09-01"

    def test_fallback_is_modern(self):
        # The hard fallback must be at least the 2026-08-01 tier so a bundle-miss
        # never regresses to an older api-version (the CLI runtime carries
        # 2026-08-01 in capi_client.rs as of 1.0.81-6).
        from hermes_cli.copilot_auth import _COPILOT_API_VERSION_FALLBACK
        assert _COPILOT_API_VERSION_FALLBACK >= "2026-08-01"
