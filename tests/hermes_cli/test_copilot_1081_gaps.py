"""Regression tests for the Copilot 1.0.81-6 gap fixes (2026-08-20).

Covers, as behavior invariants (not catalog snapshots):
  GAP-1  /responses-only Copilot models (grok-4.5/4.6, mai-code) route to
         codex_responses, not chat_completions (which 400s
         ``unsupported_api_for_model``).
  GAP-3  the offline GPT-5 reasoning-effort floor is ``none`` (gpt-5.4/5.5
         reject ``minimal`` with HTTP 400), never ``minimal``.
  GAP-7  plan-aware Copilot base-URL resolution + suffix-match host recognizers
         accept business/enterprise hosts, reject non-Copilot hosts.
"""

from unittest.mock import patch

import hermes_cli.models as M


# ── GAP-1: /responses-only routing ──────────────────────────────────────────

class TestResponsesOnlyRouting:
    def test_grok_routes_to_responses_offline(self):
        # No catalog (cold path): grok-4.5/4.6 must NOT fall through to
        # chat_completions (the Copilot proxy 400s there).
        for model in ("grok-4.5", "grok-4.6"):
            assert M.copilot_model_api_mode(model, catalog=None) == "codex_responses"

    def test_mai_code_routes_to_responses_offline(self):
        for model in ("mai-code-1.1-flash", "mai-code-1-flash-picker"):
            assert M.copilot_model_api_mode(model, catalog=None) == "codex_responses"

    def test_catalog_responses_only_routes_to_responses(self):
        # A catalog entry whose supported_endpoints is /responses-only (no
        # /chat/completions) routes to codex_responses regardless of name.
        catalog = [
            {"id": "some-new-model", "supported_endpoints": ["/responses", "ws:/responses"]},
        ]
        assert (
            M.copilot_model_api_mode("some-new-model", catalog=catalog)
            == "codex_responses"
        )

    def test_catalog_dual_endpoint_still_chat_completions(self):
        # A model that DOES list /chat/completions is not forced onto responses
        # by the new branch (gemini-style).
        catalog = [
            {"id": "gemini-x-flash", "supported_endpoints": ["/chat/completions"]},
        ]
        assert (
            M.copilot_model_api_mode("gemini-x-flash", catalog=catalog)
            == "chat_completions"
        )

    def test_claude_still_anthropic_messages(self):
        assert M.copilot_model_api_mode("claude-opus-5", catalog=None) == "anthropic_messages"

    def test_gpt5_still_codex_responses(self):
        assert M.copilot_model_api_mode("gpt-5.6-sol", catalog=None) == "codex_responses"

    def test_helper_recognizes_responses_only_prefixes(self):
        assert M._is_copilot_responses_only_model("grok-4.5")
        assert M._is_copilot_responses_only_model("grok-4.6")
        assert M._is_copilot_responses_only_model("mai-code-1.1-flash")
        assert M._is_copilot_responses_only_model("x-ai/grok-4.6")  # aggregator prefix
        # Non-responses-only models are not matched.
        assert not M._is_copilot_responses_only_model("gemini-3.7-flash")
        assert not M._is_copilot_responses_only_model("claude-opus-5")
        assert not M._is_copilot_responses_only_model("grok-3")


# ── GAP-3: offline effort floor ─────────────────────────────────────────────

class TestOfflineEffortFloor:
    def test_gpt5_floor_is_none_not_minimal(self):
        # gpt-5.4 / gpt-5.5 reject "minimal" (HTTP 400); the offline list must
        # lead with "none" (the true floor) so a catalog-cold default is valid.
        for model in ("gpt-5.4", "gpt-5.5"):
            efforts = M._github_reasoning_efforts_for_model_id(model)
            assert efforts, f"{model} should have offline efforts"
            assert efforts[0] == "none"
            assert "minimal" not in efforts

    def test_gpt56_has_max(self):
        efforts = M._github_reasoning_efforts_for_model_id("gpt-5.6-sol")
        assert efforts[0] == "none"
        assert "max" in efforts

    def test_grok_offline_efforts(self):
        # grok-4.5: low/medium/high; grok-4.6 additionally xhigh.
        e45 = M._github_reasoning_efforts_for_model_id("grok-4.5")
        e46 = M._github_reasoning_efforts_for_model_id("grok-4.6")
        assert e45 == ["low", "medium", "high"]
        assert "xhigh" in e46
        assert e46[0] == "low"  # grok has no "none"/"minimal" floor

    def test_shared_gpt5_list_has_no_minimal(self):
        # Invariant: the shared offline GPT5 effort list never contains the
        # 400-triggering "minimal".
        assert "minimal" not in M.COPILOT_REASONING_EFFORTS_GPT5
        assert "minimal" not in M.COPILOT_REASONING_EFFORTS_GPT56


# ── GAP-7: plan-aware host recognition ──────────────────────────────────────

class TestCopilotHostRecognition:
    def test_is_copilot_host_accepts_plan_scoped(self):
        assert M._is_copilot_host("https://api.githubcopilot.com")
        assert M._is_copilot_host("https://api.business.githubcopilot.com")
        assert M._is_copilot_host("https://api.enterprise.githubcopilot.com")

    def test_is_copilot_host_rejects_non_copilot(self):
        assert not M._is_copilot_host("https://api.openai.com")
        assert not M._is_copilot_host("https://api.anthropic.com")
        assert not M._is_copilot_host("")
        assert not M._is_copilot_host(None)
        # Must not be fooled by a lookalike domain suffix.
        assert not M._is_copilot_host("https://githubcopilot.com.evil.example")

    def test_anthropic_adapter_recognizer_is_suffix_aware(self):
        from agent.anthropic_adapter import _is_copilot_base_url

        assert _is_copilot_base_url("https://api.githubcopilot.com")
        assert _is_copilot_base_url("https://api.business.githubcopilot.com")
        assert _is_copilot_base_url("https://api.enterprise.githubcopilot.com")
        assert not _is_copilot_base_url("https://api.openai.com")
        assert not _is_copilot_base_url(None)


# ── GAP-7: base-URL resolution (env override + endpoints.api parsing) ────────

class TestBaseUrlResolution:
    def test_env_override_wins(self, monkeypatch):
        import hermes_cli.copilot_auth as CA

        monkeypatch.setenv("COPILOT_API_URL", "https://api.enterprise.githubcopilot.com/")
        base, plan = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.enterprise.githubcopilot.com"
        assert plan == ""

    def test_hermes_env_override_wins(self, monkeypatch):
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("COPILOT_API_URL", raising=False)
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        monkeypatch.setenv("HERMES_COPILOT_API_URL", "https://custom.githubcopilot.com")
        base, plan = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://custom.githubcopilot.com"

    def test_copilot_api_base_url_alias_honored(self, monkeypatch):
        # COPILOT_API_BASE_URL is the provider registry's base_url_env_var and
        # upstream #78378's standardized name — the plan-aware resolver must
        # honor it too, not just COPILOT_API_URL / HERMES_COPILOT_API_URL.
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("COPILOT_API_URL", raising=False)
        monkeypatch.delenv("HERMES_COPILOT_API_URL", raising=False)
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://api.enterprise.githubcopilot.com/")
        base, plan = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.enterprise.githubcopilot.com"
        assert plan == ""

    def test_dedicated_env_overrides_win_over_alias(self, monkeypatch):
        # Precedence: HERMES_COPILOT_API_URL / COPILOT_API_URL take priority
        # over the COPILOT_API_BASE_URL alias when several are set.
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("HERMES_COPILOT_API_URL", raising=False)
        monkeypatch.setenv("COPILOT_API_URL", "https://api.githubcopilot.com")
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://api.enterprise.githubcopilot.com")
        base, _ = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.githubcopilot.com"

    def test_endpoints_api_parsed_from_user_info(self, monkeypatch, tmp_path):
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("COPILOT_API_URL", raising=False)
        monkeypatch.delenv("HERMES_COPILOT_API_URL", raising=False)
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        monkeypatch.setattr(CA, "_copilot_base_url_memo", None)
        monkeypatch.setattr(CA, "_COPILOT_BASE_URL_CACHE_PATH", tmp_path / "b.json")
        monkeypatch.setattr(
            CA,
            "_resolve_copilot_user_info",
            lambda tok, timeout=10.0: {
                "copilot_plan": "business",
                "endpoints": {"api": "https://api.business.githubcopilot.com"},
            },
        )
        base, plan = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.business.githubcopilot.com"
        assert plan == "business"

    def test_falls_back_to_default_on_missing_endpoints(self, monkeypatch, tmp_path):
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("COPILOT_API_URL", raising=False)
        monkeypatch.delenv("HERMES_COPILOT_API_URL", raising=False)
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        monkeypatch.setattr(CA, "_copilot_base_url_memo", None)
        monkeypatch.setattr(CA, "_COPILOT_BASE_URL_CACHE_PATH", tmp_path / "b2.json")
        monkeypatch.setattr(CA, "_resolve_copilot_user_info", lambda tok, timeout=10.0: None)
        base, plan = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.githubcopilot.com"
        assert plan == ""

    def test_rejects_non_githubcopilot_endpoint(self, monkeypatch, tmp_path):
        # A malformed/hostile endpoints.api that isn't a *.githubcopilot.com host
        # must be ignored in favor of the safe default.
        import hermes_cli.copilot_auth as CA

        monkeypatch.delenv("COPILOT_API_URL", raising=False)
        monkeypatch.delenv("HERMES_COPILOT_API_URL", raising=False)
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        monkeypatch.setattr(CA, "_copilot_base_url_memo", None)
        monkeypatch.setattr(CA, "_COPILOT_BASE_URL_CACHE_PATH", tmp_path / "b3.json")
        monkeypatch.setattr(
            CA,
            "_resolve_copilot_user_info",
            lambda tok, timeout=10.0: {
                "copilot_plan": "business",
                "endpoints": {"api": "https://evil.example.com"},
            },
        )
        base, _ = CA.resolve_copilot_plan_and_base_url(raw_token="tok")
        assert base == "https://api.githubcopilot.com"


# ── GAP-6: identity-header parity with the real Copilot CLI 1.0.81-6 ─────────

class TestIdentityHeaderParity:
    """The header set must match the MITM-captured CLI 1.0.81-6 request 1:1.

    Matching exactly (not a minimal subset) is the anti-fingerprint posture: an
    incomplete header set is itself a flagging signal.
    """

    def _headers(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        return copilot_request_headers(is_agent_turn=True, model="gpt-5.6-sol")

    def test_carries_full_cli_header_set(self):
        h = self._headers()
        # Every header the real CLI sends on an inference call.
        assert h["Copilot-Integration-Id"] == "copilot-developer-cli"
        assert h["Editor-Version"].startswith("copilot/")
        assert h["User-Agent"].startswith("copilot/")
        assert h["Openai-Intent"] == "conversation-agent"
        assert h["X-Initiator"] == "agent"
        assert h["X-Interaction-Type"] == "conversation-user"
        assert h["Copilot-Harness-Id"] == "copilot-sdk"
        assert h["X-GitHub-Api-Version"]
        assert h["X-Client-Machine-Id"]
        assert h["X-Client-Session-Id"]
        assert h["X-Agent-Task-Id"]
        assert h["X-Interaction-Id"]
        assert h["X-GitHub-Repository-Nwo"] == "__no_repository__"
        assert h["X-GitHub-Repository-Host"] == "__no_repository__"
        assert h["X-Stainless-Helper-Method"] == "stream"

    def test_does_not_send_retired_headers(self):
        # These were Hermes-invented / VS-Code-extension headers the real CLI
        # 1.0.81-6 does NOT send; keeping them would be a fingerprint mismatch.
        h = self._headers()
        assert "Runtime-Client-Version" not in h
        assert "X-Request-Id" not in h
        assert "Editor-Plugin-Version" not in h
        # Casing matches the SDK (X-Initiator, not lowercase x-initiator).
        assert "x-initiator" not in h

    def test_machine_id_is_stable_across_calls(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        a = copilot_request_headers()["X-Client-Machine-Id"]
        b = copilot_request_headers()["X-Client-Machine-Id"]
        assert a == b  # stable per-install fingerprint, NOT per-call

    def test_per_call_ids_vary_when_not_supplied(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        a = copilot_request_headers()
        b = copilot_request_headers()
        assert a["X-Interaction-Id"] != b["X-Interaction-Id"]

    def test_explicit_session_and_task_ids_honored(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        h = copilot_request_headers(
            session_id="sess-1", agent_task_id="task-1", interaction_id="int-1"
        )
        assert h["X-Client-Session-Id"] == "sess-1"
        assert h["X-Agent-Task-Id"] == "task-1"
        assert h["X-Interaction-Id"] == "int-1"

    def test_user_turn_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        assert copilot_request_headers(is_agent_turn=False)["X-Initiator"] == "user"

    def test_vision_header_only_on_vision(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        assert "Copilot-Vision-Request" not in copilot_request_headers()
        assert copilot_request_headers(is_vision=True)["Copilot-Vision-Request"] == "true"

    def test_non_streaming_omits_stainless(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        assert "X-Stainless-Helper-Method" not in copilot_request_headers(is_streaming=False)


# ── GAP-7 latent break: plan-aware host recognition across all detection sites ─

class TestPlanAwareHostDetection:
    def test_url_to_provider_maps_business_enterprise(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.githubcopilot.com") == "copilot"
        assert _infer_provider_from_url("https://api.business.githubcopilot.com") == "copilot"
        assert _infer_provider_from_url("https://api.enterprise.githubcopilot.com") == "copilot"

    def test_max_completion_tokens_detection_plan_aware(self):
        # base_url_host_matches(..., "githubcopilot.com") must accept plan hosts.
        from utils import base_url_host_matches

        assert base_url_host_matches("https://api.business.githubcopilot.com", "githubcopilot.com")
        assert base_url_host_matches("https://api.enterprise.githubcopilot.com", "githubcopilot.com")
        assert not base_url_host_matches("https://api.openai.com", "githubcopilot.com")
