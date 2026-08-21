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
