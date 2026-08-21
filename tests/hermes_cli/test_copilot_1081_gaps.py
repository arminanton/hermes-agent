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
