"""Tests for the /fast CLI command and service-tier config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestParseServiceTierConfig(unittest.TestCase):
    def _parse(self, raw):
        cli_mod = _import_cli()
        return cli_mod._parse_service_tier_config(raw)

    def test_fast_maps_to_priority(self):
        self.assertEqual(self._parse("fast"), "priority")
        self.assertEqual(self._parse("priority"), "priority")

    def test_normal_disables_service_tier(self):
        self.assertIsNone(self._parse("normal"))
        self.assertIsNone(self._parse("off"))
        self.assertIsNone(self._parse(""))


class TestHandleFastCommand(unittest.TestCase):
    def _make_cli(self, service_tier=None):
        return SimpleNamespace(
            service_tier=service_tier,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.4",
            _fast_command_available=lambda: True,
            agent=MagicMock(),
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier=None)
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        # Bare /fast shows status, does not change config
        mock_save.assert_not_called()
        # Should have printed the status line
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("normal", printed)

    def test_no_args_shows_fast_when_enabled(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier="priority")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("fast", printed)

    def test_normal_argument_clears_service_tier(self):
        cli_mod = _import_cli()
        stub = self._make_cli(service_tier="priority")
        with (
            patch.object(cli_mod, "_cprint"),
            patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast normal")

        mock_save.assert_called_once_with("agent.service_tier", "normal")
        self.assertIsNone(stub.service_tier)
        self.assertIsNone(stub.agent)

    def test_unsupported_model_does_not_expose_fast(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            service_tier=None,
            provider="openai-codex",
            requested_provider="openai-codex",
            model="gpt-5.3-codex",
            _fast_command_available=lambda: False,
            agent=MagicMock(),
        )

        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_fast_command(stub, "/fast")

        mock_save.assert_not_called()
        self.assertTrue(mock_cprint.called)


class TestPriorityProcessingModels(unittest.TestCase):
    """Verify the expanded Priority Processing model registry."""

    def test_all_documented_models_supported(self):
        from hermes_cli.models import model_supports_fast_mode

        # All OpenAI flagship models support Priority Processing — including
        # future releases (gpt-5.5, 5.6...) via pattern matching.
        supported = [
            "gpt-5.5", "gpt-5.5-mini",
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.2",
            "gpt-5.1", "gpt-5", "gpt-5-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini",
            "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
        ]
        for model in supported:
            assert model_supports_fast_mode(model), f"{model} should support fast mode"

    def test_all_anthropic_models_supported(self):
        """Fast mode spans the opus/sonnet/haiku 4.x families (endpoint-aware).

        The model-level ``model_supports_fast_mode`` is permissive across these
        families; the *adapter* makes the final per-endpoint call (Copilot accepts
        speed=fast on 4.x; native Anthropic still 400s on 4.7/4.8 and is refused there).
        """
        from hermes_cli.models import model_supports_fast_mode

        supported = [
            "claude-opus-4-6", "claude-opus-4.6",
            "anthropic/claude-opus-4-6", "anthropic/claude-opus-4.6",
            "claude-opus-4-7", "claude-opus-4-8", "claude-opus-4.8",
            "claude-opus-4.8-fast",
            "claude-sonnet-4-6", "claude-sonnet-4.6", "claude-sonnet-4",
            "claude-haiku-4-5",
        ]
        for model in supported:
            assert model_supports_fast_mode(model), f"{model} should support fast mode"

        # Not a 4.x Claude fast family → excluded
        unsupported = [
            "claude-3-5-haiku", "claude-3-5-sonnet", "claude-2",
        ]
        for model in unsupported:
            assert not model_supports_fast_mode(model), (
                f"{model} should NOT support the speed=fast parameter"
            )

    def test_codex_models_excluded(self):
        """Codex models route through Responses API and don't accept service_tier."""
        from hermes_cli.models import model_supports_fast_mode

        for model in ["gpt-5-codex", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.1-codex-max"]:
            assert not model_supports_fast_mode(model), f"{model} is codex — should not expose /fast"

    def test_vendor_prefix_stripped(self):
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("openai/gpt-5.4") is True
        assert model_supports_fast_mode("openai/gpt-4.1") is True
        assert model_supports_fast_mode("openai/o3") is True

    def test_non_priority_models_rejected(self):
        from hermes_cli.models import model_supports_fast_mode

        # Codex-series models route through the Codex Responses API and
        # don't accept service_tier, so they're excluded.
        assert model_supports_fast_mode("gpt-5.3-codex") is False
        assert model_supports_fast_mode("gpt-5.2-codex") is False
        assert model_supports_fast_mode("gpt-5-codex") is False
        # Non-OpenAI, non-Anthropic models
        assert model_supports_fast_mode("gemini-3-pro-preview") is False
        assert model_supports_fast_mode("kimi-k2-thinking") is False
        assert model_supports_fast_mode("deepseek-chat") is False
        assert model_supports_fast_mode("") is False
        assert model_supports_fast_mode(None) is False

    def test_resolve_overrides_returns_service_tier(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("gpt-5.4")
        assert result == {"service_tier": "priority"}

        result = resolve_fast_mode_overrides("gpt-4.1")
        assert result == {"service_tier": "priority"}

    def test_resolve_overrides_none_for_unsupported(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("gpt-5.3-codex") is None
        assert resolve_fast_mode_overrides("gemini-3-pro-preview") is None
        assert resolve_fast_mode_overrides("kimi-k2-thinking") is None


class TestFastModeRouting(unittest.TestCase):
    def test_fast_command_exposed_for_model_even_when_provider_is_auto(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(provider="auto", requested_provider="auto", model="gpt-5.4", agent=None)

        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_fast_command_exposed_for_non_codex_models(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(provider="openai", requested_provider="openai", model="gpt-4.1", agent=None)
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

        stub = SimpleNamespace(provider="openrouter", requested_provider="openrouter", model="o3", agent=None)
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_turn_route_injects_overrides_without_provider_switch(self):
        """Fast mode should add request_overrides but NOT change the provider/runtime."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.4",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        # Provider should NOT have changed
        assert route["runtime"]["provider"] == "openrouter"
        assert route["runtime"]["api_mode"] == "chat_completions"
        # But request_overrides should be set
        assert route["request_overrides"] == {"service_tier": "priority"}

    def test_turn_route_keeps_primary_runtime_when_model_has_no_fast_backend(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="gpt-5.3-codex",
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "openrouter"
        assert route.get("request_overrides") is None


class TestAnthropicFastMode(unittest.TestCase):
    """Verify Anthropic Fast Mode model support and override resolution."""

    def test_anthropic_opus_supported(self):
        from hermes_cli.models import model_supports_fast_mode

        # Native Anthropic format (hyphens)
        assert model_supports_fast_mode("claude-opus-4-6") is True
        # OpenRouter format (dots)
        assert model_supports_fast_mode("claude-opus-4.6") is True
        # With vendor prefix
        assert model_supports_fast_mode("anthropic/claude-opus-4-6") is True
        assert model_supports_fast_mode("anthropic/claude-opus-4.6") is True

    def test_anthropic_non_opus46_models_excluded(self):
        """Only the opus/sonnet/haiku 4.x families are fast-capable.

        Older Claude families (3.x, 2) have no fast offering on any endpoint.
        The 4.x families are now permitted at the model level; the adapter's
        endpoint-aware gate decides whether the param actually ships.
        """
        from hermes_cli.models import model_supports_fast_mode

        assert model_supports_fast_mode("claude-3-5-sonnet") is False
        assert model_supports_fast_mode("claude-3-5-haiku") is False
        assert model_supports_fast_mode("claude-2") is False
        # 4.x families now permitted at the model level (adapter gates per-endpoint)
        assert model_supports_fast_mode("claude-sonnet-4-6") is True
        assert model_supports_fast_mode("claude-opus-4-8") is True
        assert model_supports_fast_mode("claude-haiku-4-5") is True

    def test_non_claude_models_not_anthropic_fast(self):
        """Non-Claude models should not be treated as Anthropic fast-mode."""
        from hermes_cli.models import _is_anthropic_fast_model

        assert _is_anthropic_fast_model("gpt-5.4") is False
        assert _is_anthropic_fast_model("gemini-3-pro") is False
        assert _is_anthropic_fast_model("kimi-k2-thinking") is False

    def test_anthropic_variant_tags_stripped(self):
        from hermes_cli.models import model_supports_fast_mode

        # OpenRouter variant tags after colon should be stripped
        assert model_supports_fast_mode("claude-opus-4.6:fast") is True
        assert model_supports_fast_mode("claude-opus-4.6:beta") is True

    def test_resolve_overrides_returns_speed_for_anthropic(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("claude-opus-4-6")
        assert result == {"speed": "fast"}

        result = resolve_fast_mode_overrides("anthropic/claude-opus-4.6")
        assert result == {"speed": "fast"}

    def test_resolve_overrides_speed_for_fast_capable_claude(self):
        """Fast mode now spans the opus/sonnet/haiku 4.x families (endpoint-aware).

        The model-level helper is permissive — it returns {"speed":"fast"} for any
        fast-capable Claude family because the *adapter* makes the final per-endpoint
        call (Copilot accepts it on 4.x; native Anthropic still 400s on 4.7/4.8 and is
        refused there by ``anthropic_adapter._supports_fast_mode``).
        """
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-opus-4-8") == {"speed": "fast"}
        assert resolve_fast_mode_overrides("claude-opus-4.8-fast") == {"speed": "fast"}
        assert resolve_fast_mode_overrides("claude-sonnet-4-6") == {"speed": "fast"}
        assert resolve_fast_mode_overrides("claude-haiku-4-5") == {"speed": "fast"}

    def test_resolve_overrides_returns_none_for_non_fast_claude(self):
        """Claude families with no 4.x fast offering get no speed override."""
        from hermes_cli.models import resolve_fast_mode_overrides

        assert resolve_fast_mode_overrides("claude-3-5-sonnet") is None
        assert resolve_fast_mode_overrides("claude-2") is None

    def test_resolve_overrides_returns_service_tier_for_openai(self):
        """OpenAI models should still get service_tier, not speed."""
        from hermes_cli.models import resolve_fast_mode_overrides

        result = resolve_fast_mode_overrides("gpt-5.4")
        assert result == {"service_tier": "priority"}

    def test_is_anthropic_fast_model(self):
        """Fast mode spans opus/sonnet/haiku 4.x (endpoint-aware; adapter gates final)."""
        from hermes_cli.models import _is_anthropic_fast_model

        # Supported: opus / sonnet / haiku 4.x AND newer families (sonnet-5, …),
        # in any form, incl the synthetic -fast id. Endpoint-aware; adapter gates final.
        assert _is_anthropic_fast_model("claude-opus-4-6") is True
        assert _is_anthropic_fast_model("claude-opus-4.6") is True
        assert _is_anthropic_fast_model("anthropic/claude-opus-4-6") is True
        assert _is_anthropic_fast_model("claude-opus-4.6:fast") is True
        assert _is_anthropic_fast_model("claude-opus-4-7") is True
        assert _is_anthropic_fast_model("claude-opus-4-8") is True
        assert _is_anthropic_fast_model("claude-opus-4.8-fast") is True
        assert _is_anthropic_fast_model("claude-sonnet-4-6") is True
        assert _is_anthropic_fast_model("claude-haiku-4-5") is True
        # Newer families must NOT require a per-release allow-list edit.
        assert _is_anthropic_fast_model("claude-sonnet-5") is True
        assert _is_anthropic_fast_model("claude-sonnet-5-fast") is True

        # Unsupported — genuinely-legacy Claude (2.x/3.x) and non-Claude
        assert _is_anthropic_fast_model("claude-3-5-sonnet") is False
        assert _is_anthropic_fast_model("claude-2") is False
        assert _is_anthropic_fast_model("gpt-5.4") is False
        assert _is_anthropic_fast_model("") is False

    def test_fast_command_exposed_for_anthropic_model(self):
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-opus-4-6", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_fast_command_shown_for_anthropic_sonnet(self):
        """Sonnet 4.x is fast-capable (Copilot endpoint) — /fast must be available."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-sonnet-4-6", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_fast_command_hidden_for_legacy_claude(self):
        """Pre-4.x Claude (3.x/2) has no fast offering — /fast must hide."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-3-5-sonnet", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False

    def test_fast_command_shown_for_anthropic_opus_47(self):
        """Opus 4.7/4.8 are fast-capable on the Copilot endpoint — /fast available."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="anthropic", requested_provider="anthropic",
            model="claude-opus-4-7", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is True

    def test_fast_command_hidden_for_non_claude_non_openai(self):
        """Non-Claude, non-OpenAI models should not expose /fast."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            provider="gemini", requested_provider="gemini",
            model="gemini-3-pro-preview", agent=None,
        )
        assert cli_mod.HermesCLI._fast_command_available(stub) is False

    def test_turn_route_injects_speed_for_anthropic(self):
        """Anthropic models should get speed:'fast' override, not service_tier."""
        cli_mod = _import_cli()
        stub = SimpleNamespace(
            model="claude-opus-4-6",
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            api_mode="anthropic_messages",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            service_tier="priority",
        )

        route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")

        assert route["runtime"]["provider"] == "anthropic"
        assert route["request_overrides"] == {"speed": "fast"}


class TestAnthropicFastModeAdapter(unittest.TestCase):
    """Verify build_anthropic_kwargs handles fast_mode parameter."""

    def test_fast_mode_adds_speed_and_beta(self):
        from agent.anthropic_adapter import build_anthropic_kwargs, _FAST_MODE_BETA

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
        )
        assert kwargs.get("extra_body", {}).get("speed") == "fast"
        assert "speed" not in kwargs
        assert "extra_headers" in kwargs
        assert _FAST_MODE_BETA in kwargs["extra_headers"].get("anthropic-beta", "")

    def test_fast_id_kept_verbatim_on_copilot_wire(self):
        # Live wire truth (2026-08-21): on Copilot /v1/messages a "-fast" id is a
        # REAL catalog model. It MUST be sent VERBATIM (HTTP 200, bills 2x = real
        # fast mode); stripping it would send the base id and silently bill 1x.
        # The inert speed=fast param must NOT be attached on Copilot.
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4.8-fast",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            base_url="https://api.githubcopilot.com",
            fast_mode=True,
        )
        # -fast id preserved verbatim on the wire, DOT intact (hyphenated 400s).
        self.assertEqual(kwargs["model"], "claude-opus-4.8-fast")
        # speed=fast is INERT on Copilot → never attached.
        self.assertIsNone(kwargs.get("extra_body", {}).get("speed"))

    def test_fast_mode_off_no_speed(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=False,
        )
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs

    def test_fast_mode_skipped_for_third_party_endpoint(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
            base_url="https://api.minimax.io/anthropic/v1",
        )
        # Third-party endpoints should NOT get speed or fast-mode beta
        assert kwargs.get("extra_body", {}).get("speed") is None
        assert "speed" not in kwargs
        assert "extra_headers" not in kwargs

    def test_fast_mode_kwargs_are_safe_for_sdk_unpacking(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None,
            max_tokens=None,
            reasoning_config=None,
            fast_mode=True,
        )
        assert "speed" not in kwargs
        assert kwargs.get("extra_body", {}).get("speed") == "fast"


class TestConfigDefault(unittest.TestCase):
    def test_default_config_has_service_tier(self):
        from hermes_cli.config import DEFAULT_CONFIG

        agent = DEFAULT_CONFIG.get("agent", {})
        self.assertIn("service_tier", agent)
        self.assertEqual(agent["service_tier"], "")


class TestSyntheticFastVariant(unittest.TestCase):
    """The synthetic ``…-fast`` copilot model id: a UI/selection convenience that
    auto-attaches the speed=fast knob and normalizes to the base model on the wire."""

    def test_normalize_strips_fast_suffix(self):
        from hermes_cli.models import normalize_copilot_model_id

        catalog = [{"id": "claude-opus-4.8"}, {"id": "claude-sonnet-4.6"}]
        self.assertEqual(
            normalize_copilot_model_id("claude-opus-4.8-fast", catalog=catalog),
            "claude-opus-4.8",
        )
        self.assertEqual(
            normalize_copilot_model_id("copilot/claude-sonnet-4.6-fast", catalog=catalog),
            "claude-sonnet-4.6",
        )

    def test_fast_variant_resolves_speed_override(self):
        from hermes_cli.models import resolve_fast_mode_overrides

        self.assertEqual(
            resolve_fast_mode_overrides("claude-opus-4.8-fast"), {"speed": "fast"}
        )

    def test_catalog_does_not_synthesize_fast_variants(self):
        # Live wire truth (2026-08-21): Copilot's "-fast" is a REAL catalog id, not
        # a synthetic knob. Hermes must NOT manufacture "<model>-fast" companions —
        # any id lacking a real Copilot -fast catalog entry 400s "model_not_supported".
        # Only real "-fast" ids present in the live fetch survive; nothing is injected.
        import hermes_cli.models as M

        base_items = [
            {"id": "claude-opus-4.8", "name": "Claude Opus 4.8",
             "model_picker_enabled": True, "capabilities": {"type": "chat"}},
            # A REAL -fast catalog id, as the live Copilot catalog actually returns.
            {"id": "claude-opus-4.8-fast", "name": "Claude Opus 4.8 (Fast)",
             "model_picker_enabled": True, "capabilities": {"type": "chat"}},
            {"id": "claude-sonnet-5", "name": "Claude Sonnet 5",
             "model_picker_enabled": True, "capabilities": {"type": "chat"}},
            {"id": "gpt-5", "name": "GPT-5",
             "model_picker_enabled": True, "capabilities": {"type": "chat"}},
        ]
        orig = M._fetch_github_model_catalog_items
        M._fetch_github_model_catalog_items = lambda **kw: base_items
        try:
            cat = M.fetch_github_model_catalog(api_key="x")
        finally:
            M._fetch_github_model_catalog_items = orig
        ids = [c["id"] for c in (cat or [])]
        # The REAL -fast id from the live fetch is preserved.
        self.assertIn("claude-opus-4.8-fast", ids)
        # No synthetic companions manufactured for families lacking a real -fast entry.
        self.assertNotIn("claude-sonnet-5-fast", ids)
        self.assertNotIn("claude-opus-4.8-fast-fast", ids)
        self.assertNotIn("gpt-5-fast", ids)
        # And no entry is tagged as a Hermes synthetic companion anymore.
        self.assertFalse(any(c.get("_hermes_synthetic_fast") for c in (cat or [])))

    def test_synthetic_fast_id_strips_to_base_on_wire(self):
        # A synthetic -fast id (claude-sonnet-5-fast, NOT a real catalog id) must
        # normalize to its base on the wire; a real -fast catalog id
        # (claude-opus-4.8-fast) is preserved verbatim.
        from hermes_cli.models import normalize_copilot_model_id

        catalog = [
            {"id": "claude-sonnet-5"},
            {"id": "claude-opus-4.8"},
            {"id": "claude-opus-4.8-fast"},  # real Copilot catalog id
            {"id": "claude-sonnet-5-fast", "_hermes_synthetic_fast": True},
        ]
        self.assertEqual(
            normalize_copilot_model_id("claude-sonnet-5-fast", catalog=catalog),
            "claude-sonnet-5",
        )
        self.assertEqual(
            normalize_copilot_model_id("claude-opus-4.8-fast", catalog=catalog),
            "claude-opus-4.8-fast",
        )

    def test_adapter_omits_inert_speed_for_copilot_opus48(self):
        # Live wire truth (2026-08-21): speed=fast is INERT on Copilot (bills 1x).
        # Real fast mode comes only from the -fast catalog model id. So for a BASE
        # id + fast_mode=True on Copilot, the adapter must NOT attach speed=fast.
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4.8",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None, max_tokens=None, reasoning_config=None,
            base_url="https://api.githubcopilot.com", fast_mode=True,
        )
        self.assertIsNone(kwargs.get("extra_body", {}).get("speed"))
        # Base id keeps its established normalized wire form (no synthetic -fast).
        self.assertEqual(kwargs["model"], "claude-opus-4-8")
        self.assertNotIn("-fast", kwargs["model"])

    def test_adapter_refuses_speed_for_native_opus48(self):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4.8",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=None, max_tokens=None, reasoning_config=None,
            base_url="https://api.anthropic.com", fast_mode=True,
        )
        # native Anthropic still 400s on 4.8 fast → adapter must NOT attach it
        self.assertIsNone(kwargs.get("extra_body", {}).get("speed"))
