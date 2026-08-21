"""Tests for agent.models_dev — models.dev registry integration."""
from unittest.mock import patch, MagicMock

from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    fetch_models_dev,
    get_model_capabilities,
    lookup_models_dev_context,
)


SAMPLE_REGISTRY = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-opus-4-6": {
                "id": "claude-opus-4-6",
                "limit": {"context": 1000000, "output": 128000},
            },
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "limit": {"context": 1000000, "output": 64000},
            },
            "claude-sonnet-4-0": {
                "id": "claude-sonnet-4-0",
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "models": {
            "claude-opus-4.6": {
                "id": "claude-opus-4.6",
                "limit": {"context": 128000, "output": 32000},
            },
        },
    },
    "xai": {
        "id": "xai",
        "name": "xAI",
        "models": {
            "grok-build-0.1": {
                "id": "grok-build-0.1",
                "limit": {"context": 256000, "output": 64000},
            },
        },
    },
    "kilo": {
        "id": "kilo",
        "name": "Kilo Gateway",
        "models": {
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "limit": {"context": 1000000, "output": 128000},
            },
        },
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "id": "deepseek-chat",
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "audio-only": {
        "id": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
}


class TestProviderMapping:
    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV

    def test_openai_codex_mapped_to_openai(self):
        assert PROVIDER_TO_MODELS_DEV["openai"] == "openai"
        assert PROVIDER_TO_MODELS_DEV["openai-codex"] == "openai"


class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000

    def test_zero_context_returns_none(self):
        assert _extract_context({"limit": {"context": 0}}) is None

    def test_missing_limit_returns_none(self):
        assert _extract_context({"id": "test"}) is None

    def test_missing_context_returns_none(self):
        assert _extract_context({"limit": {"output": 8192}}) is None

    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None

    def test_float_context_coerced_to_int(self):
        assert _extract_context({"limit": {"context": 131072.0}}) == 131072


class TestLookupModelsDevContext:
    @patch("agent.models_dev.fetch_models_dev")
    def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_case_insensitive_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "Claude-Opus-4-6") == 1000000

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_not_mapped(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("nous", "some-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_model_not_found(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "nonexistent-model") is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_provider_aware_context(self, mock_fetch):
        """Same model, different context per provider."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        # Anthropic direct: 1M
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000
        # GitHub Copilot: only 128K for same model
        assert lookup_models_dev_context("copilot", "claude-opus-4.6") == 128000

    @patch("agent.models_dev.fetch_models_dev")
    def test_xai_oauth_resolves_xai_context(self, mock_fetch):
        """xAI OAuth is an auth path, not a separate model catalog."""
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("xai-oauth", "grok-build-0.1") == 256000

    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None

    @patch("agent.models_dev.fetch_models_dev")
    def test_empty_registry(self, mock_fetch):
        mock_fetch.return_value = {}
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") is None


class TestFetchModelsDev:
    @patch("agent.models_dev.requests.get")
    def test_fetch_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Clear caches
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result
        assert len(result) == len(SAMPLE_REGISTRY)

    @patch("agent.models_dev.requests.get")
    def test_fetch_failure_returns_stale_cache(self, mock_get):
        mock_get.side_effect = Exception("network error")

        import agent.models_dev as md
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0  # expired

        with patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev(force_refresh=True)

        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_in_memory_cache_used(self, mock_get):
        import agent.models_dev as md
        import time
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time()  # fresh

        result = fetch_models_dev()
        mock_get.assert_not_called()
        assert result == SAMPLE_REGISTRY

    @patch("agent.models_dev.requests.get")
    def test_fresh_disk_cache_skips_network(self, mock_get):
        """When in-mem cache is empty but disk cache exists and is fresh by
        mtime (< TTL), fetch_models_dev returns disk data without ever
        making the network call.

        This is the cold-start fast path: every fresh process previously
        paid ~500 ms re-fetching a registry that was already on disk
        from an earlier run.
        """
        import agent.models_dev as md
        # Empty in-mem cache so stage 1 doesn't short-circuit.
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY):
            result = fetch_models_dev()

        # The whole point: no network call.
        mock_get.assert_not_called()
        assert "anthropic" in result
        # In-mem cache populated so subsequent calls within the same
        # process stay on stage 1.
        assert md._models_dev_cache == SAMPLE_REGISTRY

    @patch("agent.models_dev.requests.get")
    def test_stale_disk_cache_falls_through_to_network(self, mock_get):
        """When the disk cache is OLDER than TTL, we must hit the network
        (and only fall back to the stale disk data if network fails)."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Disk cache exists but is older than the TTL — must NOT short-circuit.
        with patch.object(md, "_disk_cache_age_seconds",
                          return_value=md._MODELS_DEV_CACHE_TTL + 60), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev()

        mock_get.assert_called_once()
        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_skips_disk_cache(self, mock_get):
        """force_refresh=True bypasses BOTH the in-mem cache AND the
        disk-cache fast path. Used by ``hermes config refresh`` and
        anywhere else the user explicitly asked for fresh data.
        """
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Disk cache is fresh, but force_refresh must override it.
        with patch.object(md, "_disk_cache_age_seconds", return_value=60.0), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev(force_refresh=True)

        mock_get.assert_called_once()
        assert "anthropic" in result

    @patch("agent.models_dev.requests.get")
    def test_missing_disk_cache_falls_through_to_network(self, mock_get):
        """If the disk cache file doesn't exist (first-ever run, or it
        was deleted), fall through cleanly to network."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_REGISTRY
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_save_disk_cache"):
            result = fetch_models_dev()

        mock_get.assert_called_once()
        assert "anthropic" in result


# ---------------------------------------------------------------------------
# get_model_capabilities — vision via modalities.input
# ---------------------------------------------------------------------------


CAPS_REGISTRY = {
    "google": {
        "id": "google",
        "models": {
            "gemma-4-31b-it": {
                "id": "gemma-4-31b-it",
                "attachment": False,
                "tool_call": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 128000, "output": 8192},
            },
            "gemma-3-1b": {
                "id": "gemma-3-1b",
                "tool_call": True,
                "limit": {"context": 32000, "output": 8192},
            },
            "text-only-with-stale-attachment": {
                "id": "text-only-with-stale-attachment",
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "models": {
            "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "attachment": True,
                "tool_call": True,
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
}


class TestGetModelCapabilities:
    """Tests for get_model_capabilities vision detection."""

    def test_vision_from_attachment_flag(self):
        """Models with attachment=True and no modalities should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True

    def test_vision_from_modalities_input_image(self):
        """Models with 'image' in modalities.input but attachment=False should
        still report supports_vision=True (the core fix in this PR)."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-4-31b-it")
        assert caps is not None
        assert caps.supports_vision is True

    def test_text_only_modalities_override_stale_attachment_flag(self):
        """Text-only modalities must win over stale attachment=True metadata."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "text-only-with-stale-attachment")
        assert caps is not None
        assert caps.supports_vision is False

    def test_no_vision_without_attachment_or_modalities(self):
        """Models with neither attachment nor image modality should be non-vision."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("google", "gemma-3-1b")
        assert caps is not None
        assert caps.supports_vision is False

    def test_modalities_non_dict_handled(self):
        """Non-dict modalities field should not crash."""
        registry = {
            "google": {"id": "google", "models": {
                "weird-model": {
                    "id": "weird-model",
                    "modalities": "text",  # not a dict
                    "limit": {"context": 200000, "output": 8192},
                },
            }},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("gemini", "weird-model")
        assert caps is not None
        assert caps.supports_vision is False

    def test_model_not_found_returns_none(self):
        """Unknown model should return None."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "nonexistent-model")
        assert caps is None


# ---------------------------------------------------------------------------
# Per-model metadata overrides (model_overrides config)
# ---------------------------------------------------------------------------


class TestModelOverrides:
    """Tests for the model_overrides config system (ported dafdba324a +
    de47d19f1f, adapted to compose with our probe-verified override layer)."""

    def _with_overrides(self, overrides_dict):
        import agent.models_dev as md
        return patch.object(md, "_load_model_overrides", return_value=overrides_dict)

    # --- explicit resolution ---

    def test_explicit_per_provider_model(self):
        from agent.models_dev import _explicit_model_override
        ov = {"upstage": {"solar-pro4": {"context_window": 524288}}}
        with self._with_overrides(ov):
            r = _explicit_model_override("upstage", "solar-pro4")
        assert r == {"context_window": 524288}

    def test_explicit_case_insensitive_model(self):
        from agent.models_dev import _explicit_model_override
        ov = {"upstage": {"Solar-Pro4": {"context_window": 1}}}
        with self._with_overrides(ov):
            r = _explicit_model_override("upstage", "solar-pro4")
        assert r == {"context_window": 1}

    def test_explicit_skips_default_sentinel(self):
        """A model literally named _default is not matched by the model lookup."""
        from agent.models_dev import _explicit_model_override
        ov = {"upstage": {"_default": {"context_window": 128000}}}
        with self._with_overrides(ov):
            r = _explicit_model_override("upstage", "anything")
        assert r is None

    def test_no_override_returns_none(self):
        from agent.models_dev import _explicit_model_override
        with self._with_overrides({}):
            assert _explicit_model_override("anthropic", "claude") is None

    # --- fill-gap _default semantics ---

    def test_default_is_fill_gap_only_not_explicit(self):
        """_override_context_window (explicit-only) must NOT return a _default."""
        from agent.models_dev import _override_context_window
        ov = {"upstage": {"_default": {"context_window": 128000}}}
        with self._with_overrides(ov):
            assert _override_context_window("upstage", "unknown") is None

    def test_default_fills_gap_on_catalog_miss(self):
        from agent.models_dev import _default_override_context
        ov = {"upstage": {"_default": {"context_window": 128000}}}
        with self._with_overrides(ov):
            assert _default_override_context("upstage") == 128000

    def test_global_default_fallback(self):
        from agent.models_dev import _default_override_context
        ov = {"_default": {"context_window": 65536}}
        with self._with_overrides(ov):
            assert _default_override_context("unknown-provider") == 65536

    def test_per_provider_default_beats_global(self):
        from agent.models_dev import _default_override_context
        ov = {"upstage": {"_default": {"context_window": 111}}, "_default": {"context_window": 222}}
        with self._with_overrides(ov):
            assert _default_override_context("upstage") == 111

    def test_default_does_not_clamp_known_catalog_model(self):
        """A _default: {context_window: X} must NOT override a catalog-known model."""
        registry = {"anthropic": {"id": "anthropic", "models": {
            "claude-x": {"id": "claude-x", "limit": {"context": 200000, "output": 8192}},
        }}}
        ov = {"anthropic": {"_default": {"context_window": 999}}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            assert lookup_models_dev_context("anthropic", "claude-x") == 200000

    # --- lookup_models_dev_context integration ---

    def test_explicit_wins_over_catalog_in_lookup(self):
        registry = {"anthropic": {"id": "anthropic", "models": {
            "claude-x": {"id": "claude-x", "limit": {"context": 200000}},
        }}}
        ov = {"anthropic": {"claude-x": {"context_window": 777777}}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            assert lookup_models_dev_context("anthropic", "claude-x") == 777777

    def test_default_fills_gap_for_unknown_in_lookup(self):
        registry = {"anthropic": {"id": "anthropic", "models": {}}}
        ov = {"anthropic": {"_default": {"context_window": 128000}}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            assert lookup_models_dev_context("anthropic", "brand-new") == 128000

    # --- dual id-space ---

    def test_dual_id_space_hermes_and_modelsdev(self):
        from agent.models_dev import _override_context_window
        ov = {"copilot": {"gpt-5.6-luna": {"context_window": 700000}}}
        with self._with_overrides(ov):
            # Hermes id and models.dev id both resolve the same section.
            assert _override_context_window("copilot", "gpt-5.6-luna") == 700000
            assert _override_context_window("github-copilot", "gpt-5.6-luna") == 700000

    # --- sub-dict merge (canonical schema) ---

    def test_merge_preserves_catalog_output_when_only_context_set(self):
        from agent.models_dev import _merge_catalog_entry_with_override
        raw = {"limit": {"context": 200000, "output": 8192}, "tool_call": True}
        merged = _merge_catalog_entry_with_override(raw, {"context_window": 999999})
        assert merged["limit"]["context"] == 999999
        assert merged["limit"]["output"] == 8192   # NOT clobbered
        assert merged["tool_call"] is True

    def test_merge_vision_appends_to_modalities_input(self):
        from agent.models_dev import _merge_catalog_entry_with_override
        raw = {"modalities": {"input": ["text"]}}
        merged = _merge_catalog_entry_with_override(raw, {"supports_vision": True})
        assert merged["modalities"]["input"] == ["text", "image"]

    def test_merge_vision_false_removes_image(self):
        from agent.models_dev import _merge_catalog_entry_with_override
        raw = {"modalities": {"input": ["text", "image"]}}
        merged = _merge_catalog_entry_with_override(raw, {"supports_vision": False})
        assert "image" not in merged["modalities"]["input"]

    # --- capabilities ---

    def test_capabilities_override_patches_known_model(self):
        registry = {"google": {"id": "google", "models": {
            "gm": {"id": "gm", "limit": {"context": 100, "output": 50}, "tool_call": False},
        }}}
        ov = {"google": {"gm": {"context_window": 8192, "supports_tools": True}}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("google", "gm")
        assert caps.context_window == 8192
        assert caps.supports_tools is True
        assert caps.max_output_tokens == 50   # catalog value preserved

    def test_capabilities_override_only_for_unknown_model(self):
        """A model absent from the catalog is resolvable from the override alone."""
        registry = {"custom": {"id": "custom", "models": {}}}
        ov = {"custom": {"my-llava": {
            "context_window": 8192, "supports_vision": True, "supports_reasoning": False,
        }}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry), \
             patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {"custom": "custom"}):
            caps = get_model_capabilities("custom", "my-llava")
        assert caps is not None
        assert caps.context_window == 8192
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False
        assert caps.supports_tools is True   # safe default for unknown

    # --- one-shot warning on garbage ---

    def test_invalid_value_warns_once_and_ignored(self, caplog):
        import agent.models_dev as md
        # Clear the module-level warn dedup set for a clean assertion.
        md._OVERRIDE_WARNED_KEYS.clear()
        ov = {"upstage": {"m": {"context_window": "512k"}}}
        with self._with_overrides(ov):
            import logging
            with caplog.at_level(logging.WARNING):
                assert md._override_context_window("upstage", "m") is None
                assert md._override_context_window("upstage", "m") is None  # 2nd call
        warns = [r for r in caplog.records if "ignoring invalid context_window" in r.message]
        assert len(warns) == 1   # one-shot, not per-call

    # --- probe-compose (our divergence): user config wins over probe override ---

    def test_get_model_info_user_override_wins_over_probe(self):
        import agent.models_dev as md
        registry = {"github-copilot": {"id": "github-copilot", "models": {
            "gpt-5.6-luna": {"id": "gpt-5.6-luna", "limit": {"context": 128000, "output": 8192}},
        }}}
        ov = {"copilot": {"gpt-5.6-luna": {"context_window": 1234567}}}
        # Simulate a probe override that would otherwise set a different context.
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry), \
             patch.object(md, "_resolve_probe_override",
                          return_value={"context_window": 500000}):
            info = md.get_model_info("copilot", "gpt-5.6-luna")
        assert info is not None
        assert info.context_window == 1234567   # user config beats probe (500000)

    def test_get_model_info_partial_override_preserves_other_fields(self):
        import agent.models_dev as md
        registry = {"anthropic": {"id": "anthropic", "models": {
            "claude-x": {"id": "claude-x", "limit": {"context": 200000, "output": 64000},
                         "tool_call": True, "cost": {"input": 3.0, "output": 15.0}},
        }}}
        ov = {"anthropic": {"claude-x": {"context_window": 500000}}}
        with self._with_overrides(ov), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry), \
             patch.object(md, "_resolve_probe_override", return_value=None):
            info = md.get_model_info("anthropic", "claude-x")
        assert info.context_window == 500000     # overridden
        assert info.max_output == 64000          # preserved
        assert info.tool_call is True            # preserved
        assert info.cost_input == 3.0            # preserved

    # --- model_metadata step 0b wiring (the context-length entrypoint) ---

    def test_model_metadata_step_0b_explicit_override(self):
        """get_model_context_length must honor an explicit model_overrides
        context_window at step 0b, before any probe/cache/network."""
        import agent.models_dev as md
        import agent.model_metadata as mm
        ov = {"copilot": {"gpt-5.6-luna": {"context_window": 640000}}}
        with patch.object(md, "_load_model_overrides", return_value=ov):
            ctx = mm.get_model_context_length(
                "gpt-5.6-luna", provider="copilot",
                base_url="https://api.githubcopilot.com",
            )
        assert ctx == 640000

    def test_model_metadata_step_0b_ignores_default(self):
        """A _default override must NOT short-circuit step 0b (explicit-only),
        so it can't preempt custom_providers / live probes."""
        import agent.models_dev as md
        import agent.model_metadata as mm
        ov = {"copilot": {"_default": {"context_window": 640000}}}
        custom = [{
            "base_url": "https://example.invalid/v1",
            "models": {"gpt-5.6-luna": {"context_length": 320000}},
        }]
        with patch.object(md, "_load_model_overrides", return_value=ov):
            ctx = mm.get_model_context_length(
                "gpt-5.6-luna", provider="copilot",
                base_url="https://example.invalid/v1",
                custom_providers=custom,
            )
        # The provider default is fill-gap only. It must not preempt the more
        # specific custom-provider value resolved immediately after step 0b.
        assert ctx == 320000
