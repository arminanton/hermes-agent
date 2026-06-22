"""Tests for GitHub Copilot entries shown in the /model picker."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_uses_live_catalog_when_available():
    # Pick live models that do NOT overlap the hidden-usable supplements so the
    # assertions below isolate the live-catalog contribution cleanly.
    live_models = ["gpt-5.4", "claude-sonnet-4.6", "gpt-5.5"]

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=live_models):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)

    assert copilot is not None
    # The picker uses the live catalog and then appends account-usable models the
    # live /models endpoint omits (hidden/preview slugs that work for inference),
    # deduped with live entries winning. Assert the live models all appear in
    # order at the front, and that only known hidden-usable supplements follow.
    from hermes_cli.models import _COPILOT_HIDDEN_USABLE

    assert copilot["models"][: len(live_models)] == live_models
    supplements = copilot["models"][len(live_models) :]
    assert all(m in _COPILOT_HIDDEN_USABLE for m in supplements)
    assert copilot["total_models"] == len(copilot["models"])
