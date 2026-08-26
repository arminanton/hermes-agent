"""Copilot config persistence must keep API transport model derived."""

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.model_switch import ModelSwitchResult


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._evict_cached_agent = lambda session_key: None
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_global_copilot_switch_blanks_stale_api_mode(tmp_path, monkeypatch):
    """Gateway persistence must not pin one Copilot model's wire protocol."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "claude-opus-4.8",
                    "provider": "copilot",
                    "api_mode": "anthropic_messages",
                },
                "providers": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: ModelSwitchResult(
            success=True,
            new_model="gpt-5.6-sol",
            target_provider="copilot",
            provider_changed=False,
            api_key="copilot-token",
            base_url="https://api.business.githubcopilot.com",
            api_mode="codex_responses",
            provider_label="GitHub Copilot",
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *args, **kwargs: None,
    )

    reply = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.6-sol --provider copilot --global")
    )

    assert reply is not None
    written = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "gpt-5.6-sol"
    assert written["model"]["api_mode"] == ""
