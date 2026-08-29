"""Oneshot (`hermes -z`) must pass a config-derived reasoning_config to AIAgent.

Regression cover for the bug where ``hermes_cli.oneshot._run_agent`` built its
AIAgent without ``reasoning_config``. Because
``agent.anthropic_adapter.build_anthropic_kwargs`` gates its entire ``thinking``
block on ``if reasoning_config and isinstance(reasoning_config, dict)``, a None
value meant Claude/Gemini on the Anthropic lane were never asked to think out
loud, so ``-z`` runs persisted zero reasoning while TUI sessions captured it
normally. Providers on the chat_completions lane (grok, gpt) were unaffected,
which is why the gap went unnoticed.

Run: python -m pytest tests/hermes_cli/test_oneshot_reasoning_config.py -q
"""
from __future__ import annotations

import sys
import types

import pytest

from hermes_constants import resolve_reasoning_config


# ── the shared resolver ────────────────────────────────────────────────


class TestResolveReasoningConfig:
    """hermes_constants.resolve_reasoning_config: config mapping → dict|None."""

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
    def test_every_valid_effort_round_trips(self, effort):
        assert resolve_reasoning_config({"agent": {"reasoning_effort": effort}}) == {
            "enabled": True,
            "effort": effort,
        }

    def test_none_disables_reasoning(self):
        assert resolve_reasoning_config({"agent": {"reasoning_effort": "none"}}) == {
            "enabled": False
        }

    def test_case_and_whitespace_normalized(self):
        assert resolve_reasoning_config({"agent": {"reasoning_effort": "  HIGH "}}) == {
            "enabled": True,
            "effort": "high",
        }

    @pytest.mark.parametrize(
        "cfg",
        [
            {},                                  # no agent section
            {"agent": {}},                       # no key
            {"agent": {"reasoning_effort": ""}},  # empty value
            {"agent": {"reasoning_effort": None}},
            None,                                # no config at all
            "not-a-mapping",                     # garbage
            {"agent": "oops-a-scalar"},          # malformed section
        ],
    )
    def test_absence_or_malformation_degrades_to_none(self, cfg):
        """None means 'use the provider default' — never an exception."""
        assert resolve_reasoning_config(cfg) is None

    def test_unknown_effort_warns_and_degrades(self, caplog):
        with caplog.at_level("WARNING"):
            assert resolve_reasoning_config({"agent": {"reasoning_effort": "turbo"}}) is None
        assert "turbo" in caplog.text

    def test_matches_the_interactive_cli_resolver(self):
        """Oneshot must not drift from what `hermes chat` resolves."""
        from hermes_constants import parse_reasoning_effort

        for effort in ["max", "high", "none", "", "bogus"]:
            cfg = {"agent": {"reasoning_effort": effort}}
            assert resolve_reasoning_config(cfg) == parse_reasoning_effort(effort)


class TestAllCallSitesShareOneResolver:
    """cli.py, gateway/run.py, cron and oneshot must not drift apart.

    Before this change each of the four entry points hand-rolled its own
    read-``agent.reasoning_effort``-parse-warn block. Oneshot was the one that
    forgot entirely, which is the bug. Duplicated logic is how that happens, so
    assert they now share a single implementation.
    """

    CASES = [
        {"agent": {"reasoning_effort": "max"}},
        {"agent": {"reasoning_effort": "MAX"}},
        {"agent": {"reasoning_effort": "  high  "}},
        {"agent": {"reasoning_effort": "none"}},
        {"agent": {"reasoning_effort": ""}},
        {"agent": {"reasoning_effort": None}},
        {"agent": {"reasoning_effort": "bogus"}},
        {"agent": {}},
        {},
    ]

    def test_cli_alias_matches_the_shared_resolver(self):
        import cli

        for cfg in self.CASES:
            assert cli._resolve_reasoning_config(cfg) == resolve_reasoning_config(cfg), cfg

    def test_gateway_loader_matches_the_shared_resolver(self, monkeypatch):
        from gateway import run as gateway_run

        for cfg in self.CASES:
            monkeypatch.setattr(
                gateway_run, "_load_gateway_runtime_config", lambda c=cfg: c
            )
            got = gateway_run.GatewayRunner._load_reasoning_config()
            assert got == resolve_reasoning_config(cfg), cfg

    def test_cron_scheduler_uses_the_shared_resolver(self):
        import inspect

        from cron import scheduler

        source = inspect.getsource(scheduler)
        assert "resolve_reasoning_config(_cfg)" in source
        assert "parse_reasoning_effort(effort)" not in source

    def test_oneshot_uses_the_shared_resolver(self):
        """Compare behaviour, not object identity.

        Sibling suites wipe and re-import modules, so ``is`` comparisons on a
        function object are unstable across the full run. What matters is that
        oneshot resolves identically to the shared helper.
        """
        from hermes_cli import oneshot

        assert oneshot.resolve_reasoning_config.__name__ == "resolve_reasoning_config"
        for cfg in self.CASES:
            assert oneshot.resolve_reasoning_config(cfg) == resolve_reasoning_config(cfg), cfg


# ── oneshot wiring ─────────────────────────────────────────────────────


class _FakeAgent:
    """Stands in for AIAgent, recording construction and post-init mutations."""

    last_kwargs: dict = {}
    post_init_sets: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        type(self).post_init_sets = {}
        object.__setattr__(self, "_constructed", True)

    def __setattr__(self, name, value):
        # Everything oneshot assigns after construction lands here, so a future
        # `agent.reasoning_config = ...` fixup would be visible to the tests.
        if getattr(self, "_constructed", False):
            type(self).post_init_sets[name] = value
        object.__setattr__(self, name, value)

    def chat(self, prompt):
        return f"answered: {prompt}"


@pytest.fixture
def oneshot_env(monkeypatch):
    """Drive ``_run_agent`` with every external dependency stubbed.

    Returns a setter taking the config mapping oneshot should see; calling it
    runs ``_run_agent`` and hands back the kwargs AIAgent was built with.
    """
    from hermes_cli import config as config_module
    from hermes_cli import oneshot as oneshot_module
    from hermes_cli import runtime_provider, tools_config

    # AIAgent is imported inside _run_agent via `from run_agent import AIAgent`,
    # so patch it on the source module.
    fake_run_agent = sys.modules.get("run_agent")
    if fake_run_agent is None:
        fake_run_agent = types.ModuleType("run_agent")
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(fake_run_agent, "AIAgent", _FakeAgent, raising=False)

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kw: {
            "api_key": "k",
            "base_url": "https://api.githubcopilot.com",
            "provider": "copilot",
            "api_mode": "anthropic_messages",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda cfg, plat: set())
    monkeypatch.setattr(oneshot_module, "get_fallback_chain", lambda cfg: None)
    monkeypatch.setattr(oneshot_module, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

    def _run(cfg):
        monkeypatch.setattr(config_module, "load_config", lambda: cfg)
        _FakeAgent.last_kwargs = {}
        result = oneshot_module._run_agent("what is 47*53?", model="claude-opus-5")
        assert result == "answered: what is 47*53?"
        return _FakeAgent.last_kwargs

    return _run


class TestOneshotPassesReasoningConfig:
    def test_oneshot_derives_reasoning_config_from_config(self, oneshot_env):
        """(a) The regression itself: reasoning_config is no longer absent/None."""
        kwargs = oneshot_env({"agent": {"reasoning_effort": "max"}})
        assert "reasoning_config" in kwargs, "oneshot dropped reasoning_config entirely"
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "max"}

    @pytest.mark.parametrize(
        "configured,expected",
        [
            ("low", {"enabled": True, "effort": "low"}),
            ("high", {"enabled": True, "effort": "high"}),
            ("xhigh", {"enabled": True, "effort": "xhigh"}),
            ("none", {"enabled": False}),
        ],
    )
    def test_explicit_config_value_is_honoured(self, oneshot_env, configured, expected):
        """(b) Whatever the user configured wins — nothing is hardcoded."""
        kwargs = oneshot_env({"agent": {"reasoning_effort": configured}})
        assert kwargs["reasoning_config"] == expected

    def test_absent_config_degrades_safely_to_none(self, oneshot_env):
        """(c) No config → None → provider default, exactly as before the fix.

        This is what keeps the chat_completions lanes (grok, gpt) unchanged for
        users who never set agent.reasoning_effort.
        """
        assert oneshot_env({})["reasoning_config"] is None
        assert oneshot_env({"agent": {}})["reasoning_config"] is None
        assert oneshot_env({"agent": {"reasoning_effort": ""}})["reasoning_config"] is None

    def test_unknown_effort_does_not_break_the_run(self, oneshot_env):
        kwargs = oneshot_env({"agent": {"reasoning_effort": "turbo"}})
        assert kwargs["reasoning_config"] is None

    def test_reasoning_config_not_overwritten_after_construction(self, oneshot_env):
        """Guard the post-init fixups: they must not clobber reasoning_config.

        ``_run_agent`` reaches back into the agent after building it to silence
        streaming callbacks. If a future edit added ``agent.reasoning_config =
        ...`` there it would silently defeat the constructor argument, so assert
        the attribute survives untouched.
        """
        kwargs = oneshot_env({"agent": {"reasoning_effort": "high"}})
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
        assert _FakeAgent.post_init_sets.get("reasoning_config", "unset") == "unset"
        # the fixups we DO expect still ran
        assert _FakeAgent.post_init_sets["suppress_status_output"] is True
        assert _FakeAgent.post_init_sets["stream_delta_callback"] is None


class TestReachesTheAnthropicThinkingBlock:
    """The config value must actually flip on the adapter gate that was closed."""

    def _thinking_for(self, cfg):
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=4096,
            reasoning_config=resolve_reasoning_config(cfg),
        )
        return kwargs.get("thinking")

    def test_configured_effort_produces_a_thinking_block(self):
        thinking = self._thinking_for({"agent": {"reasoning_effort": "max"}})
        assert thinking is not None, "the adapter gate is still closed"
        assert thinking["type"] == "adaptive"
        assert thinking["display"] == "summarized"

    def test_no_config_reproduces_the_old_silent_behaviour(self):
        """Documents the pre-fix state: None config → no thinking on the wire."""
        assert self._thinking_for({}) is None

    def test_effort_none_keeps_thinking_off(self):
        assert self._thinking_for({"agent": {"reasoning_effort": "none"}}) is None
