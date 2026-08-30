"""Tests for the first-turn planning prelude.

The prelude appends a short todo reminder to the model-facing copy of the first
user turn. It exists because gpt and grok open multi-step work with a todo
unprompted while claude and gemini score zero on the same prompt, and three
rounds of system-prompt changes (a 71% size cut, a depth move from 74% to 8%,
and deleting a stale fake-tool reference) moved that number not at all.

The whole safety story is the gate chain, so each gate is tested independently:
a reminder that fires on the wrong turn is worse than no reminder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import planning_prelude


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep a stray HERMES_PLANNING_PRELUDE export out of every test.

    resolve_mode() reads the env var ahead of config, so without this an
    exported value on the developer's shell would silently override the mode a
    test set up and the assertions would stop meaning what they say.
    """
    monkeypatch.delenv(planning_prelude.ENV_VAR, raising=False)


class _Store:
    def __init__(self, has: bool = False) -> None:
        self._has = has

    def has_items(self) -> bool:
        return self._has


def _agent(**over):
    """An agent stub in the state the turn prologue leaves it: first turn, todo
    loaded, empty list, prelude enabled."""
    base = dict(
        _planning_prelude_mode=planning_prelude.MODE_PREFERRED,
        valid_tool_names=["todo", "terminal"],
        _user_turn_count=1,
        _todo_store=_Store(False),
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestModeResolution:
    def test_absent_config_is_off(self, monkeypatch):
        monkeypatch.delenv(planning_prelude.ENV_VAR, raising=False)
        assert planning_prelude.resolve_mode(SimpleNamespace()) == planning_prelude.MODE_OFF

    @pytest.mark.parametrize("mode", ["off", "preferred", "always"])
    def test_valid_modes_round_trip(self, mode, monkeypatch):
        monkeypatch.delenv(planning_prelude.ENV_VAR, raising=False)
        assert planning_prelude.resolve_mode(SimpleNamespace(_planning_prelude_mode=mode)) == mode

    def test_case_and_whitespace_tolerated(self, monkeypatch):
        monkeypatch.delenv(planning_prelude.ENV_VAR, raising=False)
        agent = SimpleNamespace(_planning_prelude_mode="  Always  ")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_ALWAYS

    def test_typo_degrades_to_off_rather_than_raising(self, monkeypatch):
        # A bad config value must not break every turn.
        monkeypatch.delenv(planning_prelude.ENV_VAR, raising=False)
        agent = SimpleNamespace(_planning_prelude_mode="alwyas")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_OFF

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv(planning_prelude.ENV_VAR, "always")
        agent = SimpleNamespace(_planning_prelude_mode="off")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_ALWAYS

    def test_empty_env_var_falls_through_to_config(self, monkeypatch):
        # An exported-but-blank var must not silently disable a configured mode.
        monkeypatch.setenv(planning_prelude.ENV_VAR, "   ")
        agent = SimpleNamespace(_planning_prelude_mode="preferred")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_PREFERRED


class TestPerModelRouting:
    """The families differ, so which models get the reminder lives in config."""

    def _agent(self, model, rules, default="off"):
        return SimpleNamespace(
            model=model,
            _planning_prelude_models=rules,
            _planning_prelude_mode=default,
        )

    def test_glob_matches_family(self):
        rules = {"claude-*": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("claude-opus-5", rules)) == planning_prelude.MODE_ALWAYS

    def test_bare_substring_also_matches(self):
        # Writing "gemini" is the obvious thing; silently not matching is a trap.
        rules = {"gemini": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("gemini-3.5-flash", rules)) == planning_prelude.MODE_ALWAYS

    def test_unmatched_model_falls_back_to_global_default(self):
        rules = {"claude-*": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("gpt-5.6-sol", rules, default="off")) == planning_prelude.MODE_OFF

    def test_first_matching_rule_wins(self):
        # Declaration order lets a specific id sit above a broader family glob.
        rules = {"claude-opus-5": "preferred", "claude-*": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("claude-opus-5", rules)) == planning_prelude.MODE_PREFERRED

    def test_per_model_rule_beats_global_default(self):
        rules = {"gemini-*": "always"}
        agent = self._agent("gemini-3.5-flash", rules, default="off")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_ALWAYS

    def test_env_var_still_wins_over_per_model(self, monkeypatch):
        # The harness must be able to override everything for one process.
        monkeypatch.setenv(planning_prelude.ENV_VAR, "off")
        rules = {"claude-*": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("claude-opus-5", rules)) == planning_prelude.MODE_OFF

    def test_case_insensitive_matching(self):
        rules = {"CLAUDE-*": "always"}
        assert planning_prelude.resolve_mode(
            self._agent("claude-opus-5", rules)) == planning_prelude.MODE_ALWAYS

    def test_no_rules_configured_is_harmless(self):
        agent = SimpleNamespace(model="claude-opus-5", _planning_prelude_mode="off")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_OFF

    def test_missing_model_attribute_is_tolerated(self):
        agent = SimpleNamespace(
            _planning_prelude_models={"claude-*": "always"}, _planning_prelude_mode="off")
        assert planning_prelude.resolve_mode(agent) == planning_prelude.MODE_OFF


class TestGates:
    def test_fires_on_a_first_turn_multi_step_request(self):
        assert planning_prelude.should_inject(_agent(), "Refactor the parser and add tests") is True

    def test_gate_off_by_default(self):
        agent = _agent(_planning_prelude_mode=planning_prelude.MODE_OFF)
        assert planning_prelude.should_inject(agent, "Refactor the parser") is False

    def test_gate_todo_tool_not_loaded(self):
        # Pointing at a tool the model does not have is worse than silence.
        agent = _agent(valid_tool_names=["terminal"])
        assert planning_prelude.should_inject(agent, "Refactor the parser") is False

    def test_gate_not_first_turn(self):
        # A reminder mid-conversation is noise; the prologue increments the
        # counter before this runs, so turn one is 1.
        agent = _agent(_user_turn_count=2)
        assert planning_prelude.should_inject(agent, "Refactor the parser") is False

    def test_gate_todo_list_already_exists(self):
        agent = _agent(_todo_store=_Store(True))
        assert planning_prelude.should_inject(agent, "Refactor the parser") is False

    @pytest.mark.parametrize("text", ["What does this function do?", "Nice work!", "why?  "])
    def test_gate_questions_and_exclamations(self, text):
        # These want an answer, not a project plan.
        assert planning_prelude.should_inject(_agent(), text) is False

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_gate_empty_message(self, text):
        assert planning_prelude.should_inject(_agent(), text) is False

    def test_broken_todo_store_does_not_break_the_turn(self):
        class Exploding:
            def has_items(self):
                raise RuntimeError("store unavailable")

        agent = _agent(_todo_store=Exploding())
        # Degrades to "no list known", so the reminder still applies.
        assert planning_prelude.should_inject(agent, "Refactor the parser") is True

    def test_missing_store_attribute_is_tolerated(self):
        agent = _agent(_todo_store=None)
        assert planning_prelude.should_inject(agent, "Refactor the parser") is True


class TestApply:
    def test_preferred_appends_soft_reminder(self):
        out = planning_prelude.apply(_agent(), "Refactor the parser")
        assert out.startswith("Refactor the parser")
        assert "`todo`" in out
        assert "consider" in out.lower()

    def test_always_appends_stronger_reminder(self):
        agent = _agent(_planning_prelude_mode=planning_prelude.MODE_ALWAYS)
        out = planning_prelude.apply(agent, "Refactor the parser")
        assert out.startswith("Refactor the parser")
        assert "`todo`" in out
        assert "expects" in out.lower()

    @pytest.mark.parametrize("mode", ["preferred", "always"])
    def test_both_modes_name_skills_not_just_todo(self, mode):
        # The first A/B measured a todo-only reminder: todo adoption moved off
        # zero and skills adoption did not budge. A reminder steers what it
        # names, so the skills half has to be said out loud in both modes.
        agent = _agent(_planning_prelude_mode=mode)
        out = planning_prelude.apply(agent, "Refactor the parser")
        assert "skills_list" in out
        assert "`todo`" in out

    def test_off_returns_the_message_unchanged(self):
        agent = _agent(_planning_prelude_mode=planning_prelude.MODE_OFF)
        assert planning_prelude.apply(agent, "Refactor the parser") == "Refactor the parser"

    def test_user_text_is_never_mutated_only_appended(self):
        original = "Refactor the parser and add tests"
        out = planning_prelude.apply(_agent(), original)
        # The user's words survive verbatim as a prefix; nothing is rewritten.
        assert out[: len(original)] == original

    def test_reminder_is_bracketed_so_it_reads_as_an_environment_note(self):
        out = planning_prelude.apply(_agent(), "Refactor the parser")
        appended = out[len("Refactor the parser"):].strip()
        assert appended.startswith("[") and appended.endswith("]")

    def test_gated_turn_returns_input_identity(self):
        agent = _agent(_user_turn_count=3)
        msg = "Keep going"
        assert planning_prelude.apply(agent, msg) is msg


class TestTurnContextWiring:
    """The persisted/displayed message must stay the user's actual words."""

    def test_prologue_appends_to_model_copy_only(self):
        import inspect

        from agent import turn_context

        src = inspect.getsource(turn_context.build_turn_context)
        # original_user_message is captured BEFORE the prelude is applied, and the
        # prelude result is used only for the appended chat message.
        assert "model_facing_user_message = _planning_prelude.apply" in src
        assert '{"role": "user", "content": model_facing_user_message}' in src
        original_idx = src.index("original_user_message =")
        prelude_idx = src.index("model_facing_user_message =")
        assert original_idx < prelude_idx, "prelude must not taint original_user_message"
