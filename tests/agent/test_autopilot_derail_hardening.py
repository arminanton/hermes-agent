"""Regression tests for the autopilot cross-project derailment hardening.

Covers the three root causes diagnosed after the 2026-08-05 NuData run derailed
onto the wrong project:

  1. RESUME SEEDING — the resume kick must embed THIS session's verbatim tail and
     steer the model away from a topic-memory search, so a project-agnostic goal
     ("work on Phase A/B/C/Final") can't be re-hydrated against a different
     project that shares the same vocabulary.
  2. PANEL CHURN — the completion gate must honour ``COUNCIL_GATE_PANEL`` (a
     convergent audited panel) instead of hardcoding the never-settling ``fast``
     critic, while still degrading to fast when unset.
  3. RECEIPTS-ONLY FALLBACK — an engine that rejects an optional kwarg (older
     ``panel``/``evidence_receipts`` signatures) must NOT force the
     receipts-only fallback every tick; ``_council_run`` retries without the
     optional kwargs.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


# --------------------------------------------------------------------------- #
# Fix 1 — resume kick embeds the session tail + anti-derail steer.             #
# --------------------------------------------------------------------------- #
resume = importlib.import_module("agent.autopilot.resume")


def _jenkins_history():
    return [
        {"role": "user", "content": "refactor vars/messaging.groovy in jenkins-common into src/ classes"},
        {"role": "assistant", "content": "Extracted SlackColor + SlackMessage; gate green",
         "tool_calls": [{"function": {"name": "terminal"}}]},
        {"role": "tool", "content": "BUILD SUCCESSFUL in 34s"},
        {"role": "assistant", "content": "messaging domain done; next is C2 wave-2"},
    ]


def test_resume_kick_embeds_session_tail():
    kick = resume.build_resume_kick("work on Phase A B C Final sequencially", _jenkins_history())
    # The concrete work-in-flight must be inlined so the model continues THIS
    # project rather than inferring it from the goal wording.
    assert "jenkins-common" in kick
    assert "messaging" in kick
    assert "--- current session, last turns ---" in kick


def test_resume_kick_forbids_memory_search_grounding():
    """The specific behaviour that caused the derail (memory-grep the goal
    keywords) must be explicitly discouraged in the kick."""
    kick = resume.build_resume_kick("work on Phase A B C Final", _jenkins_history())
    low = kick.lower()
    assert "do not rely on a memory" in low or "not rely on a memory/keyword search" in low
    assert "authoritative source" in low


def test_resume_kick_surfaces_tool_calls_by_name():
    kick = resume.build_resume_kick("g", _jenkins_history())
    # "what the agent was doing" is a key anchor — tool call names are surfaced.
    assert "terminal" in kick
    assert "→called:" in kick or "called:" in kick


def test_resume_kick_cold_resume_degrades_to_bare_goal():
    """With no history there is nothing to anchor on; keep the original nudge."""
    kick = resume.build_resume_kick("some goal", [])
    assert "Resume and keep working" in kick
    assert "GROUND YOURSELF" not in kick
    assert "toward this goal: some goal" in kick


def test_resume_kick_none_history_is_safe():
    kick = resume.build_resume_kick("g", None)
    assert "Resume and keep working" in kick  # no crash, bare-goal form


def test_summarize_tail_respects_turn_limit(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_RESUME_TAIL_TURNS", "2")
    hist = [{"role": "assistant", "content": f"turn {i}"} for i in range(10)]
    tail = resume.summarize_session_tail(hist)
    # Only the last 2 substantive turns appear.
    assert "turn 9" in tail and "turn 8" in tail
    assert "turn 7" not in tail


def test_summarize_tail_disabled_with_zero(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_RESUME_TAIL_TURNS", "0")
    assert resume.summarize_session_tail(_jenkins_history()) == ""
    # …and the kick then degrades to the bare-goal form.
    kick = resume.build_resume_kick("g", _jenkins_history())
    assert "GROUND YOURSELF" not in kick


def test_summarize_tail_collapses_tool_results():
    hist = [
        {"role": "assistant", "content": "step 1"},
        {"role": "tool", "content": "x" * 5000},
        {"role": "tool", "content": "y" * 5000},
        {"role": "assistant", "content": "step 2"},
    ]
    tail = resume.summarize_session_tail(hist)
    # Tool bulk is collapsed to a marker, not inlined verbatim.
    assert "tool result(s)" in tail
    assert "x" * 100 not in tail


# --------------------------------------------------------------------------- #
# Fix 2 — completion gate honours COUNCIL_GATE_PANEL, degrades to fast.        #
# --------------------------------------------------------------------------- #
council_gate = importlib.import_module("agent.autopilot.council_gate")


def test_gate_panel_reads_env(monkeypatch):
    monkeypatch.setenv("COUNCIL_GATE_PANEL", "ship_gate_audited")
    assert council_gate._gate_panel() == "ship_gate_audited"


def test_gate_panel_empty_when_unset(monkeypatch):
    monkeypatch.delenv("COUNCIL_GATE_PANEL", raising=False)
    assert council_gate._gate_panel() == ""


def test_council_run_passes_configured_panel(monkeypatch):
    """When COUNCIL_GATE_PANEL is set, run_council receives it as `panel`."""
    seen = {}

    def fake_run_council(question, **kwargs):
        seen.update(kwargs)
        return {"verdict": "allow", "meta": {"panel": kwargs.get("panel", "fast")}}

    monkeypatch.setenv("COUNCIL_GATE_PANEL", "ship_gate_audited")
    delib = types.ModuleType("council.deliberation")
    delib.run_council = fake_run_council
    monkeypatch.setitem(sys.modules, "council.deliberation", delib)

    res = council_gate._council_run("q", mode="fast", max_tokens=100)
    assert seen.get("panel") == "ship_gate_audited"
    assert res["meta"]["panel"] == "ship_gate_audited"


def test_council_run_no_panel_when_unset(monkeypatch):
    seen = {}

    def fake_run_council(question, **kwargs):
        seen.update(kwargs)
        return {"verdict": "allow", "meta": {"panel": "fast"}}

    monkeypatch.delenv("COUNCIL_GATE_PANEL", raising=False)
    delib = types.ModuleType("council.deliberation")
    delib.run_council = fake_run_council
    monkeypatch.setitem(sys.modules, "council.deliberation", delib)

    council_gate._council_run("q", mode="fast", max_tokens=100)
    assert "panel" not in seen  # legacy fast-mode behaviour preserved


# --------------------------------------------------------------------------- #
# Fix 3 — an engine that rejects optional kwargs must not force receipts-only. #
# --------------------------------------------------------------------------- #
def test_council_run_degrades_when_engine_rejects_panel(monkeypatch):
    """Simulate an older engine whose run_council() has no `panel` kwarg.

    The retry must drop `panel` and still return a real result — NOT raise,
    which upstream would log as 'judge_criteria failed; receipts-only'.
    """
    calls = []

    def fake_run_council(question, **kwargs):
        calls.append(dict(kwargs))
        if "panel" in kwargs:
            raise TypeError("run_council() got an unexpected keyword argument 'panel'")
        return {"verdict": "allow", "meta": {"panel": "fast"}}

    monkeypatch.setenv("COUNCIL_GATE_PANEL", "ship_gate_audited")
    import types
    delib = types.ModuleType("council.deliberation")
    delib.run_council = fake_run_council
    monkeypatch.setitem(__import__("sys").modules, "council.deliberation", delib)

    res = council_gate._council_run("q", mode="fast", max_tokens=100)
    assert res["verdict"] == "allow"          # succeeded on retry
    assert len(calls) == 2                      # first with panel, retry without
    assert "panel" in calls[0]
    assert "panel" not in calls[1]


def test_council_run_degrades_when_engine_rejects_receipts(monkeypatch):
    """Older engine without `evidence_receipts` must degrade, not fail."""
    calls = []

    def fake_run_council(question, **kwargs):
        calls.append(dict(kwargs))
        if "evidence_receipts" in kwargs:
            raise TypeError("unexpected keyword argument 'evidence_receipts'")
        return {"verdict": "conditional", "meta": {"panel": "fast"}}

    monkeypatch.delenv("COUNCIL_GATE_PANEL", raising=False)
    import types
    delib = types.ModuleType("council.deliberation")
    delib.run_council = fake_run_council
    monkeypatch.setitem(__import__("sys").modules, "council.deliberation", delib)

    res = council_gate._council_run("q", mode="fast", max_tokens=100,
                                    evidence_receipts=[{"id": "C01", "exit_code": 0}])
    assert res["verdict"] == "conditional"
    assert len(calls) == 2
    assert "evidence_receipts" in calls[0]
    assert "evidence_receipts" not in calls[1]
