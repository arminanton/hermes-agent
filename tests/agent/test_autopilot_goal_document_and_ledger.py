"""Tests for the GOAL.md default-contract discovery + the auto-maintained run ledger.

GOAL.md = the conventional, project-agnostic default name for an autopilot goal
contract (the REBORN.md pattern, standardized). The ledger = an auto-maintained
GOAL-LEDGER.md recording what the run accomplished + how it concluded, the
naive-user equivalent of the hand-authored REBORN-D-LEDGER.md.
"""

import os
import types

from agent.autopilot import driver
from agent.autopilot import ledger as L


# --------------------------------------------------------------------------- #
# GOAL.md default-contract discovery                                            #
# --------------------------------------------------------------------------- #
def test_discovers_goal_md_in_workdir(tmp_path):
    (tmp_path / "GOAL.md").write_text("# My Goal\n- Build feature X and ship it\n")
    a = types.SimpleNamespace(_autopilot_verification_workdir=str(tmp_path))
    goal = driver.resolve_goal(a, "whatever the user typed in chat")
    assert "Build feature X" in goal  # the FILE governs, not the chat message


def test_explicit_goal_path_is_loaded(tmp_path):
    p = tmp_path / "GOAL.md"
    p.write_text("- Do the thing and verify it\n")
    a = types.SimpleNamespace(_autopilot_goal=str(p))
    goal = driver.resolve_goal(a, "")
    assert "Do the thing" in goal


def test_prose_goal_is_not_treated_as_path(tmp_path):
    # A normal one-sentence goal must NOT be confused for a file path.
    a = types.SimpleNamespace(_autopilot_goal="make the homepage load faster")
    assert driver.resolve_goal(a, "") == "make the homepage load faster"


def test_no_goal_doc_falls_back_to_message(tmp_path):
    a = types.SimpleNamespace(_autopilot_verification_workdir=str(tmp_path))
    assert driver.resolve_goal(a, "just a sentence") == "just a sentence"


def test_goal_doc_discovery_can_be_disabled(tmp_path):
    (tmp_path / "GOAL.md").write_text("- ignored when disabled\n")
    a = types.SimpleNamespace(_autopilot_verification_workdir=str(tmp_path),
                              _autopilot_goal_document=False)
    assert driver.resolve_goal(a, "the chat message wins") == "the chat message wins"


def test_autopilot_md_alternate_name(tmp_path):
    (tmp_path / "AUTOPILOT.md").write_text("- alternate contract name works\n")
    a = types.SimpleNamespace(_autopilot_verification_workdir=str(tmp_path))
    assert "alternate contract name" in driver.resolve_goal(a, "x")


def test_oversized_goal_doc_ignored(tmp_path):
    (tmp_path / "GOAL.md").write_text("x" * 250_000)  # >200KB cap
    a = types.SimpleNamespace(_autopilot_verification_workdir=str(tmp_path))
    # too large → ignored → falls back to the message
    assert driver.resolve_goal(a, "fallback message") == "fallback message"


# --------------------------------------------------------------------------- #
# Run ledger (GOAL-LEDGER.md)                                                   #
# --------------------------------------------------------------------------- #
def test_ledger_writes_header_and_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="s1", _autopilot_verification_workdir=str(proj))
    written = L.record_milestone(a, goal="make the homepage faster", kind="terminus",
                                 summary="Refinement-churn terminus — 4 rounds.",
                                 deliverable="LCP 4.2s -> 1.1s")
    assert written
    body = (proj / "GOAL-LEDGER.md").read_text()
    assert "# Autopilot run ledger" in body
    assert "make the homepage faster" in body
    assert "Refinement-churn terminus" in body
    assert "LCP 4.2s -> 1.1s" in body


def test_ledger_appends_multiple_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="s2", _autopilot_verification_workdir=str(proj))
    L.record_milestone(a, goal="g", kind="milestone", summary="first thing done")
    L.record_milestone(a, goal="g", kind="complete", summary="second thing done")
    body = (proj / "GOAL-LEDGER.md").read_text()
    assert "first thing done" in body and "second thing done" in body
    assert body.count("# Autopilot run ledger") == 1  # header only once


def test_ledger_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="s3", _autopilot_verification_workdir=str(proj),
                              _autopilot_ledger=False)
    assert L.record_milestone(a, goal="g", kind="terminus", summary="x") is None
    assert not (proj / "GOAL-LEDGER.md").exists()


def test_ledger_progress_written_as_run_works(tmp_path, monkeypatch):
    # record_progress writes a running turn-by-turn entry WITHOUT waiting for a
    # terminus — this is what makes the file a ledger, not an end-of-run report.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="sp", _autopilot_verification_workdir=str(proj))
    L.record_progress(a, goal="make the homepage faster", continuation=1,
                      summary="added a caching layer", directive="now measure LCP",
                      gaps_closed=2)
    L.record_progress(a, goal="make the homepage faster", continuation=2,
                      summary="measured LCP 4.2s -> 1.1s", directive="verify on mobile")
    body = (proj / "GOAL-LEDGER.md").read_text()
    assert "# Autopilot run ledger" in body            # header once
    assert body.count("# Autopilot run ledger") == 1
    assert "progress #1" in body and "progress #2" in body
    assert "added a caching layer" in body
    assert "measured LCP 4.2s -> 1.1s" in body
    assert "acceptance criteria closed this turn: 2" in body


def test_ledger_progress_then_terminus_share_one_file(tmp_path, monkeypatch):
    # Progress entries + the terminal milestone land in the SAME file under one header.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="spt", _autopilot_verification_workdir=str(proj))
    L.record_progress(a, goal="g", continuation=1, summary="first step done")
    L.record_milestone(a, goal="g", kind="complete", summary="goal verified complete")
    body = (proj / "GOAL-LEDGER.md").read_text()
    assert body.count("# Autopilot run ledger") == 1
    assert "progress #1" in body and "first step done" in body
    assert "— complete" in body and "goal verified complete" in body


def test_ledger_progress_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    a = types.SimpleNamespace(session_id="spd", _autopilot_verification_workdir=str(proj),
                              _autopilot_ledger=False)
    assert L.record_progress(a, goal="g", continuation=1, summary="x") is None
    assert not (proj / "GOAL-LEDGER.md").exists()


def test_ledger_enabled_default(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_LEDGER", raising=False)
    assert L.ledger_enabled(None) is True
    monkeypatch.setenv("AUTOPILOT_LEDGER", "0")
    assert L.ledger_enabled(None) is False


def test_ledger_terminus_written_through_driver(tmp_path, monkeypatch):
    # End-to-end: a refinement-churn terminus through maybe_continue writes the ledger.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ws"))
    proj = tmp_path / "proj"
    proj.mkdir()
    from agent.autopilot.council_gate import CompletionVerdict

    a = types.SimpleNamespace()
    a.autopilot_mode = True
    a._api_call_count = 5
    a.iteration_budget = None
    a._status = []
    a._emit_status = lambda m: a._status.append(m)
    a._autopilot_goal = "- Produce the report and ship it\n"
    a._autopilot_refinement_churn_k = 4
    a._autopilot_autodraft_checks = False
    a._autopilot_verification_workdir = str(proj)
    a._autopilot_adr_project_copy = False
    a._autopilot_structured_criteria = False
    driver.reset_turn_state(a)

    def _cond(*args, **kw):
        return CompletionVerdict(complete=False, directive="reformat it again",
                                 verdict="conditional", source="council", confidence=0.4)
    monkeypatch.setattr(driver, "judge_completion", _cond)

    def _msgs(i):
        return [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}", "function": {"name": "patch"}}]},
            {"role": "tool", "content": f"edited rev {i} ({i*5} lines)"},
        ]

    out = None
    for i in range(4):
        out = driver.maybe_continue(a, _msgs(i), f"iter {i}", a._autopilot_goal)
        if out is None:
            break
    assert out is None  # concluded on churn
    led = proj / "GOAL-LEDGER.md"
    assert led.exists()
    assert "Refinement-churn terminus" in led.read_text()
