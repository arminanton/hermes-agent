"""Tests for the autopilot ADR decision log (agent/autopilot/adr.py).

The ADR is an append-only markdown record of every autopilot decision (the
moments a human would normally be in the loop). It is OFF by default, fails
soft, and never rewrites prior records.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.autopilot import adr


class _Agent:
    """Minimal stand-in carrying the attributes the ADR reads."""

    def __init__(self, enabled=None, path=None, session_id="sess123", project_copy=False):
        if enabled is not None:
            self._autopilot_adr = enabled
        if path is not None:
            self._autopilot_adr_path = str(path)
        # Default project-copy OFF in unit tests so they assert on the canonical
        # path in isolation; the dual-write tests opt in explicitly.
        self._autopilot_adr_project_copy = project_copy
        self.session_id = session_id


def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT_ADR", raising=False)
    monkeypatch.delenv("AUTOPILOT_ADR_PATH", raising=False)
    monkeypatch.delenv("HERMES_WORKSPACE", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("AUTOPILOT_ADR_PROJECT_SUBDIR", raising=False)
    # Default project-copy OFF for env/agent=None tests so they never write a
    # stray docs/adr into the test's cwd. The dual-write tests opt in explicitly.
    monkeypatch.setenv("AUTOPILOT_ADR_PROJECT_COPY", "0")


def test_enabled_by_default(monkeypatch):
    # LOCAL DEFAULT-ON (operator opt): with no explicit setting the ADR is ON so
    # every autopilot run is auditable. Explicit opt-out is still honored.
    _clear_env(monkeypatch)
    assert adr.adr_enabled(_Agent()) is True
    assert adr.adr_enabled(None) is True
    # explicit opt-out via env
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "0")
    assert adr.adr_enabled(None) is False
    # explicit opt-out via per-agent attr
    monkeypatch.delenv("HERMES_AUTOPILOT_ADR", raising=False)
    assert adr.adr_enabled(_Agent(enabled=False)) is False


def test_enabled_via_agent_attr(monkeypatch):
    _clear_env(monkeypatch)
    assert adr.adr_enabled(_Agent(enabled=True)) is True


def test_enabled_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "1")
    assert adr.adr_enabled(None) is True
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "off")
    assert adr.adr_enabled(None) is False


def test_record_is_noop_when_disabled(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=False, path=target)
    out = adr.record_decision(agent, kind="completion", goal="do X")
    assert out is None
    assert not target.exists()


def test_record_writes_section_when_enabled(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    out = adr.record_decision(
        agent,
        kind="completion",
        goal="fix all lint errors",
        sent_for_verification="GOAL: fix all lint\nRESULT: ran ruff, 0 errors",
        verdict="allow",
        confidence=0.91,
        chosen="stop — goal verified complete",
        rationale="council verdict=allow",
        source="council",
    )
    assert out == target
    body = target.read_text()
    assert "# Autopilot decision log" in body          # header written once
    assert "## " in body and "— completion" in body
    assert "reviewer: council" in body
    assert "verdict: allow (confidence 0.91)" in body
    assert "fix all lint errors" in body


def test_record_appends_not_overwrites(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(agent, kind="completion", goal="first goal", source="aux")
    adr.record_decision(agent, kind="continue", goal="second goal", source="council")
    body = target.read_text()
    # Header appears exactly once; the goal is stamped ONCE in the header (verbatim)
    # and NOT repeated per-section, so only the first goal (header) is present.
    assert body.count("# Autopilot decision log") == 1
    assert "first goal" in body            # header goal (from the first record)
    assert "second goal" not in body       # per-section goal repetition removed
    # Both decision records still appended (two sections, identified by kind).
    assert "— completion" in body and "— continue" in body
    assert body.count("## ") >= 2


def test_record_logs_options_and_choice(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(
        agent,
        kind="clarify",
        goal="Which DB driver?",
        options=["sqlite3", "pysqlite3", "apsw"],
        chosen="sqlite3",
        rationale="stdlib, no extra dep",
        source="aux",
    )
    body = target.read_text()
    assert "options considered:" in body
    assert "sqlite3" in body and "apsw" in body
    assert "chosen path: sqlite3" in body


def test_record_logs_gap_and_required_checks(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(
        agent,
        kind="continue",
        goal="ship the feature",
        verdict="deny",
        confidence=0.7,
        gap="no tests were run on the new path",
        required_checks="run pytest on the new module; confirm 0 failures",
        source="council",
    )
    body = target.read_text()
    assert "gap found / why not passing: no tests were run" in body
    assert "required to pass: run pytest" in body


def test_record_fails_soft_on_bad_path(monkeypatch):
    _clear_env(monkeypatch)
    # Point the ADR at a path whose parent cannot be created (a file as a dir).
    # project_copy defaults off in _Agent, so this exercises the canonical-only path.
    agent = _Agent(enabled=True, path="/dev/null/cannot/exist/adr.md")
    # Must not raise; returns None on failure.
    out = adr.record_decision(agent, kind="completion", goal="x")
    assert out is None


def test_path_override_via_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "custom" / "log.md"
    monkeypatch.setenv("HERMES_AUTOPILOT_ADR", "1")
    monkeypatch.setenv("AUTOPILOT_ADR_PATH", str(target))
    assert adr.adr_path(None) == target
    adr.record_decision(None, kind="completion", goal="env-path goal", source="aux")
    assert target.exists()
    assert "env-path goal" in target.read_text()


def test_default_path_shape(monkeypatch, tmp_path):
    # Canonical ADR path = <HERMES_HOME>/autopilot/adr/ (HERMES_HOME is already the
    # workspace root, so no extra .hermes/ segment). HERMES_WORKSPACE is a legacy
    # alias used only if HERMES_HOME is unset.
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = adr.adr_path(_Agent(session_id="abc"))
    assert p.parent == tmp_path / "autopilot" / "adr"
    assert p.name.startswith("AUTOPILOT-abc-")
    assert p.suffix == ".md"


def test_default_path_legacy_workspace_alias(monkeypatch, tmp_path):
    # When HERMES_HOME is unset, fall back to the legacy HERMES_WORKSPACE alias.
    _clear_env(monkeypatch)
    monkeypatch.setenv("HERMES_WORKSPACE", str(tmp_path))
    p = adr.adr_path(_Agent(session_id="abc"))
    assert p.parent == tmp_path / "autopilot" / "adr"


# --------------------------------------------------------------------------- #
# Goal in header + dual-write (canonical + project copy)                        #
# --------------------------------------------------------------------------- #
def test_goal_stamped_in_header(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)  # project_copy off
    adr.record_decision(agent, kind="completion", goal="Fix all lint errors in foo/",
                        verdict="allow", source="council")
    body = target.read_text()
    # Goal is stamped VERBATIM (complete, untruncated) once in the header.
    assert "**Goal (verbatim):**" in body
    assert "Fix all lint errors in foo/" in body


def test_long_goal_stored_verbatim_not_truncated(monkeypatch, tmp_path):
    # A long multi-part goal must be stored COMPLETE in the header (no "…[+N chars]"
    # truncation) — the header is the one authoritative full copy of the objective.
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    long_goal = (
        "Port all counter functionality, enforce 90-line/80-col SOLID splits, "
        "add javadoc+rustdoc, wire a doc generator, and de-stale every md file. "
    ) * 12  # ~1400 chars, well past the old 500-char header cap
    adr.record_decision(agent, kind="continue", goal=long_goal, source="council")
    body = target.read_text()
    assert long_goal.strip() in body          # complete, verbatim
    assert "…[+" not in body                    # no truncation marker anywhere


def test_goal_not_repeated_per_section(monkeypatch, tmp_path):
    # The goal appears ONCE (header) — never re-emitted as a per-decision "- goal:" line.
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    for k in ("completion", "continue", "continue"):
        adr.record_decision(agent, kind=k, goal="ship the thing", source="council")
    body = target.read_text()
    assert "ship the thing" in body            # header, once
    assert "- goal:" not in body               # never a per-section goal line


def test_data_received_and_council_response_captured_verbatim(monkeypatch, tmp_path):
    # Each block records what the model produced (data received) AND what the
    # reviewer said back (council response) — verbatim, not just the verdict label.
    _clear_env(monkeypatch)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    model_out = "I rewrote all four docs to reflect verified reality at head 2adccd64.\nHere is the measured evidence: ..."
    council_reply = ("[arbiter.most_likely_wrong_point] Equating 'artifacts exist' with 'functionality ported'.\n"
                     "[skeptic] The parity claim is structural, not behavioral.")
    adr.record_decision(
        agent, kind="continue", goal="do the audit",
        data_received=model_out,
        council_response=council_reply,
        verdict="deny", confidence=0.70, source="council",
    )
    body = target.read_text()
    assert "- data received:" in body
    assert "I rewrote all four docs to reflect verified reality" in body   # model output verbatim
    assert "council response (verbatim)" in body
    assert "[arbiter.most_likely_wrong_point]" in body                     # reviewer reply verbatim
    assert "The parity claim is structural, not behavioral." in body


def test_verbatim_fields_uncapped_by_default(monkeypatch, tmp_path):
    # data_received / council_response are stored COMPLETE by default (no cap),
    # so a long model dump is captured in full — this is the whole point.
    _clear_env(monkeypatch)
    monkeypatch.delenv("AUTOPILOT_ADR_MAX_FIELD", raising=False)
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    big = "MODEL_OUTPUT_LINE " * 900  # ~16KB, far past the old 2000-char cap
    adr.record_decision(agent, kind="continue", goal="g",
                        data_received=big, source="council")
    body = target.read_text()
    assert big.strip() in body                 # complete, verbatim
    assert "truncated" not in body             # no truncation applied


def test_verbatim_field_cap_env_bounds_when_set(monkeypatch, tmp_path):
    # An operator CAN bound the verbatim fields via AUTOPILOT_ADR_MAX_FIELD, and
    # when they do, truncation is explicit and honest about how much was cut.
    _clear_env(monkeypatch)
    monkeypatch.setenv("AUTOPILOT_ADR_MAX_FIELD", "50")
    target = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=target)
    adr.record_decision(agent, kind="continue", goal="g",
                        data_received="X" * 500, source="council")
    body = target.read_text()
    assert "truncated 450 chars" in body       # explicit, honest marker


def test_dual_write_canonical_plus_project_copy(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    canonical = tmp_path / "canon" / "adr.md"
    project = tmp_path / "proj"
    project.mkdir()
    # Goal declares the project dir; project_copy on; subdir explicit for determinism.
    agent = _Agent(enabled=True, path=canonical, project_copy=True)
    agent._autopilot_adr_project_subdir = ".autopilot/adr"
    monkeypatch.setenv("HERMES_WORKSPACE", str(tmp_path))  # cwd fallback
    adr.record_decision(agent, kind="continue", goal=f"work in path: {project}",
                        verdict="deny", source="council")
    # Canonical copy written.
    assert canonical.exists()
    assert "— continue" in canonical.read_text()
    # Project copy written under the declared project dir.
    proj_files = list((project / ".autopilot" / "adr").glob("AUTOPILOT-*.md"))
    assert proj_files, "project copy not written"
    assert "— continue" in proj_files[0].read_text()


def test_project_copy_off_writes_only_canonical(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    canonical = tmp_path / "adr.md"
    agent = _Agent(enabled=True, path=canonical, project_copy=False)
    targets = adr.adr_targets(agent, goal=f"work in {tmp_path}")
    assert targets == [canonical]


def test_goal_declared_root_detected(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    d = tmp_path / "myrepo"
    d.mkdir()
    root = adr._goal_declared_root(f"build the thing in {d} fully")
    assert root == d.resolve()
    # A non-existent path is ignored.
    assert adr._goal_declared_root("do work in /nonexistent/xyz123") is None


def test_git_root_subdir_defaults_to_docs_adr(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    # A git repo root → conventional docs/adr.
    assert adr._project_subdir(None, repo) == "docs/adr"
    # A non-git dir → .autopilot/adr.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert adr._project_subdir(None, plain) == ".autopilot/adr"
