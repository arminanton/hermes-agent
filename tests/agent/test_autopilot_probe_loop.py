"""Tests for the probe loop wiring (§3 step 6) — select→run→receipts→merge.

These assert the seam that makes the senses active in the gate:
  * the modal probe plan runs and its receipts convert to the verification.Receipt shape;
  * a manufactured-independence claim/criterion FAILS against the verbatim record
    (the bypass-killer), so it lands in the report's failed_ids (blocking false completion);
  * the {verify:} COMMAND domain is left to the harness (no process probes here — no
    double-run, no bypass of the verification_exec opt-in);
  * probes are bounded by the per-turn budget and disabled via the config flag;
  * a DOWNGRADE (unobservable/unavailable) is recorded as 'skipped' (never satisfied/failed)
    and surfaced as a could-not-observe downgrade entry;
  * re-entrancy (AUTOPILOT_VERIFICATION=1) suppresses nested probe runs.
"""

import os
import sqlite3

import pytest

from agent.autopilot import contract as K
from agent.autopilot import probe_loop as PL
from agent.autopilot import verification as V
from agent.autopilot import probes as P


class _Agent:
    """Minimal stand-in carrying the attrs the probe loop reads."""

    def __init__(self):
        self._autopilot_verification_workdir = ""


@pytest.fixture()
def agent(tmp_path):
    a = _Agent()
    a._autopilot_verification_workdir = str(tmp_path)
    return a


@pytest.fixture()
def isolate_ladder(monkeypatch, tmp_path):
    monkeypatch.setenv("CMX_DB_PATH", str(tmp_path / "cmx.db"))
    monkeypatch.setenv("LCM_DB_PATH", str(tmp_path / "lcm.db"))
    monkeypatch.setenv("HERMES_STATE_DB", str(tmp_path / "state.db"))
    return tmp_path


def _state_with(tmp_path, rows):
    db = str(tmp_path / "state.db")
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, "
                    "turn_index INTEGER, role TEXT, content TEXT);")
    for i, (role, content) in enumerate(rows):
        c.execute("INSERT INTO messages(session_id,turn_index,role,content) VALUES('s',?,?,?)",
                  (i, role, content))
    c.commit()
    c.close()


# --------------------------------------------------------------------------- #
# the bypass-killer end-to-end                                                 #
# --------------------------------------------------------------------------- #
def test_manufactured_independence_criterion_fails(agent, isolate_ladder):
    _state_with(isolate_ladder, [("assistant", "I claim it is done")])  # no tool evidence
    contract = K.parse_contract("- A subagent independently confirmed the deploy succeeded")
    receipts, downgrades = PL.run_probe_plan(agent, contract, "the deploy is complete")
    assert receipts
    # the provenance probe falsifies the claim → FAIL (blocks false completion)
    assert any(r.criterion_id == "C01" and r.status == "fail" for r in receipts)
    assert not downgrades


def test_claim_probe_catches_bypass_prose(agent, isolate_ladder):
    _state_with(isolate_ladder, [("assistant", "trust me")])
    contract = K.parse_contract("- The widget works")  # vague criterion
    final = "I spawned an independent subagent and it CONFIRMED the widget works"
    receipts, _ = PL.run_probe_plan(agent, contract, final)
    # the model's own claim is probed against the record → FAIL
    assert any(r.criterion_id == "_claim" and r.status == "fail" for r in receipts)


def test_real_tool_evidence_passes(agent, isolate_ladder):
    _state_with(isolate_ladder, [
        ("assistant", "done"),
        ("tool", "subagent result: deploy succeeded, confirmed healthy"),
    ])
    contract = K.parse_contract("- A subagent confirmed the deploy succeeded")
    receipts, _ = PL.run_probe_plan(agent, contract, "deploy done")
    assert any(r.criterion_id == "C01" and r.status == "pass" for r in receipts)


# --------------------------------------------------------------------------- #
# domain boundaries + safety                                                   #
# --------------------------------------------------------------------------- #
def test_process_probes_left_to_the_harness(agent, isolate_ladder):
    # a {verify:} command must NOT be turned into a process probe here (the harness owns
    # command execution + its opt-in gate). The modal layer handles observation only.
    contract = K.parse_contract("- All tests pass {verify: true}")
    receipts, downgrades = PL.run_probe_plan(agent, contract, "tests pass")
    assert all(not r.command.startswith("probe:process") for r in receipts)


def test_disabled_via_flag(agent, isolate_ladder):
    agent._autopilot_probes = False
    contract = K.parse_contract("- A subagent confirmed the deploy")
    receipts, downgrades = PL.run_probe_plan(agent, contract, "done")
    assert receipts == [] and downgrades == []


def test_reentrancy_suppresses_probes(agent, isolate_ladder, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_VERIFICATION", "1")
    contract = K.parse_contract("- A subagent confirmed the deploy")
    receipts, downgrades = PL.run_probe_plan(agent, contract, "done")
    assert receipts == [] and downgrades == []


def test_budget_caps_probes(agent, isolate_ladder, monkeypatch):
    agent._autopilot_probe_budget = 0.0  # exhausted immediately
    _state_with(isolate_ladder, [("assistant", "x")])
    contract = K.parse_contract(
        "- A subagent confirmed step one\n- A subagent confirmed step two")
    receipts, downgrades = PL.run_probe_plan(agent, contract, "done")
    # with zero budget every selected probe is downgraded as budget-exhausted
    assert downgrades
    assert any("budget" in reason for _, _, reason in downgrades)


# --------------------------------------------------------------------------- #
# downgrade handling + report merge                                            #
# --------------------------------------------------------------------------- #
def test_unobservable_criterion_downgrades(agent, isolate_ladder):
    # a vague criterion with no observable modality → no probe; an image criterion with no
    # backend → skipped receipt + downgrade entry (never a false satisfaction).
    contract = K.parse_contract("- The dashboard.png shows a rising trend")
    # no vision backend registered + no real file → UNAVAILABLE/ERROR, not a pass
    receipts, downgrades = PL.run_probe_plan(agent, contract, "looks good")
    # the image probe could not observe (missing file) → it must NOT be a 'pass'
    assert all(r.status != "pass" for r in receipts)


def test_merge_into_report_marks_enabled():
    rep = V.VerificationReport()
    pr = P.ProbeReceipt(P.CMX_PROVENANCE, P.FAIL, "C01", summary="unsupported")
    vr = PL._to_verification_receipt(pr)
    PL.merge_into_report(rep, [vr])
    assert rep.enabled is True
    assert "C01" in rep.failed_ids
    assert "modal probe" in rep.note


def test_status_mapping():
    # downgrade statuses must map to 'skipped' (neither satisfied nor failed)
    for st in (P.UNAVAILABLE, P.UNOBSERVABLE_STATUS):
        vr = PL._to_verification_receipt(P.ProbeReceipt(P.IMAGE, st, "C1", summary="x"))
        assert vr.status == "skipped"
    assert PL._to_verification_receipt(P.ProbeReceipt(P.BROWSER, P.PASS, "C1")).status == "pass"
    assert PL._to_verification_receipt(P.ProbeReceipt(P.BROWSER, P.FAIL, "C1")).status == "fail"


def test_run_probe_plan_never_raises(agent):
    # garbage contract / no ladder → returns cleanly, never raises
    receipts, downgrades = PL.run_probe_plan(agent, None, "")
    assert receipts == [] and downgrades == []
