"""Tests for the refinement-churn terminus (Fix 4) — the diminishing-returns
"polish" loop detector that lets an autonomous run self-conclude when the
deliverable is done and the Council is only asking for presentation refinements.

These are PURE unit tests on the wording-independent tracker + decision function.
Driver-integration tests live in test_autopilot_driver.py.
"""

from agent.autopilot import contract as C


def _record_churn_round(tracker, *, label="conditional", source="council",
                        conf=0.4, closed=0, deliverable=True):
    tracker.record(
        verdict_label=label, source=source, confidence=conf,
        criteria_closed_this_round=closed, deliverable_present=deliverable,
    )


def test_pure_presentation_loop_concludes_after_k():
    # The NuData shape: every round is judged, conditional, no deny, no criterion
    # closed, low confidence, deliverable present. After K rounds → conclude.
    t = C.RefinementChurnTracker()
    for _ in range(4):
        _record_churn_round(t)
    res = C.refinement_churn_conclude(t, k=4)
    assert res.conclude is True
    assert res.reason == "refinement-churn"
    assert "presentation" in res.note.lower()


def test_below_k_does_not_conclude():
    t = C.RefinementChurnTracker()
    for _ in range(3):
        _record_churn_round(t)
    assert C.refinement_churn_conclude(t, k=4).conclude is False


def test_a_single_deny_resets_the_window():
    # A real `deny` means substantive work is missing/wrong — NOT churn. It must
    # reset the counter so the run keeps going (correctly).
    t = C.RefinementChurnTracker()
    _record_churn_round(t)
    _record_churn_round(t)
    _record_churn_round(t, label="deny")   # real failure → reset
    assert t.rounds == 0
    _record_churn_round(t)
    assert C.refinement_churn_conclude(t, k=4).conclude is False


def test_a_closed_criterion_resets_the_window():
    # If a criterion closes, real new ground was gained → not churn → reset.
    t = C.RefinementChurnTracker()
    for _ in range(3):
        _record_churn_round(t)
    _record_churn_round(t, closed=1)   # progress → reset
    assert t.rounds == 0


def test_rising_confidence_is_convergence_not_churn():
    # If the Council trends toward acceptance (confidence >= accept threshold),
    # that is genuine convergence-in-progress, NOT churn — reset so we don't cut
    # a run short right as it's about to be allowed.
    t = C.RefinementChurnTracker()
    for _ in range(3):
        _record_churn_round(t)
    _record_churn_round(t, conf=0.9)   # converging → reset
    assert t.rounds == 0


def test_unjudged_round_resets_the_window():
    # A fail-open / no-judge round means "not done yet", not "presentation polish";
    # it must not count toward churn.
    t = C.RefinementChurnTracker()
    for _ in range(3):
        _record_churn_round(t)
    _record_churn_round(t, source="fallback")   # unjudged → reset
    assert t.rounds == 0


def test_no_deliverable_does_not_accumulate():
    # With no standing deliverable (empty contract / no agent criteria) there is
    # nothing to conclude ON — the window must not fill.
    t = C.RefinementChurnTracker()
    for _ in range(6):
        _record_churn_round(t, deliverable=False)
    assert t.rounds == 0
    assert C.refinement_churn_conclude(t, k=4).conclude is False


def test_k_zero_disables_the_terminus():
    t = C.RefinementChurnTracker()
    for _ in range(10):
        _record_churn_round(t)
    assert C.refinement_churn_conclude(t, k=0).conclude is False


def test_aux_reviewer_also_counts():
    # An aux-reviewer adjudication is a real judge too (council OR aux).
    t = C.RefinementChurnTracker()
    for _ in range(4):
        _record_churn_round(t, source="aux")
    assert C.refinement_churn_conclude(t, k=4).conclude is True


def test_allow_low_confidence_still_counts_as_churn():
    # The NuData ADR showed `allow`@0.35 that still didn't stop the loop. A low-
    # confidence allow with no criterion closing is still presentation churn.
    t = C.RefinementChurnTracker()
    for _ in range(4):
        _record_churn_round(t, label="allow", conf=0.35)
    assert C.refinement_churn_conclude(t, k=4).conclude is True


def test_churn_window_k_env_override(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_REFINEMENT_CHURN_K", "6")
    assert C.churn_window_k() == 6
    monkeypatch.setenv("AUTOPILOT_REFINEMENT_CHURN_K", "0")
    assert C.churn_window_k() == 0


def test_churn_window_k_agent_attr_wins(monkeypatch):
    import types
    monkeypatch.delenv("AUTOPILOT_REFINEMENT_CHURN_K", raising=False)
    a = types.SimpleNamespace(_autopilot_refinement_churn_k=7)
    assert C.churn_window_k(a) == 7


# --------------------------------------------------------------------------- #
# Naive-user contract floor — a bare goal gets a synthesized 1-criterion contract
# --------------------------------------------------------------------------- #
def test_synthesize_minimal_contract_for_bare_goal():
    ct = C.synthesize_minimal_contract("make the homepage load faster")
    assert not ct.is_empty
    assert len(ct.agent_criteria()) == 1
    assert ct.criteria[0].satisfiability == C.AGENT_ACHIEVABLE
    assert ct.criteria[0].verify_cmd == ""  # no project check; Council judges it
    assert "homepage" in ct.criteria[0].text


def test_synthesize_minimal_contract_empty_goal_stays_empty():
    assert C.synthesize_minimal_contract("").is_empty
    assert C.synthesize_minimal_contract("   ").is_empty


def test_get_or_parse_synthesizes_floor_for_bare_goal():
    import types
    # no project checks (autodraft off) + a bare prose goal → previously empty.
    a = types.SimpleNamespace(_autopilot_autodraft_checks=False)
    ct = C.get_or_parse(a, "write a blog post about our launch")
    assert not ct.is_empty                      # floor kicked in
    assert len(ct.agent_criteria()) == 1
    # deliverable_present (the churn-terminus gate) is now true
    assert (not ct.is_empty) and bool(ct.agent_criteria())


def test_get_or_parse_floor_disabled_keeps_empty():
    import types
    a = types.SimpleNamespace(_autopilot_autodraft_checks=False,
                              _autopilot_synthesize_floor=False)
    ct = C.get_or_parse(a, "write a blog post about our launch")
    assert ct.is_empty                          # floor opted out → empty as before


def test_authored_contract_not_overwritten_by_floor():
    import types
    a = types.SimpleNamespace(_autopilot_autodraft_checks=False)
    goal = "- Implement the parser and make tests pass\n- Obtain owner sign-off\n"
    ct = C.get_or_parse(a, goal)
    # a real authored contract is preserved (2 criteria), floor does NOT replace it
    assert len(ct.criteria) == 2
    assert "C01" in {c.id for c in ct.criteria}


def test_synthesize_floor_enabled_env_override(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_SYNTH_CONTRACT_FLOOR", "0")
    assert C.synthesize_floor_enabled() is False
    monkeypatch.setenv("AUTOPILOT_SYNTH_CONTRACT_FLOOR", "1")
    assert C.synthesize_floor_enabled() is True


def test_accept_confidence_env_override(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_CHURN_ACCEPT_CONF", "0.5")
    t = C.RefinementChurnTracker()
    _record_churn_round(t, conf=0.55)   # now above the lowered threshold → reset
    assert t.rounds == 0
