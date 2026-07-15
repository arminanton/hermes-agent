"""Unit tests for the frozen acceptance contract + achievable-bar terminus
(autopilot health Fix 1/2/3 support module)."""

import types

from agent.autopilot import contract as c


# --------------------------------------------------------------------------- #
# parsing + freezing                                                           #
# --------------------------------------------------------------------------- #
def test_empty_goal_yields_empty_contract():
    assert c.parse_contract("").is_empty
    assert c.parse_contract("   \n  ").is_empty
    # A goal with no recognizable criteria lines is empty (Council-only fallback).
    assert c.parse_contract("just some prose with no bullets or verify lines").is_empty


def test_parses_bulleted_criteria():
    goal = (
        "Goal: ship the parser fix.\n"
        "- All unit tests pass with zero failures\n"
        "- The lexer handles nested quotes correctly\n"
        "- Document the new grammar in the README\n"
    )
    ct = c.parse_contract(goal)
    assert not ct.is_empty
    assert len(ct.criteria) == 3
    # all three are agent-achievable (no owner/unprovable markers)
    assert all(x.satisfiability == c.AGENT_ACHIEVABLE for x in ct.criteria)
    assert ct.content_hash  # frozen hash present


def test_parses_numbered_and_verify_lines():
    goal = (
        "1. Implement the retry wrapper\n"
        "2) Add metrics emission\n"
        "Verify: the suite is green on CI\n"
    )
    ct = c.parse_contract(goal)
    assert len(ct.criteria) == 3


def test_owner_gated_tag():
    goal = (
        "- Implement the migration script\n"
        "- Obtain owner sign-off before the live cutover\n"
    )
    ct = c.parse_contract(goal)
    tags = {x.text[:20]: x.satisfiability for x in ct.criteria}
    # the migration is agent-achievable; the sign-off is owner-gated
    assert c.OWNER_GATED in tags.values()
    assert c.AGENT_ACHIEVABLE in tags.values()


def test_unprovable_tag_beats_owner():
    # verifier-independence must classify as UNPROVABLE even if owner words co-occur
    goal = "- The result must be confirmed by an independent verifier with owner approval\n"
    ct = c.parse_contract(goal)
    assert len(ct.criteria) == 1
    assert ct.criteria[0].satisfiability == c.UNPROVABLE


def test_hash_is_stable_and_content_sensitive():
    g1 = "- do alpha thing\n- do beta thing\n"
    g2 = "- do alpha thing\n- do GAMMA thing\n"
    assert c.parse_contract(g1).content_hash == c.parse_contract(g1).content_hash
    assert c.parse_contract(g1).content_hash != c.parse_contract(g2).content_hash


def test_duplicate_and_short_lines_skipped():
    goal = (
        "- real criterion number one here\n"
        "- real criterion number one here\n"   # dup
        "- tiny\n"                              # too short (<8)
    )
    ct = c.parse_contract(goal)
    assert len(ct.criteria) == 1


# --------------------------------------------------------------------------- #
# get_or_parse: freeze-once, no mid-run redefinition                           #
# --------------------------------------------------------------------------- #
def test_get_or_parse_freezes_once():
    a = types.SimpleNamespace()
    g1 = "- implement the alpha feature fully\n"
    first = c.get_or_parse(a, g1)
    # a DIFFERENT goal text later must NOT replace the frozen contract
    second = c.get_or_parse(a, "- a totally different beta goal now\n")
    assert second is first
    assert second.content_hash == first.content_hash


# --------------------------------------------------------------------------- #
# achievable_bar_halt: the terminus rule                                        #
# --------------------------------------------------------------------------- #
def _contract_with(*specs):
    """specs: (id, text, satisfiability) tuples -> a frozen-ish AcceptanceContract."""
    crits = tuple(c.Criterion(id=i, text=t, satisfiability=s) for (i, t, s) in specs)
    return c.AcceptanceContract(criteria=crits, content_hash="h", source_len=1)


def test_no_halt_when_agent_work_open():
    ct = _contract_with(
        ("C01", "implement the thing", c.AGENT_ACHIEVABLE),
        ("C02", "owner sign-off", c.OWNER_GATED),
    )
    # C01 not satisfied -> real work remains -> do not halt
    r = c.achievable_bar_halt(ct, satisfied_ids=set())
    assert r.halt is False


def test_halt_when_only_residual_remains():
    ct = _contract_with(
        ("C01", "implement the thing", c.AGENT_ACHIEVABLE),
        ("C02", "owner sign-off for cutover", c.OWNER_GATED),
        ("C03", "independent verifier confirms", c.UNPROVABLE),
    )
    # all agent-achievable satisfied; only owner/unprovable remain -> HALT
    r = c.achievable_bar_halt(ct, satisfied_ids={"C01"}, council_denial_reason="prove independence")
    assert r.halt is True
    assert r.reason == "achievable-bar"
    assert "C02" in r.residual_text and "C03" in r.residual_text
    assert "owner sign-off required" in r.residual_text
    assert "not provable by the agent" in r.residual_text


def test_no_halt_when_no_residuals_all_done():
    # everything agent-achievable and satisfied, NO residuals -> let normal complete
    # path handle it (halt=False so we don't double-emit a terminus).
    ct = _contract_with(("C01", "do the thing", c.AGENT_ACHIEVABLE))
    r = c.achievable_bar_halt(ct, satisfied_ids={"C01"})
    assert r.halt is False


def test_empty_contract_never_halts():
    r = c.achievable_bar_halt(c.AcceptanceContract(), satisfied_ids=set())
    assert r.halt is False


# --------------------------------------------------------------------------- #
# claims_self_spawned_independence: Fix 2 detector                             #
# --------------------------------------------------------------------------- #
def test_self_spawned_independence_caught():
    txt = ("I verified completion independently: a subagent I spawned via delegate_task "
           "confirmed the 8/8 reproduction, so this is independently verified.")
    assert c.claims_self_spawned_independence(txt) is True


def test_external_verifier_allowed():
    txt = ("The Hermes Council independently verified the result and a separate run "
           "cross-confirmed the numbers.")
    assert c.claims_self_spawned_independence(txt) is False


def test_no_independence_claim_is_false():
    txt = "I edited three files and ran the tests; they pass."
    assert c.claims_self_spawned_independence(txt) is False


def test_self_spawn_but_also_external_is_allowed():
    # if it ALSO rests on a truly-external signal, don't flag it
    txt = ("My subagent re-checked it, but more importantly the owner confirmed the "
           "cutover and the Hermes Council verified independence.")
    assert c.claims_self_spawned_independence(txt) is False


def test_empty_text_is_false():
    assert c.claims_self_spawned_independence("") is False


# --------------------------------------------------------------------------- #
# {verify: <cmd>} executable-check parsing                                      #
# --------------------------------------------------------------------------- #
def test_parses_inline_verify_command():
    goal = "- All unit tests pass {verify: pytest -q}\n- Document the grammar\n"
    ct = c.parse_contract(goal)
    by_id = {x.id: x for x in ct.criteria}
    # the verify token is stripped from the visible text and captured separately
    c01 = ct.criteria[0]
    assert c01.verify_cmd == "pytest -q"
    assert "{verify" not in c01.text
    assert "pass" in c01.text
    # the criterion without a token has an empty verify_cmd
    assert ct.criteria[1].verify_cmd == ""


def test_verify_cmd_variants():
    for token, expected in (
        ("{verify: ruff check .}", "ruff check ."),
        ("{verify:cmd: go test ./...}", "go test ./..."),
        ("{verify_cmd: npm test}", "npm test"),
    ):
        ct = c.parse_contract(f"- lint is clean {token}\n")
        assert ct.criteria[0].verify_cmd == expected


def test_verifiable_criteria_accessor():
    goal = (
        "- tests pass {verify: pytest -q}\n"
        "- prose only criterion here\n"
        "- lint clean {verify: ruff check .}\n"
    )
    ct = c.parse_contract(goal)
    vids = {x.id for x in ct.verifiable_criteria()}
    assert len(vids) == 2


def test_verify_cmd_is_in_content_hash():
    # changing the command changes the frozen hash (the command is load-bearing)
    g1 = "- tests pass {verify: pytest -q}\n"
    g2 = "- tests pass {verify: pytest -x}\n"
    assert c.parse_contract(g1).content_hash != c.parse_contract(g2).content_hash


def test_criterion_default_verify_cmd_empty():
    # backward-compat: positional construction without verify_cmd still works
    crit = c.Criterion(id="C01", text="do a thing", satisfiability=c.AGENT_ACHIEVABLE)
    assert crit.verify_cmd == ""

