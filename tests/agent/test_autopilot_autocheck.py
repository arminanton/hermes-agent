"""Tests for auto-drafted verification gates (zero-effort grounding)."""

import json
import types

from agent.autopilot import autocheck as ac
from agent.autopilot import contract as c


# --------------------------------------------------------------------------- #
# enable/disable                                                               #
# --------------------------------------------------------------------------- #
def test_autodraft_enabled_by_default(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_AUTODRAFT_CHECKS", raising=False)
    assert ac.autodraft_enabled(types.SimpleNamespace()) is True


def test_autodraft_disabled_via_attr():
    assert ac.autodraft_enabled(types.SimpleNamespace(_autopilot_autodraft_checks=False)) is False


def test_autodraft_disabled_via_env(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_AUTODRAFT_CHECKS", "0")
    assert ac.autodraft_enabled(types.SimpleNamespace()) is False


# --------------------------------------------------------------------------- #
# project check detection                                                      #
# --------------------------------------------------------------------------- #
def test_detect_python_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "tests").mkdir()
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("test") == "pytest -q"


def test_detect_ruff_and_mypy(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n[tool.mypy]\n")
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("lint") == "ruff check ."
    assert checks.get("type") == "mypy ."


def test_detect_go(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("test") == "go test ./..."
    assert checks.get("build") == "go build ./..."


def test_detect_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("test") == "cargo test"


def test_detect_node_only_with_real_test_script(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("test") == "npm test"


def test_detect_node_skips_placeholder_test(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps(
        {"scripts": {"test": "echo \"Error: no test specified\" && exit 1"}}))
    checks = ac.detect_project_checks(str(tmp_path))
    assert "test" not in checks  # placeholder must NOT become a gate


def test_detect_node_pnpm_manager(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "vitest"}}))
    (tmp_path / "pnpm-lock.yaml").write_text("")
    checks = ac.detect_project_checks(str(tmp_path))
    assert checks.get("test") == "pnpm test"


def test_detect_empty_project_is_empty(tmp_path):
    assert ac.detect_project_checks(str(tmp_path)) == {}


def test_detected_commands_are_all_allowlisted(tmp_path):
    # every detector output must pass the harness allowlist
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n[tool.ruff]\n[tool.mypy]\n")
    (tmp_path / "go.mod").write_text("module x\n")
    from agent.autopilot import verification as v
    for cmd in ac.detect_project_checks(str(tmp_path)).values():
        ok, reason = v.validate_command(cmd)
        assert ok, f"{cmd!r} should be allowlisted: {reason}"


# --------------------------------------------------------------------------- #
# keyword mapping                                                              #
# --------------------------------------------------------------------------- #
def test_pick_check_by_keyword():
    checks = {"test": "pytest -q", "lint": "ruff check .", "type": "mypy ."}
    assert ac._pick_check_for("all unit tests pass", checks) == "pytest -q"
    assert ac._pick_check_for("the linter is clean", checks) == "ruff check ."
    assert ac._pick_check_for("type checks pass", checks) == "mypy ."
    assert ac._pick_check_for("write the documentation", checks) == ""  # no keyword


def test_primary_check_prefers_test():
    assert ac._primary_check({"lint": "ruff check .", "test": "pytest -q"}) == "pytest -q"
    assert ac._primary_check({"build": "go build ./..."}) == "go build ./..."


# --------------------------------------------------------------------------- #
# autodraft_contract — the zero-effort path                                     #
# --------------------------------------------------------------------------- #
def _py_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n[tool.ruff]\n")
    (tmp_path / "tests").mkdir()
    return str(tmp_path)


def test_autodraft_attaches_checks_to_bulleted_goal(tmp_path):
    root = _py_project(tmp_path)
    goal = "- All unit tests pass\n- The linter is clean\n- Document the grammar in the README\n"
    ct = c.parse_contract(goal)
    assert all(not x.verify_cmd for x in ct.criteria)  # nothing yet
    drafted = ac.autodraft_contract(ct, goal_text=goal, root=root)
    by_text = {x.text[:10]: x for x in drafted.criteria}
    # tests criterion -> pytest; lint criterion -> ruff; docs -> none (no check fits)
    assert any(x.verify_cmd == "pytest -q" for x in drafted.criteria)
    assert any(x.verify_cmd == "ruff check ." for x in drafted.criteria)
    assert any(x.verify_cmd == "" and "Document" in x.text for x in drafted.criteria)


def test_autodraft_synthesizes_criterion_for_bare_goal(tmp_path):
    # a one-line goal with NO bullet criteria still gets a grounded gate
    root = _py_project(tmp_path)
    goal = "fix the failing tests"
    ct = c.parse_contract(goal)
    assert ct.is_empty  # bare goal -> no criteria
    drafted = ac.autodraft_contract(ct, goal_text=goal, root=root)
    assert not drafted.is_empty
    assert len(drafted.criteria) == 1
    assert drafted.criteria[0].verify_cmd == "pytest -q"
    assert drafted.content_hash  # re-frozen


def test_autodraft_noop_without_project_checks(tmp_path):
    goal = "- do the alpha thing\n- do the beta thing\n"
    ct = c.parse_contract(goal)
    drafted = ac.autodraft_contract(ct, goal_text=goal, root=str(tmp_path))  # empty dir
    assert drafted is ct  # unchanged (no checks detected)


def test_autodraft_preserves_existing_verify_cmd(tmp_path):
    root = _py_project(tmp_path)
    goal = "- tests pass {verify: pytest -k smoke}\n"
    ct = c.parse_contract(goal)
    assert ct.criteria[0].verify_cmd == "pytest -k smoke"
    drafted = ac.autodraft_contract(ct, goal_text=goal, root=root)
    # an author-supplied command is NOT overwritten by auto-draft
    assert drafted.criteria[0].verify_cmd == "pytest -k smoke"


def test_autodraft_does_not_touch_owner_gated(tmp_path):
    root = _py_project(tmp_path)
    goal = "- tests pass\n- obtain owner sign-off before the live cutover\n"
    ct = c.parse_contract(goal)
    drafted = ac.autodraft_contract(ct, goal_text=goal, root=root)
    owner = [x for x in drafted.criteria if x.satisfiability == c.OWNER_GATED][0]
    assert owner.verify_cmd == ""  # never auto-check an owner-gated criterion


def test_autodraft_via_get_or_parse_end_to_end(tmp_path, monkeypatch):
    # the integration path: get_or_parse auto-drafts when enabled
    root = _py_project(tmp_path)
    agent = types.SimpleNamespace(_autopilot_verification_workdir=root)
    goal = "fix the failing tests"
    ct = c.get_or_parse(agent, goal)
    assert not ct.is_empty
    assert ct.criteria[0].verify_cmd == "pytest -q"


def test_get_or_parse_autodraft_disabled(tmp_path):
    root = _py_project(tmp_path)
    # also opt out of the naive-user floor to isolate the autodraft-off behavior
    # (the floor is a separate feature that would otherwise synthesize a criterion).
    agent = types.SimpleNamespace(_autopilot_verification_workdir=root,
                                  _autopilot_autodraft_checks=False,
                                  _autopilot_synthesize_floor=False)
    ct = c.get_or_parse(agent, "fix the failing tests")
    assert ct.is_empty  # disabled -> no synthesis
