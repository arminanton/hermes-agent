"""Unit tests for the deterministic verification harness (engine-run receipts)."""

import types

from agent.autopilot import contract as c
from agent.autopilot import verification as v


# --------------------------------------------------------------------------- #
# allowlist validation — the safety boundary                                   #
# --------------------------------------------------------------------------- #
def test_allows_readonly_test_verbs():
    for cmd in ("pytest -q", "go test ./...", "ruff check .", "npm test",
                "grep -r TODO src", "cat README.md", "test -f foo.txt",
                "python -m pytest", "git status", "git diff --stat",
                "mypy agent/", "ls -la", "cargo check"):
        ok, reason = v.validate_command(cmd)
        assert ok, f"should allow {cmd!r}: {reason}"


def test_refuses_mutating_and_network_verbs():
    for cmd in ("rm -rf /", "curl http://evil.test", "wget http://x",
                "ssh host", "sudo reboot", "dd if=/dev/zero of=/dev/sda",
                "pip install requests", "docker run x", "kubectl delete pod",
                "chmod 777 /etc", "mv a b", "cp a b"):
        ok, reason = v.validate_command(cmd)
        assert not ok, f"should refuse {cmd!r}"
        assert reason


def test_refuses_redirection_and_subshell():
    for cmd in ("pytest > /etc/passwd", "echo $(rm -rf /)", "cat `whoami`",
                "ls > out.txt", "grep x < /etc/shadow"):
        ok, reason = v.validate_command(cmd)
        assert not ok, f"should refuse {cmd!r}"


def test_refuses_unknown_verb():
    ok, reason = v.validate_command("frobnicate --all")
    assert not ok and "not allowlisted" in reason


def test_subcommand_gating():
    # read-only subcommands allowed; arbitrary code-exec subcommands refused
    assert v.validate_command("go test ./...")[0] is True
    assert v.validate_command("go run main.go")[0] is False        # runs arbitrary code
    assert v.validate_command("git log -1")[0] is True
    assert v.validate_command("git push origin main")[0] is False  # not read-only
    assert v.validate_command("npm test")[0] is True
    assert v.validate_command("npm publish")[0] is False


def test_pipeline_validates_every_segment():
    # both segments allowlisted -> ok
    assert v.validate_command("cat foo.txt | grep bar")[0] is True
    # second segment not allowlisted -> refused
    assert v.validate_command("cat foo.txt | rm -rf /")[0] is False
    assert v.validate_command("pytest -q && curl http://x")[0] is False


def test_empty_command_refused():
    assert v.validate_command("")[0] is False
    assert v.validate_command("   ")[0] is False


# --------------------------------------------------------------------------- #
# exec gating — default OFF                                                     #
# --------------------------------------------------------------------------- #
def test_exec_enabled_by_default(monkeypatch):
    # Default-ON: the harness runs unless explicitly disabled (the agent already has
    # terminal access and runs these commands anyway; the allowlist is the boundary).
    monkeypatch.delenv("AUTOPILOT_VERIFICATION_EXEC", raising=False)
    assert v.exec_enabled(types.SimpleNamespace()) is True


def test_exec_disabled_via_attr():
    a = types.SimpleNamespace(_autopilot_verification_exec=False)
    assert v.exec_enabled(a) is False


def test_exec_disabled_via_env(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_VERIFICATION_EXEC", "0")
    assert v.exec_enabled(types.SimpleNamespace()) is False


def test_exec_enabled_via_attr():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    assert v.exec_enabled(a) is True


def test_exec_enabled_via_env(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_VERIFICATION_EXEC", "1")
    assert v.exec_enabled(types.SimpleNamespace()) is True


# --------------------------------------------------------------------------- #
# run_verifications — end to end                                               #
# --------------------------------------------------------------------------- #
def _contract_with_verify(*specs):
    crits = tuple(
        c.Criterion(id=i, text=t, satisfiability=c.AGENT_ACHIEVABLE, verify_cmd=cmd)
        for (i, t, cmd) in specs
    )
    return c.AcceptanceContract(criteria=crits, content_hash="h", source_len=1)


def test_noop_when_no_verifiable_criteria():
    ct = _contract_with_verify(("C01", "do a thing", ""))  # no verify_cmd
    report = v.run_verifications(types.SimpleNamespace(_autopilot_verification_exec=True), ct)
    assert report.receipts == []
    assert "no verifiable criteria" in report.note


def test_disabled_reports_but_does_not_run(monkeypatch):
    monkeypatch.delenv("AUTOPILOT_VERIFICATION_EXEC", raising=False)
    ct = _contract_with_verify(("C01", "tests pass", "pytest -q"))
    # explicitly disable (default is now ON)
    report = v.run_verifications(types.SimpleNamespace(_autopilot_verification_exec=False), ct)
    assert report.enabled is False
    assert report.receipts == []
    assert "execution is OFF" in report.note


def test_passing_check_yields_satisfied_receipt():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "true holds", "true"))
    report = v.run_verifications(a, ct)
    assert report.enabled is True
    assert report.satisfied_ids == {"C01"}
    assert report.receipts[0].status == "pass"
    assert report.receipts[0].exit_code == 0


def test_failing_check_not_satisfied():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "false fails", "false"))
    report = v.run_verifications(a, ct)
    assert report.satisfied_ids == set()
    assert "C01" in report.failed_ids
    assert report.receipts[0].status == "fail"


def test_refused_check_not_satisfied():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "sneaky", "rm -rf /tmp/x"))
    report = v.run_verifications(a, ct)
    assert report.satisfied_ids == set()
    assert report.receipts[0].status == "refused"


def test_real_command_output_captured():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "echo works", "echo hello-from-harness"))
    report = v.run_verifications(a, ct)
    assert report.receipts[0].status == "pass"
    assert "hello-from-harness" in report.receipts[0].stdout_tail


def test_mixed_pass_fail():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(
        ("C01", "ok", "true"),
        ("C02", "bad", "false"),
        ("C03", "ok2", "test 1 = 1"),
    )
    report = v.run_verifications(a, ct)
    assert report.satisfied_ids == {"C01", "C03"}
    assert report.failed_ids == {"C02"}


def test_format_receipts_block():
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "ok", "true"), ("C02", "bad", "false"))
    report = v.run_verifications(a, ct)
    block = v.format_receipts_block(report)
    assert "ENGINE VERIFICATION RECEIPTS" in block
    assert "C01" in block and "PASS" in block
    assert "C02" in block and "FAIL" in block


def test_harness_never_raises_on_bad_contract():
    # a contract-like object that throws should degrade to an empty report
    class Boom:
        is_empty = False

        def verifiable_criteria(self):
            raise RuntimeError("boom")

    report = v.run_verifications(types.SimpleNamespace(_autopilot_verification_exec=True), Boom())
    assert report.receipts == []
    assert "harness error" in report.note


def test_reentrancy_guard_suppresses_nested_harness(monkeypatch):
    # when already inside a harness run (child inherits AUTOPILOT_VERIFICATION=1),
    # a nested harness must NOT execute — prevents recursive autopilot-in-a-check.
    monkeypatch.setenv("AUTOPILOT_VERIFICATION", "1")
    a = types.SimpleNamespace(_autopilot_verification_exec=True)
    ct = _contract_with_verify(("C01", "ok", "true"))
    report = v.run_verifications(a, ct)
    assert report.enabled is False
    assert report.receipts == []
    assert "nested harness suppressed" in report.note
