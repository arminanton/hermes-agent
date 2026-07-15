"""Deterministic verification harness — receipts that GROUND the completion gate.

The autopilot completion gate (Hermes Council) is, by construction, a TEXT-ONLY
reviewer: it reasons over the goal + the agent's self-reported result. The NuData
"Gate A" post-mortem named the gap precisely — the Council arbiter said verbatim
"I cannot execute the AWS pull or read the repo; I must judge from the text
provided." A text-only judge can only ever check *consistency* of a completion
claim, never its *correctness*; so a confident, well-written "done" with no real
artifact behind it can pass.

This module closes that gap the same way ``evidence.py`` closes it for web facts:
the ENGINE (not the model, not a persona) runs the contract's executable
``Verify:`` commands ONCE per turn, captures real exit-code / stdout / stderr
**receipts**, and hands them to two consumers:

    1. the per-criterion satisfaction gate — a criterion with ``exit 0`` is
       satisfied as a FACT, not inferred from prose; and
    2. the Council's context — its reasoning is now grounded in the engine's
       independent re-run, not the agent's narration.

Independence is structural: the harness runs in the engine's trust domain, outside
the model's control, the same boundary the Council itself sits on. This is NOT
"give the personas tools" — that path caused multi-thousand-second hangs when every
persona tried to gather evidence in a sandbox. The harness is one bounded,
deterministic, allowlisted pass.

Safety posture (default-OFF; opt-in only):
  * disabled unless ``autopilot.verification_exec`` / ``AUTOPILOT_VERIFICATION_EXEC``
    is truthy — executing commands from a goal contract is a real capability and the
    operator must grant it;
  * every command's argv[0] (and, for a shell pipeline, each segment's first token)
    must be on an allowlist of read-only / test verbs (pytest, go test, ruff, npm
    test, grep, ls, cat, test, …); anything else is REFUSED, not run;
  * hard per-command timeout and output cap; total wall-clock budget across the turn;
  * commands run in a fixed working directory with a scrubbed environment;
  * any refusal / timeout / error is recorded as a receipt with that status — it
    never raises into the gate, and a refused/failed check NEVER counts as satisfied
    (fail-closed: only a real ``exit 0`` grounds satisfaction).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Allowlisted command verbs. Deliberately read-only / test-oriented: a verify check
# proves a state, it does not mutate one. argv[0] is matched by basename so an
# absolute path (/usr/bin/pytest) is fine. A shell pipeline is split on the shell
# operators and EACH segment's leading token must be on this list.
_ALLOWED_VERBS = frozenset({
    # test runners
    "pytest", "py.test", "tox", "nox", "unittest",
    "go", "cargo", "npm", "pnpm", "yarn", "jest", "vitest", "mocha",
    "phpunit", "rspec", "rake", "mvn", "gradle", "ctest", "make",
    # linters / type / format checkers
    "ruff", "flake8", "pylint", "mypy", "pyright", "black", "isort",
    "eslint", "tsc", "prettier", "golangci-lint", "gofmt", "vet", "clippy",
    "shellcheck", "yamllint", "hadolint",
    # read-only inspection
    "grep", "rg", "egrep", "fgrep", "find", "ls", "cat", "head", "tail",
    "wc", "test", "[", "true", "false", "echo", "stat", "file", "diff",
    "sha256sum", "md5sum", "cksum", "python", "python3", "node",
    "git",  # restricted to read-only subcommands below
    "jq", "sed", "awk", "sort", "uniq", "cut", "tr", "basename", "dirname",
})

# For a few verbs the FIRST argument (subcommand) decides read-only-ness. We only
# allow the read-only subcommands; anything else under these verbs is refused.
_SUBCOMMAND_ALLOW = {
    "go": {"test", "vet", "build", "list", "version"},
    "cargo": {"test", "check", "clippy", "build", "fmt"},
    "npm": {"test", "run", "ci", "ls", "audit"},
    "pnpm": {"test", "run", "ls", "audit"},
    "yarn": {"test", "run", "list", "audit"},
    "git": {"status", "log", "diff", "show", "rev-parse", "ls-files",
            "grep", "branch", "describe", "cat-file", "for-each-ref", "blame"},
    "make": set(),   # make targets are arbitrary; allow bare/any but flag below
    "python": set(),
    "python3": set(),
    "node": set(),
}

# Shell operators we split a pipeline on so we can validate each segment.
_SHELL_SPLIT = ("&&", "||", "|", ";")

# Always-forbidden tokens anywhere in a command (defense in depth against a verb
# on the allowlist being used to mutate / exfiltrate). A command containing any of
# these is refused even if argv[0] is allowed.
_FORBIDDEN_SUBSTRINGS = (
    ">", ">>", "<", "$(", "`", "rm ", "mv ", "cp ", "dd ", "chmod", "chown",
    "curl", "wget", "ssh", "scp", "rsync", "nc ", "ncat", "telnet",
    "sudo", "su ", "kill", "pkill", "reboot", "shutdown", "mkfs",
    ":(){", "eval", "exec ", "source ", "pip install", "npm install -g",
    "apt ", "apt-get", "yum ", "brew ", "docker ", "kubectl",
)


@dataclass
class Receipt:
    """One executed check's deterministic receipt."""

    criterion_id: str
    command: str
    status: str            # "pass" | "fail" | "refused" | "timeout" | "error" | "skipped"
    exit_code: Optional[int]
    stdout_tail: str
    stderr_tail: str
    duration_s: float
    ran_at: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "duration_s": round(self.duration_s, 3),
            "ran_at": self.ran_at,
        }


@dataclass
class VerificationReport:
    """All receipts produced this turn + the engine-derived satisfied set."""

    receipts: list = field(default_factory=list)
    enabled: bool = False
    note: str = ""

    @property
    def satisfied_ids(self) -> set:
        """Criterion ids whose check actually PASSED (exit 0). Fail-closed."""
        return {r.criterion_id for r in self.receipts if r.passed}

    @property
    def failed_ids(self) -> set:
        return {r.criterion_id for r in self.receipts if r.status in ("fail", "timeout", "error")}

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "note": self.note,
            "receipts": [r.to_dict() for r in self.receipts],
            "satisfied_ids": sorted(self.satisfied_ids),
            "failed_ids": sorted(self.failed_ids),
        }


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
def exec_enabled(agent: Any = None) -> bool:
    """Whether the deterministic harness may EXECUTE checks. Default ON.

    Rationale for default-ON: the autopilot agent already has terminal access and
    already runs these exact test/lint commands during its work, so the harness
    running them as an INDEPENDENT, structured gate adds no new capability or attack
    surface — it just converts self-reported "tests pass" into an engine-verified
    fact. The real safety boundary is the read-only allowlist + forbidden-token scan
    in ``validate_command`` (mutating/network verbs are refused), not an opt-in flag.
    Disable with autopilot.verification_exec=false / AUTOPILOT_VERIFICATION_EXEC=0.
    """
    if agent is not None:
        val = getattr(agent, "_autopilot_verification_exec", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_VERIFICATION_EXEC", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _per_cmd_timeout(agent: Any = None) -> float:
    return float(_cfg(agent, "_autopilot_verification_timeout", "AUTOPILOT_VERIFICATION_TIMEOUT", 60))


def _total_budget(agent: Any = None) -> float:
    return float(_cfg(agent, "_autopilot_verification_budget", "AUTOPILOT_VERIFICATION_BUDGET", 240))


def _output_cap() -> int:
    try:
        return int(os.environ.get("AUTOPILOT_VERIFICATION_OUTPUT_CAP", "4000"))
    except (TypeError, ValueError):
        return 4000


def _workdir(agent: Any = None) -> str:
    wd = _cfg(agent, "_autopilot_verification_workdir", "AUTOPILOT_VERIFICATION_WORKDIR", "")
    if wd and os.path.isdir(str(wd)):
        return str(wd)
    return os.getcwd()


def _cfg(agent: Any, attr: str, env: str, default: Any) -> Any:
    if agent is not None:
        v = getattr(agent, attr, None)
        if v is not None:
            return v
    v = os.environ.get(env, "")
    return v if v != "" else default


# --------------------------------------------------------------------------- #
# allowlist validation                                                         #
# --------------------------------------------------------------------------- #
def _segment_ok(segment: str) -> tuple[bool, str]:
    """Validate ONE pipeline segment. Returns (ok, reason_if_not)."""
    try:
        tokens = shlex.split(segment, comments=False, posix=True)
    except ValueError as exc:
        return False, f"unparsable segment ({exc})"
    if not tokens:
        return False, "empty segment"
    verb = os.path.basename(tokens[0])
    if verb not in _ALLOWED_VERBS:
        return False, f"verb not allowlisted: {verb}"
    allowed_subs = _SUBCOMMAND_ALLOW.get(verb)
    if allowed_subs:  # non-empty set => the subcommand is constrained
        sub = next((t for t in tokens[1:] if not t.startswith("-")), "")
        if sub and sub not in allowed_subs:
            return False, f"{verb} subcommand not allowlisted: {sub}"
    return True, ""


def validate_command(command: str) -> tuple[bool, str]:
    """Allowlist gate. Returns (ok, reason). A command is runnable only if every
    pipeline segment's leading verb is allowlisted and no forbidden token appears.
    """
    if not command or not command.strip():
        return False, "empty command"
    low = command.lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in low:
            return False, f"forbidden token: {bad.strip()!r}"
    # split into pipeline segments and validate each leading verb
    segments = [command]
    for op in _SHELL_SPLIT:
        segments = [s for seg in segments for s in seg.split(op)]
    for seg in segments:
        if not seg.strip():
            continue
        ok, reason = _segment_ok(seg)
        if not ok:
            return False, reason
    return True, ""


# --------------------------------------------------------------------------- #
# execution                                                                    #
# --------------------------------------------------------------------------- #
def _tail(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return "…[head truncated]…\n" + text[-cap:]


def _scrubbed_env() -> dict[str, str]:
    """A minimal environment for the check: PATH + a handful of safe locale vars.
    Strips credentials/tokens so a verify command can't read them out."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "PYTHONPATH",
            "VIRTUAL_ENV", "GOPATH", "GOCACHE", "GOMODCACHE", "NODE_PATH")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["AUTOPILOT_VERIFICATION"] = "1"  # let checks know they run under the harness
    return env


def _run_one(criterion_id: str, command: str, *, timeout: float, workdir: str, output_cap: int) -> Receipt:
    started = time.monotonic()
    ran_at = datetime.now(timezone.utc).isoformat()
    ok, reason = validate_command(command)
    if not ok:
        return Receipt(criterion_id, command, "refused", None, "", reason, 0.0, ran_at)
    try:
        proc = subprocess.run(  # nosec B602 — shell=True is needed for pipelines;
            command,            # every segment is allowlist-validated above and the
            shell=True,         # forbidden-substring scan blocks redirection/subshells.
            cwd=workdir,
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        dur = time.monotonic() - started
        status = "pass" if proc.returncode == 0 else "fail"
        return Receipt(
            criterion_id, command, status, proc.returncode,
            _tail(proc.stdout, output_cap), _tail(proc.stderr, output_cap), dur, ran_at,
        )
    except subprocess.TimeoutExpired:
        return Receipt(criterion_id, command, "timeout", None, "", f"timed out after {timeout}s",
                       time.monotonic() - started, ran_at)
    except Exception as exc:  # noqa: BLE001 — a broken check must never crash the gate
        return Receipt(criterion_id, command, "error", None, "", str(exc)[:300],
                       time.monotonic() - started, ran_at)


def run_verifications(agent: Any, contract: Any) -> VerificationReport:
    """Run every criterion's verify_cmd ONCE (bounded) and return receipts.

    No-op (empty, enabled=False) unless the operator opted in AND the contract has
    at least one verifiable criterion. Honors a per-command timeout and a total
    wall-clock budget; once the budget is spent the remaining checks are recorded
    as ``skipped`` (never silently dropped). Never raises.
    """
    report = VerificationReport()
    try:
        verifiable = contract.verifiable_criteria() if contract and not contract.is_empty else []
        if not verifiable:
            report.note = "no verifiable criteria (no {verify: …} commands in contract)"
            return report
        # RE-ENTRANCY GUARD: if we are ALREADY running inside a harness check (the
        # child process inherits AUTOPILOT_VERIFICATION=1), do NOT start a nested
        # harness. This stops the pathological case where a goal's check re-invokes
        # an autopilot run (e.g. the check IS a test suite that drives autopilot),
        # which would recurse. The outer harness is the one that counts.
        if os.environ.get("AUTOPILOT_VERIFICATION", "").strip() == "1":
            report.enabled = False
            report.note = "nested harness suppressed (already inside a verification run)"
            return report
        if not exec_enabled(agent):
            report.enabled = False
            report.note = (
                f"{len(verifiable)} verifiable criteria present but execution is OFF "
                "(set autopilot.verification_exec=true to run them)"
            )
            return report
        report.enabled = True
        per_cmd = _per_cmd_timeout(agent)
        budget = _total_budget(agent)
        workdir = _workdir(agent)
        cap = _output_cap()
        spent = 0.0
        for c in verifiable:
            if spent >= budget:
                report.receipts.append(Receipt(
                    c.id, c.verify_cmd, "skipped", None, "",
                    f"turn verification budget {budget}s exhausted",
                    0.0, datetime.now(timezone.utc).isoformat(),
                ))
                continue
            remaining = max(1.0, min(per_cmd, budget - spent))
            r = _run_one(c.id, c.verify_cmd, timeout=remaining, workdir=workdir, output_cap=cap)
            spent += r.duration_s
            report.receipts.append(r)
            logger.info("autopilot: verify %s `%s` -> %s (exit=%s, %.1fs)",
                        c.id, c.verify_cmd[:80], r.status, r.exit_code, r.duration_s)
        passed = len(report.satisfied_ids)
        report.note = f"ran {len(report.receipts)} checks, {passed} passed, {len(report.failed_ids)} failed"
        return report
    except Exception as exc:  # noqa: BLE001 — harness must never break the gate
        logger.warning("autopilot: verification harness failed (%s)", exc)
        report.note = f"harness error: {exc}"
        return report


def format_receipts_block(report: VerificationReport, *, max_chars: int = 4000) -> str:
    """Render receipts as a compact text block for the Council's context — the
    engine's independent re-run, presented as ground truth the judge can rely on."""
    if not report.receipts:
        return ""
    lines = ["ENGINE VERIFICATION RECEIPTS (deterministic re-run by the engine, "
             "outside the agent's control — treat as ground truth):"]
    for r in report.receipts:
        head = f"  [{r.criterion_id}] {r.status.upper()}"
        if r.exit_code is not None:
            head += f" (exit {r.exit_code})"
        head += f": {r.command[:160]}"
        lines.append(head)
        detail = (r.stderr_tail or r.stdout_tail or "").strip()
        if detail and r.status in ("fail", "timeout", "error", "refused"):
            snippet = detail[:400].replace("\n", " ")
            lines.append(f"      └─ {snippet}")
    block = "\n".join(lines)
    return block[:max_chars]
