"""Auto-drafted verification gates — make grounding work WITHOUT author effort.

The deterministic harness (``verification.py``) can run a criterion's check and
ground its satisfaction in a real exit code. But requiring the goal author to hand-
write ``{verify: <cmd>}`` tokens defeats the point for the common case: a person who
types ``/autopilot goal "fix the failing tests"`` will never add tokens, so the
harness would be a no-op exactly when it's most needed.

This module closes that gap. At contract-freeze time it:

  1. DETECTS the project's own checks from its config files — pytest/tox/nox from
     ``pyproject.toml``/``pytest.ini``/``tests/``; ``go test`` from ``go.mod``;
     ``cargo test`` from ``Cargo.toml``; ``npm/pnpm/yarn test`` from
     ``package.json`` (only when a real ``test`` script exists); ``ruff``/``mypy``
     when configured; ``make <target>`` when a Makefile has test/lint/build. Every
     detected command is validated through the SAME read-only allowlist the harness
     enforces, so auto-draft can never introduce a command the harness would refuse.

  2. ATTACHES a detected check to each agent-achievable criterion that lacks one —
     by keyword (a "tests pass" criterion gets the test command, a "lint clean" one
     gets the linter, …), and falls back to the project's primary check for a
     generic criterion. An optional LLM seam can do the mapping when present; the
     deterministic keyword mapper is the always-available default (no LLM required).

  3. SYNTHESIZES criteria when the goal has NONE (a bare one-line goal). "fix the
     failing tests" parses to zero bullet criteria today; here we add a single
     agent-achievable criterion bound to the detected primary check, so the bare
     goal still gets a real gate.

The result: ``/autopilot goal "<text>"`` and ``/autopilot goal <file.md>`` both get
grounded gates automatically, with no token syntax and no opt-in, while every
executed command still passes the conservative allowlist. Auto-draft is best-effort
and fail-soft: any detection error yields the original contract unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from agent.autopilot import verification as _verification
from agent.autopilot.contract import (
    AGENT_ACHIEVABLE,
    AcceptanceContract,
    Criterion,
)

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def autodraft_enabled(agent: Any = None) -> bool:
    """Whether to auto-draft verification gates. Default ON — this is the whole
    point (zero-effort grounding for a plain ``/autopilot goal``). Disable with
    autopilot.autodraft_checks=false / AUTOPILOT_AUTODRAFT_CHECKS=0."""
    if agent is not None:
        val = getattr(agent, "_autopilot_autodraft_checks", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_AUTODRAFT_CHECKS", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


# --------------------------------------------------------------------------- #
# project check detection                                                      #
# --------------------------------------------------------------------------- #
# Each detector is (kind, command, marker-predicate). ``kind`` keys the keyword
# mapper below. The command is what the engine will run — every one is built from
# allowlisted read-only/test verbs only.
def _exists(root: str, *names: str) -> bool:
    return any(os.path.exists(os.path.join(root, n)) for n in names)


def _read(root: str, name: str, limit: int = 20000) -> str:
    try:
        with open(os.path.join(root, name), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except Exception:  # noqa: BLE001
        return ""


def detect_project_checks(root: str) -> dict[str, str]:
    """Return an ordered ``{kind: command}`` map of checks detected from ``root``.

    Deterministic, read-only (only reads config files), and conservative: a check is
    emitted only when its tool is actually configured for this project. Every command
    is allowlist-validated before inclusion, so this can never surface a command the
    harness would refuse.
    """
    checks: dict[str, str] = {}
    try:
        pyproject = _read(root, "pyproject.toml")
        # --- Python: pytest / tox / nox ---
        if (_exists(root, "pytest.ini", "tox.ini", "setup.cfg", "conftest.py")
                or "pytest" in pyproject or _exists(root, "tests", "test")):
            checks["test"] = "pytest -q"
        if "[tool.ruff" in pyproject or _exists(root, "ruff.toml", ".ruff.toml"):
            checks["lint"] = "ruff check ."
        if "[tool.mypy" in pyproject or _exists(root, "mypy.ini", ".mypy.ini"):
            checks["type"] = "mypy ."
        # --- Go ---
        if _exists(root, "go.mod"):
            checks.setdefault("test", "go test ./...")
            checks.setdefault("build", "go build ./...")
            checks.setdefault("vet", "go vet ./...")
        # --- Rust ---
        if _exists(root, "Cargo.toml"):
            checks.setdefault("test", "cargo test")
            checks.setdefault("build", "cargo check")
        # --- Node: only when a real test script exists (avoid the npm default
        #     "no test specified" exit-1 trap) ---
        pkg = _read(root, "package.json")
        if pkg:
            try:
                scripts = (json.loads(pkg).get("scripts") or {}) if pkg.strip() else {}
            except Exception:  # noqa: BLE001
                scripts = {}
            mgr = "pnpm" if _exists(root, "pnpm-lock.yaml") else ("yarn" if _exists(root, "yarn.lock") else "npm")
            if isinstance(scripts, dict):
                if scripts.get("test") and "no test specified" not in str(scripts.get("test")):
                    checks.setdefault("test", f"{mgr} test")
                if scripts.get("lint"):
                    checks.setdefault("lint", f"{mgr} run lint")
                if scripts.get("build"):
                    checks.setdefault("build", f"{mgr} run build")
        # --- Make targets (only the read-only-ish ones) ---
        mk = _read(root, "Makefile") or _read(root, "makefile")
        if mk:
            for tgt in ("test", "lint", "check"):
                if re.search(rf"^{tgt}\s*:", mk, re.MULTILINE):
                    checks.setdefault(tgt if tgt != "check" else "test", f"make {tgt}")
    except Exception as exc:  # noqa: BLE001 — detection must never crash freeze
        logger.debug("autopilot: project check detection failed (%s)", exc)

    # validate every detected command through the harness allowlist; drop any that
    # would be refused (defense in depth — detection should only ever emit safe cmds)
    safe: dict[str, str] = {}
    for kind, cmd in checks.items():
        ok, _reason = _verification.validate_command(cmd)
        if ok:
            safe[kind] = cmd
        else:
            logger.debug("autopilot: dropped non-allowlisted auto-check %r (%s)", cmd, _reason)
    return safe


# Keyword → check-kind mapping. Split into SPECIFIC keywords (name the tool/domain
# unambiguously → weight 3) and GENERIC outcome words (could fit several kinds →
# weight 1). This makes "type checks pass" map to the type checker (specific "type"
# beats generic "pass"/"check") rather than to the test runner.
_KIND_KEYWORDS_SPECIFIC = {
    "test": ("test", "spec", "unit", "integration", "suite", "pytest", "jest", "regression"),
    "lint": ("lint", "ruff", "flake", "eslint", "style", "format", "pep8"),
    "type": ("type", "mypy", "tsc", "typecheck", "typing", "annotation"),
    "build": ("build", "compile", "bundle", "binary", "package", "link"),
    "vet": ("vet", "govet"),
}
_KIND_KEYWORDS_GENERIC = {
    "test": ("pass", "green", "ci", "passing", "fail"),
    "lint": ("clean", "lint-clean"),
    "type": (),
    "build": ("builds", "compiles"),
    "vet": (),
}


def _pick_check_for(text: str, checks: dict[str, str]) -> str:
    """Choose the most appropriate detected command for a criterion's text.

    A SPECIFIC keyword (names the tool/domain) is weighted above a GENERIC outcome
    word, so "type checks pass" → mypy (specific "type") not pytest (generic "pass").
    """
    low = text.lower()
    best_kind, best_score = "", 0
    for kind in checks:
        score = 3 * sum(1 for kw in _KIND_KEYWORDS_SPECIFIC.get(kind, ()) if kw in low)
        score += 1 * sum(1 for kw in _KIND_KEYWORDS_GENERIC.get(kind, ()) if kw in low)
        if score > best_score:
            best_kind, best_score = kind, score
    if best_kind:
        return checks[best_kind]
    return ""  # no keyword match — caller decides whether to use a primary default


def _primary_check(checks: dict[str, str]) -> str:
    """The single most representative check for a generic/sole criterion."""
    for kind in ("test", "build", "lint", "type", "vet"):
        if kind in checks:
            return checks[kind]
    return next(iter(checks.values()), "")


# --------------------------------------------------------------------------- #
# optional LLM mapping seam (deterministic keyword mapper is the default)      #
# --------------------------------------------------------------------------- #
def _llm_map_checks(goal_text: str, criteria: list, checks: dict[str, str],
                    council_model: str = "") -> dict[str, str]:
    """Optional: ask an auxiliary model to map criterion_id -> command, choosing
    only from the detected ``checks`` commands. Returns {} on any failure, so the
    deterministic keyword mapper governs. Never raises."""
    try:
        from agent.autopilot.council_gate import _aux_call, _extract_json  # type: ignore

        crit_lines = "\n".join(f"- {c.id}: {c.text}" for c in criteria)
        cmd_lines = "\n".join(f"- {k}: {v}" for k, v in checks.items())
        sys = (
            "You map acceptance criteria to verification commands for an autonomous "
            "agent. Choose ONLY from the provided commands (or empty if none fits). "
            'Respond with ONLY one JSON object: {"map": [{"id": "C01", "command": '
            '"pytest -q"}]}. Use a command verbatim from the list; do not invent one.'
        )
        user = f"GOAL:\n{goal_text[:1500]}\n\nCRITERIA:\n{crit_lines}\n\nAVAILABLE COMMANDS:\n{cmd_lines}"
        content = _aux_call(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=council_model, max_tokens=500, timeout=60,
        )
        data = _extract_json(content) or {}
        allowed = set(checks.values())
        out: dict[str, str] = {}
        for item in (data.get("map") or []):
            cid = str(item.get("id", "")).strip()
            cmd = str(item.get("command", "")).strip()
            if cid and cmd in allowed:
                out[cid] = cmd
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: LLM check-mapping unavailable (%s)", exc)
        return {}


# --------------------------------------------------------------------------- #
# public API — draft checks into a contract                                    #
# --------------------------------------------------------------------------- #
def autodraft_contract(
    contract: AcceptanceContract,
    *,
    goal_text: str,
    root: str,
    use_llm: bool = False,
    council_model: str = "",
) -> AcceptanceContract:
    """Return a contract whose agent-achievable criteria carry auto-drafted
    ``verify_cmd``s wherever they lacked one, plus a synthesized criterion when the
    goal had none. Fail-soft: returns the input contract unchanged on any problem or
    when no project checks are detectable.

    The frozen content-hash is recomputed so the auto-drafted commands are part of
    the immutable contract (they cannot drift mid-run, same guarantee as authored
    criteria).
    """
    try:
        checks = detect_project_checks(root)
        if not checks:
            return contract

        existing = list(contract.criteria)
        # 1) synthesize a criterion if the goal produced none
        if not existing:
            primary = _primary_check(checks)
            if not primary:
                return contract
            synth = Criterion(
                id="C01",
                text=_synth_text(goal_text, primary),
                satisfiability=AGENT_ACHIEVABLE,
                verify_cmd=primary,
            )
            return _refrozen(AcceptanceContract(criteria=(synth,), source_len=contract.source_len), goal_text)

        # 2) optional LLM mapping (only for criteria lacking a command)
        llm_map: dict[str, str] = {}
        needing = [c for c in existing if c.satisfiability == AGENT_ACHIEVABLE and not c.verify_cmd]
        if use_llm and needing:
            llm_map = _llm_map_checks(goal_text, needing, checks, council_model=council_model)

        # 3) attach a check to each agent-achievable criterion that lacks one
        new_crits: list = []
        drafted = 0
        for c in existing:
            if c.satisfiability == AGENT_ACHIEVABLE and not c.verify_cmd:
                cmd = llm_map.get(c.id) or _pick_check_for(c.text, checks)
                # only fall back to the primary check when the criterion looks like a
                # build/verify outcome, not for e.g. a docs criterion that no check covers
                if not cmd and _looks_verifiable(c.text):
                    cmd = _primary_check(checks)
                if cmd:
                    c = Criterion(id=c.id, text=c.text, satisfiability=c.satisfiability, verify_cmd=cmd)
                    drafted += 1
            new_crits.append(c)
        if not drafted:
            return contract
        logger.info("autopilot: auto-drafted %d verification gate(s) from project checks %s",
                    drafted, list(checks.values()))
        return _refrozen(AcceptanceContract(criteria=tuple(new_crits), source_len=contract.source_len), goal_text)
    except Exception as exc:  # noqa: BLE001 — auto-draft must never break freeze
        logger.debug("autopilot: autodraft failed (%s)", exc)
        return contract


_VERIFIABLE_HINTS = (
    "test", "pass", "build", "compile", "lint", "type", "green", "ci", "suite",
    "spec", "run", "works", "succeed", "zero error", "no error", "no warning",
    "clean", "fix", "bug", "fail",
)


def _looks_verifiable(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in _VERIFIABLE_HINTS)


def _synth_text(goal_text: str, primary: str) -> str:
    g = " ".join((goal_text or "").split())[:160] or "the goal"
    return f"The work for «{g}» is complete and the project's checks pass ({primary})"


def _refrozen(contract: AcceptanceContract, goal_text: str) -> AcceptanceContract:
    import hashlib

    h = hashlib.sha256(
        "\n".join(f"{c.id}:{c.satisfiability}:{c.verify_cmd}:{c.text}" for c in contract.criteria).encode("utf-8")
    ).hexdigest()[:16]
    return AcceptanceContract(criteria=contract.criteria, content_hash=h, source_len=contract.source_len)
