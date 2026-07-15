"""Autopilot run ledger — a human-readable "what was accomplished" record.

Distinct from the ADR (``adr.py``): the ADR logs every *decision* the reviewer
made (completion/continue/clarify/deception/terminus) turn by turn — it's the
*why-it-kept-going* trail. The LEDGER is the *what-got-done* trail: a compact,
append-only summary of the run's milestones and its terminal outcome, modeled on
the hand-authored ``REBORN-D-LEDGER.md`` that the NuData contracts maintained by
hand. A naive user who just types ``/autopilot goal "..."`` gets the same
artifact automatically — the run's accomplishments + how it concluded, in one
file next to their work — without authoring a contract.

Design mirrors adr.py: default ON, fail-soft (a ledger error never breaks a run),
canonical copy under the workspace + an optional project copy next to the code.
The ledger is written at TERMINAL moments (a terminus fired, or the goal was
verified complete) so it captures the end-state, plus an optional header on first
write naming the goal.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def ledger_enabled(agent: Any = None) -> bool:
    """True when the autopilot run ledger is on. Default ON (operator opt — the
    accomplishments record is cheap, local, and useful). Disable with
    autopilot.ledger=false / AUTOPILOT_LEDGER=0.
    """
    if agent is not None:
        val = getattr(agent, "_autopilot_ledger", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_LEDGER", "").strip().lower()
    if env == "":
        return True  # default-on
    return env in _TRUTHY


def _session_id(agent: Any) -> str:
    for attr in ("session_id", "_session_id", "_autopilot_session_id"):
        val = getattr(agent, attr, "") if agent is not None else ""
        if val:
            return str(val)[:40]
    return "session"


def _workspace_root() -> Path:
    root = (
        os.environ.get("HERMES_HOME", "").strip()
        or os.environ.get("HERMES_WORKSPACE", "").strip()
        or os.getcwd()
    )
    return Path(root)


def ledger_path(agent: Any = None) -> Path:
    """Canonical ledger path: <workspace>/autopilot/ledger/GOAL-LEDGER-<session>.md.
    An explicit override is honored via AUTOPILOT_LEDGER_PATH / the agent attr.
    """
    override = ""
    if agent is not None:
        override = str(getattr(agent, "_autopilot_ledger_path", "") or "")
    override = override or os.environ.get("AUTOPILOT_LEDGER_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _workspace_root() / "autopilot" / "ledger" / f"GOAL-LEDGER-{_session_id(agent)}.md"


def _project_copy_enabled(agent: Any) -> bool:
    """Also drop the ledger next to the project (the REBORN-D-LEDGER.md location).
    Default ON, mirroring the ADR project-copy."""
    if agent is not None:
        val = getattr(agent, "_autopilot_ledger_project_copy", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_LEDGER_PROJECT_COPY", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _goal_declared_root(goal: str) -> Optional[Path]:
    if not goal:
        return None
    import re
    candidates: list[str] = []
    for m in re.finditer(r"(?:path|root|dir|cwd|repo)\s*[:=]\s*(\S+)", goal, re.IGNORECASE):
        candidates.append(m.group(1))
    for tok in goal.split():
        t = tok.strip("\"'`(),")
        if t.startswith("/") or t.startswith("~") or t.startswith("./"):
            candidates.append(t)
    for c in candidates:
        try:
            p = Path(c).expanduser()
            if p.is_dir():
                return p.resolve()
            if p.is_file():  # a GOAL.md path → its directory is the project
                return p.resolve().parent
        except Exception:  # noqa: BLE001
            continue
    return None


def _project_root(agent: Any, goal: str) -> Path:
    declared = _goal_declared_root(goal)
    if declared is not None:
        return declared
    wd = None
    if agent is not None:
        wd = getattr(agent, "_autopilot_verification_workdir", None)
    cwd = Path(wd or os.environ.get("AUTOPILOT_VERIFICATION_WORKDIR", "").strip() or os.getcwd())
    try:
        cur = cwd.resolve()
        for d in [cur, *cur.parents]:
            if (d / ".git").exists():
                return d
    except Exception:  # noqa: BLE001
        pass
    return cwd


def ledger_targets(agent: Any = None, *, goal: str = "") -> list[Path]:
    """Where the ledger is written: canonical workspace copy + optional project
    copy named GOAL-LEDGER.md next to the code (the REBORN-D-LEDGER.md spot)."""
    targets: list[Path] = [ledger_path(agent)]
    if _project_copy_enabled(agent):
        try:
            root = _project_root(agent, goal)
            proj = root / "GOAL-LEDGER.md"
            if proj.resolve() != targets[0].resolve():
                targets.append(proj)
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: ledger project-copy target failed (%s)", exc)
    return targets


def _short(text: str, n: int = 600) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _write_entry(agent: Any, goal: str, entry: str) -> Optional[list[Path]]:
    """Append one preformatted entry to every ledger target, writing the file
    header on first creation. Shared by ``record_milestone`` (terminal outcomes)
    and ``record_progress`` (the running turn-by-turn trail) so both stay in one
    file with one header. Fail-soft per target — one bad target never breaks the
    others, and the whole thing never raises.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    written: list[Path] = []
    for target in ledger_targets(agent, goal=goal):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # write a header on first creation, naming the goal
            if not target.exists():
                header = (
                    "# Autopilot run ledger\n\n"
                    f"**Goal:** {_short(goal, 800)}\n\n"
                    f"Started {ts}. Entries below are written AS THE RUN WORKS — "
                    "progress steps turn by turn plus the terminal outcome — so this "
                    "is a running record of what the agent did, not a report written "
                    "once at the end. (Companion to the turn-by-turn ADR decision "
                    "log.)\n\n---\n\n"
                )
                target.write_text(header, encoding="utf-8")
            with target.open("a", encoding="utf-8") as fh:
                fh.write(entry)
            written.append(target)
        except Exception as exc:  # noqa: BLE001 — one bad target never breaks the rest
            logger.debug("autopilot: ledger write to %s failed (%s)", target, exc)
    return written or None


def record_milestone(
    agent: Any = None,
    *,
    goal: str = "",
    kind: str = "milestone",
    summary: str = "",
    deliverable: str = "",
) -> Optional[list[Path]]:
    """Append a ledger entry. ``kind`` is a short tag (milestone / terminus /
    complete). ``summary`` is what was accomplished; ``deliverable`` names the
    artifact when there is one. Returns the paths written, or None when disabled /
    on error (never raises).
    """
    if not ledger_enabled(agent):
        return None
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        block = [f"## {ts} — {kind}"]
        if summary:
            block.append(f"- {_short(summary, 1200)}")
        if deliverable:
            block.append(f"- **Deliverable:** {_short(deliverable, 400)}")
        block.append("")
        entry = "\n".join(block) + "\n"
        return _write_entry(agent, goal, entry)
    except Exception as exc:  # noqa: BLE001 — ledger must never break a run
        logger.debug("autopilot: ledger record failed (%s)", exc)
        return None


def record_progress(
    agent: Any = None,
    *,
    goal: str = "",
    continuation: int = 0,
    summary: str = "",
    directive: str = "",
    gaps_closed: int = 0,
) -> Optional[list[Path]]:
    """Append a PROGRESS entry as the run advances — this is what makes the file a
    LEDGER (a running trail of what the agent is doing, turn by turn) instead of a
    final report written once at terminus.

    Called from the continuation seam each turn the run keeps going. Compact by
    design: one entry per advancing turn — the continuation number, how many
    acceptance criteria closed, a short summary of what the turn produced, and the
    next directive being injected. Fail-soft; never raises. Returns the paths
    written, or None when disabled / on error.
    """
    if not ledger_enabled(agent):
        return None
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        tag = f"progress #{continuation}" if continuation else "progress"
        block = [f"## {ts} — {tag}"]
        if gaps_closed:
            block.append(f"- acceptance criteria closed this turn: {gaps_closed}")
        if summary:
            block.append(f"- {_short(summary, 700)}")
        if directive:
            block.append(f"- next directive: {_short(directive, 300)}")
        block.append("")
        entry = "\n".join(block) + "\n"
        return _write_entry(agent, goal, entry)
    except Exception as exc:  # noqa: BLE001 — ledger must never break a run
        logger.debug("autopilot: ledger progress record failed (%s)", exc)
        return None
