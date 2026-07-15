"""Autopilot decision log — an append-only ADR (Architecture Decision Record)
trail of every judgment autopilot makes on a run.

When autopilot decides whether a goal is complete, whether to keep going, or how
to auto-answer a ``clarify`` question, that decision is made by an INDEPENDENT
reviewer (the Hermes Council when available, otherwise a single auxiliary
reviewer). Those decisions are exactly the moments a human would normally be in
the loop, so this module records each one to a human-readable markdown file the
user can review after an unattended run:

    * what was sent for verification (the goal + work context + candidate result),
    * what the reviewer returned (verdict, confidence, the gap it found, and the
      specific checks it said were required to reach a passing state),
    * which options were on the table and which path autopilot took.

It is OFF by default. Enable it with ``autopilot.adr: true`` in config (the CLI
bridges this to ``HERMES_AUTOPILOT_ADR=1``) or ``HERMES_AUTOPILOT_ADR=1`` in the
environment. The file is a local artifact under the workspace; it is never part
of a request to a model and never shipped anywhere.

Where it writes (dual-write):
    * ALWAYS a canonical copy under
      ``<workspace>/.hermes/autopilot/adr/AUTOPILOT-<session>-<YYYYMMDD>.md``
      (or the exact file named by ``autopilot.adr_path`` / ``AUTOPILOT_ADR_PATH``),
      so there is one durable home regardless of where the run was launched.
    * OPTIONALLY a project copy next to the code being worked on, when
      ``autopilot.adr_project_copy`` is on (default on). The project root is the
      goal's declared path if it points at a real dir, else the git top-level of
      the cwd, else the cwd. The subdir under that root is configurable via
      ``autopilot.adr_project_subdir`` and defaults to ``docs/adr`` when the root
      is a git repo (the conventional ADR home), otherwise ``.autopilot/adr``.

Design rules:
    * Pure append. Every decision is one new markdown section; nothing is ever
      rewritten, so a record is lossless and cheap.
    * Fail-soft. Any IO or formatting error is logged at debug and swallowed —
      an ADR problem must never break an autopilot run. The canonical copy and
      the project copy fail independently; one failing never blocks the other.
    * Self-contained. No imports from the rest of autopilot, so it can be unit
      tested in isolation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def adr_enabled(agent: Any = None) -> bool:
    """True when the autopilot ADR decision log is turned on.

    Reads the per-agent attribute first (set from ``config.autopilot.adr`` by the
    CLI bridge), then the ``HERMES_AUTOPILOT_ADR`` environment variable.

    LOCAL DEFAULT-ON (operator opt, 2026-06-24): on this deployment the ADR
    decision trail defaults to ON so every autopilot run is auditable. An explicit
    opt-out is still honored — set ``autopilot.adr: false`` (per-agent attr) or
    ``HERMES_AUTOPILOT_ADR=0`` to disable. NOTE: upstream/PR default stays OFF
    (public PR #51565 promise); this default flip is local to the live tree only.
    """
    if agent is not None:
        val = getattr(agent, "_autopilot_adr", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("HERMES_AUTOPILOT_ADR", "").strip().lower()
    if env == "":
        return True  # local default-on (no explicit setting)
    return env in _TRUTHY


def _session_id(agent: Any) -> str:
    for attr in ("session_id", "_session_id", "_autopilot_session_id"):
        val = getattr(agent, attr, "") if agent is not None else ""
        if val:
            return str(val)[:40]
    return "session"


def adr_path(agent: Any = None) -> Path:
    """Resolve the ADR file path.

    Priority: explicit per-agent attribute / ``AUTOPILOT_ADR_PATH`` env override,
    else ``<workspace>/.hermes/autopilot/adr/AUTOPILOT-<session>-<YYYYMMDD>.md``.
    The workspace root is ``HERMES_HOME`` (the canonical Hermes workspace var the
    wrapper + the rest of the codebase use); ``HERMES_WORKSPACE`` is honored only
    as a legacy alias, and the current working directory is a last-ditch fallback.
    """
    override = ""
    if agent is not None:
        override = str(getattr(agent, "_autopilot_adr_path", "") or "")
    override = override or os.environ.get("AUTOPILOT_ADR_PATH", "").strip()
    if override:
        return Path(override).expanduser()

    root = (
        os.environ.get("HERMES_HOME", "").strip()
        or os.environ.get("HERMES_WORKSPACE", "").strip()
        or os.getcwd()
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(root) / "autopilot" / "adr" / f"AUTOPILOT-{_session_id(agent)}-{day}.md"


def _project_copy_enabled(agent: Any) -> bool:
    """Whether to also drop a copy of the ADR next to the project. Default ON."""
    if agent is not None:
        val = getattr(agent, "_autopilot_adr_project_copy", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_ADR_PROJECT_COPY", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return True  # default on


def _looks_like_dir(token: str) -> bool:
    try:
        p = Path(token).expanduser()
        return p.is_dir()
    except Exception:  # noqa: BLE001
        return False


def _goal_declared_root(goal: str) -> Optional[Path]:
    """If the goal text names an existing directory (a declared project path),
    return it. Scans whitespace-separated tokens and a few path-ish patterns."""
    if not goal:
        return None
    import re
    # explicit "path: /x", "root: /x", "in /x", or any absolute/~ token
    candidates: list[str] = []
    for m in re.finditer(r"(?:path|root|dir|cwd|repo)\s*[:=]\s*(\S+)", goal, re.IGNORECASE):
        candidates.append(m.group(1))
    for tok in goal.split():
        t = tok.strip("\"'`(),")
        if t.startswith("/") or t.startswith("~") or t.startswith("./"):
            candidates.append(t)
    for c in candidates:
        if _looks_like_dir(c):
            return Path(c).expanduser().resolve()
    return None


def _git_toplevel(start: Path) -> Optional[Path]:
    """Walk up from ``start`` to find a git repo root (a dir containing .git)."""
    try:
        cur = start.resolve()
    except Exception:  # noqa: BLE001
        return None
    for d in [cur, *cur.parents]:
        if (d / ".git").exists():
            return d
    return None


def _project_root(agent: Any, goal: str) -> Path:
    """Resolve the project root for the project-copy: goal-declared dir, else the
    git top-level of cwd, else cwd."""
    declared = _goal_declared_root(goal)
    if declared is not None:
        return declared
    cwd = Path(os.environ.get("HERMES_WORKSPACE", "").strip() or os.getcwd())
    git_root = _git_toplevel(cwd)
    return git_root or cwd


def _project_subdir(agent: Any, root: Path) -> str:
    """The subdir under the project root for ADRs. Configurable; defaults to
    ``docs/adr`` for a git repo (conventional), else ``.autopilot/adr``."""
    override = ""
    if agent is not None:
        override = str(getattr(agent, "_autopilot_adr_project_subdir", "") or "")
    override = override or os.environ.get("AUTOPILOT_ADR_PROJECT_SUBDIR", "").strip()
    if override:
        return override.strip("/")
    return "docs/adr" if (root / ".git").exists() else ".autopilot/adr"


def adr_targets(agent: Any = None, *, goal: str = "") -> list[Path]:
    """All paths the ADR should be written to: the canonical copy always, plus the
    project copy when enabled and it resolves to a DIFFERENT file. De-duplicated."""
    targets: list[Path] = [adr_path(agent)]
    if _project_copy_enabled(agent):
        try:
            root = _project_root(agent, goal)
            sub = _project_subdir(agent, root)
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            proj = root / sub / f"AUTOPILOT-{_session_id(agent)}-{day}.md"
            if proj.resolve() != targets[0].resolve():
                targets.append(proj)
        except Exception as exc:  # noqa: BLE001 — project copy is best-effort
            logger.debug("autopilot: ADR project-copy target resolution failed (%s)", exc)
    return targets


def _trunc(text: Any, limit: int) -> str:
    s = "" if text is None else str(text).strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[+{len(s) - limit} chars]"


def _fmt_options(options: Optional[Sequence[str]]) -> str:
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    if not opts:
        return "_(open-ended; no preset options)_"
    return "\n".join(f"  - {o}" for o in opts)


def _field_cap() -> int:
    """Per-field char cap for the ADR's VERBATIM blocks (data received / council
    response). Default 0 = UNBOUNDED — capture verbatim, honoring the operator's
    'no truncation' requirement so the ADR shows exactly what the model sent to
    the reviewer and exactly what the reviewer said back. Set
    ``AUTOPILOT_ADR_MAX_FIELD`` to a positive int to bound pathologically large
    model dumps if an ADR ever grows unwieldy.
    """
    try:
        return max(0, int(os.environ.get("AUTOPILOT_ADR_MAX_FIELD", "0").strip() or "0"))
    except ValueError:
        return 0


def _verbatim(text: Any, cap: int) -> str:
    """Return ``text`` intact (verbatim) unless a positive cap is set, in which
    case truncate with an explicit, honest marker naming how much was cut."""
    s = "" if text is None else str(text).strip()
    if cap and len(s) > cap:
        return s[:cap] + f"\n…[truncated {len(s) - cap} chars; raise AUTOPILOT_ADR_MAX_FIELD to capture more]"
    return s


def _fenced(label: str, body: str) -> str:
    """A labeled fenced block for a multi-line verbatim field."""
    return f"- {label}:\n\n```\n{body}\n```"


def _ensure_header(path: Path, goal: str = "") -> None:
    """Write the file header once, on first record. Stamps the goal VERBATIM
    (complete, not truncated) so the full run objective lives at the top of the
    file exactly once — the per-decision sections below do NOT repeat it (that
    was noise; the goal is identical for every section of a run). Multiple
    concurrent runs each get their own file, so the header goal identifies the
    file without opening the body.
    """
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    goal_line = f"\n**Goal (verbatim):**\n\n{goal.strip()}\n" if goal and goal.strip() else ""
    path.write_text(
        f"# Autopilot decision log\n"
        f"{goal_line}"
        f"\nStarted {started}. Each section below is one decision made by the "
        f"independent reviewer (Hermes Council when available, otherwise the "
        f"auxiliary reviewer / options fallback). Autopilot always took the "
        f"recommended path; this log lets you review what the alternatives were "
        f"and why each path was chosen. The run goal is stated once above and is "
        f"not repeated per section.\n",
        encoding="utf-8",
    )


def record_decision(
    agent: Any,
    *,
    kind: str,
    goal: str = "",
    sent_for_verification: str = "",
    data_received: str = "",
    council_response: str = "",
    options: Optional[Sequence[str]] = None,
    chosen: str = "",
    verdict: str = "",
    confidence: float = 0.0,
    gap: str = "",
    required_checks: str = "",
    rationale: str = "",
    source: str = "",
) -> Optional[Path]:
    """Append one decision record to the ADR file. Returns the path, or None.

    ``kind`` is one of ``completion`` | ``continue`` | ``clarify`` | ``deception``
    | ``terminus``. Every field is optional so both the Council lane (rich verdict
    + gap + checks) and the fallback lane (options + recommended choice) record
    uniformly. Never raises.

    Two VERBATIM fields make each block self-contained (operator request 2026-07-14):
      * ``data_received`` — what the model actually produced this turn (the
        candidate result the reviewer judged), captured verbatim.
      * ``council_response`` — what the reviewer sent back, verbatim (its raw
        answer / rationale), so the block shows the exchange, not just the verdict
        label. Both are unbounded by default (see ``_field_cap``).
    """
    try:
        if not adr_enabled(agent):
            return None
        targets = adr_targets(agent, goal=goal)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        cap = _field_cap()
        lines: list[str] = [f"\n## {ts} — {kind}"]
        lines.append(f"- reviewer: {source or 'unknown'}")
        # DATA RECEIVED — the model's actual output this turn (what the reviewer
        # judged), captured VERBATIM. Placed right after the reviewer line so the
        # block reads "who reviewed → what they were given → what they said back".
        if data_received:
            lines.append(_fenced("data received", _verbatim(data_received, cap)))
        if verdict:
            conf = f" (confidence {confidence:.2f})" if confidence else ""
            lines.append(f"- verdict: {verdict}{conf}")
        # NOTE: the goal is intentionally NOT emitted per-section — it is written
        # once, verbatim, in the file header (_ensure_header). Repeating the goal
        # on every decision was pure noise (identical each time). The
        # sent_for_verification block below therefore OMITS the goal prefix and
        # shows only the candidate result + work context the reviewer also saw.
        if sent_for_verification:
            lines.append(_fenced("sent for verification", _verbatim(sent_for_verification, cap)))
        # COUNCIL RESPONSE — the reviewer's reply VERBATIM (raw answer/rationale),
        # so the ADR shows the exchange, not just the distilled verdict label.
        if council_response:
            lines.append(_fenced("council response (verbatim)", _verbatim(council_response, cap)))
        if gap:
            lines.append(f"- gap found / why not passing: {_verbatim(gap, cap)}")
        if required_checks:
            lines.append(f"- required to pass: {_verbatim(required_checks, cap)}")
        if options is not None:
            lines.append("- options considered:\n" + _fmt_options(options))
        if chosen:
            lines.append(f"- chosen path: {_verbatim(chosen, cap)}")
        if rationale:
            lines.append(f"- rationale: {_verbatim(rationale, cap)}")
        block = "\n".join(lines) + "\n"

        written: Optional[Path] = None
        for target in targets:
            # Each target fails independently — the canonical copy must land even
            # if the project copy's dir is read-only or unwritable, and vice versa.
            try:
                _ensure_header(target, goal=goal)
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(block)
                if written is None:
                    written = target
            except Exception as exc:  # noqa: BLE001
                logger.debug("autopilot: ADR write to %s failed (%s)", target, exc)
        return written
    except Exception as exc:  # noqa: BLE001 — ADR must never break a run
        logger.debug("autopilot: ADR record failed (%s)", exc)
        return None
