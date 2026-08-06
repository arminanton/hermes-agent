"""Autopilot resume-kick construction.

The derailment this module prevents (NuData jenkins-common, 2026-08-05): an
autopilot run was resumed with a GENERIC nudge —

    "[Autopilot] Resume and keep working toward this goal: <goal> until it is
     verifiably complete. Take the next concrete action now..."

The goal string alone ("work on Phase A, B, C and Final sequencially...") is
project-AGNOSTIC. With no anchor to the work actually in flight, the model
reflexively ran ``cmx_grep`` on the goal keywords, matched *other* projects in
durable memory that use the same "Phase A/B/C/Final" vocabulary, and silently
started working on the WRONG codebase (renamed a Hermes package instead of
continuing a Jenkins shared-library refactor). ~4 hours of a run went to a
project the user never asked about.

The fix is to seed the resume kick with THIS SESSION'S OWN VERBATIM TAIL — the
last few real turns of the conversation being resumed — and to explicitly steer
grounding to that transcript rather than to a memory search. The session tail is
ground truth for "what was I just doing"; a topic-keyword memory grep is not.

Both the CLI (``cli.py``) and the TUI gateway (``tui_gateway/server.py``) resume
paths call :func:`build_resume_kick` so the anti-derail steer is identical across
harnesses (Hermes-native and copilot-cli both drive autopilot through these two
seams).
"""

from __future__ import annotations

import os
from typing import Any, Optional

# How many trailing conversation turns to inline into the resume kick. Enough to
# re-establish the concrete work-in-flight (files, repo, phase, last action)
# without flooding the prompt. Tunable for constrained-context harnesses.
_DEFAULT_TAIL_TURNS = 6


def _tail_turns_limit() -> int:
    try:
        v = int(os.environ.get("AUTOPILOT_RESUME_TAIL_TURNS", _DEFAULT_TAIL_TURNS))
    except (TypeError, ValueError):
        return _DEFAULT_TAIL_TURNS
    # Clamp: 0 disables the tail (falls back to the bare goal kick); cap keeps
    # the injected block bounded even if misconfigured.
    return max(0, min(v, 20))


def _clip(value: Any, limit: int) -> str:
    """One-line clip of a message payload for the tail digest."""
    s = "" if value is None else str(value)
    s = s.strip().replace("\r", " ").replace("\n", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def summarize_session_tail(
    history: Optional[list],
    *,
    turns: Optional[int] = None,
    per_msg_chars: int = 500,
) -> str:
    """Render the last ``turns`` user/assistant turns as a compact digest.

    Tool-result messages are collapsed to a short marker (their bulk is rarely
    what re-orients the resume, and they bloat the kick). Assistant tool CALLS
    are surfaced by name because *what the agent was doing* (e.g. "called
    delegate_task, terminal") is exactly the signal that anchors the correct
    project. Returns "" when there is nothing usable, so callers can fall back
    to the bare-goal kick.
    """
    if not history:
        return ""
    limit = turns if turns is not None else _tail_turns_limit()
    if limit <= 0:
        return ""

    # Walk backwards collecting real user/assistant turns until we have ``limit``
    # of them, then restore chronological order. Tool messages are folded into a
    # single "(N tool results)" marker between substantive turns.
    picked: list[str] = []
    pending_tool = 0
    seen_turns = 0
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role == "tool":
            pending_tool += 1
            continue
        if role not in ("user", "assistant"):
            continue
        if pending_tool:
            picked.append(f"  … ({pending_tool} tool result(s))")
            pending_tool = 0
        tool_calls = msg.get("tool_calls") or []
        call_names = []
        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name", "")).strip()
            if name:
                call_names.append(name)
        content = _clip(msg.get("content"), per_msg_chars)
        line = f"  [{role}] {content}" if content else f"  [{role}]"
        if call_names:
            line += f"  →called: {', '.join(call_names)}"
        picked.append(line)
        seen_turns += 1
        if seen_turns >= limit:
            break

    if not picked:
        return ""
    picked.reverse()
    return "\n".join(picked)


def build_resume_kick(goal: str, history: Optional[list]) -> str:
    """Construct the autopilot resume message.

    When a session tail is available it is inlined and the model is explicitly
    told to ground the next action in THIS transcript — NOT in a memory/keyword
    search — which is the specific behaviour that caused the cross-project
    derailment. When no tail exists (cold resume) the kick degrades to the
    original bare-goal nudge.
    """
    goal = (goal or "").strip()
    target = f" toward this goal: {goal}" if goal else ""
    tail = summarize_session_tail(history)

    if not tail:
        # Cold resume (no prior turns to anchor on): original behaviour.
        return (
            f"[Autopilot] Resume and keep working{target} until it is verifiably "
            "complete. Take the next concrete action now; do not stop, "
            "summarize-and-wait, or ask the user; make the most defensible "
            "decision from context and act on it."
        )

    return (
        f"[Autopilot] Resume and keep working{target} until it is verifiably "
        "complete.\n\n"
        "GROUND YOURSELF IN THE WORK BELOW — this is the verbatim tail of the "
        "CURRENT session you are resuming. Continue THIS work, on THIS project, "
        "from where it actually left off. Do NOT infer the project from the goal "
        "wording alone, and do NOT rely on a memory/keyword search to decide what "
        "to work on — the goal phrasing may collide with other projects in "
        "memory; the transcript below is the authoritative source of what you "
        "were doing.\n\n"
        f"--- current session, last turns ---\n{tail}\n"
        "--- end of current session tail ---\n\n"
        "Take the next concrete action now to advance exactly this work; do not "
        "stop, summarize-and-wait, or ask the user; make the most defensible "
        "decision from this transcript and act on it."
    )
