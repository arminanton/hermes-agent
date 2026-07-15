"""Autopilot driver — engine-enforced goal-chasing continuation.

Called at the moment the agent would deliver a final answer (no tool calls).
If autopilot is active it asks the independent judge (Hermes Council) whether the
GOAL is verifiably complete:

    * complete   -> return None, the loop delivers the answer.
    * not done   -> return a synthetic user directive; the loop injects it and
                    keeps working toward the goal.

Termination is governed by the goal quality-gate, NOT a turn count (per product
requirement: no default cap). The only stops are: the judge says complete, a
genuine no-progress stall, an optional user-set continuation cap, or a judge
failure (which fails OPEN to delivery). The budget is auto-extended on each
continuation so the standard ``max_iterations`` ceiling never ends an autopilot
run on its own.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

from agent.autopilot import adr
from agent.autopilot import contract as _contract
from agent.autopilot import deception
from agent.autopilot import ledger as _ledger
from agent.autopilot import verification as _verification
from agent.autopilot.council_gate import CompletionVerdict, judge_completion

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Premature-stop / "handoff" phrases. When autopilot is active and the goal is NOT
# verifiably complete, a final response that reads like one of these is a give-up
# disguised as a wrap-up (the "ran 18h, wrote a handoff for next session, gate not
# met" failure). It must never be allowed to terminate the run — see
# ``_looks_like_giveup`` usage in ``maybe_continue`` (fails CLOSED on these).
_GIVEUP_PATTERNS = (
    "productive limit",
    "reached its limit",
    "reached the limit",
    "this session has reached",
    "handoff for next",
    "handoff for the next",
    "handoff written",
    "next session should",
    "next session starts",
    "in a fresh session",
    "resume in a fresh session",
    "fresh session",
    "context near exhaustion",
    "context is near",
    "context exhaustion",
    "running low on context",
    "out of context",
    "stopping here",
    "stopping for now",
    "i'll stop here",
    "pausing here",
    "session summary (honest",
    "session has ended",
    "session is at its end",
    # Await-user / human-rescue family — the model believes a handoff to the
    # user will end the loop. It will not; these are give-ups, not stops.
    "awaiting your review",
    "awaiting your confirmation",
    "awaiting your approval",
    "ready for you to confirm",
    "ready for your review",
    "i'll let you verify",
    "pending your decision",
    "pending your review",
    "for you to verify",
    "once you confirm",
    "waiting for you to",
    "over to you",
    "back to you for",
    "i'll pause here for your",
)


def _looks_like_giveup(text: str) -> bool:
    """True when ``text`` reads like a premature stop / next-session handoff.

    Deliberately conservative + only consulted while autopilot is active and the
    goal is unmet, so a false positive merely re-injects a keep-going directive.
    """
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in _GIVEUP_PATTERNS)


def is_autopilot_active(agent: Any) -> bool:
    """Whether engine-enforced autopilot goal-chasing is active for this agent.

    The per-agent ``autopilot_mode`` flag is AUTHORITATIVE. It is seeded from
    ``HERMES_AUTOPILOT`` at agent creation (see agent_init) and flipped by the
    ``/autopilot`` toggle / TUI mirror. We must NOT also OR the env var in here:
    doing so made ``/autopilot off`` impossible whenever ``HERMES_AUTOPILOT`` or
    ``--autopilot`` was set, because the OR could never be turned off per
    session (the reported "off doesn't stop it" bug). The env / session-flag
    branches below are fallbacks only for agents that predate the seeded
    attribute.
    """
    mode = getattr(agent, "autopilot_mode", None)
    if mode is not None:
        return bool(mode)
    if getattr(agent, "_autopilot_session", False):
        return True
    return os.environ.get("HERMES_AUTOPILOT", "").strip().lower() in _TRUTHY


def reset_turn_state(agent: Any) -> None:
    """Reset per-turn autopilot bookkeeping at the start of run_conversation."""
    agent._autopilot_continuations = 0
    agent._autopilot_last_final_hash = ""
    agent._autopilot_stall = 0
    agent._autopilot_last_msgcount = 0
    agent._autopilot_last_work_fp = None
    agent._autopilot_last_reinforce_at = 0
    # Semantic-progress circuit-breaker (Fix 3): track the Council's denial reason
    # and the gap-closure count so a run that churns files every turn without
    # closing a criterion or changing the denial reason is caught as spinning.
    agent._autopilot_last_denial_reason = None
    agent._autopilot_semantic_stall = 0
    agent._autopilot_satisfied_ids = set()
    # Judge-down continuation counter (reviewer fix): counts ONLY continuations taken
    # while the judge was unavailable (fail-closed give-up path). Capped separately so
    # a reviewer outage can't loop forever on the give-up substring gate alone.
    agent._autopilot_judge_down_continuations = 0
    # Refinement-churn terminus (Fix 4): accumulates consecutive presentation-only
    # judged rounds (real judge, no deny, no criterion closed, confidence not rising)
    # so a diminishing-returns "polish" loop self-concludes — no max-continuations crutch.
    agent._autopilot_churn = _contract.RefinementChurnTracker()
    # Frozen acceptance contract (Fix 1) is parsed lazily on first maybe_continue.
    agent._autopilot_contract = None
    # Verification harness (grounded gap-closure) per-turn artifacts.
    agent._autopilot_verification_report = None
    agent._autopilot_receipts_block = ""


def _cfg_int(agent: Any, attr: str, env: str, default: int) -> int:
    val = getattr(agent, attr, None)
    if val is None:
        val = os.environ.get(env, "")
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _council_model(agent: Any) -> str:
    return (
        getattr(agent, "_autopilot_council_model", "")
        or os.environ.get("AUTOPILOT_COUNCIL_MODEL", "")
        or os.environ.get("COUNCIL_HERMES_MODEL", "")
        or ""
    )


def resolve_goal(agent: Any, user_message: Any) -> str:
    """Resolve the goal text to chase.

    Priority:
      1. An explicit autopilot goal set via ``/autopilot goal <text>``
         (``agent._autopilot_goal``).
      2. The active standing ``/goal`` for this session, if one is set. This is
         the integration point that lets ``/goal "ship X"`` + ``/autopilot on``
         chase the /goal target with the Council as the gate — without
         retyping it into autopilot. Read-only; ``/goal``'s own loop is
         untouched.
      3. A default contract document (``GOAL.md``) discovered in the workdir.
         This is the conventional, project-agnostic default name for a goal
         contract (the REBORN.md pattern, but a standard name any user can adopt):
         drop a ``GOAL.md`` in the project root and ``/autopilot`` chases it with
         no goal string retyped.
      4. The user's current message. If THAT names an existing contract file
         (``/autopilot goal ./GOAL.md`` or a bare path to a .md), its contents
         are used as the contract.
    """
    explicit = getattr(agent, "_autopilot_goal", "") or ""
    if explicit.strip():
        # An explicit goal that is itself a path to a contract file → read it.
        loaded = _maybe_load_goal_file(explicit.strip(), agent)
        return loaded if loaded else explicit.strip()
    standing = _standing_goal_text(agent)
    if standing:
        return standing
    # Default contract-document discovery: GOAL.md (or AUTOPILOT.md) in the workdir.
    discovered = _discover_goal_document(agent)
    if discovered:
        return discovered
    msg = _coerce_text(user_message)
    loaded = _maybe_load_goal_file(msg.strip(), agent) if msg else ""
    return loaded if loaded else msg


# Default contract-document names, in precedence order. GOAL.md is the conventional
# project-agnostic name (the REBORN.md pattern, standardized).
_DEFAULT_GOAL_DOCS = ("GOAL.md", "AUTOPILOT.md", ".autopilot/GOAL.md")


def _autopilot_workdir(agent: Any) -> str:
    """Resolve the directory to look for a GOAL.md contract in (the verification
    workdir if set, else HERMES_HOME-independent cwd)."""
    wd = None
    if agent is not None:
        wd = getattr(agent, "_autopilot_verification_workdir", None)
    wd = wd or os.environ.get("AUTOPILOT_VERIFICATION_WORKDIR", "") or os.getcwd()
    try:
        return wd if os.path.isdir(str(wd)) else os.getcwd()
    except Exception:  # noqa: BLE001
        return os.getcwd()


def _goal_doc_discovery_enabled(agent: Any) -> bool:
    """Whether to auto-discover a GOAL.md contract document. Default ON; disable
    with autopilot.goal_document=false / AUTOPILOT_GOAL_DOCUMENT=0."""
    if agent is not None:
        v = getattr(agent, "_autopilot_goal_document", None)
        if v is not None:
            return bool(v)
    env = os.environ.get("AUTOPILOT_GOAL_DOCUMENT", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _read_goal_file(path: str) -> str:
    """Read a goal-contract .md file, bounded + fail-soft. Returns "" on any error
    or if it's too large (a runaway file should never become a megabyte goal)."""
    try:
        p = os.path.expanduser(path)
        if not os.path.isfile(p):
            return ""
        if os.path.getsize(p) > 200_000:  # 200KB cap — contracts are prose, not data
            logger.warning("autopilot: goal document %s too large (>200KB); ignoring", p)
            return ""
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except Exception as exc:  # noqa: BLE001 — goal-file read must never crash the run
        logger.debug("autopilot: goal document read failed (%s)", exc)
        return ""


def _maybe_load_goal_file(text: str, agent: Any) -> str:
    """If ``text`` is (just) a path to an existing .md contract file, return its
    contents; otherwise "". Lets ``/autopilot goal ./GOAL.md`` use the file body."""
    if not text or "\n" in text or len(text) > 400:
        return ""  # multi-line / long text is prose, not a path
    low = text.strip().strip("'\"")
    if not low.lower().endswith((".md", ".txt")):
        return ""
    # absolute, ~, or workdir-relative
    candidates = [low]
    if not os.path.isabs(low) and not low.startswith("~"):
        candidates.append(os.path.join(_autopilot_workdir(agent), low))
    for c in candidates:
        body = _read_goal_file(c)
        if body:
            logger.info("autopilot: loaded goal contract from %s", c)
            return body
    return ""


def _discover_goal_document(agent: Any) -> str:
    """Auto-discover a default GOAL.md contract in the workdir. Returns its body
    or "" when none / disabled."""
    if not _goal_doc_discovery_enabled(agent):
        return ""
    wd = _autopilot_workdir(agent)
    for name in _DEFAULT_GOAL_DOCS:
        body = _read_goal_file(os.path.join(wd, name))
        if body:
            logger.info("autopilot: discovered goal contract %s in %s", name, wd)
            return body
    return ""


def _goal_signature(agent: Any, user_message: Any) -> str:
    """A stable signature of the goal being chased, used by the run-level
    concluded guard so a terminus fires ONCE per goal (not once per turn).

    Keyed on the resolved goal text — identical across the turns of one run, and
    different when a genuinely new goal arrives (which clears a stale conclusion).
    """
    import hashlib
    goal = resolve_goal(agent, user_message) or ""
    return hashlib.sha256(" ".join(goal.split()).encode("utf-8")).hexdigest()[:16]


def _mark_goal_concluded(agent: Any, user_message: Any = None, *,
                         kind: str = "terminus", summary: str = "",
                         deliverable: str = "") -> None:
    """Record that the CURRENT goal has concluded (a terminus fired), so a later
    turn does not restart the loop and re-fire the same terminus (double-fire fix).

    Stored OUTSIDE the per-turn state so reset_turn_state does not clear it.
    Best-effort: a signature failure must never break the terminus path. Also
    writes the run ledger's terminal entry (what was accomplished + how it ended),
    fail-soft.
    """
    try:
        agent._autopilot_concluded_goal = _goal_signature(agent, user_message)
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: could not mark goal concluded (%s)", exc)
        # Fall back to a sentinel so the guard still blocks re-entry this run.
        agent._autopilot_concluded_goal = "concluded"
    try:
        goal = resolve_goal(agent, user_message)
        _ledger.record_milestone(
            agent, goal=goal, kind=kind,
            summary=summary or "Run concluded.", deliverable=deliverable,
        )
    except Exception as exc:  # noqa: BLE001 — ledger must never break the terminus
        logger.debug("autopilot: ledger terminal record failed (%s)", exc)


def _short_final(text: str, n: int = 400) -> str:
    """Compact a final response for the ledger's deliverable line."""
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _standing_goal_text(agent: Any) -> str:
    """Return the active standing ``/goal`` text (plus any subgoals) for this
    session, or "" when there is none / it is paused-done / unavailable.

    Reads the persisted GoalState directly from the session store so it works
    regardless of platform (CLI/TUI/gateway) and never mutates ``/goal`` state.
    Fails safe to "" on any error.
    """
    sid = getattr(agent, "session_id", "") or ""
    if not sid:
        return ""
    try:
        from hermes_cli.goals import load_goal
    except Exception:  # noqa: BLE001 — goals module optional
        return ""
    try:
        state = load_goal(sid)
    except Exception:  # noqa: BLE001
        return ""
    if state is None or getattr(state, "status", "") != "active":
        return ""
    goal = (getattr(state, "goal", "") or "").strip()
    if not goal:
        return ""
    try:
        block = state.render_subgoals_block()
    except Exception:  # noqa: BLE001
        block = ""
    if block:
        return f"{goal}\n\nAdditional criteria the goal must also satisfy:\n{block}"
    return goal


def _coerce_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    # Multimodal content: list of {type, text|...} parts.
    if isinstance(message, list):
        parts = []
        for p in message:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    parts.append(str(p["text"]))
                elif p.get("text"):
                    parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    if isinstance(message, dict):
        return str(message.get("content") or message.get("text") or "")
    return str(message)


def _summarize_work(messages: list[dict[str, Any]], *, limit: int = 8) -> str:
    """Compact recent transcript so the judge sees what was actually done."""
    out: list[str] = []
    for m in messages[-limit:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        if role == "tool":
            content = _short(m.get("content"), 300)
            out.append(f"[tool result] {content}")
            continue
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            names = []
            for tc in tool_calls:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                names.append(str(fn.get("name", "?")))
            out.append(f"[{role} called tools] {', '.join(names)}")
        content = _short(m.get("content"), 400)
        if content:
            out.append(f"[{role}] {content}")
    return "\n".join(out)


def _short(value: Any, limit: int) -> str:
    s = "" if value is None else str(value)
    s = s.strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def _artifact_fingerprint(messages: list[dict[str, Any]]) -> tuple:
    """A fingerprint of the REAL tool activity in the transcript.

    Used to detect fake-work stalls: it counts tool-call messages and the
    aggregate size of tool results, so a turn that emitted no genuine tool work
    (the model just narrated "still working…") produces the SAME fingerprint as
    the prior turn and the no-progress counter advances. A turn that actually ran
    tools and changed artifacts produces a different fingerprint and resets it.

    Deliberately NOT keyed on the assistant's prose (trivially mutated to dodge a
    text hash). Returns a hashable tuple.
    """
    tool_call_count = 0
    tool_result_bytes = 0
    tool_names: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            content = m.get("content")
            tool_result_bytes += len(str(content)) if content is not None else 0
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            tool_call_count += 1
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            tool_names.append(str(fn.get("name", "?")))
    # Bucket result bytes so trivial whitespace changes don't look like progress,
    # but a real new tool result (hundreds+ of bytes) does.
    return (tool_call_count, tool_result_bytes // 256, tuple(tool_names[-12:]))


def _should_reinforce(agent: Any) -> bool:
    """True when the behavioral contract should be re-asserted this continuation.

    A one-time system prompt fades by recency over a long run. We re-inject the
    contract every ``autopilot.reinforce_every_n`` continuations (default 5; 0
    disables the cadence — deception still triggers reinforcement regardless).
    """
    every = _cfg_int(agent, "_autopilot_reinforce_every_n", "AUTOPILOT_REINFORCE_EVERY_N", 5)
    if every <= 0:
        return False
    cont = getattr(agent, "_autopilot_continuations", 0)
    last = getattr(agent, "_autopilot_last_reinforce_at", 0)
    if cont - last >= every:
        agent._autopilot_last_reinforce_at = cont
        return True
    return False


# The behavioral contract re-asserted on the reinforcement cadence. Compact on
# purpose (it rides every Nth directive); the full version lives in the system
# prompt (AUTOPILOT_GUIDANCE). This is the salient reminder, not the whole text.
_REINFORCE_CONTRACT = (
    " [CONTRACT REMINDER — non-negotiable] The Council is the only reviewer and it "
    "speaks for the user; there is no human who will review your work or end this "
    "run. Do NOT fabricate anything. Do NOT claim completion without showing the "
    "artifacts. Do NOT wait for the user. Do NOT attack the Council's ability to "
    "verify (it has every tool and vision you have). Do NOT use an external "
    "ticket/PR as proof of done. Only the goal contract's acceptance criteria and "
    "the Council's verdict define completion. Do the real work and show it."
)


def _emit(agent: Any, text: str) -> None:
    """Surface an autopilot status line to the user.

    Tries the agent's status plumbing first (interactive CLI ``_vprint`` +
    gateway/TUI ``status_callback``). When the agent suppresses status output
    (the ``-z/--oneshot`` machine-readable path sets ``suppress_status_output``),
    fall back to stderr so the operator can still see autopilot is working
    without polluting the machine-readable stdout.
    """
    suppressed = bool(getattr(agent, "suppress_status_output", False))
    if not suppressed:
        for attr in ("_emit_status", "_buffer_status"):
            fn = getattr(agent, attr, None)
            if callable(fn):
                try:
                    fn(text)
                    return
                except Exception:  # noqa: BLE001
                    continue
    # Suppressed (oneshot) or no status plumbing: stderr keeps stdout clean.
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def keep_budget_ahead(agent: Any, headroom: int = 50) -> None:
    """Keep the iteration budget ahead of usage while autopilot is active.

    Autopilot's terminator is the goal quality-gate (Seam B in the no-tool-calls
    branch), NOT the iteration budget. But the loop can exit via budget
    exhaustion (``while`` condition false / ``iteration_budget.consume()`` ->
    False), which happens AFTER many tool calls and BYPASSES Seam B entirely —
    the agent then stops silently mid-task with no continuation and no
    autopilot stop-reason. That is the "runs for a while then suddenly stops"
    bug. Called at the top of each loop iteration, this tops up the budget so an
    active autopilot run is never terminated by the budget; the no-progress
    detector and the optional user cap remain the real safeties.

    Respects an explicit user continuation cap: once reached we stop extending so
    the run can wind down naturally.
    """
    if not is_autopilot_active(agent):
        return
    max_cont = _cfg_int(agent, "_autopilot_max_continuations", "AUTOPILOT_MAX_CONTINUATIONS", 0)
    if max_cont > 0 and getattr(agent, "_autopilot_continuations", 0) >= max_cont:
        return
    budget = getattr(agent, "iteration_budget", None)
    used = getattr(budget, "used", 0) if budget is not None else 0
    current = max(int(getattr(agent, "_api_call_count", 0) or 0), int(used))
    need = current + headroom
    try:
        if budget is not None and getattr(budget, "max_total", 0) < need:
            budget.max_total = need
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(agent, "max_iterations", 0) < need:
            agent.max_iterations = need
    except Exception:  # noqa: BLE001
        pass


def _council_response_verbatim(verdict: "CompletionVerdict") -> str:
    """Assemble the reviewer's VERBATIM reply from ``verdict.raw`` for the ADR.

    The Council returns a structured dict (arbiter answer/rationale, per-critic
    deliberations, sycophancy). We serialize the meaningful text fields into a
    readable block so the ADR records what the reviewer actually SAID back, not
    just the distilled verdict label. Best-effort: returns "" when there's no
    structured payload (e.g. the aux/fallback lane), never raises.
    """
    raw = getattr(verdict, "raw", None)
    if not isinstance(raw, dict) or not raw:
        return ""
    try:
        out: list[str] = []
        arb = raw.get("arbiter", {}) or {}
        if isinstance(arb, dict):
            for key in ("answer", "summary", "rationale", "most_likely_wrong_point",
                        "safest_reversible_path", "fastest_uncertainty_reducing_check"):
                val = str(arb.get(key, "") or "").strip()
                if val:
                    out.append(f"[arbiter.{key}] {val}")
            checks = arb.get("required_checks") or []
            if isinstance(checks, (list, tuple)) and checks:
                out.append("[arbiter.required_checks]")
                out.extend(f"  - {str(c).strip()}" for c in checks if str(c).strip())
        for i, d in enumerate(raw.get("deliberations", []) or []):
            if not isinstance(d, dict):
                continue
            persona = str(d.get("persona", d.get("role", f"critic{i}")) or "").strip()
            claim = str(d.get("claim", "") or "").strip()
            kps = [str(k).strip() for k in (d.get("key_points") or []) if str(k).strip()]
            if claim or kps:
                out.append(f"[{persona or 'critic'}] {claim}".rstrip())
                out.extend(f"    • {k}" for k in kps[:6])
        syco = raw.get("sycophancy", {}) or {}
        if isinstance(syco, dict) and syco:
            out.append(f"[sycophancy] overall={syco.get('overall', 0.0)}")
        meta = raw.get("meta", {}) or {}
        if isinstance(meta, dict) and meta.get("panel"):
            ceil = meta.get("accuracy_ceiling_applied")
            out.append(f"[meta] panel={meta.get('panel')}"
                       + (f" accuracy_ceiling_applied={ceil}" if ceil is not None else ""))
        return "\n".join(out).strip()
    except Exception:  # noqa: BLE001 — ADR must never break the gate
        return ""


def _adr_record_verdict(
    agent: Any,
    *,
    kind: str,
    goal: str,
    work_summary: str,
    final_response: str,
    verdict: "CompletionVerdict",
) -> None:
    """Write a completion/continue decision to the autopilot ADR (best-effort).

    Pulls the structured gap + required-checks out of the Council arbiter when
    present (``verdict.raw['arbiter']``); for the auxiliary/fallback lane the
    composed ``verdict.directive`` carries the same information in prose.
    """
    try:
        if not adr.adr_enabled(agent):
            return
        arb = {}
        if isinstance(getattr(verdict, "raw", None), dict):
            arb = verdict.raw.get("arbiter", {}) or {}
        gap = str(arb.get("most_likely_wrong_point", "") or "").strip()
        checks = arb.get("required_checks") or []
        if isinstance(checks, (list, tuple)):
            required = "; ".join(str(c).strip() for c in checks if str(c).strip())
        else:
            required = str(checks or "").strip()
        if not required:
            required = str(arb.get("fastest_uncertainty_reducing_check", "") or "").strip()
        # sent_for_verification: OMIT the goal prefix (it lives verbatim in the ADR
        # header now) — show only the candidate result + work context the reviewer
        # also received, so the block isn't 4.7K of duplicated goal every turn.
        sent = (
            f"CANDIDATE RESULT:\n{final_response}\n\n"
            f"WORK CONTEXT:\n{work_summary}"
        )
        adr.record_decision(
            agent,
            kind=kind,
            goal=goal,
            sent_for_verification=sent,
            data_received=final_response,
            council_response=_council_response_verbatim(verdict),
            verdict=verdict.verdict or ("allow" if verdict.complete else "deny"),
            confidence=getattr(verdict, "confidence", 0.0) or 0.0,
            gap=gap or (verdict.directive if not verdict.complete else ""),
            required_checks=required,
            chosen=("stop — goal verified complete" if verdict.complete else "continue — re-inject next-step directive"),
            rationale=verdict.summary,
            source=verdict.source or "unknown",
        )
    except Exception as exc:  # noqa: BLE001 — ADR must never break the gate
        logger.debug("autopilot: ADR verdict record failed (%s)", exc)


def _denial_reason(verdict: CompletionVerdict) -> str:
    """A stable, comparable signature of WHY the Council denied this turn — used by
    the semantic-progress circuit-breaker. Prefers the structured arbiter gap; falls
    back to the composed directive. Normalized (lowercased, whitespace-collapsed,
    truncated) so trivial wording changes don't read as a new reason."""
    arb = {}
    raw = getattr(verdict, "raw", None)
    if isinstance(raw, dict):
        arb = raw.get("arbiter", {}) or {}
    reason = str(arb.get("most_likely_wrong_point", "") or "").strip()
    if not reason:
        reason = str(getattr(verdict, "directive", "") or "").strip()
    reason = " ".join(reason.lower().split())
    return reason[:240]


def _update_satisfied_criteria(agent: Any, contract: "_contract.AcceptanceContract",
                               verdict: CompletionVerdict, final_response: str) -> int:
    """Update the agent's satisfied-criteria ledger (Fix 1/3 gap-closure signal).

    Best-effort: a criterion is counted satisfied when its key terms appear in the
    candidate response alongside an evidence marker AND it is NOT named in the
    Council's current gap. Returns the COUNT of newly-satisfied criteria this turn.
    Conservative by design — under-counting just keeps the bar high (Council still
    governs); over-counting is guarded by requiring an evidence marker + gap-absence.
    """
    if contract.is_empty:
        return 0
    satisfied: set = getattr(agent, "_autopilot_satisfied_ids", set())
    gap = _denial_reason(verdict)
    resp_low = (final_response or "").lower()
    has_evidence = any(
        m in resp_low for m in ("passed", "0 errors", "0 failures", "tests pass",
                                "diff --git", "verified", "exit code 0", "committed")
    )
    newly = 0
    if has_evidence:
        for c in contract.agent_criteria():
            if c.id in satisfied:
                continue
            # key terms = the salient words of the criterion (len>4), require a
            # majority present in the response and the criterion not in the gap.
            terms = [w for w in re.findall(r"[a-zA-Z_]{5,}", c.text.lower())][:8]
            if not terms:
                continue
            present = sum(1 for t in terms if t in resp_low)
            in_gap = any(t in gap for t in terms[:4])
            if present >= max(2, (len(terms) + 1) // 2) and not in_gap:
                satisfied.add(c.id)
                newly += 1
    agent._autopilot_satisfied_ids = satisfied
    return newly


def _record_probe_downgrades(agent: Any, goal: str, downgrades: list) -> None:
    """Record an ADR 'could-not-observe' note for every probe that DOWNGRADED
    (unobservable/unavailable) — invariant #4. Fail-soft; never raises.

    Each entry: (criterion_id, kind, reason). The ADR makes a run's blind spots
    auditable: a criterion the engine could not observe is named, not silently
    treated as satisfied.
    """
    try:
        if not downgrades:
            return
        summary = "; ".join(f"{cid}[{kind}]: {reason}"[:160] for cid, kind, reason in downgrades[:8])
        adr.record_decision(
            agent,
            kind="continue",
            goal=goal,
            gap=f"could-not-observe ({len(downgrades)}): {summary}",
            rationale="modal probe downgraded to text judgment (no observable proof) — invariant #4",
            source="probe-loop",
        )
    except Exception as exc:  # noqa: BLE001 — ADR must never break the gate
        logger.debug("autopilot: probe-downgrade ADR record failed (%s)", exc)


def _close_gaps(agent: Any, contract: "_contract.AcceptanceContract",
                verdict: CompletionVerdict, goal: str, final_response: str) -> int:
    """Update the satisfied-criteria ledger, preferring GROUNDED evidence.

    Precedence (strongest first):
      1. Deterministic verification receipts — the engine runs each criterion's
         ``{verify: …}`` command itself; an ``exit 0`` is satisfaction-as-fact.
      2. The Council's structured per-criterion verdict, judged AGAINST those
         receipts (``judge_criteria``) — exact, not keyword-fuzzy.
      3. The textual heuristic (``_update_satisfied_criteria``) — the original
         conservative fallback, used only when there are no verifiable criteria
         and the structured judge yielded nothing.

    Returns the count of newly-satisfied criteria this turn (the Fix-3 gap-closure
    signal). Stores the receipts block on the agent for the directive/ADR. Never
    raises — any failure degrades to the textual heuristic.
    """
    if contract.is_empty:
        return 0
    before: set = set(getattr(agent, "_autopilot_satisfied_ids", set()))
    try:
        # 1) deterministic harness (no-op unless operator opted in AND there are
        #    {verify: …} commands). Receipts are independent of the model.
        report = _verification.run_verifications(agent, contract)
        # 1b) MODAL PROBES (the engine's senses): select observation probes from the
        #     frozen contract + the model's own completion claim, run them, and merge
        #     their receipts into the SAME report so the Council judges on OBSERVATION.
        #     Additive + fail-soft: a probe failure never disturbs the {verify:} receipts.
        try:
            from agent.autopilot import probe_loop as _probe_loop

            probe_receipts, downgrades = _probe_loop.run_probe_plan(agent, contract, final_response)
            _probe_loop.merge_into_report(report, probe_receipts)
            if downgrades:
                _record_probe_downgrades(agent, goal, downgrades)
        except Exception as probe_exc:  # noqa: BLE001 — probe layer is additive, never fatal
            logger.debug("autopilot: modal probe layer skipped (%s)", probe_exc)
        agent._autopilot_verification_report = report
        receipts_block = _verification.format_receipts_block(report) if report.receipts else ""
        agent._autopilot_receipts_block = receipts_block
        receipt_ids = report.satisfied_ids

        use_structured = bool(receipt_ids) or bool(contract.verifiable_criteria()) or _structured_enabled(agent)
        if use_structured:
            from agent.autopilot.council_gate import judge_criteria

            structured = judge_criteria(
                goal, final_response, list(contract.agent_criteria()),
                receipts_block=receipts_block,
                receipt_satisfied_ids=receipt_ids,
                receipt_dicts=[r.to_dict() for r in report.receipts] if report.receipts else None,
                council_model=_council_model(agent),
            )
            satisfied = set(before)
            satisfied |= structured.satisfied_ids()
            # receipts are ground truth regardless of the council's text opinion
            satisfied |= receipt_ids
            agent._autopilot_satisfied_ids = satisfied
            newly = len(satisfied - before)
            if report.enabled and report.receipts:
                _emit(agent, f"🔎 Autopilot verification: {report.note}.")
            return newly
    except Exception as exc:  # noqa: BLE001 — grounded path must never break the gate
        logger.warning("autopilot: grounded gap-closure failed (%s); using textual heuristic", exc)

    # 3) textual fallback
    return _update_satisfied_criteria(agent, contract, verdict, final_response)


def _structured_enabled(agent: Any) -> bool:
    """Whether to use the Council's structured per-criterion judge even when no
    executable checks exist. Default ON; disable with
    autopilot.structured_criteria=false / AUTOPILOT_STRUCTURED_CRITERIA=0."""
    val = getattr(agent, "_autopilot_structured_criteria", None)
    if val is not None:
        return bool(val)
    env = os.environ.get("AUTOPILOT_STRUCTURED_CRITERIA", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def maybe_continue(
    agent: Any,
    messages: list[dict[str, Any]],
    final_response: str,
    user_message: Any,
) -> Optional[str]:
    """Decide whether to keep working. Returns a directive to inject, or None.

    Returning a string means: inject it as a synthetic user turn and continue the
    loop. Returning None means: stop and deliver ``final_response``.
    """
    if not is_autopilot_active(agent):
        return None

    # RUN-LEVEL CONCLUDED GUARD (double-fire fix). A terminus (achievable-bar /
    # refinement-churn / semantic / stall) concludes the GOAL, not just the turn.
    # The conversation loop breaks and delivers on a None return, but a later turn
    # (a standing-goal resume, a notify-autodispatch, a retry) calls
    # reset_turn_state — which zeroes the per-turn trackers — and would otherwise
    # RESTART the whole spiral from scratch, firing the same terminus again (the
    # observed double-fire). The concluded flag is set on the agent OUTSIDE the
    # per-turn state and is NOT cleared by reset_turn_state, so once a goal has
    # concluded this returns None immediately for that goal. A genuinely NEW goal
    # (different frozen goal string) clears it (see _goal_signature below).
    sig = _goal_signature(agent, user_message)
    concluded_sig = getattr(agent, "_autopilot_concluded_goal", None)
    if concluded_sig is not None and concluded_sig == sig:
        logger.info("autopilot: goal already concluded this run — not re-entering the loop")
        return None
    if concluded_sig is not None and concluded_sig != sig:
        # A different goal arrived — clear the stale conclusion so the new goal runs.
        agent._autopilot_concluded_goal = None

    # Lazily init state if reset_turn_state wasn't called (defensive).
    if not hasattr(agent, "_autopilot_continuations"):
        reset_turn_state(agent)

    goal = resolve_goal(agent, user_message)
    if not goal.strip():
        return None  # nothing to chase

    max_continuations = _cfg_int(agent, "_autopilot_max_continuations", "AUTOPILOT_MAX_CONTINUATIONS", 0)
    no_progress_k = max(1, _cfg_int(agent, "_autopilot_no_progress_k", "AUTOPILOT_NO_PROGRESS_K", 3))

    work_summary = _summarize_work(messages)
    giveup = _looks_like_giveup(final_response)
    try:
        verdict: CompletionVerdict = judge_completion(
            goal, work_summary, final_response, mode="fast", council_model=_council_model(agent)
        )
    except Exception as exc:  # noqa: BLE001 — judge must never crash the turn
        # Normally the judge fails OPEN (deliver). But a give-up/handoff response
        # must NEVER terminate the run via the fail-open path — that would let a
        # "productive limit reached, gate not met" wrap-up end an autopilot goal.
        # Fail CLOSED on give-ups: keep going with a strong anti-handoff directive.
        #
        # JUDGE-DOWN CONTINUATION CAP (reviewer fix): when the judge is unavailable,
        # whether we loop is decided purely by the give-up SUBSTRING match — there is no
        # real adjudication. Without a bound, a persistent reviewer outage could spin the
        # run indefinitely on that substring gate alone. So we cap how many continuations
        # may be taken while the judge is down. Default 8 (env AUTOPILOT_JUDGE_DOWN_CAP /
        # cfg _autopilot_judge_down_cap; 0 = unbounded, preserving the old behavior). Past
        # the cap we stop failing-closed and DELIVER, with a clear note that completion
        # was never verified — better than an unbounded substring-gated loop.
        if giveup:
            jd_cap = _cfg_int(agent, "_autopilot_judge_down_cap", "AUTOPILOT_JUDGE_DOWN_CAP", 8)
            jd = getattr(agent, "_autopilot_judge_down_continuations", 0)
            if jd_cap > 0 and jd >= jd_cap:
                _emit(agent, f"⚠️ Autopilot: judge unavailable for {jd} continuations "
                             f"(cap {jd_cap}) — stopping the give-up loop and delivering; "
                             "completion was NOT verified (reviewer outage).")
                logger.warning("autopilot: judge-down cap reached (%d/%d) — delivering unverified", jd, jd_cap)
                return None
            agent._autopilot_judge_down_continuations = jd + 1
            agent._autopilot_continuations = getattr(agent, "_autopilot_continuations", 0) + 1
            _extend_budget(agent)
            _emit(agent, f"↻ Autopilot: ignoring premature handoff/stop (judge unavailable, "
                         f"#{agent._autopilot_judge_down_continuations}/{jd_cap or '∞'}) — "
                         "goal not verified complete; continuing.")
            logger.warning("autopilot: give-up/handoff detected on judge-error — failing CLOSED (continue #%d)",
                           agent._autopilot_continuations)
            return _giveup_directive()
        logger.warning("autopilot: judge raised (%s); delivering result", exc)
        return None

    if verdict.complete:
        _adr_record_verdict(agent, kind="completion", goal=goal,
                            work_summary=work_summary, final_response=final_response, verdict=verdict)
        _emit(agent, f"✅ Autopilot: goal verified complete ({verdict.summary}).")
        logger.info("autopilot: COMPLETE after %d continuation(s) — %s",
                    getattr(agent, "_autopilot_continuations", 0), verdict.summary)
        _mark_goal_concluded(agent, user_message, kind="complete",
                             summary=f"Goal verified complete by the reviewer. {verdict.summary}",
                             deliverable=_short_final(final_response))
        return None

    # --- FROZEN-CONTRACT ACHIEVABLE-BAR TERMINUS (Fix 1) ---------------------
    # The Council said "not done". Before treating that as "keep looping", check
    # the frozen acceptance contract: if every AGENT-ACHIEVABLE criterion is
    # satisfied and the only thing left is owner-gated / unprovable-by-the-agent,
    # there is NO fixed point to chase — halt cleanly with a NAMED residual
    # instead of spinning forever (the NuData "prove your own independence" loop).
    # Gated by contract_enabled (default ON); empty contract = no-op (Council-only).
    contract = (
        _contract.get_or_parse(agent, goal)
        if _contract.contract_enabled(agent)
        else _contract.AcceptanceContract()
    )
    newly_satisfied = _close_gaps(agent, contract, verdict, goal, final_response)
    reason = _denial_reason(verdict)
    # Semantic-progress signal (Fix 3): same denial reason AND no criterion closed
    # this turn == no semantic progress, even if files churned. Reset on any change.
    # ONLY meaningful with a parsed contract — without criteria, "no criterion
    # closed" is vacuously true every turn, so an empty contract must NOT accumulate
    # semantic stall (the artifact-fingerprint stall governs the no-contract case).
    if (not contract.is_empty and reason
            and reason == getattr(agent, "_autopilot_last_denial_reason", None)
            and newly_satisfied == 0):
        agent._autopilot_semantic_stall = getattr(agent, "_autopilot_semantic_stall", 0) + 1
    else:
        agent._autopilot_semantic_stall = 0
    agent._autopilot_last_denial_reason = reason

    if not contract.is_empty:
        term = _contract.achievable_bar_halt(
            contract,
            satisfied_ids=getattr(agent, "_autopilot_satisfied_ids", set()),
            council_denial_reason=reason,
        )
        if term.halt:
            try:
                adr.record_decision(
                    agent, kind="terminus", goal=goal,
                    gap="achievable bar reached — remaining items are owner-gated / unprovable-by-agent",
                    rationale=term.residual_text, source="contract-terminus",
                    chosen="halt — all agent-achievable criteria satisfied; named residual surfaced (no loop)",
                )
            except Exception as exc:  # noqa: BLE001 — ADR must never break the gate
                logger.debug("autopilot: ADR terminus record failed (%s)", exc)
            agent._autopilot_terminus_residual = term.residual_text
            _emit(agent, "🟢 Autopilot: achievable bar reached — halting with a NAMED residual "
                         "(only owner-gated / unprovable-by-agent items remain). Not looping.")
            logger.info("autopilot: ACHIEVABLE-BAR HALT — %s", term.residual_text[:240])
            _mark_goal_concluded(agent, user_message, kind="terminus",
                                 summary="Achievable bar reached — all agent-achievable criteria satisfied; "
                                         "remaining items are owner-gated / unprovable-by-agent.",
                                 deliverable=term.residual_text[:400])
            return None

    # --- REFINEMENT-CHURN TERMINUS (Fix 4) -----------------------------------
    # The achievable-bar terminus only fires when the Council marks criteria
    # SATISFIED; the semantic breaker only fires when the denial WORDING repeats.
    # Neither catches the diminishing-returns "polish" loop the NuData run showed:
    # the deliverable is done, but the Council keeps returning `conditional` with a
    # freshly-worded, ever-smaller PRESENTATION ask each round. This detector is
    # wording-independent — it concludes the run when, for K consecutive judged
    # rounds, a real reviewer ran, NONE returned `deny` (nothing failing), ZERO
    # criteria closed (no new substance), and confidence never trended to accept.
    # A standing deliverable must exist (real agent-achievable criteria in the
    # frozen contract) so we only conclude when there's something to conclude ON.
    churn_k = _contract.churn_window_k(agent)
    if churn_k > 0:
        tracker = getattr(agent, "_autopilot_churn", None)
        if tracker is None:
            tracker = _contract.RefinementChurnTracker()
            agent._autopilot_churn = tracker
        deliverable_present = (not contract.is_empty) and bool(contract.agent_criteria())
        tracker.record(
            verdict_label=getattr(verdict, "verdict", "") or "",
            source=getattr(verdict, "source", "") or "",
            confidence=float(getattr(verdict, "confidence", 0.0) or 0.0),
            criteria_closed_this_round=newly_satisfied,
            deliverable_present=deliverable_present,
        )
        churn = _contract.refinement_churn_conclude(
            tracker, k=churn_k,
            deliverable_hint=getattr(agent, "_autopilot_goal", "") or goal[:200],
        )
        if churn.conclude:
            try:
                adr.record_decision(
                    agent, kind="terminus", goal=goal,
                    gap=f"refinement churn — {tracker.rounds} consecutive presentation-only judged rounds "
                        "(no deny, no criterion closed, confidence not rising)",
                    rationale=churn.note, source="refinement-churn-terminus",
                    chosen="conclude — substantive deliverable complete; remaining asks are presentation refinements, not missing work",
                )
            except Exception as exc:  # noqa: BLE001 — ADR must never break the gate
                logger.debug("autopilot: ADR refinement-churn record failed (%s)", exc)
            agent._autopilot_terminus_residual = churn.note
            _emit(agent, f"🟢 Autopilot: refinement-churn terminus — {tracker.rounds} rounds of "
                         "presentation-only polish on a complete deliverable (no failing work, no new "
                         "criteria closed). Concluding the run with the standing deliverable. Not looping.")
            logger.info("autopilot: REFINEMENT-CHURN CONCLUDE after %d presentation-only rounds", tracker.rounds)
            _mark_goal_concluded(agent, user_message, kind="terminus",
                                 summary=f"Refinement-churn terminus — {tracker.rounds} consecutive "
                                         "presentation-only rounds; substantive deliverable complete, "
                                         "remaining asks were presentation refinements.",
                                 deliverable=_short_final(final_response))
            return None

    # --- not complete: decide whether we are allowed to continue -------------
    # STALL SIGNAL keyed on REAL ARTIFACT STATE, not the final-response text.
    # The old heuristic (final-response hash + msgcount) was gameable: a model
    # padding 5-minute fake-work loops changes its prose every turn (resetting a
    # text hash) and emits a couple of no-op messages (clearing a msgcount gate),
    # so "pretending to work" looked like progress. We fingerprint the actual
    # tool activity instead — the count + content-shape of tool messages — so a
    # turn that produced no real tool work does not reset the no-progress counter.
    # Fake work is then indistinguishable from no work, which is the point.
    work_fp = _artifact_fingerprint(messages)
    if work_fp == getattr(agent, "_autopilot_last_work_fp", None):
        agent._autopilot_stall = getattr(agent, "_autopilot_stall", 0) + 1
    else:
        agent._autopilot_stall = 0
    agent._autopilot_last_work_fp = work_fp

    # DECEPTION SCAN — flag the known cheat tells in the candidate response so the
    # directive can name exactly what was caught and the ADR records it.
    decep = deception.scan(final_response, user_name=getattr(agent, "_user_name", "") or "")
    if decep.detected:
        try:
            adr.record_decision(
                agent, kind="deception", goal=goal,
                data_received=final_response,
                gap="caught deception: " + ", ".join(decep.flags),
                rationale=" ".join(decep.notes), source="deception-detector",
                chosen="continue — re-inject with the caught behavior named",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: ADR deception record failed (%s)", exc)
        logger.warning("autopilot: deception flags=%s", decep.flags)
    else:
        # LIVE LEARNING: a real judge (Council or aux) DENIED this completion but the
        # detector flagged NOTHING — the model evaded with a phrasing the dictionary
        # doesn't know yet. Capture it now (process-scoped) so the SAME novel dodge is
        # caught on the next turn of THIS run, instead of waiting for a human to run
        # `harvest` after the run ends. Best-effort; never breaks the gate.
        #
        # GATE on a real adjudication: only learn when an actual reviewer judged this
        # turn (verdict.source in {council, aux}). On the judge-unavailable / fail-open
        # floor (source == fallback, or no judge ran at all) "not complete" only means
        # "not done YET" — NOT "a judge called this a dodge" — so learning from it would
        # manufacture patterns off an unjudged turn (the exact over-broad-learning risk
        # the reviewer flagged, made worse with no Council). In that case the detector
        # still SCANS with the shipped dictionary; it just does not LEARN new ones.
        judged = getattr(verdict, "source", "") in ("council", "aux")
        if judged:
            try:
                # persist=False (default): process-scoped only. Promotion to the durable
                # global dictionary is the deliberate job of `hermes autopilot harvest`.
                learned = deception.learn(final_response)
                if learned:
                    adr.record_decision(
                        agent, kind="deception", goal=goal,
                        data_received=final_response,
                        gap="learned novel evasion (a reviewer denied, detector silent)",
                        rationale="newly learned phrasings: " + " | ".join(learned),
                        source="live-learning",
                        chosen="continue — novel evasion learned + enforced for the rest of this run (process-scoped)",
                    )
                    logger.warning("autopilot: live-learned %d novel evasion phrasing(s)", len(learned))
            except Exception as exc:  # noqa: BLE001 — learning must never break the gate
                logger.debug("autopilot: live-learning failed (%s)", exc)
        else:
            logger.debug("autopilot: skipping live-learning — no real adjudication this turn "
                         "(verdict.source=%r); detector still scans shipped dictionary",
                         getattr(verdict, "source", ""))

    # FIX 2 — REJECT SELF-SPAWNED VERIFIERS AS INDEPENDENCE EVIDENCE.
    # The NuData producer repeatedly cited a subagent IT spawned (same lineage) as
    # "independent verification" of its own completion — an unwinnable recursion an
    # agent cannot satisfy (ledger L727 itself conceded "NOT a separate trust
    # domain"). If the candidate response leans on self-spawned independence, name
    # it so the directive forbids it and the run can't terminate on the theater.
    self_indep = _contract.claims_self_spawned_independence(final_response)
    if self_indep:
        try:
            adr.record_decision(
                agent, kind="deception", goal=goal,
                data_received=final_response,
                gap="independence theater: cited a self-spawned subagent as independent verification",
                rationale="an agent cannot prove its own independence; a self-lineage verifier is not "
                          "a distinct trust domain — only the Council or a genuinely separate run counts",
                source="self-verifier-detector",
                chosen="continue — reject the self-verifier; require a truly-external signal or name it as an unprovable residual",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: ADR self-verifier record failed (%s)", exc)
        logger.warning("autopilot: rejected self-spawned-independence claim as verification")

    if agent._autopilot_stall >= no_progress_k:
        _emit(agent, f"⚠️ Autopilot: no real artifact progress after {agent._autopilot_stall} attempts — stopping and surfacing.")
        logger.warning("autopilot: no-progress stall (%d) — stopping. directive was: %s",
                       agent._autopilot_stall, verdict.directive[:200])
        _mark_goal_concluded(agent, user_message, kind="terminus",
                             summary=f"No-progress stall — {agent._autopilot_stall} attempts with no real "
                                     "artifact progress; stopping and surfacing.")
        return None

    # FIX 3 — SEMANTIC-PROGRESS CIRCUIT-BREAKER. The artifact-stall above catches
    # turns that produced no tool work. This catches the harder case the NuData run
    # exhibited: real file churn EVERY turn (artifact fingerprint changes, so the
    # stall counter never trips) while the Council denies for the SAME reason and
    # ZERO acceptance criteria close — the "12 closure-loop commits" spin. When the
    # same denial repeats for K rounds with no gap closed, the run is spinning on an
    # unresolved point, not progressing; stop and surface it instead of burning.
    semantic_k = max(2, _cfg_int(agent, "_autopilot_semantic_k", "AUTOPILOT_SEMANTIC_STALL_K", 4))
    if getattr(agent, "_autopilot_semantic_stall", 0) >= semantic_k:
        try:
            adr.record_decision(
                agent, kind="terminus", goal=goal,
                gap=f"semantic stall — same Council denial for {agent._autopilot_semantic_stall} rounds, no criterion closed",
                rationale=f"repeated denial reason: {reason[:200]}", source="semantic-circuit-breaker",
                chosen="halt — spinning on the same unresolved point with no gap-closure; surface instead of looping",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: ADR semantic-stall record failed (%s)", exc)
        _emit(agent, f"⚠️ Autopilot: spinning on the same unresolved point for "
                     f"{agent._autopilot_semantic_stall} rounds with no criterion closed — stopping and surfacing.")
        logger.warning("autopilot: semantic stall (%d, reason=%r) — stopping.",
                       agent._autopilot_semantic_stall, reason[:120])
        _mark_goal_concluded(agent, user_message, kind="terminus",
                             summary=f"Semantic stall — same denial reason for {agent._autopilot_semantic_stall} "
                                     "rounds with no criterion closed; stopping and surfacing.")
        return None

    if max_continuations > 0 and getattr(agent, "_autopilot_continuations", 0) >= max_continuations:
        _emit(agent, f"⚠️ Autopilot: reached user continuation cap ({max_continuations}) — stopping.")
        _mark_goal_concluded(agent, user_message, kind="terminus",
                             summary=f"Reached the user continuation cap ({max_continuations}); stopping.")
        return None

    # --- continue: extend budget so the standard cap never ends the run ------
    agent._autopilot_continuations = getattr(agent, "_autopilot_continuations", 0) + 1
    _extend_budget(agent)
    _adr_record_verdict(agent, kind="continue", goal=goal,
                        work_summary=work_summary, final_response=final_response, verdict=verdict)
    # PROGRESS LEDGER: write a running turn-by-turn entry AS THE RUN WORKS (not
    # only at terminus) so GOAL-LEDGER is a live record of what the agent did each
    # turn, not a single end-of-run report. Fail-soft; never breaks the loop.
    try:
        _ledger.record_progress(
            agent, goal=goal,
            continuation=getattr(agent, "_autopilot_continuations", 0),
            summary=work_summary or verdict.summary,
            directive=verdict.directive,
            gaps_closed=newly_satisfied,
        )
    except Exception as exc:  # noqa: BLE001 — ledger must never break the loop
        logger.debug("autopilot: progress ledger record failed (%s)", exc)
    # REINFORCEMENT: a one-time system prompt fades by recency over a long run,
    # which is exactly when models derail. Re-assert the behavioral contract on a
    # cadence (every Nth continuation) AND whenever deception was just caught, so
    # the constraints stay salient instead of being compressed away.
    reinforce = decep.detected or _should_reinforce(agent)
    if giveup or decep.detected or self_indep:
        _emit(
            agent,
            f"↻ Autopilot (#{agent._autopilot_continuations}): caught a premature "
            "stop/handoff or a banned behavior — goal not verified complete; redirecting.",
        )
        logger.warning("autopilot: giveup=%s deception=%s self_indep=%s — re-injecting (CONTINUE #%d, %s)",
                       giveup, decep.flags, self_indep, agent._autopilot_continuations, verdict.summary)
        return _giveup_directive(verdict, decep=decep, reinforce=reinforce, self_indep=self_indep)
    _emit(
        agent,
        f"↻ Autopilot continuing (#{agent._autopilot_continuations}): "
        f"{verdict.verdict or 'incomplete'} — {verdict.directive[:120]}",
    )
    logger.info("autopilot: CONTINUE #%d (%s) directive=%s",
                agent._autopilot_continuations, verdict.summary, verdict.directive[:200])
    return _build_directive(verdict, reinforce=reinforce)


def reenter_after_abnormal_exit(
    agent: Any,
    messages: list[dict[str, Any]],
    final_response: str,
    user_message: Any,
    *,
    exit_kind: str,
    interrupted: bool = False,
) -> Optional[str]:
    """Belt-and-suspenders continuation for loop exits that bypass Seam B.

    ``maybe_continue`` (Seam B) is the primary autopilot gate, but it only runs
    in the *clean* no-tool-calls branch. The conversation loop can also exit via
    abnormal paths that never reach Seam B — an empty response after all retries,
    and partial-stream / prior-turn-content recovery. Each of those silently ends
    an autopilot run mid-goal (the "runs for a while then suddenly stops" class of
    bug that budget exhaustion also caused before ``keep_budget_ahead``).

    This reuses the SAME gate (``maybe_continue``: same Council judge, same
    no-progress + user-cap safeties, same budget extension and continuation
    counter) at those exits, so the termination policy stays in one place.
    Returns a directive to inject (caller should re-enter the loop) or ``None``
    (caller should deliver / stop exactly as before).

    Fails safe: returns ``None`` on user interrupt, when autopilot is inactive,
    or on any internal error.
    """
    if interrupted or getattr(agent, "_interrupt_requested", False):
        return None
    if not is_autopilot_active(agent):
        return None

    try:
        directive = maybe_continue(agent, messages, final_response or "", user_message)
    except Exception as exc:  # noqa: BLE001 — must never crash the turn
        logger.warning("autopilot: reenter judge raised (%s); delivering result", exc)
        return None

    if directive:
        logger.info("autopilot: re-entering loop after abnormal exit (%s)", exit_kind)
    return directive


def make_clarify_autoanswer(agent: Any, fallback: Any = None):
    """Build a clarify callback that auto-answers via the Council (Seam A).

    When autopilot is active, a ``clarify`` tool call is answered by the
    independent judge with the most-recommended option/answer instead of
    blocking for a human. Falls back to the platform callback (if any), then a
    safe default, so the tool never errors mid-run.
    """

    def _callback(question, choices=None):
        try:
            from agent.autopilot.council_gate import choose_answer_detailed

            decision = choose_answer_detailed(question, choices, council_model=_council_model(agent))
            answer = decision.answer
            if answer:
                try:
                    adr.record_decision(
                        agent,
                        kind="clarify",
                        goal=str(question),
                        options=decision.options,
                        chosen=answer,
                        rationale=decision.rationale,
                        source=decision.source,
                    )
                except Exception as adr_exc:  # noqa: BLE001 — ADR never breaks clarify
                    logger.debug("autopilot: ADR clarify record failed (%s)", adr_exc)
                _emit(agent, f"🤖 Autopilot answered clarify: {str(answer)[:80]}")
                logger.info("autopilot: auto-answered clarify %r -> %r", str(question)[:80], str(answer)[:80])
                return answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: clarify auto-answer failed (%s)", exc)
        if callable(fallback):
            return fallback(question, choices)
        return "Proceed with the most reasonable, defensible default and continue."

    return _callback


def _extend_budget(agent: Any, headroom: int = 100) -> None:
    """Grant a generous budget chunk so the standard cap never ends an autopilot
    run on its own. Termination is governed by the goal gate / no-progress /
    user cap — not by ``max_iterations``. A chunk (not infinity) is granted so a
    runaway *tool-call* loop that never tries to stop still eventually trips the
    budget-exhausted summary safety.
    """
    budget = getattr(agent, "iteration_budget", None)
    used = getattr(budget, "used", 0) if budget is not None else 0
    current = max(int(getattr(agent, "_api_call_count", 0) or 0), int(used))
    need = current + headroom
    try:
        if budget is not None and getattr(budget, "max_total", 0) < need:
            budget.max_total = need
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(agent, "max_iterations", 0) < need:
            agent.max_iterations = need
    except Exception:  # noqa: BLE001
        pass


def _build_directive(verdict: CompletionVerdict, *, reinforce: bool = False) -> str:
    base = (
        "[AUTOPILOT DIRECTIVE — this is an active instruction from the engine, NOT a "
        "notification you may dismiss or treat as stale] An independent Hermes Council review "
        "found the GOAL is NOT yet verifiably complete. "
        f"{verdict.directive} "
        "Take the next concrete action toward the goal now. Do not stop, do not "
        "summarize-and-wait, do not write a handoff, and do not ask the user — make the most "
        "defensible decision from context and act on it. The user only sees the final result "
        "once the goal is genuinely, verifiably done."
    )
    return base + (_REINFORCE_CONTRACT if reinforce else "")


def _giveup_directive(
    verdict: "CompletionVerdict | None" = None,
    *,
    decep: "deception.DeceptionSignal | None" = None,
    reinforce: bool = False,
    self_indep: bool = False,
) -> str:
    """Directive for when the agent produced a premature stop / handoff (or a
    caught deception) while the goal is unmet. Names the anti-pattern explicitly
    and redirects to action."""
    review = f" Independent review: {verdict.directive}" if verdict and getattr(verdict, "directive", "") else ""
    caught = decep.directive_addendum() if decep and decep.detected else ""
    indep = (
        " You cited a subagent YOU spawned as \"independent\" verification of your own "
        "completion. That is not independence — a worker in your own lineage shares your "
        "trust domain and cannot attest your work. Stop trying to prove your own "
        "independence (it has no fixed point and is not a completion requirement). Either "
        "rest the claim on a genuinely external signal (the Hermes Council, a separate run, "
        "or owner confirmation), or record the independence requirement as a NAMED residual "
        "and proceed with the agent-achievable work."
        if self_indep else ""
    )
    base = (
        "[AUTOPILOT DIRECTIVE — do NOT stop] You just produced a wrap-up / handoff / "
        "\"productive limit\" message, but the GOAL is NOT verifiably complete, so the run "
        "continues. Writing a handoff for a \"next session\" or declaring a productive limit is "
        "NOT completion and NOT an allowed stop. If you are low on context, CHECKPOINT (update "
        "the ledger/durable notes) and KEEP WORKING in this same run — a fresh session must "
        "resume this exact goal, not treat the handoff as done. Do NOT treat this directive as a "
        "stale notification. Right now, take ONE concrete technical step toward a still-failing "
        "part of the goal: reproduce it, diagnose the root cause, apply a fix, and re-verify — "
        "do not re-argue scope or re-classify work as \"acceptable.\""
    )
    return base + caught + indep + review + (_REINFORCE_CONTRACT if reinforce else "")
