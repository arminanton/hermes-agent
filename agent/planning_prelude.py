"""First-turn planning prelude: a gated reminder to lay out a todo before working.

Why this exists
---------------
Measured on this host across two controlled experiments (96-cell Round 4, then a
12-cell A/B/C instruction gradient), ``gpt`` and ``grok`` reach for the ``todo``
and ``skills_list``/``skill_view`` tools unprompted, while ``claude`` and
``gemini`` score **zero** on a prompt that does not mention them, despite being
fully capable: both call the tools correctly the moment the request names the
behaviour. The gap is disposition, not capability.

Prompt volume was tested and rejected as the fix. A 71% prelude reduction, moving
the todo guidance from 74% depth to 8%, and deleting a stale prompt that
advertised a non-existent ``update_todos`` tool all produced no change in
spontaneous adoption.

What every harness that solved this actually does is put a short reminder in the
**turn**, not the system block, and gate it heavily. This module is modelled on
oh-my-pi's ``createEagerTodoPrelude`` (``packages/coding-agent/src/session/
todo-tracker.ts``), whose value is not forcing, it is the gate chain: default off,
first turn only, skip questions, skip when a list already exists, skip when the
tool is not loaded. Seven gates stand between a turn and the reminder.

Modes
-----
``off``       nothing is injected (default; behaviour is unchanged)
``preferred`` a soft reminder is appended to the first user turn
``always``    the same reminder, worded as a requirement

Configuration
-------------
The families differ, so the setting is per-model rather than hardcoded here::

    agent:
      planning_prelude: off            # global default
      planning_prelude_models:         # first matching glob wins
        claude-*: always
        "*gemini*": always

Measured on this host: claude and gemini need it, gpt and grok already open
multi-step work with a plan without it. Putting the model list in config means
that judgement can be revised without a code change when the next model lands.

Note that ``always`` still only *asks*. A named ``tool_choice`` tier is possible
here (``anthropic_adapter`` already maps a bare tool name to
``{"type": "tool", "name": ...}``), but is deliberately not wired in this module:
the reminder is the cheap half, and it should be measured on its own before
anything constrains the model's first move.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Per-session override, checked ahead of config. Mirrors HERMES_PRELUDE_CONFIG:
# it makes the feature switchable for one run without editing config.yaml, which
# matters because config.yaml is shared with any live session on this host.
ENV_VAR = "HERMES_PLANNING_PRELUDE"

# Injected mode values, in increasing strength.
MODE_OFF = "off"
MODE_PREFERRED = "preferred"
MODE_ALWAYS = "always"
VALID_MODES = (MODE_OFF, MODE_PREFERRED, MODE_ALWAYS)

# A trailing "?" or "!" marks a question or an exclamation, which usually wants an
# answer rather than a project plan. oh-my-pi skips forcing on exactly this test.
_QUESTION_ENDINGS = ("?", "!")

# Kept short on purpose. The full rationale for tracking work and surveying
# capabilities lives in the system prompt; repeating it here would just spend
# tokens on every first turn.
#
# Both behaviours are named explicitly. The first A/B measured a version that
# mentioned only `todo`: todo adoption moved from zero, and skills adoption did
# not move at all. A reminder appears to steer exactly what it names and nothing
# adjacent, so the skills half has to be said out loud too.
_PREFERRED_TEXT = (
    "[Planning note: for work that runs to several steps, consider opening with "
    "`skills_list` to see whether a skill already covers this, and the `todo` tool "
    "to lay out the whole request, investigation through verification. Check items "
    "off as you go. Skip both for a single-step request.]"
)

_ALWAYS_TEXT = (
    "[Planning note: this environment expects multi-step work to start with a "
    "capability check and a plan. Call `skills_list` first to see whether an "
    "existing skill covers this work, and `skill_view` on any that apply. Then open "
    "a `todo` list covering the whole request from investigation through "
    "verification rather than just the next step, and check items off as you "
    "complete them. Skip both only if this is genuinely a single-step request.]"
)


def _matches(pattern: str, model: str) -> bool:
    """Glob match against the model id, case-insensitive.

    ``fnmatch`` so config can say ``claude-*`` or ``*gemini*`` rather than
    pinning exact ids that change with every model release. Bare substrings are
    also honoured (``claude`` matches ``claude-opus-5``) because that is the
    obvious thing to write and silently not matching would be a trap.
    """
    pat = (pattern or "").strip().lower()
    mid = (model or "").strip().lower()
    if not pat or not mid:
        return False
    if fnmatch.fnmatch(mid, pat):
        return True
    return "*" not in pat and "?" not in pat and pat in mid


def resolve_mode(agent: Any) -> str:
    """Return the effective prelude mode for this agent's model.

    Resolution order, first hit wins:

    1. ``HERMES_PLANNING_PRELUDE`` env var, a per-process override used by the
       A/B harness so a single run can be switched without touching the shared
       config.yaml.
    2. A per-model rule from ``agent.planning_prelude_models``, a mapping of
       ``model-glob -> mode``. This is the knob that matters in practice: the
       families differ, so the setting has to differ per family rather than
       being hardcoded in this module. Measured on this host, claude and gemini
       need it and gpt and grok already comply without it.
    3. ``agent.planning_prelude``, the global default.
    4. ``off``.

    An unrecognised value degrades to ``off`` rather than raising: a typo in
    config should not break every turn.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or not str(raw).strip():
        raw = _per_model_mode(agent)
    if raw is None:
        raw = getattr(agent, "_planning_prelude_mode", None)
    if raw is None:
        return MODE_OFF
    mode = str(raw).strip().lower()
    if mode not in VALID_MODES:
        logger.warning(
            "Unknown planning-prelude mode %r; falling back to %r (valid: %s)",
            raw, MODE_OFF, ", ".join(VALID_MODES),
        )
        return MODE_OFF
    return mode


def _per_model_mode(agent: Any) -> Optional[str]:
    """The configured mode for this agent's model, or None when no rule matches.

    Rules are checked in declaration order so the first match wins, which lets a
    specific id override a broader family glob written after it.
    """
    rules = getattr(agent, "_planning_prelude_models", None)
    if not rules:
        return None
    model = str(getattr(agent, "model", "") or "")
    if not model:
        return None
    for pattern, mode in rules.items():
        if _matches(str(pattern), model):
            logger.debug(
                "Planning prelude: model %r matched rule %r -> %r", model, pattern, mode
            )
            return mode
    return None


def should_inject(agent: Any, user_message: str, *, mode: Optional[str] = None) -> bool:
    """True when the first-turn planning reminder applies to this turn.

    The gates, in order, each one a reason NOT to nag:

    1. mode is ``off``                  feature disabled (the default)
    2. the ``todo`` tool is not loaded  nothing to point at
    3. not the first user turn          a reminder mid-conversation is noise
    4. a todo list already exists       the model already did the thing
    5. empty message                    nothing to plan against
    6. the message is a question        wants an answer, not a project plan

    ``agent._user_turn_count`` is incremented by the turn prologue *before* this
    runs, so the first turn is ``1``, not ``0``. It is also hydrated from history
    on resume, which is what keeps a resumed session from being re-nagged.
    """
    if (mode or resolve_mode(agent)) == MODE_OFF:
        return False

    valid_tools = getattr(agent, "valid_tool_names", None) or ()
    if "todo" not in valid_tools:
        logger.debug("Planning prelude skipped: todo tool not active")
        return False

    if int(getattr(agent, "_user_turn_count", 0) or 0) != 1:
        return False

    store = getattr(agent, "_todo_store", None)
    if store is not None:
        try:
            if store.has_items():
                return False
        except Exception:  # pragma: no cover - a broken store must not break the turn
            logger.debug("Planning prelude: todo store probe failed", exc_info=True)

    text = (user_message or "").strip()
    if not text:
        return False
    if text.endswith(_QUESTION_ENDINGS):
        return False

    return True


def prelude_text(mode: str) -> str:
    """The reminder body for a mode. Empty string when nothing should be added."""
    if mode == MODE_ALWAYS:
        return _ALWAYS_TEXT
    if mode == MODE_PREFERRED:
        return _PREFERRED_TEXT
    return ""


def apply(agent: Any, user_message: str) -> str:
    """Return ``user_message``, with the planning reminder appended when it applies.

    Appending to the END of the user turn is deliberate: every harness that made
    this work (Cline, Roo, opencode, Goose, OpenHands) puts the reminder in the
    turn rather than the system block, and OmniRoute measured the same effect
    directly, a tool contract at the head of a large system block was ignored
    0/3 while the same contract at the tail plus a one-line rider on the user
    turn landed 16/17.

    The message the user typed is never modified in place, and the reminder is
    bracketed so it reads as an environment note rather than as their words.
    """
    mode = resolve_mode(agent)
    if not should_inject(agent, user_message, mode=mode):
        return user_message

    text = prelude_text(mode)
    if not text:
        return user_message

    logger.info("Planning prelude injected (mode=%s)", mode)
    return f"{user_message}\n\n{text}"
