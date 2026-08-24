"""Offload an oversized pasted user message to ONE retrievable file.

The daily pain this fixes: when a user pastes a large blob into the chat,
downstream context handling elides the middle into a lossy
``[... N chars ...]`` marker that the agent has **no way to retrieve**.
The model then complains that the content "came broken" with references it
cannot open.

The fix intercepts the paste at ingestion, the single point where every
non-interactive input path (piped stdin, ``-p``, the gateway, and the HTTP
API) turns raw user text into the current turn's user message, BEFORE any
lossy elision can happen.  The ENTIRE paste is written to one stable file
under ``$HERMES_HOME/pastes/`` and the message is replaced with a single
RESOLVABLE reference: an absolute path the ``read_file`` tool can open, plus
an explicit instruction telling the model how to retrieve the full content.

Nothing is elided.  The reference round-trips to the complete original bytes.

The interactive TUI already collapses bracketed pastes to a file reference
(see ``cli.py`` ``handle_paste``); this module closes the same gap for every
other ingestion path so the behaviour is uniform.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Detection default: a paste at or above this many characters is offloaded.
# Chosen to clear ordinary prose/code messages comfortably while catching the
# "dumped a whole log / minified JSON / giant file" case that produces the
# broken ``[...]`` references.  Overridable via config
# ``oversized_input.char_threshold``.
DEFAULT_CHAR_THRESHOLD = 50_000


def _paste_dir() -> Path:
    """Resolve ``$HERMES_HOME/pastes`` as a real, absolute directory.

    Uses :func:`hermes_constants.get_hermes_home` (the single source of truth
    that honours the ``HERMES_HOME`` env var / active profile), never the
    display-only ``~/.hermes`` string, so the reference the model receives is a
    genuinely openable absolute path.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "pastes"


def _stable_paste_path(content: str) -> Path:
    """Return a stable, unique file path for this paste.

    The name embeds a content hash so re-pasting the identical blob reuses one
    file (no directory bloat) and a timestamp so distinct pastes never collide.
    """
    digest = hashlib.sha256(
        content.encode("utf-8", "surrogatepass")
    ).hexdigest()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _paste_dir() / f"paste_{stamp}_{digest[:12]}.txt"


def _build_reference(path: Path, content: str) -> str:
    """Build the single resolvable reference that replaces the paste.

    The text is explicit about (a) what happened, (b) the absolute path, and
    (c) exactly how to retrieve the full, unabridged content, so a model that
    needs the body knows precisely how to open it.
    """
    n_chars = len(content)
    n_lines = content.count("\n") + 1
    return (
        f"[Large pasted content was saved to a file to keep the conversation "
        f"readable ({n_chars:,} characters, {n_lines:,} lines). This is the "
        f"COMPLETE, unabridged paste, nothing was elided or truncated.\n"
        f"Full content file (absolute path): {path}\n"
        f"To read it, call the read_file tool with "
        f"path=\"{path}\". read_file paginates via offset/limit, so use those "
        f"to page through it if it is very large. Prefer search_files over "
        f"read_file if you only need to find a specific section.]"
    )


def _threshold(agent: Any) -> int:
    """Effective character threshold for this agent (config-overridable)."""
    raw = getattr(agent, "_oversized_input_char_threshold", None)
    if raw is None:
        return DEFAULT_CHAR_THRESHOLD
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CHAR_THRESHOLD
    # A threshold of 0 or below disables offloading (documented opt-out).
    return val


def _enabled(agent: Any) -> bool:
    """Whether ingestion offloading is enabled for this agent."""
    return bool(getattr(agent, "_oversized_input_enabled", True))


def should_offload(agent: Any, content: Any) -> bool:
    """True when ``content`` is a plain string large enough to offload."""
    if not _enabled(agent):
        return False
    if not isinstance(content, str) or not content:
        return False
    threshold = _threshold(agent)
    if threshold <= 0:
        return False
    return len(content) >= threshold


def write_paste_file(content: str) -> Optional[Path]:
    """Persist the ENTIRE paste to one file and return its absolute path.

    Returns ``None`` (fail-soft) if the write fails, so the caller falls
    through to the unchanged, pre-existing behaviour.
    """
    try:
        path = _stable_paste_path(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: an identical prior paste already wrote these exact bytes.
        if not path.exists():
            path.write_text(content, encoding="utf-8", errors="surrogatepass")
        return path.resolve()
    except Exception as exc:  # pragma: no cover, disk edge
        logger.warning("oversized-paste offload write failed: %s", exc)
        return None


def maybe_offload_oversized_message(
    agent: Any,
    user_message: Any,
    persist_user_message: Any = None,
) -> Tuple[Any, Any, Optional[Path]]:
    """Offload an oversized string user message to one retrievable file.

    Returns ``(user_message, persist_user_message, path)``.  When the message
    is offloaded, both the API-facing and persisted content become the single
    resolvable reference (so a reload does not re-inflate and re-trigger), and
    ``path`` is the absolute paste file.  When nothing is offloaded the inputs
    are returned unchanged with ``path=None``.
    """
    if not should_offload(agent, user_message):
        return user_message, persist_user_message, None

    path = write_paste_file(user_message)
    if path is None:
        return user_message, persist_user_message, None

    reference = _build_reference(path, user_message)
    logger.info(
        "Offloaded oversized paste (%d chars) to %s",
        len(user_message),
        path,
    )
    # Persisted content also becomes the reference: the file IS the durable
    # home of the full bytes, and re-hydrating the raw blob on reload would
    # just re-trigger this path.
    new_persist = (
        reference
        if isinstance(persist_user_message, str)
        else persist_user_message
    )
    return reference, new_persist, path
