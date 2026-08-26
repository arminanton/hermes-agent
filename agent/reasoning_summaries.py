"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models can emit one completed summary part per
``reasoning_content`` delta.  Chat-completions-compatible wires do not carry
the Responses API's ``summary_index``, so clients that concatenate those
deltas can glue adjacent markdown headings together.  This module restores
that dropped paragraph boundary without changing ordinary token streams.
"""

from __future__ import annotations

import re

__all__ = ["separate_glued_reasoning_blocks"]

_COMPLETE_SUMMARY_HEADING = re.compile(r"^\*\*([A-Z][A-Za-z]+ing\b[^*\n]*)\*\*$")
_SUMMARY_BLOCK_OPENER = re.compile(
    r"^\*\*([A-Z][A-Za-z]+ing\b[^*\n]*)\*\*(?:\r?\n\r?\n|$)"
)


def separate_glued_reasoning_blocks(previous: str, delta: str) -> str:
    """Prefix *delta* with a paragraph break at a dropped summary boundary."""
    if not previous or not delta:
        return delta
    if previous[-1].isspace():
        return delta
    # Chat Completions carries no summary_index. Repair only the one shape
    # that is unambiguous without metadata: a complete bold summary heading
    # immediately followed by another complete bold heading. Prose followed
    # by ordinary emphasis, labels, links, and partial Markdown stay intact.
    if not _COMPLETE_SUMMARY_HEADING.fullmatch(previous.lstrip("\r\n")):
        return delta
    if not _SUMMARY_BLOCK_OPENER.match(delta):
        return delta
    return f"\n\n{delta}"
