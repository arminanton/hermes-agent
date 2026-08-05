"""Sentence-boundary streaming normalizer for stream_delta_callback chunks.

Streamed text arrives one chunk at a time, and providers split chunks at
ARBITRARY positions, mid-word, mid-phrase, anywhere. A normalizer that
runs per-chunk will:

  1. Miss patterns that span chunk boundaries (e.g. "frank" + "ly,").
  2. Leak state across reasoning blocks if it carries a tail buffer with
     no reset hook.
  3. Silently lose its trailing buffer if there is no flush hook.

This module solves all three by buffering until a SEMANTIC boundary, the
end of a sentence (``[.!?]`` followed by whitespace) or a newline, then
normalizing the complete sentence and emitting it. The buffer is empty at
every boundary, so there is no cross-block leakage by construction.

Lifecycle hooks the runtime MUST call:

  * ``feed(chunk)`` , for every incoming chunk; returns the normalized
    text to emit (may be empty while still accumulating).
  * ``flush()`` , at end-of-stream; emits whatever is still buffered
    (a final mid-sentence remainder), normalized.
  * ``reset()`` , at the start of a new logical block / new turn;
    discards any partial buffer.

The boundary regex is intentionally conservative: ``[.!?]`` + whitespace,
or a bare newline. This avoids splitting numbers like ``1.5`` mid-token,
because ``1.5`` has no whitespace after the dot. The trade-off is a long
sentence with no internal punctuation will stay buffered until the model
emits its terminal punctuation, which is exactly the behaviour the user
asked for (no mid-sentence flushes during long thinking pauses).
"""
from __future__ import annotations

import re
from typing import Callable

# Sentence boundary: terminal punctuation + optional closing quote/bracket
# + whitespace, OR a bare newline. Whitespace AFTER the punctuation prevents
# splitting numbers like "1.5" or "v2.3" mid-token. Newlines flush
# unconditionally so list-style thinking (each item on its own line) does
# not stall waiting for a `.`.
_BOUNDARY_RE = re.compile(r'(?:[.!?]["\')\]]?\s|\n)')


class StreamNormalizer:
    """Sentence-boundary buffer that applies a normalizer to complete units.

    Usage::

        sn = StreamNormalizer(normalize_fn)
        for chunk in stream:
            out = sn.feed(chunk)
            if out:
                ui.write(out)
        tail = sn.flush()                # at end-of-stream
        if tail:
            ui.write(tail)
        sn.reset()                       # at the start of a new block
    """

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn
        self._buf = ""

    def feed(self, chunk):
        """Append chunk; emit and normalize any complete sentences.

        Returns the normalized text covering every complete sentence
        currently in the buffer. Returns the empty string when the buffer
        does not yet contain a terminal sentence boundary, the caller
        should not emit anything in that case (the chunks are still
        accumulating into the next sentence).
        """
        if not chunk:
            return ""
        self._buf += chunk
        # Find the LAST boundary in the buffer; flush everything up to and
        # including it as one batch. Anything after the last boundary stays
        # buffered for the next call.
        last_end = -1
        for m in _BOUNDARY_RE.finditer(self._buf):
            last_end = m.end()
        if last_end < 0:
            return ""
        ready, self._buf = self._buf[:last_end], self._buf[last_end:]
        try:
            return self._fn(ready)
        except Exception:
            # Defensive: never drop bytes because the normalizer raised.
            # Returning the raw text is safer than silently swallowing it.
            return ready

    def flush(self):
        """Stream ended; emit and normalize whatever is still buffered.

        Called at end-of-stream so a final mid-sentence remainder reaches
        the UI. After ``flush()`` the buffer is empty; subsequent ``feed()``
        calls behave as if the normalizer was fresh.
        """
        out, self._buf = self._buf, ""
        if not out:
            return ""
        try:
            return self._fn(out)
        except Exception:
            return out

    def reset(self):
        """New logical block / new turn starting; discard any partial buffer.

        Use at boundaries where the prior buffer content is NOT semantically
        a prefix of the new block (e.g. a new reasoning block starts after a
        tool call, or a new user turn begins). Without this, a tail held back
        from the previous block would leak into the start of the new one.
        """
        self._buf = ""


def chain_normalizers(*fns: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple ``str -> str`` normalizers left-to-right.

    The dash plugin and the honesty plugin each have their own normalizer;
    in streaming we want both to apply. Composition is order-sensitive but
    safe here: both normalizers are idempotent and operate on disjoint
    surface forms (dashes vs words).
    """
    def _composed(text):
        for fn in fns:
            text = fn(text)
        return text
    return _composed
