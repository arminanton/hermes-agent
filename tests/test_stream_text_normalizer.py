"""Unit + integration tests for the sentence-boundary StreamNormalizer.

Validates the four properties the runtime depends on:
  (1) byte-perfect: sum of feeds + flush == normalize(full text), across many
      random chunkings (proves chunk boundaries do not lose bytes);
  (2) cross-block isolation: reset() drops a buffered partial so a new
      block does not inherit the previous block tail (the "tance role flow"
      leak we shipped + caught + fixed);
  (3) cross-chunk pattern match: a multi-word pattern like "frankly,"
      split across three chunks still gets normalized as one unit;
  (4) flush-on-end: a long buffered string with no sentence terminator is
      delivered when flush() is called at end-of-stream (no silent loss).
"""
import random
import re

from agent.stream_text_normalizer import StreamNormalizer, chain_normalizers


def _strip_honesty(text):
    return re.sub(
        r"\b(?:honestly|frankly|candidly)\b\s*,?\s*",
        "",
        text,
        flags=re.I,
    )


def _normalize_dashes(text):
    return text.replace("\u2014", ", ").replace("\u2013", "-")


_NORMALIZE = chain_normalizers(_normalize_dashes, _strip_honesty)


def _chunk_randomly(text, seed):
    """Split text into 3-15 char chunks like a real SSE stream."""
    random.seed(seed)
    chunks = []
    i = 0
    while i < len(text):
        n = random.randint(3, 15)
        chunks.append(text[i:i + n])
        i += n
    return chunks


_REPRESENTATIVE_INPUTS = [
    # Sentences with terminator + multi-word pattern
    "I'm thinking honestly. The core issue is that patterns might split.",
    # Long sentence with no internal punctuation (buffers for a while)
    "The tradeoff is UX a long sentence at typical generation speeds.",
    # Multi-paragraph (newlines as flush points)
    "First paragraph.\nSecond paragraph.\n\nThird with a question? Yes!",
    # Numbers that look like sentence ends but are not (no whitespace after dot)
    "The version is 1.5 and v2.3 was released. That works.",
]


def test_byte_perfect_round_trip_many_seeds():
    """Sum of feed outputs + flush == normalize(full text), for any chunking."""
    for input_idx, text in enumerate(_REPRESENTATIVE_INPUTS):
        expected = _NORMALIZE(text)
        for seed in range(20):
            sn = StreamNormalizer(_NORMALIZE)
            emitted = []
            for chunk in _chunk_randomly(text, seed):
                out = sn.feed(chunk)
                if out:
                    emitted.append(out)
            tail = sn.flush()
            if tail:
                emitted.append(tail)
            actual = "".join(emitted)
            assert actual == expected, (
                f"input={input_idx} seed={seed}: expected {expected!r} got {actual!r}"
            )


def test_reset_drops_partial_buffer_no_cross_block_leak():
    """reset() throws away buffered partial so block N+1 starts clean."""
    sn = StreamNormalizer(_NORMALIZE)
    for chunk in ["This is an", " incomplete senten"]:
        assert sn.feed(chunk) == ""
    sn.reset()
    # New block starts; first emit must NOT carry "This is an incomplete senten"
    assert sn.feed("New block here. ") == "New block here. "


def test_multi_word_pattern_across_chunks_still_strips():
    """A pattern split across three chunks still gets normalized as one unit."""
    sn = StreamNormalizer(_NORMALIZE)
    emitted = []
    # 'frankly,' split as 'fr' + 'ankly' + ', yes...'
    for chunk in ["He said, fr", "ankly", ", yes. Then he left."]:
        out = sn.feed(chunk)
        if out:
            emitted.append(out)
    tail = sn.flush()
    if tail:
        emitted.append(tail)
    result = "".join(emitted)
    assert "frankly" not in result.lower(), result


def test_flush_emits_buffered_text_with_no_terminator():
    """Long text with no terminator buffers, then flush() delivers it."""
    sn = StreamNormalizer(_NORMALIZE)
    for chunk in ["This is a very ", "long sentence with no ", "terminator at all"]:
        assert sn.feed(chunk) == ""
    assert sn.flush() == "This is a very long sentence with no terminator at all"


def test_empty_input_is_safe():
    """feed('') and feed(None) are no-ops and never raise."""
    sn = StreamNormalizer(_NORMALIZE)
    assert sn.feed("") == ""
    assert sn.feed(None) == ""
    assert sn.flush() == ""


def test_normalizer_exception_does_not_drop_bytes():
    """If the normalizer raises, the raw text is returned (defensive)."""
    def broken(text):
        raise RuntimeError("nope")

    sn = StreamNormalizer(broken)
    out = sn.feed("Hello world. ")
    assert out == "Hello world. "  # raw, not dropped
