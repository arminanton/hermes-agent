"""Per-tool-result character cap.

A successful compaction can be undone within a couple of turns by a handful of
oversized tool results. Observed live in session 20260828_022304_aed0ee, minutes
after a compaction that took the session from 545 messages down to ~118k tokens:

    cmx_grep     73,016 chars
    cmx_grep     69,871 chars
    cmx_grep     63,914 chars
    skill_view   55,666 chars
    skill_view   49,480 chars

Five calls, roughly 78k tokens, straight back into a just-emptied window.

The cap keeps the head and the tail (the shape of a result lives at its edges:
the first hits, and the summary or verdict) and elides the middle with a marker
that names the tool and the exact number of characters removed, so the model can
distinguish "that was everything" from "there is more, go narrow it".
"""
import agent.tool_dispatch_helpers as tdh


def _with_cap(monkeypatch, cap):
    monkeypatch.setattr(tdh, "_tool_result_char_cap", lambda: cap)


# --------------------------------------------------------------------------
# disabled by default
# --------------------------------------------------------------------------

def test_cap_disabled_passes_content_through(monkeypatch):
    _with_cap(monkeypatch, 0)
    big = "x" * 500_000
    assert tdh._cap_tool_result("cmx_grep", big) == big


def test_content_under_cap_is_untouched(monkeypatch):
    _with_cap(monkeypatch, 1000)
    small = "y" * 999
    assert tdh._cap_tool_result("cmx_grep", small) == small


def test_content_exactly_at_cap_is_untouched(monkeypatch):
    _with_cap(monkeypatch, 1000)
    exact = "y" * 1000
    assert tdh._cap_tool_result("cmx_grep", exact) == exact


# --------------------------------------------------------------------------
# capping behaviour
# --------------------------------------------------------------------------

def test_oversized_result_is_bounded(monkeypatch):
    _with_cap(monkeypatch, 1000)
    out = tdh._cap_tool_result("cmx_grep", "z" * 73_016)
    # head + tail + a bounded marker; nowhere near the original.
    assert len(out) < 2_000
    assert len(out) < 73_016


def test_head_and_tail_are_preserved(monkeypatch):
    _with_cap(monkeypatch, 100)
    content = "HEAD_MARKER" + ("m" * 5_000) + "TAIL_MARKER"
    out = tdh._cap_tool_result("skill_view", content)
    assert out.startswith("HEAD_MARKER")
    assert out.endswith("TAIL_MARKER")


def test_elision_names_the_tool_and_the_amount(monkeypatch):
    """The model must be able to tell truncation happened, and by how much."""
    _with_cap(monkeypatch, 100)
    out = tdh._cap_tool_result("cmx_grep", "q" * 10_000)
    assert "cmx_grep" in out
    assert "elided" in out
    assert "10,000" in out          # the true original size
    assert "characters" in out


def test_middle_is_what_gets_dropped(monkeypatch):
    _with_cap(monkeypatch, 100)
    content = "A" * 2_000 + "SECRET_MIDDLE" + "B" * 2_000
    out = tdh._cap_tool_result("terminal", content)
    assert "SECRET_MIDDLE" not in out


# --------------------------------------------------------------------------
# type safety
# --------------------------------------------------------------------------

def test_multimodal_content_is_not_capped(monkeypatch):
    """Content lists must keep their structure for vision adapters."""
    _with_cap(monkeypatch, 10)
    parts = [{"type": "image_url", "image_url": {"url": "data:..."}}]
    assert tdh._cap_tool_result("browser_vision", parts) is parts


def test_none_content_survives(monkeypatch):
    _with_cap(monkeypatch, 10)
    assert tdh._cap_tool_result("x", None) is None


# --------------------------------------------------------------------------
# integration with the message builder
# --------------------------------------------------------------------------

def test_make_tool_result_message_applies_the_cap(monkeypatch):
    _with_cap(monkeypatch, 200)
    msg = tdh.make_tool_result_message("cmx_grep", "w" * 50_000, "call_1")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert len(msg["content"]) < 50_000
    assert "elided" in msg["content"]


def test_untrusted_wrapper_survives_capping(monkeypatch):
    """Capping runs BEFORE wrapping.

    Truncating after wrapping could sever the closing delimiter and leave the
    model reading untrusted content as if it were trusted.
    """
    _with_cap(monkeypatch, 200)
    msg = tdh.make_tool_result_message("web_extract", "p" * 50_000, "call_2")
    content = msg["content"]
    wrapped_probe = tdh._maybe_wrap_untrusted("web_extract", "tiny")
    if wrapped_probe != "tiny":
        # This tool IS wrapped, so the closing delimiter must still be present.
        tail_marker = wrapped_probe.split("tiny")[-1].strip()
        if tail_marker:
            assert content.rstrip().endswith(tail_marker.rstrip()[-20:])


def test_schema_fields_are_intact_after_capping(monkeypatch):
    _with_cap(monkeypatch, 100)
    msg = tdh.make_tool_result_message("skill_view", "v" * 9_999, "call_3")
    assert set(msg) >= {"role", "name", "tool_name", "content", "tool_call_id"}
    assert msg["name"] == "skill_view"
    assert msg["tool_name"] == "skill_view"
