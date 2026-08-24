"""Oversized-paste ingestion offload tests (reworked #50073).

The daily pain being fixed: pasting a large blob produced broken
``[... N chars ...]`` references the agent could not retrieve.  The rework
intercepts at ingestion and turns the WHOLE paste into ONE file with a single
resolvable reference (an absolute path ``read_file`` can open).

Coverage:
  * oversized paste  -> exactly one file on disk holding the FULL bytes, and
    the message carries a resolvable absolute path that round-trips.
  * under-threshold paste -> untouched (no file, message unchanged).
  * config defaults present (enabled + char_threshold) and behaviour toggles.
"""
import types
from pathlib import Path

import pytest


# ── config defaults ──────────────────────────────────────────────────
def test_oversized_input_config_defaults_present():
    """The oversized_input section ships enabled with a char threshold."""
    from hermes_cli.config import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["oversized_input"]
    assert section["enabled"] is True
    assert section["char_threshold"] == 50_000


# ── helpers ──────────────────────────────────────────────────────────
def _make_agent(enabled=True, threshold=50_000):
    """Minimal duck-typed agent exposing just the offload knobs."""
    agent = types.SimpleNamespace()
    agent._oversized_input_enabled = enabled
    agent._oversized_input_char_threshold = threshold
    return agent


def _isolate_home(monkeypatch, tmp_path):
    """Point HERMES_HOME at a real absolute dir and move CWD elsewhere so a
    buggy relative write would be detectable."""
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    # get_hermes_home caches nothing but has a context override; ensure clean.
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home_override", lambda: None, raising=False
    )
    return hermes_home, cwd


# ── oversized paste -> one file + resolvable reference ────────────────
def test_oversized_paste_writes_one_file_with_full_content(tmp_path, monkeypatch):
    hermes_home, cwd = _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=1_000)
    # A big, multi-part blob: the kind that used to get chopped into [...].
    big = "".join(f"line {i} of pasted content\n" for i in range(5_000))
    assert len(big) >= 1_000

    new_msg, new_persist, path = maybe_offload_oversized_message(
        agent, big, big
    )

    # A path was returned, it is absolute, and it exists.
    assert path is not None
    assert path.is_absolute()
    assert path.exists()

    # Exactly ONE file under $HERMES_HOME/pastes, holding the FULL bytes.
    paste_dir = hermes_home / "pastes"
    spills = list(paste_dir.glob("paste_*.txt"))
    assert len(spills) == 1, f"expected 1 paste file, got {spills}"
    assert spills[0].read_text(encoding="utf-8") == big
    # Round-trip the returned path too (the reference the model receives).
    assert path.read_text(encoding="utf-8") == big

    # The message is now a SINGLE resolvable reference, not lossy fragments.
    assert isinstance(new_msg, str)
    assert len(new_msg) < 1_000
    assert str(path) in new_msg
    assert "read_file" in new_msg
    # No lossy elision marker leaked in.
    assert "chars omitted" not in new_msg
    assert "..." not in new_msg or "read_file" in new_msg
    # Persisted content also became the reference (no re-inflation on reload).
    assert new_persist == new_msg

    # Nothing leaked under the CWD.
    assert not (cwd / "~").exists()
    assert not (cwd / "pastes").exists()
    assert list(cwd.rglob("paste_*.txt")) == []


def test_reference_path_round_trips_full_bytes(tmp_path, monkeypatch):
    """Parse the path out of the reference exactly as a model would, and read
    the complete original content back."""
    _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=100)
    big = "PAYLOAD-\u00e9\n" * 20_000  # includes non-ascii to test encoding
    new_msg, _persist, _path = maybe_offload_oversized_message(agent, big)

    # Reference format: "...(absolute path): <PATH>\n..."
    marker = "(absolute path): "
    assert marker in new_msg
    ref = new_msg.split(marker, 1)[1].splitlines()[0].strip()
    p = Path(ref)
    assert p.is_absolute()
    assert p.exists()
    assert p.read_text(encoding="utf-8") == big


def test_identical_paste_reuses_one_file(tmp_path, monkeypatch):
    """Re-pasting the identical blob reuses the same content-addressed file
    (no directory bloat), still holding the full bytes."""
    hermes_home, _ = _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=100)
    big = "SAME\n" * 10_000
    maybe_offload_oversized_message(agent, big)
    maybe_offload_oversized_message(agent, big)

    paste_dir = hermes_home / "pastes"
    spills = list(paste_dir.glob("paste_*.txt"))
    assert len(spills) == 1
    assert spills[0].read_text(encoding="utf-8") == big


# ── under-threshold paste -> untouched ────────────────────────────────
def test_under_threshold_paste_is_untouched(tmp_path, monkeypatch):
    hermes_home, _ = _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=50_000)
    small = "just a normal message\n" * 10  # well under 50k chars
    new_msg, new_persist, path = maybe_offload_oversized_message(
        agent, small, small
    )

    assert path is None
    assert new_msg == small
    assert new_persist == small
    # No file written at all.
    paste_dir = hermes_home / "pastes"
    assert not paste_dir.exists() or list(paste_dir.glob("paste_*.txt")) == []


def test_disabled_flag_skips_offload(tmp_path, monkeypatch):
    hermes_home, _ = _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(enabled=False, threshold=100)
    big = "X\n" * 100_000
    new_msg, _persist, path = maybe_offload_oversized_message(agent, big)

    assert path is None
    assert new_msg == big
    paste_dir = hermes_home / "pastes"
    assert not paste_dir.exists() or list(paste_dir.glob("paste_*.txt")) == []


def test_zero_threshold_disables_offload(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=0)
    big = "Y\n" * 100_000
    new_msg, _persist, path = maybe_offload_oversized_message(agent, big)
    assert path is None
    assert new_msg == big


def test_non_string_message_is_untouched(tmp_path, monkeypatch):
    """Multimodal (list) content is not a plain paste; leave it alone."""
    _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import maybe_offload_oversized_message

    agent = _make_agent(threshold=100)
    content = [{"type": "text", "text": "hi"}, {"type": "image", "x": "..."}]
    new_msg, _persist, path = maybe_offload_oversized_message(agent, content)
    assert path is None
    assert new_msg is content


def test_should_offload_predicate(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)
    from agent.oversized_paste import should_offload

    agent = _make_agent(threshold=1_000)
    assert should_offload(agent, "z" * 1_000) is True
    assert should_offload(agent, "z" * 999) is False
    assert should_offload(agent, "") is False
    assert should_offload(agent, None) is False


def test_write_failure_fails_soft(tmp_path, monkeypatch):
    """If the file write fails, the original message flows through unchanged."""
    _isolate_home(monkeypatch, tmp_path)
    import agent.oversized_paste as op

    agent = _make_agent(threshold=100)
    monkeypatch.setattr(op, "write_paste_file", lambda content: None)
    big = "Z\n" * 100_000
    new_msg, new_persist, path = op.maybe_offload_oversized_message(
        agent, big, big
    )
    assert path is None
    assert new_msg == big
    assert new_persist == big
