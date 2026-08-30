"""Compression split resilience to corrupt session end stamps.

Regression coverage for the "compression retry loop" failure: a session row
stamped ``ended_at = started_at + 3600`` with ``end_reason = NULL`` (a synthetic
finalizer stamp, not produced by any legitimate ender) made
``split_session_for_compression`` abort on its ``ended_at IS NULL`` guard. The
auto-compress loop then retried the same no-op up to ``max_attempts`` times,
each attempt doing ~85s of work before failing, producing a ~15-20 minute hang
that ended in "Context length exceeded: max compression attempts reached".

These tests pin three behaviors:

* a corrupt-stamped parent (ended_at set, end_reason NULL) is repaired
  in-transaction and the split proceeds (the fix);
* a legitimately-ended parent (end_reason set) still aborts the split with no
  orphan child (the anti-orphan guard the atomic split was built for);
* ``repair_corrupt_end_stamps`` neutralizes corrupt rows idempotently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB


PARENT = "corrupt-parent"
CHILD = "rescued-child"


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "continuing"},
    ]


def _corrupt_stamp(db: SessionDB, session_id: str) -> None:
    """Reproduce the wild corruption: ended_at = started_at + 3600, reason NULL.

    Writes through the same ``_execute_write`` path a real writer would use,
    bypassing ``end_session`` (which always sets a reason) so the row lands in
    the exact corrupt state observed in production.
    """
    started_at = db._conn.execute(
        "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()[0]
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (started_at + 3600.0, session_id),
        )
    )


def _split(db: SessionDB, parent: str, child: str) -> None:
    db.split_session_for_compression(
        parent_session_id=parent,
        child_session_id=child,
        source="subagent",
        messages=_messages(),
        model="test/model",
        system_prompt="compressed system prompt",
    )


def test_split_rescues_parent_with_corrupt_end_stamp(tmp_path: Path) -> None:
    """A live parent carrying a bogus ended_at + NULL reason is repaired, not aborted."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(PARENT, source="subagent")
        db.append_message(PARENT, role="user", content="parent tail")
        _corrupt_stamp(db, PARENT)

        # Precondition: the row is in the exact corrupt state.
        parent = db.get_session(PARENT)
        assert parent["ended_at"] is not None
        assert parent["end_reason"] is None

        # The split must NOT raise; it repairs the stamp and rotates.
        _split(db, PARENT, CHILD)

        parent = db.get_session(PARENT)
        child = db.get_session(CHILD)
        assert parent["end_reason"] == "compression"
        assert parent["ended_at"] is not None
        assert child is not None
        assert child["parent_session_id"] == PARENT
        assert child["message_count"] == 2
    finally:
        db.close()


def test_split_still_aborts_on_legitimately_ended_parent(tmp_path: Path) -> None:
    """A parent ended with a real reason keeps the anti-orphan guard: split aborts."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(PARENT, source="cli")
        db.append_message(PARENT, role="user", content="parent tail")
        # Legitimate end — sets a real reason (mimics a concurrent compression
        # path that already committed, /new, session reset, etc.).
        db.end_session(PARENT, "new_session")

        with pytest.raises(RuntimeError):
            _split(db, PARENT, "should-not-exist")

        # No orphan child may be created.
        assert db.get_session("should-not-exist") is None
        # Parent's original end reason is preserved (not overwritten).
        assert db.get_session(PARENT)["end_reason"] == "new_session"
    finally:
        db.close()


def test_split_aborts_on_missing_parent(tmp_path: Path) -> None:
    """A genuinely missing parent is still fatal (no orphan child)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with pytest.raises(RuntimeError):
            _split(db, "no-such-parent", "should-not-exist")
        assert db.get_session("should-not-exist") is None
    finally:
        db.close()


def test_repair_corrupt_end_stamps_neutralizes_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """The startup reconciler retags corrupt rows and never touches clean ones."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # Two corrupt rows.
        db.create_session("corrupt-a", source="subagent")
        db.create_session("corrupt-b", source="cli")
        _corrupt_stamp(db, "corrupt-a")
        _corrupt_stamp(db, "corrupt-b")
        # One legitimately-ended row (must be left alone).
        db.create_session("legit", source="cli")
        db.end_session("legit", "cli_close")
        # One live row (ended_at NULL — must be left alone).
        db.create_session("live", source="tui")

        repaired = db.repair_corrupt_end_stamps()
        assert repaired == 2

        assert db.get_session("corrupt-a")["end_reason"] == "stale_stamp_repaired"
        assert db.get_session("corrupt-b")["end_reason"] == "stale_stamp_repaired"
        assert db.get_session("legit")["end_reason"] == "cli_close"
        assert db.get_session("live")["ended_at"] is None
        assert db.get_session("live")["end_reason"] is None

        # No corrupt rows remain, and re-running is a no-op.
        remaining = db._conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE ended_at IS NOT NULL AND end_reason IS NULL"
        ).fetchone()[0]
        assert remaining == 0
        assert db.repair_corrupt_end_stamps() == 0
    finally:
        db.close()


def test_repaired_parent_can_still_be_compressed(tmp_path: Path) -> None:
    """After reconciler repair, a formerly-corrupt live session can compress.

    The reconciler retags a corrupt stamp to 'stale_stamp_repaired'. If that
    same session is still live and later needs compression, the split's
    anti-orphan guard now treats it as legitimately ended (real reason), so the
    caller reopens it first. This asserts the reconciler + reopen path lets a
    real continuation proceed rather than dead-ending.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(PARENT, source="subagent")
        db.append_message(PARENT, role="user", content="tail")
        _corrupt_stamp(db, PARENT)
        db.repair_corrupt_end_stamps()
        # Now reason is 'stale_stamp_repaired' (a real reason) — a live session
        # that wants to compress must reopen first, as the caller does.
        db.reopen_session(PARENT)
        assert db.get_session(PARENT)["ended_at"] is None

        _split(db, PARENT, CHILD)
        assert db.get_session(CHILD) is not None
        assert db.get_session(PARENT)["end_reason"] == "compression"
    finally:
        db.close()
