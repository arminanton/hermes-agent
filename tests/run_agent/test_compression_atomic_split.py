"""Atomic SQLite transition for compression parent and child sessions."""

from __future__ import annotations

import multiprocessing
import os
import signal
from pathlib import Path

import pytest

from hermes_state import SessionDB


PARENT = "compression-parent"
CHILD = "compression-child"


def _compressed_messages() -> list[dict]:
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {
            "role": "assistant",
            "content": "I will inspect the browser.",
            "tool_calls": [
                {
                    "id": "call-browser",
                    "type": "function",
                    "function": {
                        "name": "browser_snapshot",
                        "arguments": "{}",
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]


def _seed_parent(db: SessionDB) -> None:
    db.create_session(PARENT, source="tui")
    db.append_message(PARENT, role="user", content="durable parent tail")


def _split(db: SessionDB, child_id: str = CHILD) -> None:
    db.split_session_for_compression(
        parent_session_id=PARENT,
        child_session_id=child_id,
        source="tui",
        messages=_compressed_messages(),
        model="test/model",
        model_config={"reasoning_config": {"enabled": False}},
        system_prompt="compressed system prompt",
    )


def _kill_inside_split(db_path: str) -> None:
    db = SessionDB(db_path=Path(db_path))

    def kill_process() -> int:
        os.kill(os.getpid(), signal.SIGKILL)
        return 0

    assert db._conn is not None
    db._conn.create_function("kill_process", 0, kill_process)
    db._conn.execute(
        """
        CREATE TRIGGER kill_compression_child_message
        BEFORE INSERT ON messages
        WHEN NEW.session_id = 'compression-child-kill'
          AND EXISTS (
              SELECT 1 FROM sessions
              WHERE id = 'compression-parent'
                AND end_reason = 'compression'
                AND ended_at IS NOT NULL
          )
          AND EXISTS (
              SELECT 1 FROM sessions
              WHERE id = 'compression-child-kill'
                AND parent_session_id = 'compression-parent'
          )
        BEGIN
            SELECT kill_process();
        END
        """
    )
    db._conn.commit()
    _split(db, "compression-child-kill")
    raise AssertionError("split returned instead of killing the process")


def _kill_on_commit(db_path: str) -> None:
    db = SessionDB(db_path=Path(db_path))

    def trace(statement: str) -> None:
        if statement.strip().upper() == "COMMIT":
            os.kill(os.getpid(), signal.SIGKILL)

    assert db._conn is not None
    db._conn.set_trace_callback(trace)
    _split(db, "compression-child-commit-kill")
    raise AssertionError("split returned instead of killing on COMMIT")


def _kill_during_wal_checkpoint(db_path: str) -> None:
    db = SessionDB(db_path=Path(db_path))

    def trace(statement: str) -> None:
        if statement.strip().upper().startswith("PRAGMA WAL_CHECKPOINT"):
            os.kill(os.getpid(), signal.SIGKILL)

    assert db._conn is not None
    db._conn.set_trace_callback(trace)
    setattr(db, "_CHECKPOINT_EVERY_N_WRITES", 1)
    _split(db, "compression-child-checkpoint-kill")
    raise AssertionError("split returned instead of killing during checkpoint")


def test_session_db_uses_crash_durable_sqlite_pragmas(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._conn is not None
        journal_mode = str(db._conn.execute("PRAGMA journal_mode").fetchone()[0])
        synchronous = int(db._conn.execute("PRAGMA synchronous").fetchone()[0])
        assert journal_mode.lower() in {"wal", "delete"}
        assert synchronous >= 1  # NORMAL=1, FULL=2, EXTRA=3; OFF=0 is unsafe.
    finally:
        db.close()


def test_atomic_compression_split_commits_complete_child(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_parent(db)
        _split(db)

        parent = db.get_session(PARENT)
        child = db.get_session(CHILD)
        assert parent is not None
        assert child is not None
        assert parent["end_reason"] == "compression"
        assert child["parent_session_id"] == PARENT
        assert child["system_prompt"] == "compressed system prompt"
        assert [row["content"] for row in db.get_messages(CHILD)] == [
            "[CONTEXT COMPACTION] summary",
            "I will inspect the browser.",
        ]
        assert child["message_count"] == 2
        assert child["tool_call_count"] == 1
    finally:
        db.close()


def test_atomic_compression_split_rolls_back_on_insert_failure(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_parent(db)
        assert db._conn is not None
        db._conn.execute(
            """
            CREATE TRIGGER fail_compression_child_message
            BEFORE INSERT ON messages
            WHEN NEW.session_id = 'compression-child'
            BEGIN
                SELECT RAISE(ABORT, 'injected child message failure');
            END
            """
        )
        db._conn.commit()

        with pytest.raises(Exception, match="injected child message failure"):
            _split(db)

        parent = db.get_session(PARENT)
        assert parent is not None
        assert parent["ended_at"] is None
        assert parent["end_reason"] is None
        assert db.get_session(CHILD) is None
        assert [row["content"] for row in db.get_messages(PARENT)] == [
            "durable parent tail"
        ]
    finally:
        db.close()


def test_atomic_compression_split_survives_sigkill_inside_transaction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed_parent(db)
    db.close()

    process = multiprocessing.get_context("spawn").Process(
        target=_kill_inside_split,
        args=(str(db_path),),
    )
    process.start()
    process.join(timeout=15)

    assert not process.is_alive()
    assert process.exitcode == -signal.SIGKILL

    recovered = SessionDB(db_path=db_path)
    try:
        parent = recovered.get_session(PARENT)
        assert parent is not None
        assert parent["ended_at"] is None
        assert parent["end_reason"] is None
        assert recovered.get_session("compression-child-kill") is None
        assert [row["content"] for row in recovered.get_messages(PARENT)] == [
            "durable parent tail"
        ]
        assert recovered.resolve_resume_session_id(PARENT) == PARENT
    finally:
        recovered.close()


def test_atomic_compression_split_is_coherent_when_killed_on_commit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed_parent(db)
    db.close()

    process = multiprocessing.get_context("spawn").Process(
        target=_kill_on_commit,
        args=(str(db_path),),
    )
    process.start()
    process.join(timeout=15)

    assert not process.is_alive()
    assert process.exitcode == -signal.SIGKILL

    recovered = SessionDB(db_path=db_path)
    try:
        parent = recovered.get_session(PARENT)
        child = recovered.get_session("compression-child-commit-kill")
        assert parent is not None
        if child is None:
            assert parent["ended_at"] is None
            assert parent["end_reason"] is None
        else:
            assert parent["end_reason"] == "compression"
            assert child["message_count"] == 2
            assert len(recovered.get_messages(child["id"])) == 2
        assert recovered.resolve_resume_session_id(PARENT) == PARENT
        assert [row["content"] for row in recovered.get_messages(PARENT)] == [
            "durable parent tail"
        ]
    finally:
        recovered.close()


def test_atomic_compression_child_survives_kill_during_wal_checkpoint(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed_parent(db)
    db.close()

    process = multiprocessing.get_context("spawn").Process(
        target=_kill_during_wal_checkpoint,
        args=(str(db_path),),
    )
    process.start()
    process.join(timeout=15)

    assert not process.is_alive()
    assert process.exitcode == -signal.SIGKILL

    recovered = SessionDB(db_path=db_path)
    try:
        parent = recovered.get_session(PARENT)
        child = recovered.get_session("compression-child-checkpoint-kill")
        assert parent is not None
        assert child is not None
        assert parent["end_reason"] == "compression"
        assert child["message_count"] == 2
        assert len(recovered.get_messages(child["id"])) == 2
    finally:
        recovered.close()
