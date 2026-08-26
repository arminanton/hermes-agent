"""Regression tests for flush_pending_to_db (durability net for reaped sessions).

Background: between the conversation loop's periodic ``_persist_session``
checkpoints, a long turn (deep delegation, many tool calls) accumulates
assistant/tool messages that live only on the in-process agent's
``_session_messages`` reference. If the in-memory session is torn down before
the turn finalizes cleanly (a transient stream/WS drop letting the gateway
finalize the session, or the standalone TUI's gateway child exiting on user
quit), those messages die with the agent and a later resume silently rewinds
to the last checkpoint.

``AIAgent.flush_pending_to_db()`` is the net: called from the gateway's
finalize chokepoint before ``agent.close()`` drops the reference, it flushes
whatever the agent is holding, reusing the identity-tracked
``_flush_messages_to_session_db`` so already-written rows are never duplicated.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

SESSION_ID = "test-flush-pending"


def _make_agent(session_db, session_id=SESSION_ID):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


def _contents(db, session_id=SESSION_ID):
    return [row["content"] for row in db.get_messages(session_id)]


class TestFlushPendingToDb:
    def test_flushes_inflight_tail_held_only_on_agent(self):
        """The un-checkpointed tail on _session_messages reaches the DB."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)

                # Simulate a turn that checkpointed the user+first-answer pair,
                # then kept working (delegation) accumulating more messages that
                # were never flushed — they live only on the agent.
                messages = [
                    {"role": "user", "content": "kick off the long task"},
                    {"role": "assistant", "content": "starting delegation"},
                    {"role": "assistant", "content": "delegation result A"},
                    {"role": "assistant", "content": "delegation result B"},
                ]
                agent._session_messages = messages

                assert agent.flush_pending_to_db() is True

                contents = _contents(db)
                assert "delegation result A" in contents
                assert "delegation result B" in contents
            finally:
                db.close()

    def test_returns_false_when_database_append_fails(self):
        """Callers must distinguish a durable checkpoint from a swallowed error."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._session_messages = [
                    {"role": "user", "content": "must remain retryable"}
                ]
                with patch.object(
                    db, "append_message", side_effect=RuntimeError("disk unavailable")
                ):
                    assert agent.flush_pending_to_db() is False
            finally:
                db.close()

    def test_checkpoint_preserves_operational_message_metadata(self):
        """Normal checkpoints retain the metadata carried by atomic splits."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._session_messages = [
                    {
                        "role": "assistant",
                        "content": "durable answer",
                        "token_count": 37,
                        "message_id": "provider-message-1",
                        "observed": True,
                    }
                ]

                assert agent.flush_pending_to_db() is True

                rows = db.get_messages(SESSION_ID)
                assert len(rows) == 1
                assert rows[0]["token_count"] == 37
                assert rows[0]["platform_message_id"] == "provider-message-1"
                assert bool(rows[0]["observed"]) is True
            finally:
                db.close()

    def test_explicit_platform_message_id_and_timestamp_survive_checkpoint(self):
        """Explicit platform identity wins over the alias without losing time."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._session_messages = [
                    {
                        "role": "user",
                        "content": "two identifiers",
                        "token_count": 17,
                        "platform_message_id": "explicit-id",
                        "message_id": "legacy-alias",
                        "observed": True,
                        "timestamp": 1_725_000_000.25,
                    }
                ]

                assert agent.flush_pending_to_db() is True
                row = db.get_messages(SESSION_ID)[0]
                assert row["token_count"] == 17
                assert row["platform_message_id"] == "explicit-id"
                assert bool(row["observed"]) is True
                assert row["timestamp"] == 1_725_000_000.25
            finally:
                db.close()

    def test_idempotent_no_duplicate_rows(self):
        """Re-flushing (multiple teardown paths) writes each message once."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                messages = [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
                agent._session_messages = messages

                agent.flush_pending_to_db()
                agent.flush_pending_to_db()
                agent.flush_pending_to_db()

                contents = _contents(db)
                assert contents.count("q1") == 1
                assert contents.count("a1") == 1
            finally:
                db.close()

    def test_does_not_duplicate_already_checkpointed_messages(self):
        """A normal _persist_session checkpoint then a finalize flush = no dupes."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                messages = [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
                # The loop's periodic checkpoint wrote these and set the live ref.
                agent._persist_session(messages)
                # Finalize-time flush must not re-append them.
                agent.flush_pending_to_db()

                contents = _contents(db)
                assert contents.count("hello") == 1
                assert contents.count("world") == 1
            finally:
                db.close()

    def test_persist_disabled_fork_never_writes(self):
        """A persistence-isolated fork (curator/memory review) must not write."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._persist_disabled = True
                agent._session_messages = [
                    {"role": "user", "content": "harness prompt"},
                    {"role": "assistant", "content": "curator output"},
                ]

                assert agent.flush_pending_to_db() is False
                assert "curator output" not in _contents(db)
            finally:
                db.close()

    def test_noop_when_no_messages_held(self):
        """Empty/absent in-flight list is a safe no-op returning False."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db)
                agent._session_messages = []
                assert agent.flush_pending_to_db() is False
            finally:
                db.close()
