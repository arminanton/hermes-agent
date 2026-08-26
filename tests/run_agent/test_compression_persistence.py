"""Tests for context compression persistence in the gateway.

Verifies that when context compression fires during run_conversation(),
the compressed messages are properly persisted to both SQLite (via the
agent) and JSONL (via the gateway).

Bug scenario (pre-fix):
  1. Gateway loads 200-message history, passes to agent
  2. Agent's run_conversation() compresses to ~30 messages mid-run
  3. _compress_context() resets _last_flushed_db_idx = 0
  4. On exit, _flush_messages_to_session_db() calculates:
     flush_from = max(len(conversation_history=200), _last_flushed_db_idx=0) = 200
  5. messages[200:] is empty (only ~30 messages after compression)
  6. Nothing written to new session's SQLite — compressed context lost
  7. Gateway's history_offset was still 200, producing empty new_messages
  8. Fallback wrote only user/assistant pair — summary lost
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch



# ---------------------------------------------------------------------------
# Part 1: Agent-side — _flush_messages_to_session_db after compression
# ---------------------------------------------------------------------------

class TestFlushAfterCompression:
    """Verify that compressed messages are flushed to the new session's SQLite
    even when conversation_history (from the original session) is longer than
    the compressed messages list."""

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    @staticmethod
    def _install_compressor(agent, compressed):
        compressor = MagicMock()
        compressor.compress.return_value = compressed
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        agent._compression_feasibility_checked = True

    def test_compression_publishes_and_persists_only_the_compressed_child(self):
        """Finalize after rotation must not copy the full parent into the child."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db)
                db.create_session("original-session", source="tui")
                agent._session_db_created = True
                original = [
                    {"role": "user", "content": f"old-{i}"}
                    for i in range(8)
                ]
                assert agent.checkpoint_session_messages(original) is True

                compacted = [
                    {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                    {"role": "assistant", "content": "recent answer"},
                ]
                self._install_compressor(agent, compacted)
                real_split = db.split_session_for_compression
                published_ids = []

                def observe_split(**kwargs):
                    published_ids.append(getattr(agent, "session_id"))
                    return real_split(**kwargs)

                setattr(db, "split_session_for_compression", observe_split)

                result, _ = agent._compress_context(
                    original, "system", approx_tokens=120_000
                )
                child_id = getattr(agent, "session_id")

                assert published_ids == ["original-session"]
                assert result == compacted
                assert agent._session_messages == compacted
                assert [row["content"] for row in db.get_messages(child_id)] == [
                    "[CONTEXT COMPACTION] summary",
                    "recent answer",
                ]

                agent.flush_pending_to_db()
                assert [row["content"] for row in db.get_messages(child_id)] == [
                    "[CONTEXT COMPACTION] summary",
                    "recent answer",
                ]

            finally:
                db.close()

    def test_compression_does_not_rotate_when_parent_checkpoint_fails(self):
        """The old tail must be durable before its session is ended."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db)
                db.create_session("original-session", source="tui")
                agent._session_db_created = True
                original = [{"role": "user", "content": "must survive"}]
                compacted = [
                    {"role": "user", "content": "[CONTEXT COMPACTION] summary"}
                ]
                self._install_compressor(agent, compacted)
                agent.checkpoint_session_messages = MagicMock(return_value=False)

                result, _ = agent._compress_context(
                    original, "system", approx_tokens=120_000
                )

                assert result is original
                assert getattr(agent, "session_id") == "original-session"
                parent_row = db.get_session("original-session")
                assert parent_row is not None
                assert parent_row["ended_at"] is None
                conn = db._conn
                assert conn is not None
                children = conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ?",
                    ("original-session",),
                ).fetchall()
                assert children == []
            finally:
                db.close()

    def test_atomic_split_failure_keeps_parent_available_for_cold_resume(self):
        """An atomic split error must leave the durable parent resumable."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            try:
                agent = self._make_agent(db)
                db.create_session("original-session", source="tui")
                agent._session_db_created = True
                original = [
                    {"role": "user", "content": "inspect the stale browser"},
                    {
                        "role": "assistant",
                        "content": "I will inspect it now.",
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
                    },
                ]
                compacted = [
                    {"role": "user", "content": "[CONTEXT COMPACTION] summary"}
                ]
                self._install_compressor(agent, compacted)
                db.split_session_for_compression = MagicMock(
                    side_effect=RuntimeError("injected atomic split failure")
                )

                result, _ = agent._compress_context(
                    original, "system", approx_tokens=120_000
                )

                assert result is original
                assert getattr(agent, "session_id") == "original-session"
                assert agent._session_messages is original
                parent_row = db.get_session("original-session")
                assert parent_row is not None
                assert parent_row["ended_at"] is None
                conn = db._conn
                assert conn is not None
                children = conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ?",
                    ("original-session",),
                ).fetchall()
                assert children == []

                resumed_id = db.resolve_resume_session_id("original-session")
                assert resumed_id == "original-session"
                resumed = db.get_messages_as_conversation(resumed_id)
                assert "inspect the stale browser" in str(resumed)
            finally:
                db.close()

    def test_flush_after_compression_with_long_history(self):
        """The actual bug: conversation_history longer than compressed messages.

        Before the fix, flush_from = max(len(conversation_history), 0) = 200,
        but messages only has ~30 entries, so messages[200:] is empty.
        After the fix, conversation_history is cleared to None after compression,
        so flush_from = max(0, 0) = 0, and ALL compressed messages are written.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate the original long history (200 messages)
            original_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(200)
            ]

            # First, flush original messages to the original session
            agent._flush_messages_to_session_db(original_history, [])
            original_rows = db.get_messages("original-session")
            assert len(original_rows) == 200

            # Now simulate compression: new session, reset idx, shorter messages
            agent.session_id = "compressed-session"
            db.create_session(session_id="compressed-session", source="test")
            agent._last_flushed_db_idx = 0

            # The compressed messages (summary + tail + new turn)
            compressed_messages = [
                {"role": "user", "content": "[CONTEXT COMPACTION] Summary of work..."},
                {"role": "user", "content": "What should we do next?"},
                {"role": "assistant", "content": "Let me check..."},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            # THE BUG: passing the original history as conversation_history
            # causes flush_from = max(200, 0) = 200, skipping everything.
            # After the fix, conversation_history should be None.
            agent._flush_messages_to_session_db(compressed_messages, None)

            new_rows = db.get_messages("compressed-session")
            assert len(new_rows) == 5, (
                f"Expected 5 compressed messages in new session, got {len(new_rows)}. "
                f"Compression persistence bug: messages not written to SQLite."
            )

    def test_flush_with_stale_history_loses_messages(self):
        """Stale conversation_history no longer causes data loss."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate compression reset
            agent.session_id = "new-session"
            db.create_session(session_id="new-session", source="test")
            agent._last_flushed_db_idx = 0

            compressed = [
                {"role": "user", "content": "summary"},
                {"role": "assistant", "content": "continuing..."},
            ]

            # Stale history longer than messages: the old positional flush
            # sliced past the end and dropped both messages (#46053).
            stale_history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
            agent._flush_messages_to_session_db(compressed, stale_history)

            rows = db.get_messages("new-session")
            assert len(rows) == 2
            assert [row["content"] for row in rows] == ["summary", "continuing..."]


# ---------------------------------------------------------------------------
# Part 2: Gateway-side — history_offset after session split
# ---------------------------------------------------------------------------

class TestGatewayHistoryOffsetAfterSplit:
    """Verify that when the agent creates a new session during compression,
    the gateway uses history_offset=0 so all compressed messages are written
    to the JSONL transcript."""

    def test_history_offset_zero_on_session_split(self):
        """When agent.session_id differs from the original, history_offset must be 0."""
        # This tests the logic in gateway/run.py run_sync():
        # _session_was_split = agent.session_id != session_id
        # _effective_history_offset = 0 if _session_was_split else len(agent_history)

        original_session_id = "session-abc"
        agent_session_id = "session-compressed-xyz"  # Different = compression happened
        agent_history_len = 200

        # Simulate the gateway's offset calculation (post-fix)
        _session_was_split = (agent_session_id != original_session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is True
        assert _effective_history_offset == 0

    def test_history_offset_preserved_without_split(self):
        """When no compression happened, history_offset is the original length."""
        session_id = "session-abc"
        agent_session_id = "session-abc"  # Same = no compression
        agent_history_len = 200

        _session_was_split = (agent_session_id != session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is False
        assert _effective_history_offset == 200

    def test_new_messages_extraction_after_split(self):
        """After compression with offset=0, new_messages should be ALL agent messages."""
        # Simulates the gateway's new_messages calculation
        agent_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] Summary..."},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]
        history_offset = 0  # After fix: 0 on session split

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 5, (
            f"Expected all 5 messages with offset=0, got {len(new_messages)}"
        )

    def test_new_messages_empty_with_stale_offset(self):
        """Demonstrates the bug: stale offset produces empty new_messages."""
        agent_messages = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "answer"},
        ]
        # Bug: offset is the pre-compression history length
        history_offset = 200

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 0, (
            "Expected 0 messages with stale offset=200 (demonstrates the bug)"
        )
