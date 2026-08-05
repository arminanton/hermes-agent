"""Tests for the per-subagent control helpers in ``tools.delegate_tool``.

Covers the /agents overlay control surface added for soft/hard kill, steer,
interrupt-with-message, and per-child pause/resume:

* soft-kill calls ``interrupt()`` with NO message (graceful stop);
* steer calls ``steer()`` only (child keeps running);
* interrupt-with-message does BOTH steer + interrupt (refocus, not stop);
* hard-kill fires interrupt then kills ONLY this child's task-scoped
  tool subprocesses (never a system/hermes/tmux process);
* every control is RUNNING-only (a completed child is never touched);
* pause/resume flip the cooperative ``_subagent_hold`` flag + record.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tools import delegate_tool as dt


class _FakeAgent:
    """Minimal stand-in for AIAgent exposing the control surface."""

    def __init__(self) -> None:
        self.interrupt_calls: list = []
        self.steer_calls: list = []
        self._subagent_hold = False

    def interrupt(self, message: str | None = None) -> None:
        self.interrupt_calls.append(message)

    def steer(self, text: str) -> bool:
        self.steer_calls.append(text)
        return True


def _register(sid: str, agent, status: str = "running") -> None:
    dt._register_subagent(
        {
            "subagent_id": sid,
            "parent_id": None,
            "depth": 0,
            "goal": "test goal",
            "model": "test-model",
            "session_id": f"sess-{sid}",
            "started_at": 0.0,
            "status": status,
            "tool_count": 0,
            "agent": agent,
        }
    )


class SubagentControlsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate the module-level registry for each test.
        with dt._active_subagents_lock:
            dt._active_subagents.clear()

    def tearDown(self) -> None:
        with dt._active_subagents_lock:
            dt._active_subagents.clear()

    # ── soft kill ────────────────────────────────────────────────────
    def test_soft_kill_interrupts_with_no_message(self) -> None:
        agent = _FakeAgent()
        _register("a1", agent)
        self.assertTrue(dt.soft_kill_subagent("a1"))
        # Graceful stop = interrupt() with NO message.
        self.assertEqual(agent.interrupt_calls, [None])
        self.assertEqual(agent.steer_calls, [])

    def test_soft_kill_missing_returns_false(self) -> None:
        self.assertFalse(dt.soft_kill_subagent("nope"))

    def test_soft_kill_skips_completed(self) -> None:
        agent = _FakeAgent()
        _register("done", agent, status="completed")
        self.assertFalse(dt.soft_kill_subagent("done"))
        self.assertEqual(agent.interrupt_calls, [])

    def test_soft_kill_all_only_running(self) -> None:
        run1, run2, done = _FakeAgent(), _FakeAgent(), _FakeAgent()
        _register("r1", run1)
        _register("r2", run2)
        _register("d1", done, status="completed")
        self.assertEqual(dt.soft_kill_all_subagents(), 2)
        self.assertEqual(run1.interrupt_calls, [None])
        self.assertEqual(run2.interrupt_calls, [None])
        self.assertEqual(done.interrupt_calls, [])

    # ── steer ────────────────────────────────────────────────────────
    def test_steer_calls_steer_only(self) -> None:
        agent = _FakeAgent()
        _register("s1", agent)
        self.assertTrue(dt.steer_subagent("s1", "focus on X"))
        self.assertEqual(agent.steer_calls, ["focus on X"])
        self.assertEqual(agent.interrupt_calls, [])  # does NOT stop the child

    def test_steer_empty_text_rejected(self) -> None:
        agent = _FakeAgent()
        _register("s1", agent)
        self.assertFalse(dt.steer_subagent("s1", "   "))
        self.assertEqual(agent.steer_calls, [])

    def test_steer_skips_completed(self) -> None:
        agent = _FakeAgent()
        _register("done", agent, status="completed")
        self.assertFalse(dt.steer_subagent("done", "hi"))

    # ── interrupt with message ───────────────────────────────────────
    def test_interrupt_message_does_steer_then_interrupt(self) -> None:
        agent = _FakeAgent()
        _register("i1", agent)
        self.assertTrue(
            dt.interrupt_subagent_with_message("i1", "switch tasks")
        )
        # Steer stashes the message; interrupt(message) aborts the in-flight
        # tool so the child refocuses and CONTINUES.
        self.assertEqual(agent.steer_calls, ["switch tasks"])
        self.assertEqual(agent.interrupt_calls, ["switch tasks"])

    def test_interrupt_message_empty_rejected(self) -> None:
        agent = _FakeAgent()
        _register("i1", agent)
        self.assertFalse(dt.interrupt_subagent_with_message("i1", ""))
        self.assertEqual(agent.interrupt_calls, [])

    # ── hard kill (scoped subprocess termination) ────────────────────
    def test_hard_kill_interrupts_then_kills_scoped_procs(self) -> None:
        agent = _FakeAgent()
        _register("h1", agent)

        # A tracked process owned by THIS child (task_id == subagent_id).
        proc = MagicMock()
        proc.id = "proc-h1"
        proc.exited = False
        fake_registry = MagicMock()
        fake_registry.list_sessions.return_value = [proc]
        fake_registry.kill_process.return_value = {"status": "killed"}

        with patch.dict(
            "sys.modules",
            {"tools.process_registry": MagicMock(process_registry=fake_registry)},
        ):
            res = dt.hard_kill_subagent("h1")

        self.assertTrue(res["found"])
        self.assertTrue(res["interrupted"])
        self.assertEqual(res["procs_killed"], 1)
        # interrupt fired once as just-in-case (no wait).
        self.assertEqual(agent.interrupt_calls, [None])
        # CRITICAL: only the child's own task_id was targeted — never a
        # broad/system scan that could catch hermes or tmux.
        fake_registry.list_sessions.assert_called_once_with(task_id="h1")
        fake_registry.kill_process.assert_called_once_with(
            "proc-h1", source="subagent.hard_kill"
        )

    def test_hard_kill_skips_already_exited_procs(self) -> None:
        agent = _FakeAgent()
        _register("h2", agent)
        dead = MagicMock()
        dead.id = "dead"
        dead.exited = True
        fake_registry = MagicMock()
        fake_registry.list_sessions.return_value = [dead]

        with patch.dict(
            "sys.modules",
            {"tools.process_registry": MagicMock(process_registry=fake_registry)},
        ):
            res = dt.hard_kill_subagent("h2")

        self.assertEqual(res["procs_killed"], 0)
        fake_registry.kill_process.assert_not_called()

    def test_hard_kill_missing_returns_not_found(self) -> None:
        res = dt.hard_kill_subagent("nope")
        self.assertFalse(res["found"])
        self.assertEqual(res["procs_killed"], 0)

    def test_hard_kill_all_aggregates(self) -> None:
        _register("r1", _FakeAgent())
        _register("r2", _FakeAgent())
        _register("d1", _FakeAgent(), status="completed")
        fake_registry = MagicMock()
        fake_registry.list_sessions.return_value = []
        with patch.dict(
            "sys.modules",
            {"tools.process_registry": MagicMock(process_registry=fake_registry)},
        ):
            res = dt.hard_kill_all_subagents()
        self.assertEqual(res["count"], 2)  # only the 2 running ones

    # ── pause / resume ───────────────────────────────────────────────
    def test_pause_sets_hold_flag_and_record(self) -> None:
        agent = _FakeAgent()
        _register("p1", agent)
        self.assertTrue(dt.pause_subagent("p1"))
        self.assertTrue(agent._subagent_hold)
        with dt._active_subagents_lock:
            self.assertTrue(dt._active_subagents["p1"]["paused"])

    def test_resume_clears_hold_flag(self) -> None:
        agent = _FakeAgent()
        _register("p1", agent)
        dt.pause_subagent("p1")
        self.assertTrue(dt.resume_subagent("p1"))
        self.assertFalse(agent._subagent_hold)
        with dt._active_subagents_lock:
            self.assertFalse(dt._active_subagents["p1"]["paused"])

    def test_pause_skips_completed(self) -> None:
        agent = _FakeAgent()
        _register("done", agent, status="completed")
        self.assertFalse(dt.pause_subagent("done"))
        self.assertFalse(agent._subagent_hold)

    # ── registry snapshot exposes session_id + strips the agent ──────
    def test_list_active_exposes_session_id_no_agent_handle(self) -> None:
        _register("x1", _FakeAgent())
        snap = dt.list_active_subagents()
        self.assertEqual(len(snap), 1)
        row = snap[0]
        self.assertEqual(row["session_id"], "sess-x1")
        self.assertNotIn("agent", row)  # never leak the live handle


if __name__ == "__main__":
    unittest.main()
