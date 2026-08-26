"""Quota-free process-restart coverage for TUI transcript durability.

The test kills a real AIAgent process after its assistant tool-call checkpoint,
starts a new WebSocket gateway process, and cold-resumes the persisted session.
It does not model a renderer reconnect to a still-live gateway, and it does not
model the future fresh-agent conversation cutover.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "inflight-restart-fixture"
COMMENTARY = "Starting the deterministic fixture operation."
TOOL_NAME = "fixture_operation"


def _tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Run a deterministic fixture operation.",
                "parameters": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            },
        }
    ]


def _tool_response():
    from types import SimpleNamespace

    tool_call = SimpleNamespace(
        id="call_fixture_operation",
        type="function",
        function=SimpleNamespace(
            name=TOOL_NAME,
            arguments='{"label":"restart-fixture"}',
        ),
    )
    message = SimpleNamespace(content=COMMENTARY, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _child(home: Path, marker: Path) -> int:
    from hermes_state import SessionDB
    import run_agent

    db = SessionDB(db_path=home / "state.db")
    with patch.object(run_agent, "get_tool_definitions", lambda **_kwargs: _tool_schema()), patch.object(
        run_agent, "check_toolset_requirements", lambda: {}
    ):
        agent = run_agent.AIAgent(
            model="test/model",
            provider="custom",
            api_mode="chat_completions",
            base_url="http://127.0.0.1:9/v1",
            api_key="fixture-token",
            quiet_mode=True,
            max_iterations=4,
            session_db=db,
            session_id=SESSION_ID,
            skip_context_files=True,
            skip_memory=True,
        )

    def _no_cleanup(task_id: str) -> None:
        del task_id

    def _fake_api_call(api_kwargs: dict, on_first_delta=None):
        del api_kwargs
        if on_first_delta is not None:
            on_first_delta()
        return _tool_response()

    def _block_in_tool(
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        del assistant_message, messages, effective_task_id, api_call_count
        marker.write_text("tool-started", encoding="utf-8")
        while True:
            time.sleep(1)

    setattr(agent, "_cleanup_task_resources", _no_cleanup)
    setattr(agent, "_save_trajectory", lambda *_args, **_kwargs: None)
    setattr(agent, "_interruptible_api_call", _fake_api_call)
    setattr(agent, "_interruptible_streaming_api_call", _fake_api_call)
    setattr(agent, "_execute_tool_calls", _block_in_tool)
    agent.run_conversation("Run the deterministic fixture operation.")
    return 3


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_marker(child: subprocess.Popen, marker: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.exists():
        if child.poll() is not None:
            output = child.stdout.read().decode(errors="replace") if child.stdout else ""
            raise RuntimeError(f"agent child exited before tool start: {output}")
        time.sleep(0.05)
    if not marker.exists():
        raise TimeoutError("agent never entered the fixture tool")


def _wait_listening(proc: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"ws_host exited during startup: {output}")
        with socket.socket() as sock:
            sock.settimeout(0.2)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.05)
    raise TimeoutError("ws_host did not listen")


async def _resume(url: str) -> dict:
    import websockets

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session.resume",
                    "params": {"session_id": SESSION_ID, "cols": 120},
                }
            )
        )
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if message.get("id") == 1:
                return message


def test_cold_resume_surfaces_interrupted_tool_call(tmp_path):
    """A killed agent's unmatched call is visible and explicitly interrupted."""
    from hermes_state import SessionDB

    home = tmp_path / "hermes-home"
    home.mkdir()
    marker = home / "tool-started"
    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(home),
            "HERMES_IGNORE_RULES": "1",
            "OPENAI_API_KEY": "fixture-token",
            "OPENROUTER_API_KEY": "fixture-token",
            "PYTHONPATH": str(ROOT),
        }
    )
    child = subprocess.Popen(
        [sys.executable, __file__, "--child", str(home), str(marker)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    gateway = None
    try:
        _wait_for_marker(child, marker)
        child.kill()
        child.wait(timeout=5)

        db = SessionDB(db_path=home / "state.db")
        try:
            rows = db.get_messages(SESSION_ID)
        finally:
            db.close()
        assert any(row["role"] == "assistant" and row["tool_calls"] for row in rows)

        port = _free_port()
        credential = "restart-fixture-credential"
        ws_env = dict(env)
        ws_env.update(
            {
                "HERMES_TUI_WS_HOST": "127.0.0.1",
                "HERMES_TUI_WS_PORT": str(port),
                "HERMES_TUI_WS_INTERNAL_CREDENTIAL": credential,
            }
        )
        gateway = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.ws_host"],
            cwd=ROOT,
            env=ws_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_listening(gateway, port)
        url = f"ws://127.0.0.1:{port}/api/ws?internal={credential}"
        response = asyncio.run(_resume(url))
        result = response.get("result") or {}
        messages = result.get("messages") or []

        assert result.get("resumed") == SESSION_ID
        assert COMMENTARY in json.dumps(messages)
        assert any(
            message.get("role") == "tool"
            and message.get("name") == TOOL_NAME
            and message.get("status") == "interrupted"
            and "pending" not in message
            for message in messages
        )
    finally:
        if gateway is not None and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait(timeout=5)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    raise SystemExit(_child(Path(sys.argv[2]), Path(sys.argv[3])))
