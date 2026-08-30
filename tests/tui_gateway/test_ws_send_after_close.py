"""Regression tests for WSTransport send-after-close.

Observed in production (2026-08-28): when three long-lived TUI clients went
away at once, the gateway logged, for one peer, three cascading warnings for a
single dead socket:

    ws send failed peer=127.0.0.1:48192 error_type=WebSocketDisconnect
    ws response send failed peer=127.0.0.1:48192 id=r1475 method=session.active_list
    ws send failed peer=127.0.0.1:48192 error_type=RuntimeError
        error=Cannot call "send" once a close message has been sent.

The RuntimeError is the bug. ``write()`` checks ``self._closed`` and then
SCHEDULES ``_safe_send`` on the event loop; by the time that coroutine runs, a
sibling frame (or an explicit ``close()``) may already have latched the
transport dead. ``_safe_send`` re-sends anyway, starlette raises, and the
handler logs a second, misleading failure for a disconnect that was already
recorded.

These tests pin the two guarantees of the fix:
  1. A frame scheduled while open but executed after the latch does not touch
     the socket at all.
  2. Concurrent in-flight frames on a dying socket produce exactly one warning,
     not one per frame.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from tui_gateway.ws import WSTransport


class _FakeWS:
    """Minimal websocket double that models starlette's post-close behaviour."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._fail_with = fail_with

    async def send_text(self, line: str) -> None:
        if self.closed:
            # Exactly what starlette raises once a close frame has gone out.
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        if self._fail_with is not None:
            raise self._fail_with
        self.sent.append(line)


@pytest.mark.asyncio
async def test_frame_scheduled_before_close_does_not_send_after_latch():
    """A queued frame must not hit the socket once the transport is latched."""
    ws = _FakeWS()
    transport = WSTransport(ws, asyncio.get_running_loop(), peer="test:1")

    # Frame is accepted while the transport is open...
    assert transport.write({"jsonrpc": "2.0", "id": 1}) is True

    # ...but the peer goes away (or a sibling frame failed) before the loop
    # gets to run the scheduled coroutine.
    transport.close()
    ws.closed = True

    # Let the fire-and-forget task run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert ws.sent == [], "send was attempted on an already-closed transport"


@pytest.mark.asyncio
async def test_concurrent_failing_frames_log_one_warning(caplog):
    """One disconnect must produce one warning, not one per in-flight frame."""
    ws = _FakeWS(fail_with=ConnectionResetError("peer went away"))
    transport = WSTransport(ws, asyncio.get_running_loop(), peer="test:2")

    with caplog.at_level(logging.WARNING, logger="tui_gateway.ws"):
        # Three frames racing onto a socket that is about to die — the
        # session.active_list poll plus its response is exactly this shape.
        await asyncio.gather(
            transport._safe_send("a"),
            transport._safe_send("b"),
            transport._safe_send("c"),
        )

    failures = [r for r in caplog.records if "ws send failed" in r.getMessage()]
    assert len(failures) == 1, (
        f"expected a single disconnect warning, got {len(failures)}: "
        f"{[r.getMessage() for r in failures]}"
    )
    assert transport._closed is True


@pytest.mark.asyncio
async def test_healthy_send_still_delivers():
    """The guard must not suppress normal traffic."""
    ws = _FakeWS()
    transport = WSTransport(ws, asyncio.get_running_loop(), peer="test:3")

    assert await transport.write_async({"jsonrpc": "2.0", "id": 7}) is True
    assert len(ws.sent) == 1
    assert '"id": 7' in ws.sent[0]
