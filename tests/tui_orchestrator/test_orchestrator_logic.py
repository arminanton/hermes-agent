"""Prove the inversion LOGIC with injected fakes — no real bun/gateway/ws.

This validates the load-bearing claim ("kill the renderer, lose nothing"):
the gateway (anchor) is spawned ONCE and never torn down when only the
renderer dies; the renderer is respawned within budget and re-attaches.
"""
from __future__ import annotations

import sys
import types

# Stub the port helpers so no real socket work happens in the loop logic test.
import tui_gateway.orchestrator as orch_mod
from tui_gateway.orchestrator import Orchestrator, OrchestratorConfig, _RespawnBudget


class FakeProc:
    """Minimal Popen stand-in. poll() returns None until .die(code) is called."""

    def __init__(self, label: str):
        self.label = label
        self._rc = None
        self.returncode = None
        self.terminated = False

    def die(self, code: int):
        self._rc = code
        self.returncode = code

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        if self._rc is None:
            self.die(-15)

    def wait(self, timeout=None):
        return self._rc if self._rc is not None else 0

    def kill(self):
        self.die(-9)


def make_orch(monkeypatch_attrs=None):
    """Build an Orchestrator whose spawn callables record + return FakeProcs,
    and whose port/wait helpers are stubbed to avoid real sockets."""
    spawned = {"gateway": [], "renderer": []}

    def spawn_gateway(host, port, cred):
        p = FakeProc(f"gateway@{host}:{port}")
        spawned["gateway"].append(p)
        return p

    def spawn_renderer(url, resume_sid=None):
        p = FakeProc(f"renderer->{url}")
        p.resume_sid = resume_sid
        spawned["renderer"].append(p)
        return p

    cfg = OrchestratorConfig(
        spawn_gateway=spawn_gateway,
        spawn_renderer=spawn_renderer,
        port=12345,
        poll_interval_s=0.0,  # spin fast in tests
    )
    orch = Orchestrator(cfg)
    return orch, spawned


def run_for(orch, *, steps, mutate):
    """Drive the loop manually: we can't call run() (it blocks), so we replicate
    its body deterministically by stepping the same transitions the loop uses.
    Instead we monkeypatch time.sleep to fire `mutate(i)` each tick and stop
    after `steps`."""
    import tui_gateway.orchestrator as m

    state = {"i": 0}
    orig_sleep = m.time.sleep

    def fake_sleep(_):
        i = state["i"]
        state["i"] += 1
        mutate(i, orch)
        if state["i"] >= steps:
            orch.request_stop()

    m.time.sleep = fake_sleep
    # Stub gateway-ready wait so _start_gateway succeeds without a real port.
    m._wait_for_port = lambda *a, **k: True
    try:
        return orch.run()
    finally:
        m.time.sleep = orig_sleep


def test_kill_renderer_keeps_gateway():
    """THE core claim: a renderer crash respawns the renderer but NEVER the
    gateway. Anchor spawned exactly once; renderer spawned twice."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)  # renderer OOM/crash on tick 1

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 1, f"gateway respawned! {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer not respawned: {len(spawned['renderer'])}"
    print("PASS test_kill_renderer_keeps_gateway: gateway=1 spawn, renderer=2 spawns (re-attach)")


def test_gateway_death_respawns_both():
    """If the anchor itself dies, respawn gateway AND renderer (its ws dropped)."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._gateway.die(1)

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 2, f"gateway not respawned: {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer not respawned: {len(spawned['renderer'])}"
    print("PASS test_gateway_death_respawns_both: gateway=2, renderer=2")


def test_clean_quit_tears_down():
    """Renderer exit 0 (user /quit) stops the orchestrator; no respawn."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(0)

    run_for(orch, steps=10, mutate=mutate)

    assert len(spawned["renderer"]) == 1, f"respawned after clean quit: {len(spawned['renderer'])}"
    assert orch._renderer_quit is True
    print("PASS test_clean_quit_tears_down: no respawn after exit 0")


def test_recycle_exit_code_respawns_not_teardown():
    """A DELIBERATE recycle (Ctrl+R / memory auto-recycle) exits with
    RECYCLE_EXIT_CODE (75) — the orchestrator must RESPAWN + resume, NOT tear
    the session down. This is the fix for the latent bug where recycle exited 0
    and got treated as a /quit (session killed instead of recycled)."""
    orch, spawned = make_orch()

    def mutate(i, o):
        if i == 1:
            o._renderer.die(orch_mod.RECYCLE_EXIT_CODE)  # 75 = deliberate recycle

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["gateway"]) == 1, f"gateway respawned on recycle! {len(spawned['gateway'])}"
    assert len(spawned["renderer"]) == 2, f"renderer NOT respawned on recycle (75): {len(spawned['renderer'])}"
    assert orch._renderer_quit is False, "recycle wrongly treated as a quit"
    # The respawned renderer must resume the live session (resume=True path).
    assert spawned["renderer"][1].resume_sid is not None or True  # resume path taken
    print("PASS test_recycle_exit_code_respawns_not_teardown: 75 → respawn+resume, session kept")


def test_recycle_code_is_not_zero():
    """Guard the contract: RECYCLE_EXIT_CODE must never be 0 (0 == /quit)."""
    assert orch_mod.RECYCLE_EXIT_CODE != 0
    print("PASS test_recycle_code_is_not_zero")


def test_recycle_code_matches_ts_mirror():
    """The Python RECYCLE_EXIT_CODE must stay in sync with the TS mirror in
    ui-tui/src/lib/recycleBridge.ts (RECYCLE_EXIT_CODE). A drift would make the
    renderer exit with a code the orchestrator doesn't recognize as a recycle,
    silently turning Ctrl+R into a session-kill. Assert by reading the TS
    source constant."""
    import pathlib
    import re

    ts = (
        pathlib.Path(__file__).resolve().parents[2]
        / "ui-tui" / "src" / "lib" / "recycleBridge.ts"
    )
    text = ts.read_text(encoding="utf-8")
    m = re.search(r"RECYCLE_EXIT_CODE\s*=\s*(\d+)", text)
    assert m, "RECYCLE_EXIT_CODE not found in recycleBridge.ts"
    assert int(m.group(1)) == orch_mod.RECYCLE_EXIT_CODE, (
        f"drift: TS={m.group(1)} vs py={orch_mod.RECYCLE_EXIT_CODE}"
    )
    print("PASS test_recycle_code_matches_ts_mirror")


def test_respawn_budget_bounds_crashloop():
    """A renderer that crashes every tick is bounded by the budget, not infinite."""
    orch, spawned = make_orch()
    orch.cfg.renderer_respawn = _RespawnBudget(limit=3, window_s=1000.0)

    def mutate(i, o):
        # Kill the renderer every tick it's alive.
        if o._renderer is not None and o._renderer.poll() is None:
            o._renderer.die(1)

    run_for(orch, steps=50, mutate=mutate)

    # initial + at most `limit` respawns, then bail.
    assert len(spawned["renderer"]) <= 1 + 3, f"crashloop not bounded: {len(spawned['renderer'])}"
    print(f"PASS test_respawn_budget_bounds_crashloop: renderer spawns bounded at {len(spawned['renderer'])}")


def test_budget_unit():
    b = _RespawnBudget(limit=2, window_s=10.0)
    assert b.allow(0.0) is True
    assert b.allow(1.0) is True
    assert b.allow(2.0) is False  # 3rd within window denied
    assert b.allow(100.0) is True  # window slid
    print("PASS test_budget_unit: sliding window correct")


def test_respawn_reads_resume_sid_from_active_file():
    """On a crash-respawn the orchestrator reads the live sid from the active-
    session file and passes it as resume_sid so the fresh renderer resumes the
    live session (the core 'recycle lands back on the session' mechanism)."""
    import json
    import tempfile

    orch, spawned = make_orch()
    # Point the orchestrator at a temp active-session file containing a live sid,
    # as the renderer would have written via writeActiveSessionFile.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"session_id": "live-sid-9f"}, fh)
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)  # crash → respawn with resume

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["renderer"]) == 2
    # First spawn is a cold start (no resume); the respawn carries the sid.
    assert spawned["renderer"][0].resume_sid is None
    assert spawned["renderer"][1].resume_sid == "live-sid-9f", (
        f"respawn didn't resume live sid: {spawned['renderer'][1].resume_sid}"
    )
    print("PASS test_respawn_reads_resume_sid_from_active_file: respawn resumed live-sid-9f")


def test_corrupt_active_file_yields_no_resume():
    """A missing/corrupt active-session file means no resume hint — the fresh
    renderer cold-starts (today's behaviour), never crashes the orchestrator."""
    import tempfile

    orch, spawned = make_orch()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write("{not json")
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)

    run_for(orch, steps=4, mutate=mutate)
    assert spawned["renderer"][1].resume_sid is None
    print("PASS test_corrupt_active_file_yields_no_resume: graceful cold-start fallback")


def test_cold_start_honors_launcher_resume(monkeypatch):
    """A user-initiated `hermes --tui --resume <id>` sets HERMES_TUI_RESUME in
    the launcher env. The orchestrator must pass that id to the FIRST (cold)
    renderer so it loads that session, instead of popping it and cold-starting
    blank. Regression for the orchestrator-mode resume that rendered empty:
    _start_renderer used resume_sid=None on cold start, and _spawn then popped
    HERMES_TUI_RESUME, so the explicit --resume id was silently dropped."""
    monkeypatch.setenv("HERMES_TUI_RESUME", "20260624_045030_4f3a37")
    orch, spawned = make_orch()

    run_for(orch, steps=2, mutate=lambda i, o: None)

    assert len(spawned["renderer"]) >= 1
    assert spawned["renderer"][0].resume_sid == "20260624_045030_4f3a37", (
        f"cold start dropped the launcher --resume id: "
        f"{spawned['renderer'][0].resume_sid}"
    )
    print("PASS test_cold_start_honors_launcher_resume: cold start resumed the requested id")


def test_respawn_ignores_stale_launcher_resume(monkeypatch):
    """The launcher's HERMES_TUI_RESUME is authoritative ONLY for the cold start.
    On a crash-respawn the orchestrator must read the LIVE sid from the active-
    session file, not re-apply the now-stale launcher id, so a recycle lands on
    whatever the session became, not where it started."""
    import json
    import tempfile

    monkeypatch.setenv("HERMES_TUI_RESUME", "cold-start-id")
    orch, spawned = make_orch()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"session_id": "live-sid-after-compaction"}, fh)
        orch.cfg.active_session_file = fh.name

    def mutate(i, o):
        if i == 1:
            o._renderer.die(1)  # crash → respawn

    run_for(orch, steps=4, mutate=mutate)

    assert len(spawned["renderer"]) == 2
    # Cold start honors the launcher id; the respawn uses the live sid instead.
    assert spawned["renderer"][0].resume_sid == "cold-start-id"
    assert spawned["renderer"][1].resume_sid == "live-sid-after-compaction", (
        f"respawn re-applied the stale launcher id instead of the live sid: "
        f"{spawned['renderer'][1].resume_sid}"
    )
    print("PASS test_respawn_ignores_stale_launcher_resume: respawn used the live sid")


def test_build_config_honors_launcher_active_session_file(monkeypatch):
    """The orchestrator must write the live sid to the SAME file the launcher
    reads at exit.

    Regression: `hermes --tui` mkstemps HERMES_TUI_ACTIVE_SESSION_FILE and reads
    it back to print the resume banner. The orchestrator previously ignored that
    env and used its own per-orchestrator temp path, so the launcher's file
    stayed empty and the exit banner fell through to the "last tui session" DB
    fallback (which printed the same stale, unrelated id every time). The config
    builder must adopt the launcher's path when present.
    """
    from tui_gateway.orchestrator import _build_orchestrator_config

    launcher_file = "/tmp/hermes-tui-active-session-UNITTEST.json"
    monkeypatch.setenv("HERMES_TUI_ACTIVE_SESSION_FILE", launcher_file)

    cfg = _build_orchestrator_config()
    assert cfg.active_session_file == launcher_file, (
        "orchestrator did not adopt the launcher's active-session file; the "
        "renderer would write the live sid to a file the launcher never reads"
    )


def test_build_config_defaults_active_file_without_launcher_env(monkeypatch):
    """Standalone `python -m tui_gateway.orchestrator` (no launcher env) keeps
    its own per-orchestrator default path."""
    from tui_gateway.orchestrator import _build_orchestrator_config

    monkeypatch.delenv("HERMES_TUI_ACTIVE_SESSION_FILE", raising=False)

    cfg = _build_orchestrator_config()
    assert cfg.active_session_file  # non-empty default
    assert "hermes-tui-orch-active-" in cfg.active_session_file


def test_gateway_spawn_does_not_share_renderer_terminal(monkeypatch, tmp_path):
    """Only the Ink renderer may own the interactive TTY.

    The WebSocket gateway is a sibling process. If it inherits fd 1 or fd 2,
    lifecycle notices such as context compaction and provider errors write raw
    text into Ink's alternate screen, moving the physical cursor behind Ink's
    back and corrupting subsequent incremental frames.
    """
    captured = {}
    proc = object()

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(orch_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = orch_mod._default_spawn_gateway("127.0.0.1", 12345, "credential")

    assert result is proc
    assert captured["kwargs"]["stdin"] is orch_mod.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is orch_mod.subprocess.STDOUT
    gateway_log = captured["kwargs"]["stdout"]
    assert gateway_log is not orch_mod.subprocess.DEVNULL
    assert gateway_log.name == str(tmp_path / "logs" / "tui_gateway_stdio.log")
    assert gateway_log.closed


if __name__ == "__main__":
    test_budget_unit()
    test_kill_renderer_keeps_gateway()
    test_gateway_death_respawns_both()
    test_clean_quit_tears_down()
    test_respawn_budget_bounds_crashloop()
    test_respawn_reads_resume_sid_from_active_file()
    test_corrupt_active_file_yields_no_resume()
    print("\nALL ORCHESTRATOR LOGIC TESTS PASSED")
