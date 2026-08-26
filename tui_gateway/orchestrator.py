"""TUI session orchestrator — durable gateway anchor + disposable renderer.

THE INVERSION
-------------
Today the bun renderer is the PARENT of the gateway+worker, so killing the
leaky renderer kills the session. This orchestrator makes them SIBLINGS:

    orchestrator (this process)
      |- gateway ws-host  (durable anchor: owns the session, survives renderer death)
      `- bun renderer      (disposable client: attaches over ws, killable/respawnable)

The renderer already supports attach mode: when HERMES_TUI_GATEWAY_URL is set it
connects to an existing gateway over WebSocket (gatewayClient.startAttachedGateway)
instead of spawning its own. The gateway already serves the full session over ws
(tui_gateway.ws.handle_ws reuses server.dispatch verbatim — same RPC, slash,
approval, agent-event handlers as stdio). This module wires the two together and
supervises them.

WHAT THIS GUARANTEES
--------------------
- Kill the renderer (OOM, manual, recycle): the gateway + agent run keep going.
  A fresh renderer started with the same HERMES_TUI_GATEWAY_URL re-attaches to
  the live session — zero work lost. ("kill the renderer, lose nothing")
- The orchestrator supervises both: a dead renderer is respawned (bounded), a
  dead gateway is respawned and the renderer re-attaches and resumes by sid.
- Reaper housekeeping (orphans/dupes) folds into the same supervisor loop.

Runtime is irrelevant: bun is node-compatible (just faster); the renderer is
launched exactly as today, only with HERMES_TUI_GATEWAY_URL pointed at our host.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from hermes_constants import get_hermes_home


# ── Bounded respawn policy (mirrors gatewayRecovery.planGatewayRecovery on the
# TS side so a crash-loop can't fork-bomb). Window + limit are deliberately the
# same shape: at most LIMIT respawns within WINDOW seconds, else give up and let
# the orchestrator exit so a human/outer supervisor notices.
RESPAWN_LIMIT = 5
RESPAWN_WINDOW_S = 60.0

# Distinct exit code the renderer uses for a DELIBERATE recycle (Ctrl+R, or the
# memory-pressure auto-recycle). It must NOT be 0 — the renderer's normal quit
# path (`die`) exits 0, and the orchestrator tears the session down on 0. A
# recycle means "respawn me on the still-live gateway and resume the session",
# so it needs its own code the supervisor recognizes. Mirrored on the TS side
# as RECYCLE_EXIT_CODE in lib/recycleBridge.ts — keep the two in sync.
RECYCLE_EXIT_CODE = 75


@dataclass
class _RespawnBudget:
    """Sliding-window respawn budget. Pure + unit-testable."""

    limit: int = RESPAWN_LIMIT
    window_s: float = RESPAWN_WINDOW_S
    attempts: list[float] = field(default_factory=list)

    def allow(self, now: float) -> bool:
        self.attempts = [t for t in self.attempts if now - t < self.window_s]
        if len(self.attempts) >= self.limit:
            return False
        self.attempts.append(now)
        return True


def _free_loopback_port() -> int:
    """Bind :0 on loopback to claim a free port, then release it. The gateway
    host re-binds it immediately; the tiny race is acceptable for a local,
    single-user orchestrator (no external contention)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    """Poll until the gateway ws-host accepts TCP, so we don't launch the
    renderer against a not-yet-listening socket."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


@dataclass
class OrchestratorConfig:
    """All inputs the orchestrator needs. Kept as data so the supervise loop is
    testable with injected spawn callables."""

    # How to launch the gateway ws-host. Returns a Popen.
    spawn_gateway: Callable[[str, int, str], "subprocess.Popen[bytes]"]
    # How to launch the renderer attached to the gateway ws. Receives the attach
    # URL and an optional resume sid (set on respawn so a recycled renderer
    # resumes the live session instead of forging a new one).
    spawn_renderer: Callable[[str, Optional[str]], "subprocess.Popen[bytes]"]
    host: str = "127.0.0.1"
    port: int = 0  # 0 → pick a free loopback port
    # Multi-use internal credential the renderer presents on (re)connect; the
    # gateway host accepts it for the orchestrator's own children only.
    internal_credential: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    # File the renderer writes its live sid to (HERMES_TUI_ACTIVE_SESSION_FILE),
    # and that the orchestrator reads on respawn to resume the same session. A
    # per-orchestrator temp path by default.
    active_session_file: str = field(
        default_factory=lambda: os.path.join(
            tempfile.gettempdir(), f"hermes-tui-orch-active-{secrets.token_hex(4)}.json"
        )
    )
    # File the renderer touches on a timer (HERMES_TUI_HEARTBEAT_FILE) so the
    # reaper can detect a frozen-but-alive renderer (stale mtime). Per-orchestrator.
    heartbeat_file: str = field(
        default_factory=lambda: os.path.join(
            tempfile.gettempdir(), f"hermes-tui-orch-hb-{secrets.token_hex(4)}"
        )
    )
    gateway_ready_timeout_s: float = 20.0
    renderer_respawn: _RespawnBudget = field(default_factory=_RespawnBudget)
    gateway_respawn: _RespawnBudget = field(default_factory=_RespawnBudget)
    poll_interval_s: float = 0.5
    # Stage 3 reaper: how often to scan /proc for orphan/frozen/dupe debris.
    # 0 disables the reaper entirely (the orchestrator's own lifecycle handling
    # is unaffected). Default 30s — cheap, and debris is not urgent.
    reaper_interval_s: float = 30.0


class Orchestrator:
    """Supervises a durable gateway ws-host and a disposable renderer.

    The renderer exiting is NORMAL (recycle/OOM/manual): the orchestrator
    respawns it within budget and it re-attaches to the live session. The
    gateway exiting is the rare event: respawn it (within budget), the renderer
    notices the dropped ws and reconnects (it already retries attach), and the
    session resumes from state.db by sid.
    """

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.port = cfg.port or _free_loopback_port()
        self._gateway: Optional["subprocess.Popen[bytes]"] = None
        self._renderer: Optional["subprocess.Popen[bytes]"] = None
        self._stop = threading.Event()
        # Set when the renderer exits 0 voluntarily (user /quit) — we then stop
        # the whole orchestrator instead of respawning a renderer nobody wants.
        self._renderer_quit = False
        self._last_reap = 0.0
        # The launcher's explicit `--resume <id>` arrives as HERMES_TUI_RESUME in
        # our env. It is authoritative ONLY for the first (cold) renderer launch:
        # a user asked to resume that session. Capture it once here so the cold
        # start can honor it. Respawns deliberately ignore it and read the live
        # sid from the active-session file instead (a stale env id must not
        # hijack a recycle). Without this, the cold-start renderer was spawned
        # with resume_sid=None and _spawn popped HERMES_TUI_RESUME, so
        # `hermes --tui --resume <id>` silently cold-started a blank session.
        cold = str(os.environ.get("HERMES_TUI_RESUME") or "").strip()
        self._cold_resume_sid: Optional[str] = cold or None

    def _live_pids(self) -> list[int]:
        """Pids the orchestrator currently owns + tracks — NEVER reaped."""
        pids = [os.getpid()]
        for proc in (self._gateway, self._renderer):
            if proc is not None and proc.poll() is None and proc.pid is not None:
                pids.append(proc.pid)
        return pids

    def _maybe_reap(self, now: float) -> None:
        """Periodic Stage 3 housekeeping: scan /proc for orphan/frozen/dupe
        Hermes-TUI debris and SIGTERM it. Best-effort and fully isolated from the
        orchestrator's own lifecycle (live pids are excluded). Any error here must
        never disturb supervision, so the whole thing is guarded.

        ``now`` is a MONOTONIC clock value (for the interval throttle only). The
        scan itself uses wall-clock time internally because heartbeat freshness is
        computed from file mtimes (wall clock) — mixing the two would corrupt the
        age calc, so we let scan_processes default to time.time()."""
        if self.cfg.reaper_interval_s <= 0:
            return
        if now - self._last_reap < self.cfg.reaper_interval_s:
            return
        self._last_reap = now
        try:
            from tui_gateway import reaper

            snapshot = reaper.scan_processes()  # wall-clock inside, for mtimes
            # Alive orchestrators = the owner pids in the snapshot that still
            # exist, plus ourselves. A child whose owner is NOT alive = orphan.
            owners = {p.owner_pid for p in snapshot if p.owner_pid is not None}
            alive_orchs = {os.getpid()} | {o for o in owners if reaper.pid_alive(o)}
            plan = reaper.plan_reap(
                snapshot,
                my_orchestrator_pid=os.getpid(),
                live_pids=self._live_pids(),
                alive_orchestrator_pids=alive_orchs,
            )
            if not plan.is_empty():
                reaper.execute_reap(plan, log=lambda m: None)
        except Exception:
            # Reaper failures are non-fatal — supervision continues.
            pass

    @property
    def attach_url(self) -> str:
        return (
            f"ws://{self.cfg.host}:{self.port}/api/ws"
            f"?internal={self.cfg.internal_credential}"
        )

    def _start_gateway(self) -> bool:
        self._gateway = self.cfg.spawn_gateway(self.cfg.host, self.port, self.cfg.internal_credential)
        if not _wait_for_port(self.cfg.host, self.port, self.cfg.gateway_ready_timeout_s):
            return False
        return True

    def _read_resume_sid(self) -> Optional[str]:
        """Read the live sid the renderer last wrote to the active-session file.

        The renderer writes {"session_id": "..."} via writeActiveSessionFile on
        every resume/activate. On respawn we read it so the fresh renderer
        resumes the SAME session (HERMES_TUI_RESUME) instead of forging a new
        one — this is what makes a recycle land back on the live session. Best-
        effort: a missing/corrupt file means no resume hint (fresh renderer
        forges/auto-resumes per config, today's cold-start behaviour).
        """
        try:
            with open(self.cfg.active_session_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sid = data.get("session_id")
            return sid if isinstance(sid, str) and sid else None
        except (OSError, ValueError, KeyError, AttributeError):
            return None

    def _start_renderer(self, *, resume: bool = False) -> None:
        # On a respawn (resume=True) read the last-known sid from the active-
        # session file so the fresh renderer resumes the live session the gateway
        # still holds. On the FIRST (cold) launch, honor the launcher's explicit
        # `--resume <id>` captured at construction (self._cold_resume_sid) so
        # `hermes --tui --resume <id>` actually loads that session instead of
        # cold-starting blank. The active-session file does not exist yet at cold
        # start, so the two sources never conflict.
        if resume:
            resume_sid = self._read_resume_sid()
        else:
            resume_sid = self._cold_resume_sid
        self._renderer = self.cfg.spawn_renderer(self.attach_url, resume_sid)

    def _terminate(self, proc: Optional["subprocess.Popen[bytes]"], *, grace_s: float = 5.0) -> None:
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=grace_s)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        """Blocking supervise loop. Returns a process exit code.

        Loop: keep the gateway alive (it is the anchor), keep a renderer alive
        attached to it. A renderer exit respawns a renderer (re-attach). A
        gateway exit respawns the gateway AND the renderer (the renderer's ws
        died with it). Budgets bound both so a crash-loop can't fork-bomb.
        """
        if not self._start_gateway():
            self._terminate(self._gateway)
            return 70  # EX_SOFTWARE: gateway never came up
        self._start_renderer()

        try:
            while not self._stop.is_set():
                time.sleep(self.cfg.poll_interval_s)

                # Stage 3 housekeeping — cheap, throttled internally to
                # reaper_interval_s, fully isolated from the lifecycle below.
                self._maybe_reap(time.monotonic())

                gw_dead = self._gateway is not None and self._gateway.poll() is not None
                rn_dead = self._renderer is not None and self._renderer.poll() is not None

                if rn_dead and not gw_dead:
                    code = self._renderer.returncode if self._renderer else 0
                    # A clean voluntary exit (user /quit via `die`) tears down
                    # everything. Only exit code 0 means "the user wants out".
                    if code == 0:
                        self._renderer_quit = True
                        break
                    # A DELIBERATE recycle (Ctrl+R or memory-pressure auto-
                    # recycle) exits with RECYCLE_EXIT_CODE. Respawn + resume the
                    # live session. Still budget-bounded so a stuck recycle key
                    # can't fork-bomb, but it's the intended, seamless path.
                    # Any OTHER non-zero code is a crash/OOM — same treatment
                    # (respawn within budget, re-attach, resume by sid).
                    if self.cfg.renderer_respawn.allow(time.monotonic()):
                        self._start_renderer(resume=True)
                    else:
                        break  # renderer crash/recycle-looping; bail so it's noticed

                elif gw_dead:
                    # The anchor died (rare). Respawn it within budget, then the
                    # renderer (its ws dropped). The renderer resumes by sid.
                    self._terminate(self._renderer)
                    if not self.cfg.gateway_respawn.allow(time.monotonic()):
                        break
                    if not self._start_gateway():
                        break
                    self._start_renderer(resume=True)
        finally:
            self._terminate(self._renderer)
            self._terminate(self._gateway)

        return 0 if self._renderer_quit else 0


def _default_spawn_gateway(host: str, port: int, internal_credential: str) -> "subprocess.Popen[bytes]":
    """Launch the standalone gateway ws-host (tui_gateway.ws_host) as a child."""
    env = dict(os.environ)
    env["HERMES_TUI_WS_HOST"] = host
    env["HERMES_TUI_WS_PORT"] = str(port)
    env["HERMES_TUI_WS_INTERNAL_CREDENTIAL"] = internal_credential
    # Stage 3 reaper marker: stamp THIS orchestrator's pid + role so the reaper
    # can positively identify our children in a /proc scan (never a PID guess).
    env["HERMES_TUI_ORCH_OWNER_PID"] = str(os.getpid())
    env["HERMES_TUI_ORCH_ROLE"] = "gateway"
    # The Ink renderer is the sole owner of the interactive terminal. The
    # gateway communicates over WebSocket and never needs terminal stdio.
    # Inheriting these descriptors lets any Python print, warning, traceback,
    # or uvicorn log write directly into Ink's alternate screen. That moves the
    # physical cursor behind Ink's frame model and corrupts later incremental
    # paints. Preserve those diagnostics in a profile-local file instead.
    stdio_log = None
    try:
        stdio_path = get_hermes_home() / "logs" / "tui_gateway_stdio.log"
        stdio_path.parent.mkdir(parents=True, exist_ok=True)
        stdio_log = stdio_path.open("ab", buffering=0)
    except OSError:
        pass

    try:
        return subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.ws_host"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdio_log if stdio_log is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if stdio_log is not None else subprocess.DEVNULL,
        )
    finally:
        if stdio_log is not None:
            stdio_log.close()


def _make_default_spawn_renderer(active_session_file: str, heartbeat_file: str = ""):
    """Build a renderer-spawner bound to the active-session-file path.

    The renderer writes its live sid to ``active_session_file``
    (HERMES_TUI_ACTIVE_SESSION_FILE); on respawn the orchestrator reads it back
    and passes ``resume_sid`` so the fresh renderer resumes the live session
    (HERMES_TUI_RESUME) instead of forging a new one. It also touches
    ``heartbeat_file`` (HERMES_TUI_HEARTBEAT_FILE) for frozen-detection.
    """

    def _spawn(attach_url: str, resume_sid: Optional[str]) -> "subprocess.Popen[bytes]":
        env = dict(os.environ)
        env["HERMES_TUI_GATEWAY_URL"] = attach_url
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file
        if heartbeat_file:
            env["HERMES_TUI_HEARTBEAT_FILE"] = heartbeat_file
        # Stage 3 reaper marker: stamp owner pid + role so a /proc scan can
        # positively identify this renderer as our child.
        env["HERMES_TUI_ORCH_OWNER_PID"] = str(os.getpid())
        env["HERMES_TUI_ORCH_ROLE"] = "renderer"
        if resume_sid:
            env["HERMES_TUI_RESUME"] = resume_sid
        else:
            # A cold start must NOT carry a stale resume from the parent env.
            env.pop("HERMES_TUI_RESUME", None)
        # Prefer the launcher-resolved renderer argv (bun/node/tsx, --dev, the
        # right entry path + NODE_OPTIONS) handed over via HERMES_TUI_RENDERER_ARGV
        # so runtime selection and dev-mode are honored. Fall back to the bun +
        # dist/entry.js default for a standalone `python -m tui_gateway.orchestrator`.
        renderer_argv_raw = env.get("HERMES_TUI_RENDERER_ARGV")
        if renderer_argv_raw:
            try:
                renderer_argv = json.loads(renderer_argv_raw)
                if not (isinstance(renderer_argv, list) and renderer_argv):
                    raise ValueError("empty renderer argv")
            except (ValueError, TypeError):
                renderer_argv = None
        else:
            renderer_argv = None
        if renderer_argv is None:
            bun = env.get("HERMES_BUN") or "bun"
            root = env.get("HERMES_PYTHON_SRC_ROOT") or os.getcwd()
            entry = os.path.join(root, "ui-tui", "dist", "entry.js")
            renderer_argv = [bun, entry]
        # This is the interactive Ink/Node TUI renderer itself — it must
        # inherit the real terminal's stdin/stdout to receive user
        # keystrokes and paint the UI. DEVNULL would break the TUI
        # entirely (no keyboard input could ever reach it).
        return subprocess.Popen(renderer_argv, env=env)  # noqa: subprocess-stdin

    return _spawn


def _build_orchestrator_config() -> "OrchestratorConfig":
    """Build the orchestrator config, honoring the launcher's env seams.

    Honor the launcher-provided HERMES_TUI_ACTIVE_SESSION_FILE when present.
    ``hermes --tui`` (hermes_cli/main.py) mkstemps this path, passes it in the
    env, and reads IT BACK at exit to print the resume epilogue. If the
    orchestrator picked its own default path instead, the renderer would write
    the live sid to the orchestrator's file while the launcher read its own
    (forever-empty) file, so the exit banner fell through to the "last tui
    session" DB fallback and printed a stale, unrelated session id on every
    exit. Sharing the one path the launcher owns keeps the written sid and the
    read sid the same file. The per-orchestrator default still applies for a
    standalone ``python -m tui_gateway.orchestrator`` with no launcher env.
    """
    launcher_active_file = str(
        os.environ.get("HERMES_TUI_ACTIVE_SESSION_FILE") or ""
    ).strip()
    if launcher_active_file:
        cfg = OrchestratorConfig(
            spawn_gateway=_default_spawn_gateway,
            spawn_renderer=lambda url, sid: url,  # placeholder, replaced below
            active_session_file=launcher_active_file,
        )
    else:
        cfg = OrchestratorConfig(
            spawn_gateway=_default_spawn_gateway,
            spawn_renderer=lambda url, sid: url,  # placeholder, replaced below
        )
    cfg.spawn_renderer = _make_default_spawn_renderer(
        cfg.active_session_file, cfg.heartbeat_file
    )
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    # Build cfg first so the renderer-spawner can bind to its active-session
    # file path (the seam the recycle/resume uses).
    cfg = _build_orchestrator_config()
    orch = Orchestrator(cfg)

    def _on_signal(_signum: int, _frame: object) -> None:
        orch.request_stop()

    for _sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(Exception):
            signal.signal(_sig, _on_signal)

    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())
