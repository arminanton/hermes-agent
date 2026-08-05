"""`hermes install` — install the SSH-layer media/file bridge client (shellctl).

`hermes install shellctl` is run on the HERMES HOST (the box you SSH into). It:

  1. ensures the bridge assets exist under the workspace,
  2. generates (or reuses) a shared bridge token,
  3. prints the exact `~/.ssh/config` snippet + the one-line client install command
     the user pastes on THEIR machine (Mac/Linux/WSL/PuTTY host).

Design: zero-dependency client (Python stdlib only), no sudo, no ControlMaster
requirement (per-connection RemoteForward works for plain `ssh` and tmux-wrapping
`sshp` alike). The client is served for copy via `hermes install shellctl --print-client`.
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path

# Bridge assets live in the workspace so they survive Hermes updates.
_SHELLCTL_DIR = Path(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
) / "shellctl"
_TOKEN_FILE = _SHELLCTL_DIR / "bridge-token"
_CLIENT_FILE = _SHELLCTL_DIR / "hermes-shellctl"
_BRIDGE_FILE = _SHELLCTL_DIR / "hermes-shellbridge"
_DEFAULT_PORT = 8765

# Canonical source location (ships with the hermes_cli package).
_CANONICAL_DIR = Path(__file__).resolve().parent / "shellctl_assets"


def _ensure_assets() -> None:
    _SHELLCTL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("hermes-shellctl", "hermes-shellbridge"):
        src = _CANONICAL_DIR / name
        dst = _SHELLCTL_DIR / name
        # When HERMES_HOME points somewhere whose shellctl dir IS the canonical
        # dir, src == dst — nothing to copy, just ensure perms.
        if src.resolve() != dst.resolve():
            if src.is_file() and (not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime):
                shutil.copy2(src, dst)
        if dst.is_file():
            dst.chmod(0o755)


def _get_or_make_token() -> str:
    if _TOKEN_FILE.is_file():
        tok = _TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    tok = secrets.token_hex(24)
    _TOKEN_FILE.write_text(tok + "\n")
    _TOKEN_FILE.chmod(0o600)
    return tok


def _print_client() -> int:
    if not _CLIENT_FILE.is_file():
        _ensure_assets()
    if not _CLIENT_FILE.is_file():
        print("error: client asset missing", file=sys.stderr)
        return 1
    sys.stdout.write(_CLIENT_FILE.read_text())
    return 0


def cmd_install_shellctl(args: argparse.Namespace) -> int:
    if getattr(args, "print_client", False):
        return _print_client()

    _ensure_assets()
    token = _get_or_make_token()
    port = int(getattr(args, "port", _DEFAULT_PORT) or _DEFAULT_PORT)
    # The hostname/alias the user SSHes to — best effort; user edits to taste.
    host_hint = getattr(args, "ssh_host", "") or "your-hermes-host"

    bar = "=" * 72
    print(bar)
    print(" Hermes shellctl — SSH-layer media/file bridge")
    print(bar)
    print()
    print("Bridge token (shared secret, keep private):")
    print(f"  {token}")
    print()
    print("STEP 1 — On YOUR machine (Mac/Linux/WSL), save the client:")
    print(f"  ssh {host_hint} 'hermes install shellctl --print-client' > ~/.hermes-shellctl")
    print("  chmod +x ~/.hermes-shellctl")
    print()
    print("STEP 2 — Add this ONE block to ~/.ssh/config on YOUR machine")
    print("         (works for plain `ssh` AND tmux-wrapping helpers like sshp;")
    print("          no ControlMaster required):")
    print()
    print(f"  Host {host_hint}")
    print(f"      RemoteForward 127.0.0.1:{port} 127.0.0.1:{port}")
    print()
    print("STEP 3 — Start the client daemon on YOUR machine (one command, leave it running")
    print("         in a tab — or add to login items):")
    print(f"  HERMES_SHELLCTL_TOKEN={token} \\")
    print(f"    python3 ~/.hermes-shellctl daemon --port {port}")
    print()
    print("STEP 4 — SSH in normally. The reverse forward makes your machine's bridge")
    print("         reachable at 127.0.0.1:%d ON the Hermes host. Verify:" % port)
    print(f"  HERMES_SHELLCTL_TOKEN={token} \\")
    print(f"    HERMES_SHELLCTL_URL=http://127.0.0.1:{port} \\")
    print(f"    python3 {_BRIDGE_FILE} ping")
    print()
    print("Then in the TUI:  /get <local-path> · /send <file> · /say <text> · /listen · /paste")
    print(bar)

    # Persist the resolved config so the TUI bridge glue can read it.
    cfg = _SHELLCTL_DIR / "bridge.env"
    cfg.write_text(
        f"HERMES_SHELLCTL_URL=http://127.0.0.1:{port}\n"
        f"HERMES_SHELLCTL_TOKEN={token}\n"
        f"HERMES_SHELLCTL_PORT={port}\n"
    )
    cfg.chmod(0o600)
    return 0


def register_cli(install_parser: argparse.ArgumentParser) -> None:
    """Attach `install` subcommands to the given parser."""
    sub = install_parser.add_subparsers(dest="install_target", required=True)
    sc = sub.add_parser(
        "shellctl",
        help="Install the SSH media/file bridge client (image/pdf/audio over SSH)",
    )
    sc.add_argument("--port", type=int, default=_DEFAULT_PORT,
                    help=f"bridge port (default {_DEFAULT_PORT})")
    sc.add_argument("--ssh-host", default="",
                    help="the Host alias you SSH to (for the printed snippet)")
    sc.add_argument("--print-client", action="store_true",
                    help="print the client script to stdout (for piping to a file)")
    sc.set_defaults(func=cmd_install_shellctl)


def install_command(args: argparse.Namespace) -> int:
    func = getattr(args, "func", None)
    if func is None or func is install_command:
        print("usage: hermes install shellctl", file=sys.stderr)
        return 2
    return func(args)
