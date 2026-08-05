"""Tests for the hermes-shellctl daemon's lossless clipboard grab.

The daemon is a standalone stdlib-only script (no .py extension) that runs on the
USER's machine and serves clipboard/file bytes to the remote Hermes host over the
SSH bridge. These tests load it into a real module object (so its globals can be
monkeypatched) and verify the lossless-transfer contract:

  - a FILE on the clipboard (Finder copy / file URL) is read byte-for-byte off
    disk and returned with its real name + content-type (any type: pdf/exe/...),
  - only when there is NO backing file does it fall back to a raw image bitmap,
  - the content-type mapping is correct,
  - the legacy _clipboard_image() 2-tuple shim still works.
"""

import hashlib
import os
import tempfile
import types
from pathlib import Path

import pytest

_DAEMON_PATH = (
    Path(__file__).resolve().parents[2]
    / "hermes_cli" / "shellctl_assets" / "hermes-shellctl"
)


@pytest.fixture
def daemon():
    """Load the extension-less daemon script as a real module object."""
    mod = types.ModuleType("shellctl_under_test")
    src = _DAEMON_PATH.read_text(encoding="utf-8")
    exec(compile(src, str(_DAEMON_PATH), "exec"), mod.__dict__)  # noqa: S102 - trusted in-repo file
    return mod


def test_guess_ctype_covers_common_types(daemon):
    assert daemon._guess_ctype("shot.png") == "image/png"
    assert daemon._guess_ctype("scan.pdf") == "application/pdf"
    assert daemon._guess_ctype("photo.jpg") == "image/jpeg"
    assert daemon._guess_ctype("clip.heic") == "image/heic"
    # Unknown/binary extensions fall back to octet-stream, never guessed as text,
    # so the remote side saves them verbatim.
    assert daemon._guess_ctype("tool.exe") == "application/octet-stream"
    assert daemon._guess_ctype("noext") == "application/octet-stream"


def test_clipboard_grab_prefers_original_file_bytes(daemon, monkeypatch):
    """A file on the clipboard is returned byte-for-byte with its real name."""
    payload = os.urandom(300_000)  # 300 KB, larger than any thumbnail proxy
    src = os.path.join(tempfile.mkdtemp(), "report.pdf")
    with open(src, "wb") as f:
        f.write(payload)

    monkeypatch.setattr(daemon, "_IS_MAC", True)
    monkeypatch.setattr(daemon, "_macos_clipboard_file_paths", lambda: [src])

    data, name, ctype, msg = daemon._clipboard_grab()

    assert msg == "ok"
    assert name == "report.pdf"
    assert ctype == "application/pdf"
    # Byte-identity: no re-encode, no downscale, no proxy.
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_clipboard_grab_falls_back_to_image_bitmap(daemon, monkeypatch):
    """With no backing file, a raw clipboard image is returned as PNG."""
    png_bytes = b"\x89PNG\r\n\x1a\n" + os.urandom(1024)

    def _fake_run(argv, **kw):
        with open(argv[-1], "wb") as f:  # pngpaste writes to the dest path
            f.write(png_bytes)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(daemon, "_IS_MAC", True)
    monkeypatch.setattr(daemon, "_macos_clipboard_file_paths", lambda: [])
    monkeypatch.setattr(daemon, "_which", lambda *a: "/usr/local/bin/pngpaste")
    monkeypatch.setattr(daemon.subprocess, "run", _fake_run)

    data, name, ctype, msg = daemon._clipboard_grab()

    assert msg == "ok"
    assert ctype == "image/png"
    assert name.endswith(".png")
    assert data == png_bytes


def test_clipboard_grab_rejects_oversize_file(daemon, monkeypatch):
    big = os.path.join(tempfile.mkdtemp(), "huge.bin")
    with open(big, "wb") as f:
        f.write(b"\0")

    monkeypatch.setattr(daemon, "_IS_MAC", True)
    monkeypatch.setattr(daemon, "_macos_clipboard_file_paths", lambda: [big])
    monkeypatch.setattr(daemon.os.path, "getsize", lambda _p: daemon._MAX_BYTES + 1)

    data, name, ctype, msg = daemon._clipboard_grab()

    assert data is None
    assert "too large" in msg


def test_legacy_clipboard_image_shim_returns_two_tuple(daemon, monkeypatch):
    """The old (data, msg) contract must still hold for any legacy caller."""
    monkeypatch.setattr(
        daemon, "_clipboard_grab",
        lambda: (b"abc", "clip.png", "image/png", "ok"),
    )
    assert daemon._clipboard_image() == (b"abc", "ok")
