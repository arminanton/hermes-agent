"""Google Gemini CLI fingerprint helpers (User-Agent / X-Goog-Api-Client).

The legitimate API-key Gemini provider (``agent/gemini_native_adapter.py``,
``generativelanguage.googleapis.com``) presents a User-Agent that matches the
real ``@google/gemini-cli`` wire format so Google's backend recognizes it.

These helpers were relocated here from the now-removed
``agent/google_user_agent.py`` (which belonged to the deleted Google-OAuth /
Cloud Code Assist provider category) so the surviving Gemini provider keeps
working. Only the UA / X-Goog-Api-Client helpers and their private deps are
retained; the clientMetadata / OAuth fingerprint machinery went with the
removed provider.

The real ``@google/gemini-cli`` builds its UA like::

    User-Agent: GeminiCLI/<pkg.version>/<model> (<process.platform>; <process.arch>; <surface>)
    X-Goog-Api-Client: gl-node/<process.versions.node> gccl/<pkg.version>

Env overrides
-------------
- ``HERMES_GEMINI_CLI_VERSION``   pin pkg version (skip npm lookup)
- ``HERMES_GEMINI_NODE_VERSION``  pin reported node version
- ``HERMES_GEMINI_CLI_SURFACE``   override the ``surface`` segment
                                  (default ``hermes``)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Hardcoded fallbacks. Bumped opportunistically; runtime resolution prefers
# the live npm registry so these only matter when offline / blocked.
_GEMINI_CLI_VERSION_FALLBACK = "0.44.1"
_NODE_VERSION_FALLBACK = "24.0.0"
_DEFAULT_SURFACE = "hermes"

_GEMINI_CLI_NPM_URL = "https://registry.npmjs.org/@google/gemini-cli/latest"
_VERSION_CACHE_TTL = 24 * 60 * 60  # 24h, matches copilot_auth pattern
_VERSION_CACHE_PATH = Path.home() / ".cache" / "hermes" / "gemini_cli_version.json"

# In-process memo so the hot path doesn't touch disk on every header build.
_version_memo: Optional[tuple[str, float]] = None


def _gemini_cli_version() -> str:
    """Return the latest ``@google/gemini-cli`` version on npm.

    Resolution order:
      1. ``HERMES_GEMINI_CLI_VERSION`` env override.
      2. In-process memo (TTL ``_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_VERSION_CACHE_PATH``.
      4. ``GET https://registry.npmjs.org/@google/gemini-cli/latest``.
      5. Hard fallback ``_GEMINI_CLI_VERSION_FALLBACK``.

    Network failures are swallowed; we always return *something*.
    """
    override = os.getenv("HERMES_GEMINI_CLI_VERSION", "").strip()
    if override:
        return override

    global _version_memo
    now = time.time()
    if _version_memo and now - _version_memo[1] < _VERSION_CACHE_TTL:
        return _version_memo[0]

    cache_path = _VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VERSION_CACHE_TTL:
                _version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("gemini-cli version cache read failed: %s", exc)

    ver = _GEMINI_CLI_VERSION_FALLBACK
    try:
        import urllib.request

        req = urllib.request.Request(
            _GEMINI_CLI_NPM_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "gemini-cli",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode())
        latest = str(payload.get("version") or "").lstrip("v").strip()
        if latest:
            ver = latest
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now}),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("gemini-cli version cache write failed: %s", exc)
    except Exception as exc:
        logger.debug(
            "failed to fetch latest @google/gemini-cli version, using fallback %s: %s",
            _GEMINI_CLI_VERSION_FALLBACK,
            exc,
        )

    _version_memo = (ver, now)
    return ver


def _node_version() -> str:
    """Return the node version we report in the UA / X-Goog-Api-Client."""
    return os.getenv("HERMES_GEMINI_NODE_VERSION", "").strip() or _NODE_VERSION_FALLBACK


def _surface() -> str:
    """Return the ``surface`` segment of the UA.

    The real gemini-cli reads this from ``GEMINI_CLI_SURFACE`` and falls
    back to detection (``vscode``, ``cursor``, …) or ``unknown``. We default
    to ``hermes`` so Google's logs can attribute traffic correctly.
    """
    return os.getenv("HERMES_GEMINI_CLI_SURFACE", "").strip() or _DEFAULT_SURFACE


def _process_platform() -> str:
    """Mimic Node's ``process.platform`` (``linux``/``darwin``/``win32``)."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "win32"
    return sys.platform


def _process_arch() -> str:
    """Mimic Node's ``process.arch`` (``x64``/``arm64``/``ia32``)."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("i386", "i686", "x86"):
        return "ia32"
    return machine or "x64"


def gemini_cli_user_agent(model: str = "") -> str:
    """Return a User-Agent string matching gemini-cli's wire format.

    Examples::

        GeminiCLI/0.44.1 (linux; x64)
        GeminiCLI/0.44.1/gemini-2.5-pro (linux; x64)
    """
    ver = _gemini_cli_version()
    head = f"GeminiCLI/{ver}/{model}" if model else f"GeminiCLI/{ver}"
    return f"{head} ({_process_platform()}; {_process_arch()}; {_surface()})"


def gemini_cli_x_goog_api_client() -> str:
    """Return the ``X-Goog-Api-Client`` header value gemini-cli sends."""
    return f"gl-node/{_node_version()} gccl/{_gemini_cli_version()}"
