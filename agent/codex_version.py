"""Resolve the codex CLI semver to advertise on chatgpt.com backend calls.

The Cloudflare layer in front of ``chatgpt.com/backend-api/codex/*``
allowlists requests whose ``originator`` is one of ``codex_cli_rs`` /
``codex_vscode`` / ``codex_sdk_ts`` (or starts with ``Codex``) and whose
``User-Agent`` is shaped like ``codex_cli_rs/MAJOR.MINOR.PATCH``. The same
value is sent as the ``client_version`` query parameter on ``/models`` and
mirrored to the local app-server ``initialize`` handshake so codex's own
diagnostics see a consistent peer.

To present a *current* identity that tracks the real codex release train
without shipping a stale hard-coded number, the version is resolved in this
precedence (each tier fails soft to the next, this never raises):

  1. **Live npm registry.** Fetch ``@openai/codex`` ``latest`` from
     ``registry.npmjs.org`` under a short timeout. This is the authoritative
     released version (e.g. ``0.149.1``).
  2. **Weekly cache.** The fetched value is written to a small JSON file
     under ``$HERMES_HOME/.cache/codex_version.json`` with a 7-day TTL, so
     the network call happens at most once a week. A fresh cache short
     circuits the fetch entirely, we never block on the network when the
     cache is still valid.
  3. **Installed codex CLI.** ``codex --version`` (parsed as
     ``codex-cli X.Y.Z``) is used *only if* it is at or above the floor,
     so a stale local install can never drag the advertised identity below
     a known-good baseline.
  4. **Hard-coded floor.** ``_FLOOR_CODEX_CLI_VERSION`` as a last resort.

The resolved value is memoized at module level so repeated hot-path calls in
one process neither re-read disk nor re-hit the network.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Known-good baseline. The installed-CLI tier is ignored below this, and it
# is the final fallback when every other tier fails. Kept in step with the
# codex release train Hermes is validated against.
_FLOOR_CODEX_CLI_VERSION = "0.149.0"

# npm registry endpoint for the released codex CLI. The JSON payload's
# ``version`` field is the authoritative latest semver.
_NPM_LATEST_URL = "https://registry.npmjs.org/@openai/codex/latest"

# Network fetch is best-effort on a hot path: keep it short.
_NPM_FETCH_TIMEOUT_SECONDS = 2.5

# Subprocess ``codex --version`` timeout.
_VERSION_QUERY_TIMEOUT_SECONDS = 10.0

# Cache freshness window: fetch the registry at most once per week.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# In-process memo so repeated calls neither re-read disk nor re-fetch.
_memo: Optional[str] = None


def _parse_version_tuple(text: str) -> Optional[tuple[int, int, int]]:
    """Parse ``MAJOR.MINOR.PATCH`` out of ``text``. Returns a tuple or None.

    Reused for both npm ``version`` strings and ``codex --version`` output
    (``codex-cli 0.149.0``), tolerating any surrounding/trailing metadata.
    """
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )


def _cache_path() -> Path:
    """Return the on-disk cache location under ``$HERMES_HOME/.cache``.

    Resolves ``HERMES_HOME`` from the environment, falling back to
    ``~/.hermes`` so the cache is always writable to a stable location.
    """
    home = os.environ.get("HERMES_HOME", "").strip()
    base = Path(home) if home else (Path.home() / ".hermes")
    return base / ".cache" / "codex_version.json"


def _read_cache() -> Optional[str]:
    """Return the cached version if present and fresh, else None.

    A read/parse error, a missing ``fetched_at``/``version``, or an entry
    older than the TTL all yield None so the caller falls through to a live
    fetch.
    """
    path = _cache_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    fetched_at = data.get("fetched_at")
    if not isinstance(version, str) or not version.strip():
        return None
    if not isinstance(fetched_at, (int, float)):
        return None
    if (time.time() - float(fetched_at)) >= _CACHE_TTL_SECONDS:
        return None  # stale
    if _parse_version_tuple(version) is None:
        return None
    return version.strip()


def _write_cache(version: str) -> None:
    """Persist ``version`` with the current timestamp. Best-effort."""
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "fetched_at": time.time()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("codex_version: cache write failed: %s", exc)


def _fetch_latest_from_npm() -> Optional[str]:
    """Fetch the latest codex version from npm. Returns semver or None.

    Best-effort: any network error, timeout, non-200, or unparseable
    payload returns None so the caller falls through.
    """
    try:
        import httpx

        resp = httpx.get(
            _NPM_LATEST_URL,
            timeout=_NPM_FETCH_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            logger.debug(
                "codex_version: npm returned %s", resp.status_code
            )
            return None
        data = resp.json()
        version = data.get("version") if isinstance(data, dict) else None
        if not isinstance(version, str):
            return None
        parsed = _parse_version_tuple(version)
        if parsed is None:
            return None
        return ".".join(str(part) for part in parsed)
    except Exception as exc:  # network, import, decode, etc.
        logger.debug("codex_version: npm fetch failed: %s", exc)
        return None


def _default_codex_bin() -> str:
    """Return the codex executable Hermes drives (``HERMES_CODEX_BIN``)."""
    return (os.environ.get("HERMES_CODEX_BIN") or "codex").strip() or "codex"


def _query_installed_version(codex_bin: str) -> Optional[str]:
    """Run ``<codex_bin> --version`` and return MAJOR.MINOR.PATCH, or None.

    Any failure (missing binary, non-zero exit, timeout, unparseable
    output) returns None so the caller can fall back.
    """
    try:
        proc = subprocess.run(
            [codex_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_VERSION_QUERY_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            logger.debug(
                "codex_version: %r --version exited %s",
                codex_bin,
                proc.returncode,
            )
            return None
        parsed = _parse_version_tuple(proc.stdout)
        if parsed is None:
            return None
        return ".".join(str(part) for part in parsed)
    except FileNotFoundError:
        logger.debug("codex_version: %r not found on PATH", codex_bin)
        return None
    except Exception as exc:  # subprocess timeout, etc.
        logger.debug("codex_version: version query failed: %s", exc)
        return None


def _resolve() -> str:
    """Resolve the codex CLI version through the tier precedence.

    Order: fresh weekly cache, else live npm fetch (cached on success),
    else installed CLI if at/above the floor, else the floor constant.
    """
    floor_tuple = _parse_version_tuple(_FLOOR_CODEX_CLI_VERSION)

    # Tier 1+2: a fresh cache short-circuits the network entirely.
    cached = _read_cache()
    if cached is not None:
        return cached

    # Tier 1: live fetch (cache stale/missing). Persist on success so the
    # next call within the TTL avoids the network.
    fetched = _fetch_latest_from_npm()
    if fetched is not None:
        _write_cache(fetched)
        return fetched

    # Tier 3: installed CLI, but only if it meets the floor.
    installed = _query_installed_version(_default_codex_bin())
    if installed is not None and floor_tuple is not None:
        installed_tuple = _parse_version_tuple(installed)
        if installed_tuple is not None and installed_tuple >= floor_tuple:
            return installed

    # Tier 4: hard-coded floor.
    return _FLOOR_CODEX_CLI_VERSION


def get_codex_cli_version() -> str:
    """Return the codex CLI semver to advertise on backend calls.

    Always returns a ``MAJOR.MINOR.PATCH`` string, never raises. The result
    is memoized at module level so repeated hot-path calls in one process
    neither re-read the disk cache nor re-hit the network.
    """
    global _memo
    if _memo is not None:
        return _memo
    try:
        resolved = _resolve()
    except Exception as exc:  # defensive: never raise into the caller
        logger.debug("codex_version: resolution failed: %s", exc)
        resolved = _FLOOR_CODEX_CLI_VERSION
    _memo = resolved
    return resolved


__all__ = ["get_codex_cli_version"]
