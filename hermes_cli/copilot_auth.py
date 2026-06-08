"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API.

Token type support (per GitHub docs):
  gho_          OAuth token           ✓  (default via copilot login)
  github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
  ghu_          GitHub App token      ✓  (via environment variable)
  ghp_          Classic PAT           ✗  NOT SUPPORTED

Credential search order (matching Copilot CLI behaviour):
  1. COPILOT_GITHUB_TOKEN env var
  2. GH_TOKEN env var
  3. GITHUB_TOKEN env var
  4. gh auth token  CLI fallback

Catalog discovery can optionally extend that path with the Copilot
credential pool and records skipped invalid sources for auditability.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


@dataclass(frozen=True)
class CopilotIdentitySkip:
    """One invalid source encountered while resolving the Copilot identity."""

    source: str
    reason: str


@dataclass(frozen=True)
class CopilotIdentityAudit:
    """Structured Copilot identity resolution result."""

    token: str = ""
    source: str = ""
    source_kind: str = ""
    skipped_sources: tuple[CopilotIdentitySkip, ...] = field(default_factory=tuple)
    error: str = ""


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API.

    Returns (valid, message).
    """
    token = token.strip()
    if not token:
        return False, "Empty token"

    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
        )

    return True, "OK"


def resolve_copilot_identity_audit(
    *,
    include_credential_pool: bool = False,
    exchange_pool_tokens: bool = False,
) -> CopilotIdentityAudit:
    """Resolve the active Copilot identity and retain an audit trail.

    ``resolve_copilot_token()`` wraps this helper for the compatibility path.
    Discovery code can opt into the credential pool and exchange behavior
    with ``include_credential_pool`` and ``exchange_pool_tokens``.
    """
    skipped_sources: list[CopilotIdentitySkip] = []

    # 1. Check env vars in priority order.
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if not val:
            continue
        valid, msg = validate_copilot_token(val)
        if not valid:
            logger.warning(
                "Token from %s is not supported: %s", env_var, msg
            )
            skipped_sources.append(CopilotIdentitySkip(source=env_var, reason=msg))
            continue
        return CopilotIdentityAudit(
            token=val,
            source=env_var,
            source_kind="env",
            skipped_sources=tuple(skipped_sources),
        )

    # 2. Optionally inspect the Copilot credential pool before gh auth.
    if include_credential_pool:
        try:
            from hermes_cli.auth import read_credential_pool
        except Exception as exc:
            logger.debug("Copilot credential pool lookup unavailable: %s", exc)
        else:
            try:
                pool_entries = read_credential_pool("copilot")
            except Exception as exc:
                logger.debug("Copilot credential pool lookup failed: %s", exc)
                skipped_sources.append(
                    CopilotIdentitySkip(
                        source="credential_pool:copilot",
                        reason=f"Failed to read credential pool: {exc}",
                    )
                )
            else:
                for index, entry in enumerate(pool_entries):
                    entry_source = f"credential_pool:copilot[{index}]"
                    if not isinstance(entry, dict):
                        skipped_sources.append(
                            CopilotIdentitySkip(
                                source=entry_source,
                                reason="Non-dict credential pool entry",
                            )
                        )
                        continue

                    raw = str(entry.get("access_token") or "").strip()
                    if not raw:
                        skipped_sources.append(
                            CopilotIdentitySkip(
                                source=entry_source,
                                reason="Missing access_token",
                            )
                        )
                        continue

                    valid, msg = validate_copilot_token(raw)
                    if not valid:
                        skipped_sources.append(
                            CopilotIdentitySkip(source=entry_source, reason=msg)
                        )
                        continue

                    if exchange_pool_tokens:
                        try:
                            api_token, _expires_at = exchange_copilot_token(raw)
                        except Exception as exc:
                            skipped_sources.append(
                                CopilotIdentitySkip(
                                    source=entry_source,
                                    reason=f"Copilot token exchange failed: {exc}",
                                )
                            )
                            continue
                        if not api_token:
                            skipped_sources.append(
                                CopilotIdentitySkip(
                                    source=entry_source,
                                    reason="Copilot token exchange returned empty token",
                                )
                            )
                            continue
                        return CopilotIdentityAudit(
                            token=api_token,
                            source=entry_source,
                            source_kind="credential_pool",
                            skipped_sources=tuple(skipped_sources),
                        )

                    return CopilotIdentityAudit(
                        token=raw,
                        source=entry_source,
                        source_kind="credential_pool",
                        skipped_sources=tuple(skipped_sources),
                    )

    # 3. Fall back to gh auth token.
    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            return CopilotIdentityAudit(
                skipped_sources=tuple(skipped_sources),
                error=(
                    "Token from `gh auth token` is a classic PAT (ghp_*). "
                    f"{msg}"
                ),
            )
        return CopilotIdentityAudit(
            token=token,
            source="gh auth token",
            source_kind="gh_auth",
            skipped_sources=tuple(skipped_sources),
        )

    return CopilotIdentityAudit(skipped_sources=tuple(skipped_sources))


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for Copilot API use.

    Returns (token, source) where source describes where the token came from.
    Raises ValueError if only a classic PAT is available from ``gh auth token``.
    """
    audit = resolve_copilot_identity_audit()
    if audit.error:
        raise ValueError(audit.error)
    return audit.token, audit.source


def _gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            candidates.append(candidate)

    return candidates


def _try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available.

    When COPILOT_GH_HOST is set, passes ``--hostname`` so gh returns the
    correct host's token. When COPILOT_GH_USER is set, also passes ``--user``
    so multi-account setups resolve to the intended account regardless of
    which one is currently active. Also strips GITHUB_TOKEN / GH_TOKEN from
    the subprocess environment so ``gh`` reads from its own credential store
    (hosts.yml) instead of just echoing the env var back.
    """
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()
    username = os.getenv("COPILOT_GH_USER", "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        if username:
            cmd += ["--user", username]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=clean_env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


# ─── OAuth Device Code Flow ────────────────────────────────────────────────

def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot.

    Prints instructions for the user, polls for completion, and returns
    the OAuth access token on success, or None on failure/cancellation.

    This replicates the flow used by opencode and the Copilot CLI.
    """
    import urllib.request
    import urllib.parse

    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    # Step 1: Request device code
    data = urllib.parse.urlencode({
        "client_id": COPILOT_OAUTH_CLIENT_ID,
        "scope": "read:user",
    }).encode()

    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": f"GitHubCopilotChat/{_latest_copilot_chat_version()}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            device_data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None

    # Step 2: Show instructions
    print()
    print(f"  Open this URL in your browser: {verification_uri}")
    print(f"  Enter this code: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    # Step 3: Poll for completion
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()

        poll_req = urllib.request.Request(
            access_token_url,
            data=poll_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": f"GitHubCopilotChat/{_latest_copilot_chat_version()}",
            },
        )

        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception:
            print(".", end="", flush=True)
            continue

        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]

        error = result.get("error", "")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval
            server_interval = result.get("interval")
            if isinstance(server_interval, (int, float)) and server_interval > 0:
                interval = int(server_interval)
            else:
                interval += 5
            print(".", end="", flush=True)
            continue
        elif error == "expired_token":
            print()
            print("  ✗ Device code expired. Please try again.")
            return None
        elif error == "access_denied":
            print()
            print("  ✗ Authorization was denied.")
            return None
        elif error:
            print()
            print(f"  ✗ Authorization failed: {error}")
            return None

    print()
    print("  ✗ Timed out waiting for authorization.")
    return None


# ─── Copilot Token Exchange ────────────────────────────────────────────────

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch).
_jwt_cache: dict[str, tuple[str, float]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint and headers (matching VS Code Copilot Chat).
# We intentionally identify as VS Code (not copilot-cli) because the
# vscode-chat integration is on a more generous token budget for Copilot
# subscribers. The editor version is fetched dynamically from the VS Code
# GitHub releases so we always look like the latest stable build.
# NOTE: the exchange endpoint itself is no longer used by the official
# Copilot CLI for /chat/completions or /models — those accept the raw gh
# token as a Bearer credential directly. Kept for opt-in compatibility
# (HERMES_COPILOT_FORCE_EXCHANGE=1).
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_VSCODE_VERSION_FALLBACK = "1.104.1"
_VSCODE_RELEASES_URL = "https://api.github.com/repos/microsoft/vscode/releases/latest"
_VSCODE_VERSION_CACHE_TTL = 24 * 60 * 60  # 24h
_VSCODE_VERSION_CACHE_PATH = Path.home() / ".cache" / "hermes" / "vscode_version.json"

_COPILOT_CHAT_VERSION_FALLBACK = "0.26.7"
_COPILOT_CHAT_MARKETPLACE_URL = (
    "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery"
)
_COPILOT_CHAT_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "copilot_chat_version.json"
)

# X-GitHub-Api-Version sent on Copilot API calls. Sourced (in priority order)
# from the locally-installed `@github/copilot` npm bundle, which bakes it in
# as a constant and is updated whenever the user runs `npm i -g @github/copilot`.
# Fallback is the value shipped by @github/copilot @ 1.0.57 (today's date).
_COPILOT_API_VERSION_FALLBACK = "2026-06-01"
_COPILOT_API_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "copilot_api_version.json"
)

# Copilot-Integration-Id sent on Copilot API inference calls. The official
# @github/copilot CLI uses "copilot-cli"; verified live (2026-06-07, account
# e126380_magh) this integrator exposes the FULL model catalog — including
# gemini-3.1-pro-preview / gemini-3.5-flash at 1M context — and the account's
# true per-model limits and reasoning-effort range (opus low..max). The legacy
# "vscode-chat" value hides gemini-3.x from the catalog and is not what a CLI
# agent should present as. Override via HERMES_COPILOT_INTEGRATION_ID when a
# different integrator is required (e.g. copilot-developer-cli, vscode-chat).
_COPILOT_INTEGRATION_ID_DEFAULT = "copilot-cli"


def _copilot_integration_id() -> str:
    """Return the Copilot-Integration-Id to send (env-overridable)."""
    override = os.getenv("HERMES_COPILOT_INTEGRATION_ID", "").strip()
    return override or _COPILOT_INTEGRATION_ID_DEFAULT
# Candidate paths for the @github/copilot CLI bundle (global npm install).
_COPILOT_CLI_BUNDLE_CANDIDATES = (
    "/usr/local/lib/node_modules/@github/copilot/sdk/index.js",
    "/usr/lib/node_modules/@github/copilot/sdk/index.js",
)

# In-process caches so we don't hit disk on every header build.
_vscode_version_memo: tuple[str, float] | None = None
_copilot_chat_version_memo: tuple[str, float] | None = None
_copilot_api_version_memo: tuple[str, float] | None = None


def _latest_vscode_version() -> str:
    """Return the latest stable VS Code version (e.g. ``1.104.1``).

    Resolution order:
      1. ``HERMES_VSCODE_VERSION`` env override (if set).
      2. In-process memo (TTL ``_VSCODE_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_VSCODE_VERSION_CACHE_PATH`` (same TTL).
      4. ``GET https://api.github.com/repos/microsoft/vscode/releases/latest``.
      5. Hard fallback ``_VSCODE_VERSION_FALLBACK``.

    Network failures are swallowed; we always return *something*.
    """
    override = os.getenv("HERMES_VSCODE_VERSION", "").strip()
    if override:
        return override

    global _vscode_version_memo
    now = time.time()
    if _vscode_version_memo and now - _vscode_version_memo[1] < _VSCODE_VERSION_CACHE_TTL:
        return _vscode_version_memo[0]

    cache_path = _VSCODE_VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VSCODE_VERSION_CACHE_TTL:
                _vscode_version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("vscode version cache read failed: %s", exc)

    ver = _VSCODE_VERSION_FALLBACK
    try:
        import urllib.request

        req = urllib.request.Request(
            _VSCODE_RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "vscode",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode())
        tag = str(payload.get("tag_name") or "").lstrip("v").strip()
        if tag:
            ver = tag
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now})
                )
            except Exception as exc:
                logger.debug("vscode version cache write failed: %s", exc)
    except Exception as exc:
        logger.debug(
            "failed to fetch latest VS Code version, using fallback %s: %s",
            _VSCODE_VERSION_FALLBACK,
            exc,
        )

    _vscode_version_memo = (ver, now)
    return ver


def _latest_copilot_chat_version() -> str:
    """Return the latest published GitHub.copilot-chat extension version.

    Resolution order mirrors :func:`_latest_vscode_version`:
      1. ``HERMES_COPILOT_CHAT_VERSION`` env override.
      2. In-process memo (TTL ``_VSCODE_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_COPILOT_CHAT_VERSION_CACHE_PATH``.
      4. VS Marketplace extensionquery API.
      5. Hard fallback ``_COPILOT_CHAT_VERSION_FALLBACK``.
    """
    override = os.getenv("HERMES_COPILOT_CHAT_VERSION", "").strip()
    if override:
        return override

    global _copilot_chat_version_memo
    now = time.time()
    if (
        _copilot_chat_version_memo
        and now - _copilot_chat_version_memo[1] < _VSCODE_VERSION_CACHE_TTL
    ):
        return _copilot_chat_version_memo[0]

    cache_path = _COPILOT_CHAT_VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VSCODE_VERSION_CACHE_TTL:
                _copilot_chat_version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("copilot-chat version cache read failed: %s", exc)

    ver = _COPILOT_CHAT_VERSION_FALLBACK
    try:
        import urllib.request

        body = json.dumps({
            "filters": [{
                "criteria": [{"filterType": 7, "value": "GitHub.copilot-chat"}],
            }],
            "flags": 914,
        }).encode()
        req = urllib.request.Request(
            _COPILOT_CHAT_MARKETPLACE_URL,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json;api-version=7.2-preview.1;excludeUrls=true",
                "Content-Type": "application/json",
                "User-Agent": "VSCode",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode())
        latest = (
            payload.get("results", [{}])[0]
            .get("extensions", [{}])[0]
            .get("versions", [{}])[0]
            .get("version", "")
        )
        latest = str(latest).lstrip("v").strip()
        if latest:
            ver = latest
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now})
                )
            except Exception as exc:
                logger.debug("copilot-chat version cache write failed: %s", exc)
    except Exception as exc:
        logger.debug(
            "failed to fetch latest copilot-chat version, using fallback %s: %s",
            _COPILOT_CHAT_VERSION_FALLBACK,
            exc,
        )

    _copilot_chat_version_memo = (ver, now)
    return ver


def _discover_copilot_cli_bundles() -> list[Path]:
    """Locate plausible ``@github/copilot/sdk/index.js`` paths on this host.

    Walks common global-npm locations (incl. nvm). Returns existing files only.
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen or not rp.is_file():
            return
        seen.add(rp)
        out.append(rp)

    # Static candidates.
    for s in _COPILOT_CLI_BUNDLE_CANDIDATES:
        _add(Path(s))

    # nvm-managed installs: ~/.nvm/versions/node/*/lib/node_modules/@github/copilot/sdk/index.js
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        try:
            for node_dir in nvm_root.iterdir():
                _add(node_dir / "lib" / "node_modules" / "@github" / "copilot" / "sdk" / "index.js")
        except Exception as exc:
            logger.debug("nvm scan failed: %s", exc)

    # User-local global install (npm prefix override).
    _add(Path.home() / ".npm-global" / "lib" / "node_modules" / "@github" / "copilot" / "sdk" / "index.js")

    return out


def _extract_api_version_from_bundle(bundle: Path) -> str | None:
    """Grep the Copilot CLI bundle for the X-GitHub-Api-Version constant.

    The bundle defines it as e.g. ``Mss="X-GitHub-Api-Version",Oss="2026-06-01"``.
    We extract every adjacent date literal, drop the github.com REST date
    ``2022-11-28`` (used only for gist/asset uploads), and return the newest.
    """
    import re
    try:
        text = bundle.read_text(errors="ignore")
    except Exception as exc:
        logger.debug("copilot CLI bundle read failed (%s): %s", bundle, exc)
        return None
    matches = re.findall(
        r'"X-GitHub-Api-Version"\s*,\s*[A-Za-z0-9_$]+\s*=\s*"(\d{4}-\d{2}-\d{2})"',
        text,
    )
    # Filter out the github.com REST API version (different surface).
    candidates = sorted({m for m in matches if m != "2022-11-28"}, reverse=True)
    return candidates[0] if candidates else None


def _latest_copilot_api_version() -> str:
    """Return the X-GitHub-Api-Version value used by the Copilot API.

    Resolution order:
      1. ``HERMES_COPILOT_API_VERSION`` env override.
      2. In-process memo (TTL ``_VSCODE_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_COPILOT_API_VERSION_CACHE_PATH``.
      4. Local ``@github/copilot`` npm bundle (the live source of truth —
         updates whenever the user runs ``npm i -g @github/copilot``).
      5. Hard fallback ``_COPILOT_API_VERSION_FALLBACK``.
    """
    override = os.getenv("HERMES_COPILOT_API_VERSION", "").strip()
    if override:
        return override

    global _copilot_api_version_memo
    now = time.time()
    if (
        _copilot_api_version_memo
        and now - _copilot_api_version_memo[1] < _VSCODE_VERSION_CACHE_TTL
    ):
        return _copilot_api_version_memo[0]

    cache_path = _COPILOT_API_VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VSCODE_VERSION_CACHE_TTL:
                _copilot_api_version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("copilot api-version cache read failed: %s", exc)

    ver = _COPILOT_API_VERSION_FALLBACK
    for bundle in _discover_copilot_cli_bundles():
        extracted = _extract_api_version_from_bundle(bundle)
        if extracted:
            ver = extracted
            logger.debug("copilot api-version %s from %s", ver, bundle)
            break
    else:
        logger.debug(
            "no @github/copilot bundle found, using fallback api-version %s",
            _COPILOT_API_VERSION_FALLBACK,
        )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"version": ver, "fetched_at": now}))
    except Exception as exc:
        logger.debug("copilot api-version cache write failed: %s", exc)

    _copilot_api_version_memo = (ver, now)
    return ver


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def exchange_copilot_token(raw_token: str, *, timeout: float = 10.0) -> tuple[str, float]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    import urllib.request

    fp = _token_fingerprint(raw_token)

    # Check cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at

    req = urllib.request.Request(
        _TOKEN_EXCHANGE_URL,
        method="GET",
        headers={
            "Authorization": f"Bearer {raw_token}",
            "User-Agent": f"GitHubCopilotChat/{_latest_copilot_chat_version()}",
            "Accept": "application/json",
            "Editor-Version": f"vscode/{_latest_vscode_version()}",
            "Copilot-Integration-Id": "vscode-chat",
            "X-GitHub-Api-Version": "2026-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise ValueError(f"Copilot token exchange failed: {exc}") from exc

    api_token = data.get("token", "")
    expires_at = data.get("expires_at", 0)
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")

    # Convert expires_at to float if needed
    expires_at = float(expires_at) if expires_at else time.time() + 1800

    _jwt_cache[fp] = (api_token, expires_at)
    logger.debug(
        "Copilot token exchanged, expires_at=%s",
        expires_at,
    )
    return api_token, expires_at


def get_copilot_api_token(raw_token: str) -> str:
    """Return the API token to use against ``api.githubcopilot.com``.

    The Copilot API accepts the raw GitHub OAuth/PAT token directly as
    ``Authorization: Bearer <token>`` — no exchange step is required.
    This was verified against the official ``@github/copilot`` CLI bundle:
    its SDK calls Copilot endpoints with ``Bearer <gh-token>`` directly.

    The legacy ``GET /copilot_internal/v2/token`` exchange endpoint on
    ``api.github.com`` is not used by the CLI and now returns 404 (the
    REST router treats ``/copilot_internal/...`` as a repo sub-path).

    Set ``HERMES_COPILOT_FORCE_EXCHANGE=1`` to opt in to the legacy
    exchange flow (will fall back to the raw token on failure).
    """
    if not raw_token:
        return raw_token
    if os.getenv("HERMES_COPILOT_FORCE_EXCHANGE", "").strip() in ("1", "true", "yes"):
        try:
            api_token, _ = exchange_copilot_token(raw_token)
            return api_token
        except Exception as exc:
            logger.debug("Copilot token exchange failed, using raw token: %s", exc)
    return raw_token


# ─── Copilot API Headers ───────────────────────────────────────────────────

def copilot_request_headers(
    *,
    is_agent_turn: bool = True,
    is_vision: bool = False,
    model: str = "",
    intent: str = "conversation-panel",
    interaction_id: Optional[str] = None,
) -> dict[str, str]:
    """Build the standard headers for Copilot API requests.

    Replicates the header set used by the github.copilot-chat extension
    in VS Code Insiders (RE 2026-06-04, Worker-A wave1 findings).
    """
    chat_ver = _latest_copilot_chat_version()
    import uuid as _uuid
    headers: dict[str, str] = {
        "Editor-Version": f"vscode/{_latest_vscode_version()}",
        "Editor-Plugin-Version": f"copilot-chat/{chat_ver}",
        "User-Agent": "rest-book",
        "Copilot-Integration-Id": _copilot_integration_id(),
        "Openai-Intent": intent,
        # Mirror of Openai-Intent (extension sends both unless overridden).
        "X-Interaction-Type": intent,
        "X-GitHub-Api-Version": _latest_copilot_api_version(),
        "x-initiator": "agent" if is_agent_turn else "user",
        # Per-call request id + stable per-session interaction id (Worker-A
        # RE: the chat extension always sets both, server uses them for
        # trace/log correlation and may key some quotas off X-Interaction-Id).
        "X-Request-Id": str(_uuid.uuid4()),
        "X-Interaction-Id": interaction_id or str(_uuid.uuid4()),
    }

    # NOTE: hermes previously injected `X-Copilot-Agent-Slug: copilot-1m-context`
    # here, believing it mapped the token to the developer-app integrator and
    # unlocked 1M context / Gemini-3.x. Live probing (2026-06-07) proved that
    # slug is INERT — it changes neither catalog visibility nor per-model limits.
    # What actually exposes gemini-3.x and the full limits is the
    # Copilot-Integration-Id (now `copilot-cli`, matching the official CLI). The
    # slug was removed to avoid sending a misleading no-op header. The official
    # @github/copilot CLI sends `copilot-developer-sandbox` only on specific
    # (non-inference) endpoints; we don't need it for chat/messages/responses.

    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers
