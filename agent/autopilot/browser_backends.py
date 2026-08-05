"""Browser backend auto-detection — the `browser` modality's backend layer (§3.3).

The `browser` probe must observe the RENDERED page (DOM/console/screenshot — the
source-vs-render trap). Several drivers can do that; this module picks the best AVAILABLE
one in the owner-specified preference order, **always preferring camoufox** for its
anti-bot-detection, and degrades fail-soft (no driver → the caller downgrades, never a
crash).

Preference order (design §3.3):
  1. camoufox REST shim   — ``CAMOFOX_URL`` set + ``{url}/health`` 200 (anti-detection,
                            Hermes-native). ``BROWSER_CDP_URL`` overrides it (CDP wins),
                            exactly as ``tools/browser_camofox.py`` does.
  2. camoufox via API/CLI — the python ``camoufox`` lib if importable (absent on this
                            host's venv today; recorded for completeness).
  3. playwright+chromium  — node + the global playwright module (the proven path today).
  4. selenium+webdriver   — selenium + geckodriver/firefox or chromedriver/chromium
                            (last-resort).

This module is detection-only: it answers "which backend, and is it reachable?" The
actual drive logic per backend lives in probes.py (playwright today; camoufox/selenium
drivers are additive). Detection never raises — every failure path returns a lower rung
or ``NONE``.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Backend identifiers (ordered best→worst).
CAMOFOX_SHIM = "camofox_shim"        # the REST server (anti-detection, preferred)
CAMOUFOX_LIB = "camoufox_lib"        # python camoufox lib (API/CLI)
PLAYWRIGHT = "playwright"            # node + playwright chromium (proven path)
SELENIUM = "selenium"               # selenium + geckodriver/chromedriver
NONE = "none"                       # nothing available → caller downgrades

_PREFERENCE = (CAMOFOX_SHIM, CAMOUFOX_LIB, PLAYWRIGHT, SELENIUM)

# Global playwright module candidate (the verified-present path; same as probes.py).
_PW_INDEX_CANDIDATES = (
    os.path.expanduser("~/.nvm/versions/node/v22.22.3/lib/node_modules/playwright/index.js"),
)


@dataclass
class BrowserBackend:
    """The selected browser backend and how to reach it."""

    kind: str                          # one of the identifiers above
    detail: str = ""                   # human-readable: url / path / driver
    camofox_url: str = ""              # for CAMOFOX_SHIM
    playwright_index: str = ""         # for PLAYWRIGHT
    webdriver: str = ""                # for SELENIUM: 'geckodriver' | 'chromedriver'

    @property
    def available(self) -> bool:
        return self.kind != NONE

    @property
    def anti_detection(self) -> bool:
        """True when the chosen backend is camoufox (the anti-bot-detection path)."""
        return self.kind in (CAMOFOX_SHIM, CAMOUFOX_LIB)


# --------------------------------------------------------------------------- #
# per-backend availability probes (each fail-soft)                              #
# --------------------------------------------------------------------------- #
def _camofox_url() -> str:
    """The configured camoufox REST URL, or '' — honoring the CDP override (CDP wins)."""
    # BROWSER_CDP_URL takes priority over camoufox, exactly as tools/browser_camofox.py
    # does: when the user explicitly connected a real CDP browser, don't shadow it.
    if os.environ.get("BROWSER_CDP_URL", "").strip():
        return ""
    return os.environ.get("CAMOFOX_URL", "").strip().rstrip("/")


def _camofox_shim_reachable(url: str, *, timeout: float = 4.0) -> bool:
    """Health-check the camoufox shim (GET {url}/health == 200). Fail-soft."""
    if not url:
        return False
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/health", headers={"User-Agent": "HermesAutopilot/probe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - operator-configured URL
            return resp.status == 200
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: camofox shim health-check failed (%s)", exc)
        return False


def _camoufox_lib_available() -> bool:
    """The python camoufox lib, if importable (absent on this host today)."""
    try:
        import importlib.util

        return importlib.util.find_spec("camoufox") is not None
    except Exception:  # noqa: BLE001
        return False


def _playwright_index() -> Optional[str]:
    """Locate the global playwright module index.js (node driver). Fail-soft."""
    if not shutil.which("node"):
        return None
    for p in _PW_INDEX_CANDIDATES:
        if os.path.exists(p):
            return p
    try:
        import subprocess

        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True,
                              timeout=10, stdin=subprocess.DEVNULL).stdout.strip()
        cand = os.path.join(root, "playwright", "index.js")
        if os.path.exists(cand):
            return cand
    except Exception:  # noqa: BLE001
        pass
    return None


def _selenium_webdriver() -> Optional[str]:
    """Return an available selenium webdriver name, or None. Prefers gecko (firefox)."""
    try:
        import importlib.util

        if importlib.util.find_spec("selenium") is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    if shutil.which("geckodriver") and (shutil.which("firefox") or shutil.which("firefox-esr")):
        return "geckodriver"
    if shutil.which("chromedriver") and (shutil.which("chromium") or shutil.which("chromium-browser")
                                         or shutil.which("google-chrome")):
        return "chromedriver"
    return None


# --------------------------------------------------------------------------- #
# the detection ladder                                                         #
# --------------------------------------------------------------------------- #
def detect_browser_backend(*, prefer: tuple = _PREFERENCE) -> BrowserBackend:
    """Pick the best AVAILABLE browser backend in the owner-specified order.

    Always prefers camoufox (anti-detection). Returns ``BrowserBackend(kind=NONE)`` when
    nothing is reachable, so the browser probe DOWNGRADES (never crashes). The ``prefer``
    order can be overridden (e.g. to force a specific backend in a test).
    """
    for kind in prefer:
        if kind == CAMOFOX_SHIM:
            url = _camofox_url()
            if url and _camofox_shim_reachable(url):
                return BrowserBackend(CAMOFOX_SHIM, detail=url, camofox_url=url)
        elif kind == CAMOUFOX_LIB:
            if _camoufox_lib_available():
                return BrowserBackend(CAMOUFOX_LIB, detail="python camoufox lib")
        elif kind == PLAYWRIGHT:
            idx = _playwright_index()
            if idx:
                return BrowserBackend(PLAYWRIGHT, detail=idx, playwright_index=idx)
        elif kind == SELENIUM:
            drv = _selenium_webdriver()
            if drv:
                return BrowserBackend(SELENIUM, detail=drv, webdriver=drv)
    return BrowserBackend(NONE, detail="no browser backend reachable")


def describe_backends() -> dict:
    """A diagnostic snapshot of which backends are present (for ADR/doctor)."""
    return {
        CAMOFOX_SHIM: bool(_camofox_url() and _camofox_shim_reachable(_camofox_url())),
        CAMOUFOX_LIB: _camoufox_lib_available(),
        PLAYWRIGHT: bool(_playwright_index()),
        SELENIUM: bool(_selenium_webdriver()),
    }
