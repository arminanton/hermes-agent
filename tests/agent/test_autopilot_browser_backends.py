"""Tests for browser backend auto-detection (§3.3) and the camoufox-shim driver.

Two layers:
  1. ``browser_backends.detect_browser_backend`` picks the best AVAILABLE backend in the
     owner-specified order (camoufox shim → camoufox lib → playwright → selenium), and
     downgrades to NONE fail-soft. Tested by stubbing each rung's availability.
  2. ``probes._run_browser_camofox`` drives the camoufox REST shim. Tested against a
     FAITHFUL in-process mock HTTP server implementing the shim's documented endpoints
     (POST /tabs, POST /tabs/{id}/navigate, GET /tabs/{id}/snapshot, GET .../screenshot,
     GET /health) — so we prove the driver against the real contract without the live
     Docker shim, the same discipline as the in-memory provenance store.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.autopilot import browser_backends as BB
from agent.autopilot import probes as P


# --------------------------------------------------------------------------- #
# 1. detection ladder — preference order + fail-soft                            #
# --------------------------------------------------------------------------- #
def _stub_all_absent(monkeypatch):
    monkeypatch.setattr(BB, "_camofox_url", lambda: "")
    monkeypatch.setattr(BB, "_camoufox_lib_available", lambda: False)
    monkeypatch.setattr(BB, "_playwright_index", lambda: None)
    monkeypatch.setattr(BB, "_selenium_webdriver", lambda: None)


def test_camofox_shim_is_preferred(monkeypatch):
    _stub_all_absent(monkeypatch)
    monkeypatch.setattr(BB, "_camofox_url", lambda: "http://localhost:9377")
    monkeypatch.setattr(BB, "_camofox_shim_reachable", lambda url, timeout=4.0: True)
    # even with playwright ALSO present, camoufox wins (anti-detection preference)
    monkeypatch.setattr(BB, "_playwright_index", lambda: "/x/playwright/index.js")
    b = BB.detect_browser_backend()
    assert b.kind == BB.CAMOFOX_SHIM
    assert b.anti_detection is True
    assert b.camofox_url == "http://localhost:9377"


def test_cdp_override_suppresses_camofox(monkeypatch):
    # BROWSER_CDP_URL set → camoufox is NOT chosen (CDP wins, as the shim itself does)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://localhost:9222")
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    assert BB._camofox_url() == ""           # the override blanks camoufox


def test_falls_through_to_playwright(monkeypatch):
    _stub_all_absent(monkeypatch)
    monkeypatch.setattr(BB, "_playwright_index", lambda: "/x/playwright/index.js")
    b = BB.detect_browser_backend()
    assert b.kind == BB.PLAYWRIGHT
    assert b.anti_detection is False
    assert b.playwright_index.endswith("index.js")


def test_falls_through_to_selenium(monkeypatch):
    _stub_all_absent(monkeypatch)
    monkeypatch.setattr(BB, "_selenium_webdriver", lambda: "geckodriver")
    b = BB.detect_browser_backend()
    assert b.kind == BB.SELENIUM
    assert b.webdriver == "geckodriver"


def test_none_when_all_absent_is_failsoft(monkeypatch):
    _stub_all_absent(monkeypatch)
    b = BB.detect_browser_backend()
    assert b.kind == BB.NONE
    assert b.available is False             # → caller downgrades, never crashes


def test_camofox_unreachable_skips_to_next(monkeypatch):
    # CAMOFOX_URL set but the health check FAILS → skip to playwright (don't pick a dead shim)
    _stub_all_absent(monkeypatch)
    monkeypatch.setattr(BB, "_camofox_url", lambda: "http://localhost:9377")
    monkeypatch.setattr(BB, "_camofox_shim_reachable", lambda url, timeout=4.0: False)
    monkeypatch.setattr(BB, "_playwright_index", lambda: "/x/playwright/index.js")
    b = BB.detect_browser_backend()
    assert b.kind == BB.PLAYWRIGHT


def test_describe_backends_shape(monkeypatch):
    _stub_all_absent(monkeypatch)
    monkeypatch.setattr(BB, "_playwright_index", lambda: "/x/index.js")
    snap = BB.describe_backends()
    assert snap[BB.PLAYWRIGHT] is True
    assert snap[BB.SELENIUM] is False
    assert set(snap) == {BB.CAMOFOX_SHIM, BB.CAMOUFOX_LIB, BB.PLAYWRIGHT, BB.SELENIUM}


# --------------------------------------------------------------------------- #
# 2. camoufox-shim driver — against a faithful mock of the REST contract        #
# --------------------------------------------------------------------------- #
class _CamofoxMock(BaseHTTPRequestHandler):
    """A faithful in-process mock of the camofox-browser REST shim.

    Serves the endpoints probes._run_browser_camofox calls. The 'page' it renders is
    controlled by the class attr ``snapshot_text`` so a test can simulate a working vs
    broken page (the source-vs-render observation).
    """

    snapshot_text = "counter value: 3 — OK"   # default: a 'good' rendered page

    def log_message(self, format, *args):  # silence  # noqa: A002
        pass

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if "/snapshot" in self.path:
            return self._send(200, {"snapshot": type(self).snapshot_text})
        if "/screenshot" in self.path:
            return self._send(200, b"\x89PNG\r\n\x1a\nFAKE", content_type="image/png")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        if self.path == "/tabs":
            return self._send(200, {"tabId": "tab-123"})
        if "/navigate" in self.path or "/click" in self.path:
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


@pytest.fixture()
def camofox_server():
    srv = HTTPServer(("127.0.0.1", 0), _CamofoxMock)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield url
    srv.shutdown()


@pytest.fixture()
def wd(tmp_path):
    return str(tmp_path)


def test_camofox_driver_observes_good_page(camofox_server, wd):
    _CamofoxMock.snapshot_text = "counter value: 3 — OK"
    spec = P.ProbeSpec(kind=P.BROWSER, target="http://example.test/page",
                       assert_text="counter value: 3", screenshot=True, criterion_id="C1")
    r = P._run_browser_camofox(spec, camofox_server, timeout=10, workdir=wd)
    assert r.status == P.PASS
    assert "[camoufox]" in r.summary
    assert r.screenshot_path and r.screenshot_path.endswith(".png")
    assert r.observed["backend"] == "camofox_shim"


def test_camofox_driver_observes_broken_page(camofox_server, wd):
    # the rendered page does NOT contain the expected text → FAIL (source-vs-render catch)
    _CamofoxMock.snapshot_text = "counter value: 0 — handler threw"
    spec = P.ProbeSpec(kind=P.BROWSER, target="http://example.test/page",
                       assert_text="counter value: 3", screenshot=False, criterion_id="C1")
    r = P._run_browser_camofox(spec, camofox_server, timeout=10, workdir=wd)
    assert r.status == P.FAIL
    assert "present=False" in r.summary


def test_camofox_driver_downgrades_on_dead_shim(wd):
    # shim URL points nowhere → the driver returns UNAVAILABLE (downgrade), never crashes
    spec = P.ProbeSpec(kind=P.BROWSER, target="http://example.test/page",
                       assert_text="x", criterion_id="C1")
    r = P._run_browser_camofox(spec, "http://127.0.0.1:1", timeout=2, workdir=wd)
    assert r.status == P.UNAVAILABLE
    assert r.is_downgrade is True


def test_run_browser_routes_to_camofox_when_detected(camofox_server, wd, monkeypatch):
    # end-to-end: when detection picks the camoufox shim, _run_browser routes to it.
    monkeypatch.setattr(P, "browser_backend",
                        lambda: BB.BrowserBackend(BB.CAMOFOX_SHIM, detail=camofox_server,
                                                  camofox_url=camofox_server))
    _CamofoxMock.snapshot_text = "hello rendered world"
    spec = P.ProbeSpec(kind=P.BROWSER, target="http://example.test/x",
                       assert_text="rendered", screenshot=False, criterion_id="C1")
    r = P.run_probe(spec, workdir=wd)
    assert r.status == P.PASS
    assert "[camoufox]" in r.summary
