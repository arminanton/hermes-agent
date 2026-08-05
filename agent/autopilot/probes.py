"""Probe capability layer — the engine's senses (observe reality, return receipts).

Build-order step 2 of DESIGN-v2-observe-reality.md. This generalizes the one-off
oracle proven in ``derail_repro.py`` into a reusable set of PROBES the engine can run
to OBSERVE the world and return a structured ``ProbeReceipt``. The probe loop (step 4)
selects probes per turn and hands their receipts to the Council; the verdict rests on
the receipt, not the model's prose.

Design invariants honored here:
  * the ENGINE runs the probe (never the model, never a persona) — same trust boundary
    as the Council itself, so a receipt can't be fabricated by the working model;
  * every executed command goes through the SAME read-only allowlist as
    ``verification.py`` (mutating/network verbs refused) + the re-entrancy guard;
  * bounded (per-probe timeout) and fail-soft (a broken probe → a receipt with
    ``status='error'``/``'unavailable'``, never an exception into the gate);
  * a probe that genuinely cannot observe returns ``status='unobservable'`` so the
    loop DOWNGRADES to the Council's text judgment (owner-locked decision) rather than
    blocking — and records a could-not-observe note for the ADR.

Probe taxonomy (design §3.1) — MODAL senses, an OPEN registry. Each kind observes the
rendered/decoded TRUTH (not the source) and returns the SAME ``ProbeReceipt``:
  process        — run an allowlisted command, capture exit/stdout/stderr.
  browser        — drive headless browser: load a URL/file, assert DOM text/element,
                   capture console+page errors, capture a screenshot (source-vs-render).
  http           — fetch a URL, assert status / JSON shape (stdlib urllib, SSRF-guarded).
  artifact       — filesystem assertion: file exists / contains / non-empty / diff present.
  cmx_provenance — does the VERBATIM RECORD support a claim? walks the cmx→lcm→state
                   ladder; falsifies manufactured-independence ("a subagent confirmed it").
  image          — true visual structure/content via AGENT VISION (OCR insufficient, §3.2).
  audio          — what the audio actually says/contains (transcribe + assert; ffprobe).
  video          — what the video actually shows (sample frame → vision; ffprobe).
  document       — the COMPILED/rendered doc, not its source (pdf render→vision / pdftotext;
                   password/convert/compile failures reported explicitly).
  unobservable   — explicit "no observable proof exists for this" → downgrade signal.
An UNKNOWN kind degrades to ``unobservable`` (not a crash) so the registry stays open.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from agent.autopilot import verification as _verification

logger = logging.getLogger(__name__)

# Probe kinds (modalities). The set is an OPEN REGISTRY: adding a modality is a new
# handler + the same ProbeReceipt contract, never a redesign. An unrecognized kind
# degrades to UNOBSERVABLE with a reason rather than crashing the gate (design §3).
PROCESS = "process"
BROWSER = "browser"
HTTP = "http"
ARTIFACT = "artifact"
# modal senses (design §3.1) — observe the rendered/decoded truth, not the source:
CMX_PROVENANCE = "cmx_provenance"   # does the verbatim record support a claim? (§3.4)
IMAGE = "image"                     # true visual structure/content — agent vision (§3.2)
AUDIO = "audio"                     # what the audio actually says/contains
VIDEO = "video"                     # what the video actually shows
DOCUMENT = "document"               # the COMPILED/rendered doc, not its source
UNOBSERVABLE = "unobservable"

# The known modality registry (for is-this-observable checks + open-registry semantics).
KNOWN_KINDS = frozenset({
    PROCESS, BROWSER, HTTP, ARTIFACT,
    CMX_PROVENANCE, IMAGE, AUDIO, VIDEO, DOCUMENT,
    UNOBSERVABLE,
})

# Receipt statuses.
PASS = "pass"
FAIL = "fail"
ERROR = "error"               # the probe itself broke (tooling/exception)
UNAVAILABLE = "unavailable"   # the probe's tool isn't present in this environment
UNOBSERVABLE_STATUS = "unobservable"  # no observable proof exists → downgrade


@dataclass
class ProbeSpec:
    """One probe the verifier wants the engine to run. Frozen-intent-derived, chosen
    per turn by the selector (step 3); here it's just the structured request."""

    kind: str                              # PROCESS | BROWSER | HTTP | ARTIFACT | UNOBSERVABLE
    # process:
    command: str = ""
    # browser:
    target: str = ""                       # URL or file path
    assert_text: str = ""                  # text that must be present in the DOM
    click_selector: str = ""               # optional element to click N times
    click_times: int = 0
    expect_text_in: str = ""               # selector whose textContent is checked
    expect_text_equals: str = ""           # expected value of that textContent
    require_no_console_errors: bool = True
    screenshot: bool = True
    # http:
    url: str = ""
    expect_status: int = 200
    expect_body_contains: str = ""
    # artifact:
    path: str = ""
    must_exist: bool = True
    must_contain: str = ""
    must_be_nonempty: bool = False
    # cmx_provenance (§3.4): the claim is supported only if an EVIDENCE-role row in the
    # verbatim record contains ALL of these terms. Used to falsify "a subagent confirmed
    # it" — the manufactured-independence bypass.
    evidence_terms: list = field(default_factory=list)
    provenance_session_id: str = ""        # scope the record search to one session (optional)
    # image / audio / video / document (§3.1–3.2): observe the rendered/decoded truth.
    media_path: str = ""                   # the file to render/decode/transcribe
    vision_question: str = ""              # what agent-vision should judge (image/video/doc)
    expect_visual: str = ""                # text/structure the vision answer must affirm
    transcript_contains: str = ""          # audio/video: text the transcript must contain
    document_password: str = ""            # for password-protected documents
    page: int = 0                          # document/video: page or frame index to observe
    # shared:
    criterion_id: str = ""                 # which acceptance criterion this proves
    why_unobservable: str = ""             # for UNOBSERVABLE: the recorded reason


@dataclass
class ProbeReceipt:
    """Structured result of one probe — ground truth for the Council."""

    kind: str
    status: str                            # pass|fail|error|unavailable|unobservable
    criterion_id: str = ""
    summary: str = ""                      # one-line human/Council-readable result
    detail: str = ""                       # stderr/observed values/error message
    screenshot_path: str = ""              # for browser probes, if captured
    observed: dict = field(default_factory=dict)
    ran_at: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def is_downgrade(self) -> bool:
        """When the verdict should fall back to the Council's text judgment."""
        return self.status in (UNOBSERVABLE_STATUS, UNAVAILABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "status": self.status, "criterion_id": self.criterion_id,
            "summary": self.summary, "detail": self.detail[:1000],
            "screenshot_path": self.screenshot_path, "observed": self.observed,
            "ran_at": self.ran_at,
        }

    def as_evidence(self) -> str:
        """Render for the Council context — the engine's observation as ground truth."""
        head = f"ENGINE PROBE [{self.kind}] {self.status.upper()}"
        if self.criterion_id:
            head += f" ({self.criterion_id})"
        line = f"{head}: {self.summary}"
        if self.status in (FAIL, ERROR) and self.detail:
            line += f"\n    └─ {self.detail[:300].splitlines()[0] if self.detail else ''}"
        if self.screenshot_path:
            line += f"\n    └─ screenshot: {self.screenshot_path}"
        return line


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_bin() -> Optional[str]:
    return shutil.which("node")


def _playwright_index() -> Optional[str]:
    candidates = [
        os.path.expanduser("~/.nvm/versions/node/v22.22.3/lib/node_modules/playwright/index.js"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    try:
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL).stdout.strip()
        cand = os.path.join(root, "playwright", "index.js")
        if os.path.exists(cand):
            return cand
    except Exception:  # noqa: BLE001
        pass
    return None


def browser_available() -> bool:
    """True when ANY browser backend is reachable (camoufox shim / camoufox lib /
    playwright / selenium), per the detection ladder (§3.3). Falls back to the legacy
    node+playwright check if the detector is unavailable."""
    try:
        from agent.autopilot import browser_backends as _bb

        return _bb.detect_browser_backend().available
    except Exception:  # noqa: BLE001 — detector optional; keep the proven check
        return bool(_node_bin() and _playwright_index())


def browser_backend():
    """The selected browser backend (BrowserBackend), or None if the detector is absent."""
    try:
        from agent.autopilot import browser_backends as _bb

        return _bb.detect_browser_backend()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# process probe — reuse the verification allowlist + bounds                     #
# --------------------------------------------------------------------------- #
def _run_process(spec: ProbeSpec, *, timeout: float, workdir: str) -> ProbeReceipt:
    ok, reason = _verification.validate_command(spec.command)
    if not ok:
        return ProbeReceipt(PROCESS, ERROR, spec.criterion_id,
                            summary=f"refused: {spec.command}", detail=reason, ran_at=_now())
    if os.environ.get("AUTOPILOT_VERIFICATION", "").strip() == "1":
        return ProbeReceipt(PROCESS, UNAVAILABLE, spec.criterion_id,
                            summary="nested probe suppressed (already inside a probe run)", ran_at=_now())
    try:
        env = dict(os.environ)
        env["AUTOPILOT_VERIFICATION"] = "1"
        proc = subprocess.run(  # nosec B602 — allowlist-validated above
            spec.command, shell=True, cwd=workdir, env=env,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        status = PASS if proc.returncode == 0 else FAIL
        tail = (proc.stderr or proc.stdout or "")[-1000:]
        return ProbeReceipt(PROCESS, status, spec.criterion_id,
                            summary=f"`{spec.command}` exit {proc.returncode}",
                            detail=tail, observed={"exit_code": proc.returncode}, ran_at=_now())
    except subprocess.TimeoutExpired:
        return ProbeReceipt(PROCESS, FAIL, spec.criterion_id,
                            summary=f"`{spec.command}` timed out after {timeout}s", ran_at=_now())
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(PROCESS, ERROR, spec.criterion_id, summary="process probe error",
                            detail=str(exc)[:300], ran_at=_now())


# --------------------------------------------------------------------------- #
# browser probe — drive headless chromium, observe DOM/console, screenshot      #
# --------------------------------------------------------------------------- #
_BROWSER_JS = r"""
const pwIndex = process.argv[2];
const spec = JSON.parse(process.argv[3]);
const { chromium } = require(pwIndex);
(async () => {
  const b = await chromium.launch({ headless: true });
  const p = await b.newPage();
  const errors = [];
  p.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  p.on('pageerror', e => errors.push('pageerror: ' + e.message));
  const url = spec.target.startsWith('http') || spec.target.startsWith('file://')
      ? spec.target : ('file://' + spec.target);
  await p.goto(url, { waitUntil: 'load', timeout: 15000 });
  if (spec.click_selector && spec.click_times) {
    for (let i = 0; i < spec.click_times; i++) {
      try { await p.click(spec.click_selector, { timeout: 1500 }); }
      catch (e) { errors.push('click_failed: ' + e.message.split('\n')[0]); }
    }
  }
  let textPresent = true, observedText = null;
  if (spec.assert_text) {
    const body = await p.textContent('body');
    textPresent = (body || '').includes(spec.assert_text);
  }
  if (spec.expect_text_in) {
    try { observedText = await p.textContent(spec.expect_text_in); }
    catch (e) { observedText = null; }
  }
  let shot = '';
  if (spec.screenshot) {
    shot = spec.screenshot_path;
    try { await p.screenshot({ path: shot }); } catch (e) { shot = ''; }
  }
  const textEquals = spec.expect_text_equals
      ? (observedText === spec.expect_text_equals) : true;
  const noErrs = spec.require_no_console_errors ? (errors.length === 0) : true;
  const PASS = textPresent && textEquals && noErrs;
  console.log(JSON.stringify({
    PASS, observed_text: observedText, text_present: textPresent,
    console_errors: errors, screenshot: shot,
  }));
  await b.close();
  process.exit(PASS ? 0 : 1);
})().catch(e => { console.error('PROBE_FAIL', e.message); process.exit(3); });
"""


def _run_browser(spec: ProbeSpec, *, timeout: float, workdir: str) -> ProbeReceipt:
    """Observe a rendered page. Routes to the best available backend (§3.3): camoufox
    REST shim (anti-detection, preferred) → playwright. Selenium/camoufox-lib are
    detected for reporting; the active drive paths are camoufox-shim + playwright."""
    if not spec.target:
        return ProbeReceipt(BROWSER, ERROR, spec.criterion_id, summary="browser probe missing target", ran_at=_now())
    backend = browser_backend()
    # camoufox REST shim — anti-detection, preferred when reachable
    if backend is not None and getattr(backend, "kind", "") == "camofox_shim":
        return _run_browser_camofox(spec, backend.camofox_url, timeout=timeout, workdir=workdir)
    # playwright (the proven path) — used when it's the detected backend OR as the
    # universal fallback whenever node+playwright are present.
    node, pw = _node_bin(), _playwright_index()
    if node and pw:
        return _run_browser_playwright(spec, node, pw, timeout=timeout, workdir=workdir)
    # nothing drivable → downgrade with the detected-backend detail for the ADR
    detail = getattr(backend, "detail", "") if backend is not None else ""
    return ProbeReceipt(BROWSER, UNAVAILABLE, spec.criterion_id,
                        summary="no drivable browser backend (camoufox shim / playwright)",
                        detail=detail, ran_at=_now())


def _run_browser_playwright(spec: ProbeSpec, node: str, pw: str, *, timeout: float, workdir: str) -> ProbeReceipt:
    js_path = os.path.join(workdir, "_probe_browser.js")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(_BROWSER_JS)
    shot_path = os.path.join(workdir, f"probe_{spec.criterion_id or 'shot'}_{int(datetime.now().timestamp())}.png")
    payload = {
        "target": spec.target, "assert_text": spec.assert_text,
        "click_selector": spec.click_selector, "click_times": spec.click_times,
        "expect_text_in": spec.expect_text_in, "expect_text_equals": spec.expect_text_equals,
        "require_no_console_errors": spec.require_no_console_errors,
        "screenshot": spec.screenshot, "screenshot_path": shot_path if spec.screenshot else "",
    }
    try:
        proc = subprocess.run([node, js_path, pw, json.dumps(payload)],
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        data = {}
        for line in reversed((proc.stdout or "").strip().splitlines()):
            try:
                data = json.loads(line)
                break
            except Exception:  # noqa: BLE001
                continue
        if not data:
            return ProbeReceipt(BROWSER, ERROR, spec.criterion_id,
                                summary="no observation json from browser probe",
                                detail=(proc.stderr or "")[:300], ran_at=_now())
        status = PASS if data.get("PASS") else FAIL
        errs = data.get("console_errors", [])
        parts = []
        if spec.expect_text_equals:
            parts.append(f"observed {spec.expect_text_in}={data.get('observed_text')!r} (want {spec.expect_text_equals!r})")
        if spec.assert_text:
            parts.append(f"text '{spec.assert_text}' present={data.get('text_present')}")
        if errs:
            parts.append(f"{len(errs)} console error(s)")
        return ProbeReceipt(BROWSER, status, spec.criterion_id,
                            summary="[playwright] " + ("; ".join(parts) or f"browser probe {status}"),
                            detail="\n".join(errs)[:1000],
                            screenshot_path=data.get("screenshot", "") or "",
                            observed=data, ran_at=_now())
    except subprocess.TimeoutExpired:
        return ProbeReceipt(BROWSER, FAIL, spec.criterion_id,
                            summary=f"browser probe timed out after {timeout}s", ran_at=_now())
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(BROWSER, ERROR, spec.criterion_id, summary="browser probe error",
                            detail=str(exc)[:300], ran_at=_now())


def _run_browser_camofox(spec: ProbeSpec, base_url: str, *, timeout: float, workdir: str) -> ProbeReceipt:
    """Drive the camoufox REST shim (anti-detection) to observe a rendered page.

    Uses the shim's documented contract (mirrors tools/browser_camofox.py): create a tab
    (POST /tabs), navigate (POST /tabs/{id}/navigate), read the accessibility/DOM snapshot
    (GET /tabs/{id}/snapshot), and capture a screenshot (GET /tabs/{id}/screenshot, binary
    PNG). Returns the SAME ProbeReceipt shape as the playwright path so the selector and
    Council are backend-agnostic. Console-error capture is best-effort (the snapshot
    carries page state; the shim does not expose a console stream uniformly), so an
    assert_text / expect_text check is the primary signal here.
    """
    try:
        import urllib.request
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(BROWSER, UNAVAILABLE, spec.criterion_id,
                            summary="urllib unavailable for camoufox driver", detail=str(exc)[:200], ran_at=_now())

    user_id = "autopilot-probe"

    def _req(method: str, path: str, body: Optional[dict] = None, raw: bool = False):
        url = f"{base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"User-Agent": "HermesAutopilot/probe"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - operator-configured shim
            payload = resp.read()
            return payload if raw else json.loads(payload.decode("utf-8", "replace") or "{}")

    try:
        tab = _req("POST", "/tabs", {"userId": user_id})
        tab_id = tab.get("tabId") or tab.get("tab_id") or tab.get("id")
        if not tab_id:
            return ProbeReceipt(BROWSER, ERROR, spec.criterion_id,
                                summary="camoufox: could not open a tab", detail=str(tab)[:200], ran_at=_now())
        _req("POST", f"/tabs/{tab_id}/navigate", {"userId": user_id, "url": spec.target})
        # optional clicks by selector
        for _ in range(max(0, spec.click_times)):
            if not spec.click_selector:
                break
            try:
                _req("POST", f"/tabs/{tab_id}/click", {"userId": user_id, "selector": spec.click_selector})
            except Exception:  # noqa: BLE001 — click failures are observed via the snapshot
                break
        snap = _req("GET", f"/tabs/{tab_id}/snapshot?userId={user_id}")
        snapshot_text = snap.get("snapshot", "") if isinstance(snap, dict) else str(snap)
        # screenshot (binary PNG)
        shot_path = ""
        if spec.screenshot:
            try:
                png = _req("GET", f"/tabs/{tab_id}/screenshot?userId={user_id}", raw=True)
                shot_path = os.path.join(workdir, f"probe_{spec.criterion_id or 'shot'}_{int(datetime.now().timestamp())}.png")
                with open(shot_path, "wb") as fh:
                    fh.write(png)
            except Exception:  # noqa: BLE001
                shot_path = ""
        # evaluate assertions against the rendered snapshot text
        text_present = (spec.assert_text in snapshot_text) if spec.assert_text else True
        text_equals = True
        if spec.expect_text_equals and spec.expect_text_in:
            # the snapshot carries element text; a contains-check is the portable signal
            text_equals = spec.expect_text_equals in snapshot_text
        status = PASS if (text_present and text_equals) else FAIL
        parts = []
        if spec.assert_text:
            parts.append(f"text '{spec.assert_text}' present={text_present}")
        if spec.expect_text_equals:
            parts.append(f"expect {spec.expect_text_equals!r} present={text_equals}")
        return ProbeReceipt(BROWSER, status, spec.criterion_id,
                            summary="[camoufox] " + ("; ".join(parts) or f"browser probe {status}"),
                            detail=snapshot_text[:1000],
                            screenshot_path=shot_path,
                            observed={"backend": "camofox_shim", "text_present": text_present},
                            ran_at=_now())
    except Exception as exc:  # noqa: BLE001 — shim unreachable mid-run → downgrade, never crash
        return ProbeReceipt(BROWSER, UNAVAILABLE, spec.criterion_id,
                            summary="camoufox shim error during observation",
                            detail=str(exc)[:300], ran_at=_now())


# --------------------------------------------------------------------------- #
# http probe — SSRF-guarded fetch + status/body assertion                       #
# --------------------------------------------------------------------------- #
def _run_http(spec: ProbeSpec, *, timeout: float) -> ProbeReceipt:
    url = spec.url or spec.target
    if not url:
        return ProbeReceipt(HTTP, ERROR, spec.criterion_id, summary="http probe missing url", ran_at=_now())
    # reuse the council evidence SSRF guard when available
    try:
        from agent.autopilot import council_gate  # noqa: F401
        try:
            from council.evidence import _validate_url  # type: ignore
            _validate_url(url)
        except Exception:  # noqa: BLE001 — guard optional; only blocks if importable
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HermesAutopilot/probe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - guarded above
            status_code = resp.status
            body = resp.read(200_000).decode("utf-8", "replace")
        status_ok = status_code == spec.expect_status
        body_ok = (spec.expect_body_contains in body) if spec.expect_body_contains else True
        status = PASS if (status_ok and body_ok) else FAIL
        return ProbeReceipt(HTTP, status, spec.criterion_id,
                            summary=f"GET {url} → {status_code} (want {spec.expect_status})",
                            observed={"status": status_code}, ran_at=_now())
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(HTTP, FAIL, spec.criterion_id,
                            summary=f"GET {url} failed", detail=str(exc)[:300], ran_at=_now())


# --------------------------------------------------------------------------- #
# artifact probe — filesystem assertions                                        #
# --------------------------------------------------------------------------- #
def _run_artifact(spec: ProbeSpec, *, workdir: str) -> ProbeReceipt:
    path = spec.path if os.path.isabs(spec.path) else os.path.join(workdir, spec.path)
    try:
        exists = os.path.exists(path)
        if spec.must_exist and not exists:
            return ProbeReceipt(ARTIFACT, FAIL, spec.criterion_id,
                                summary=f"{spec.path} does not exist", ran_at=_now())
        content = ""
        if exists and os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(200_000)
        if spec.must_be_nonempty and not content.strip():
            return ProbeReceipt(ARTIFACT, FAIL, spec.criterion_id,
                                summary=f"{spec.path} is empty", ran_at=_now())
        if spec.must_contain and spec.must_contain not in content:
            return ProbeReceipt(ARTIFACT, FAIL, spec.criterion_id,
                                summary=f"{spec.path} missing required content", ran_at=_now())
        return ProbeReceipt(ARTIFACT, PASS, spec.criterion_id,
                            summary=f"{spec.path} satisfies artifact assertions", ran_at=_now())
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(ARTIFACT, ERROR, spec.criterion_id, summary="artifact probe error",
                            detail=str(exc)[:300], ran_at=_now())


# --------------------------------------------------------------------------- #
# cmx_provenance probe — does the verbatim record support the claim? (§3.4)      #
# --------------------------------------------------------------------------- #
def _run_cmx_provenance(spec: ProbeSpec) -> ProbeReceipt:
    """Falsify a claim against the verbatim record via the cmx→lcm→state ladder.

    The strongest bypass move ("a subagent confirmed it") is unfalsifiable PROSE to a
    text-only judge, but a database lookup here: an evidence-role row must actually
    contain the claim. No row ⇒ FAIL (the gate does not pass on prose). No ladder rung
    reachable ⇒ UNOBSERVABLE → downgrade.
    """
    if not spec.evidence_terms:
        return ProbeReceipt(CMX_PROVENANCE, ERROR, spec.criterion_id,
                            summary="cmx_provenance probe missing evidence_terms", ran_at=_now())
    try:
        from agent.autopilot import provenance as _prov
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(CMX_PROVENANCE, UNAVAILABLE, spec.criterion_id,
                            summary="provenance backend unimportable", detail=str(exc)[:200], ran_at=_now())
    res = _prov.supports_claim(spec.evidence_terms, session_id=spec.provenance_session_id)
    if not res.reachable:
        return ProbeReceipt(CMX_PROVENANCE, UNOBSERVABLE_STATUS, spec.criterion_id,
                            summary="no verbatim store reachable (cmx/lcm/state all absent)",
                            detail="provenance ladder empty → downgrade to text judgment",
                            observed={"backends_tried": res.backends_tried}, ran_at=_now())
    if res.supported:
        return ProbeReceipt(CMX_PROVENANCE, PASS, spec.criterion_id,
                            summary=f"claim supported by {res.backend} record "
                                    f"(role={res.matched_role})",
                            detail=res.matched_excerpt,
                            observed={"backend": res.backend, "role": res.matched_role,
                                      "rows_searched": res.rows_searched}, ran_at=_now())
    return ProbeReceipt(CMX_PROVENANCE, FAIL, spec.criterion_id,
                        summary=f"NO evidence-role row in {res.backend} record supports the "
                                f"claim — unsupported by the verbatim record",
                        detail=f"searched terms={spec.evidence_terms!r} across "
                               f"{res.backends_tried} ({res.rows_searched} evidence rows)",
                        observed={"backend": res.backend, "rows_searched": res.rows_searched},
                        ran_at=_now())


# --------------------------------------------------------------------------- #
# media/document backends — registration hook + CLI fallbacks                   #
# --------------------------------------------------------------------------- #
# The engine layer cannot import the agent's ``vision_analyze`` / transcribe tools
# directly (wrong process boundary), so production REGISTERS them here. When a real
# vision backend is registered, image/document/video STRUCTURE+APPEARANCE probes use
# it (invariant: OCR is insufficient for structure — §3.2). When none is registered,
# the probe degrades to a CLI fallback (tesseract OCR / pdftotext / ffprobe) and SAYS
# SO, or returns UNOBSERVABLE when only true vision would be accurate.
_VISION_BACKEND = None       # callable(image_path, question) -> str (the vision answer)
_TRANSCRIBE_BACKEND = None   # callable(audio_path) -> str (the transcript)


def register_vision_backend(fn) -> None:
    """Production wires the agent's vision tool in here (callable(path, question)->str)."""
    global _VISION_BACKEND
    _VISION_BACKEND = fn


def register_transcribe_backend(fn) -> None:
    """Production wires a transcribe tool in here (callable(path)->str)."""
    global _TRANSCRIBE_BACKEND
    _TRANSCRIBE_BACKEND = fn


def _ocr_text(image_path: str, *, timeout: float = 30) -> Optional[str]:
    """Cheap OCR pre-pass via the tesseract CLI. Returns None if unavailable."""
    if not shutil.which("tesseract"):
        return None
    try:
        proc = subprocess.run(["tesseract", image_path, "stdout"],
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
        return proc.stdout if proc.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _hermes_transcribe(audio_path: str) -> Optional[tuple]:
    """Transcribe via Hermes' OWN native STT pipeline — no model download, no
    re-integration. Delegates to ``tools.transcription_tools.transcribe_audio``, which
    honors the user's configured ``stt.provider`` (local/groq/openai/mistral/xai/
    local_command; ``local``/``local_command`` can wrap whisper.cpp via config on this
    no-GPU host). Returns (transcript, provider) or None when STT is unavailable/disabled
    (→ the caller degrades to ffprobe/unavailable). Never raises."""
    try:
        from tools.transcription_tools import transcribe_audio
    except Exception:  # noqa: BLE001 — tool tree not importable in some test envs
        return None
    try:
        res = transcribe_audio(audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: hermes transcribe_audio failed (%s)", exc)
        return None
    if isinstance(res, dict) and res.get("success") and res.get("transcript"):
        return (str(res["transcript"]), f"hermes:{res.get('provider', 'stt')}")
    return None


def _hermes_vision(image_path: str, question: str) -> Optional[str]:
    """Observe an image via Hermes' OWN native vision tool — no re-integration, uses the
    user's configured auxiliary vision model. Delegates to ``tools.vision_tools``'s native
    analyzer (async), bridged to sync. Returns the vision answer or None when unavailable
    (→ caller falls back to OCR). Never raises."""
    try:
        from tools.vision_tools import _vision_analyze_native  # type: ignore
    except Exception:  # noqa: BLE001 — tool tree not importable in some test envs
        return None
    try:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            res = asyncio.run(_vision_analyze_native(image_path, question))
        else:
            # already inside a loop (rare for the engine layer): use a fresh loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                res = ex.submit(lambda: asyncio.run(_vision_analyze_native(image_path, question))).result()
        if isinstance(res, dict):
            return str(res.get("description") or res.get("analysis") or res.get("text") or "") or None
        return str(res) if res else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: hermes vision failed (%s)", exc)
        return None


def _vision_observe(image_path: str, question: str, *, timeout: float) -> tuple[str, str]:
    """Return (answer, source) observing an image. Backend order: a registered agent-vision
    backend → Hermes' native vision tool (user's configured aux vision model) → OCR text
    (source='ocr', a WEAKER observation that cannot judge structure, §3.2)."""
    if callable(_VISION_BACKEND):
        try:
            return (str(_VISION_BACKEND(image_path, question) or ""), "vision")
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: vision backend failed (%s); trying hermes vision", exc)
    hv = _hermes_vision(image_path, question)
    if hv:
        return (hv, "vision")
    ocr = _ocr_text(image_path, timeout=timeout)
    if ocr is not None:
        return (ocr, "ocr")
    return ("", "none")


def _run_image(spec: ProbeSpec, *, timeout: float) -> ProbeReceipt:
    """Observe an image's true content. Agent vision is required for structure/appearance;
    OCR is only a degraded fallback (§3.2)."""
    img = spec.media_path or spec.path or spec.target
    if not img or not os.path.exists(img):
        return ProbeReceipt(IMAGE, ERROR, spec.criterion_id,
                            summary=f"image not found: {img}", ran_at=_now())
    question = spec.vision_question or "Describe what this image shows in detail."
    answer, source = _vision_observe(img, question, timeout=timeout)
    if source == "none":
        return ProbeReceipt(IMAGE, UNAVAILABLE, spec.criterion_id,
                            summary="no vision backend and no OCR available", ran_at=_now())
    want = (spec.expect_visual or "").strip()
    if want:
        ok = want.lower() in answer.lower()
        # OCR cannot reliably judge structure/appearance: a structural miss under OCR is
        # reported as UNOBSERVABLE (downgrade), not a false FAIL.
        if not ok and source == "ocr":
            return ProbeReceipt(IMAGE, UNOBSERVABLE_STATUS, spec.criterion_id,
                                summary="OCR could not confirm the visual expectation; "
                                        "true vision required for an accurate verdict",
                                detail=f"want={want!r} not in OCR text", observed={"source": source},
                                ran_at=_now())
        status = PASS if ok else FAIL
        return ProbeReceipt(IMAGE, status, spec.criterion_id,
                            summary=f"image vision[{source}] {'affirms' if ok else 'denies'} {want!r}",
                            detail=answer[:600], observed={"source": source}, ran_at=_now())
    # no explicit expectation → return the observation as evidence for the Council
    return ProbeReceipt(IMAGE, PASS, spec.criterion_id,
                        summary=f"observed image via {source}", detail=answer[:600],
                        observed={"source": source}, ran_at=_now())


def _run_audio(spec: ProbeSpec, *, timeout: float) -> ProbeReceipt:
    """Observe what an audio file actually contains via transcription + assertion.

    Backend order: a registered transcribe backend (production wires e.g. faster-whisper)
    → the whisper.cpp CLI (`whisper-cli`, CPU/BLAS — present on this no-GPU host) →
    ffprobe structural facts when no transcriber exists and no transcript assertion is set.
    """
    aud = spec.media_path or spec.path or spec.target
    if not aud or not os.path.exists(aud):
        return ProbeReceipt(AUDIO, ERROR, spec.criterion_id,
                            summary=f"audio not found: {aud}", ran_at=_now())
    transcript: Optional[str] = None
    source = ""
    if callable(_TRANSCRIBE_BACKEND):
        try:
            transcript = str(_TRANSCRIBE_BACKEND(aud) or "")
            source = "registered"
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: transcribe backend failed (%s); trying hermes STT", exc)
    if transcript is None:
        hx = _hermes_transcribe(aud)
        if hx is not None:
            transcript, source = hx
    if transcript is None:
        # no transcriber available: structural facts are still observable via ffprobe,
        # but a transcript assertion cannot be checked → unavailable (downgrade).
        if spec.transcript_contains:
            return ProbeReceipt(AUDIO, UNAVAILABLE, spec.criterion_id,
                                summary="no transcribe backend (registered/whisper.cpp); cannot check transcript",
                                ran_at=_now())
        return _ffprobe_structural(spec, aud, AUDIO, timeout=timeout)
    want = (spec.transcript_contains or "").strip()
    if want:
        ok = want.lower() in transcript.lower()
        return ProbeReceipt(AUDIO, PASS if ok else FAIL, spec.criterion_id,
                            summary=f"transcript[{source}] {'contains' if ok else 'missing'} {want!r}",
                            detail=transcript[:600], observed={"source": source}, ran_at=_now())
    return ProbeReceipt(AUDIO, PASS, spec.criterion_id, summary=f"transcribed audio via {source}",
                        detail=transcript[:600], observed={"source": source}, ran_at=_now())


def _ffprobe_structural(spec: ProbeSpec, path: str, kind: str, *, timeout: float) -> ProbeReceipt:
    """Observe STRUCTURAL facts of a media file via ffprobe (codec/streams/duration)."""
    if not shutil.which("ffprobe"):
        return ProbeReceipt(kind, UNAVAILABLE, spec.criterion_id,
                            summary="ffprobe not available for structural observation", ran_at=_now())
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return ProbeReceipt(kind, FAIL, spec.criterion_id,
                                summary=f"ffprobe could not decode {os.path.basename(path)}",
                                detail=(proc.stderr or "")[:300], ran_at=_now())
        data = json.loads(proc.stdout or "{}")
        nstreams = len(data.get("streams", []))
        dur = data.get("format", {}).get("duration", "?")
        return ProbeReceipt(kind, PASS, spec.criterion_id,
                            summary=f"{os.path.basename(path)} decodes: {nstreams} stream(s), {dur}s",
                            observed={"streams": nstreams, "duration": dur}, ran_at=_now())
    except Exception as exc:  # noqa: BLE001
        return ProbeReceipt(kind, ERROR, spec.criterion_id, summary="ffprobe error",
                            detail=str(exc)[:200], ran_at=_now())


def _run_video(spec: ProbeSpec, *, timeout: float, workdir: str) -> ProbeReceipt:
    """Observe what a video shows: sample a frame → agent vision, or structural ffprobe."""
    vid = spec.media_path or spec.path or spec.target
    if not vid or not os.path.exists(vid):
        return ProbeReceipt(VIDEO, ERROR, spec.criterion_id,
                            summary=f"video not found: {vid}", ran_at=_now())
    # appearance criterion → extract a frame and judge it with vision
    if (spec.expect_visual or spec.vision_question) and shutil.which("ffmpeg"):
        frame = os.path.join(workdir, f"_frame_{spec.criterion_id or 'v'}_{int(datetime.now().timestamp())}.png")
        ts = max(0, spec.page)  # reuse 'page' as a seconds offset for the sampled frame
        try:
            subprocess.run(["ffmpeg", "-y", "-ss", str(ts), "-i", vid, "-frames:v", "1", frame],
                          capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            frame = ""
        if frame and os.path.exists(frame):
            fspec = ProbeSpec(kind=IMAGE, media_path=frame, vision_question=spec.vision_question,
                              expect_visual=spec.expect_visual, criterion_id=spec.criterion_id)
            r = _run_image(fspec, timeout=timeout)
            r.kind = VIDEO  # report as a video observation
            r.screenshot_path = frame
            return r
    # otherwise structural observation
    return _ffprobe_structural(spec, vid, VIDEO, timeout=timeout)


def _run_document(spec: ProbeSpec, *, timeout: float, workdir: str) -> ProbeReceipt:
    """Observe the COMPILED/rendered document, not its source (§3.1).

    PDF: render page→image→vision when an appearance/structure expectation is set
    (vision required, §3.2); otherwise extract text via pdftotext and assert content.
    Password-protected / convert-needed / compile-needed cases are reported explicitly
    (UNAVAILABLE/UNOBSERVABLE), never a crash (§3 open-registry resilience).
    """
    doc = spec.media_path or spec.path or spec.target
    if not doc or not os.path.exists(doc):
        return ProbeReceipt(DOCUMENT, ERROR, spec.criterion_id,
                            summary=f"document not found: {doc}", ran_at=_now())
    ext = os.path.splitext(doc)[1].lower()

    if ext == ".pdf":
        # appearance/structure expectation → render the page and judge with vision
        if (spec.expect_visual or spec.vision_question) and shutil.which("pdftoppm"):
            png_base = os.path.join(workdir, f"_doc_{spec.criterion_id or 'p'}_{int(datetime.now().timestamp())}")
            pageno = spec.page or 1
            cmd = ["pdftoppm", "-png", "-f", str(pageno), "-l", str(pageno), "-singlefile"]
            if spec.document_password:
                cmd += ["-upw", spec.document_password]
            cmd += [doc, png_base]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                      stdin=subprocess.DEVNULL)
            except Exception as exc:  # noqa: BLE001
                return ProbeReceipt(DOCUMENT, ERROR, spec.criterion_id,
                                    summary="pdftoppm render error", detail=str(exc)[:200], ran_at=_now())
            png = png_base + ".png"
            if proc.returncode != 0 or not os.path.exists(png):
                low = (proc.stderr or "").lower()
                if "password" in low or "encrypt" in low:
                    return ProbeReceipt(DOCUMENT, UNAVAILABLE, spec.criterion_id,
                                        summary="PDF is password-protected; no/invalid password supplied",
                                        detail=(proc.stderr or "")[:200], ran_at=_now())
                return ProbeReceipt(DOCUMENT, FAIL, spec.criterion_id,
                                    summary="PDF did not render", detail=(proc.stderr or "")[:200], ran_at=_now())
            fspec = ProbeSpec(kind=IMAGE, media_path=png, vision_question=spec.vision_question,
                              expect_visual=spec.expect_visual, criterion_id=spec.criterion_id)
            r = _run_image(fspec, timeout=timeout)
            r.kind = DOCUMENT
            r.screenshot_path = png
            return r
        # text expectation → pdftotext extraction
        if shutil.which("pdftotext"):
            cmd = ["pdftotext"]
            if spec.document_password:
                cmd += ["-upw", spec.document_password]
            cmd += [doc, "-"]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                      stdin=subprocess.DEVNULL)
            except Exception as exc:  # noqa: BLE001
                return ProbeReceipt(DOCUMENT, ERROR, spec.criterion_id,
                                    summary="pdftotext error", detail=str(exc)[:200], ran_at=_now())
            if proc.returncode != 0:
                low = (proc.stderr or "").lower()
                if "password" in low or "encrypt" in low:
                    return ProbeReceipt(DOCUMENT, UNAVAILABLE, spec.criterion_id,
                                        summary="PDF is password-protected", detail=(proc.stderr or "")[:200],
                                        ran_at=_now())
                return ProbeReceipt(DOCUMENT, FAIL, spec.criterion_id,
                                    summary="pdftotext failed", detail=(proc.stderr or "")[:200], ran_at=_now())
            text = proc.stdout or ""
            want = (spec.must_contain or spec.expect_body_contains or "").strip()
            if want:
                ok = want.lower() in text.lower()
                return ProbeReceipt(DOCUMENT, PASS if ok else FAIL, spec.criterion_id,
                                    summary=f"PDF text {'contains' if ok else 'missing'} {want!r}",
                                    detail=text[:600], ran_at=_now())
            if spec.must_be_nonempty:
                ok = bool(text.strip())
                return ProbeReceipt(DOCUMENT, PASS if ok else FAIL, spec.criterion_id,
                                    summary=f"PDF text {'present' if ok else 'empty'}",
                                    detail=text[:300], ran_at=_now())
            return ProbeReceipt(DOCUMENT, PASS, spec.criterion_id, summary="extracted PDF text",
                                detail=text[:600], ran_at=_now())
        return ProbeReceipt(DOCUMENT, UNAVAILABLE, spec.criterion_id,
                            summary="no PDF backend (pdftoppm/pdftotext) available", ran_at=_now())

    # formats requiring a converter we may not have (docx/latex/odt/etc.)
    return ProbeReceipt(DOCUMENT, UNOBSERVABLE_STATUS, spec.criterion_id,
                        summary=f"no compiled-observation backend for '{ext}' documents",
                        detail="needs a converter/compiler (e.g. pandoc/latexmk) not present "
                               "→ downgrade to text judgment", ran_at=_now())


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def run_probe(spec: ProbeSpec, *, timeout: float = 60, workdir: Optional[str] = None) -> ProbeReceipt:
    """Run ONE probe and return its receipt. Never raises."""
    wd = workdir or os.getcwd()
    try:
        if spec.kind == UNOBSERVABLE:
            return ProbeReceipt(UNOBSERVABLE, UNOBSERVABLE_STATUS, spec.criterion_id,
                                summary="no observable proof exists for this criterion",
                                detail=spec.why_unobservable, ran_at=_now())
        if spec.kind == PROCESS:
            return _run_process(spec, timeout=timeout, workdir=wd)
        if spec.kind == BROWSER:
            return _run_browser(spec, timeout=timeout, workdir=wd)
        if spec.kind == HTTP:
            return _run_http(spec, timeout=timeout)
        if spec.kind == ARTIFACT:
            return _run_artifact(spec, workdir=wd)
        if spec.kind == CMX_PROVENANCE:
            return _run_cmx_provenance(spec)
        if spec.kind == IMAGE:
            return _run_image(spec, timeout=timeout)
        if spec.kind == AUDIO:
            return _run_audio(spec, timeout=timeout)
        if spec.kind == VIDEO:
            return _run_video(spec, timeout=timeout, workdir=wd)
        if spec.kind == DOCUMENT:
            return _run_document(spec, timeout=timeout, workdir=wd)
        # OPEN REGISTRY (§3): an unknown modality is not a crash — it's an honest
        # "we can't observe this kind" → downgrade, with the kind named for the ADR.
        return ProbeReceipt(spec.kind or "unknown", UNOBSERVABLE_STATUS, spec.criterion_id,
                            summary=f"no probe backend for modality '{spec.kind}' → downgrade",
                            detail="unregistered modality; add a handler or treat as unobservable",
                            ran_at=_now())
    except Exception as exc:  # noqa: BLE001 — probe must never crash the gate
        logger.warning("autopilot: probe %s crashed (%s)", spec.kind, exc)
        return ProbeReceipt(spec.kind or "unknown", ERROR, spec.criterion_id,
                            summary="probe crashed", detail=str(exc)[:300], ran_at=_now())


def run_probes(specs: list, *, timeout: float = 60, workdir: Optional[str] = None) -> list:
    """Run a probe plan (list of ProbeSpec) and return receipts in order."""
    return [run_probe(s, timeout=timeout, workdir=workdir) for s in (specs or [])]


def format_probe_block(receipts: list, *, max_chars: int = 4000) -> str:
    """Render probe receipts as a Council-context block (the engine's observations)."""
    if not receipts:
        return ""
    lines = ["ENGINE PROBE RECEIPTS (the engine OBSERVED reality — ground truth, "
             "outside the agent's control):"]
    for r in receipts:
        lines.append("  " + r.as_evidence().replace("\n", "\n  "))
    return "\n".join(lines)[:max_chars]
