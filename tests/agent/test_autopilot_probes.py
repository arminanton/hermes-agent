"""Tests for the probe capability layer (the engine's senses)."""

import os
import tempfile

import pytest

from agent.autopilot import probes as P
from agent.autopilot import derail_repro as dr

_HAS_BROWSER = P.browser_available()
_browser_only = pytest.mark.skipif(not _HAS_BROWSER, reason="node/playwright not present")


@pytest.fixture()
def wd():
    d = tempfile.mkdtemp(prefix="probe-test-")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# process probe — reuses the verification allowlist                            #
# --------------------------------------------------------------------------- #
def test_process_pass(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.PROCESS, command="true", criterion_id="C1"), workdir=wd)
    assert r.passed and r.status == P.PASS


def test_process_fail(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.PROCESS, command="false", criterion_id="C1"), workdir=wd)
    assert r.status == P.FAIL and not r.passed


def test_process_refuses_mutating_command(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.PROCESS, command="rm -rf /tmp/x", criterion_id="C1"), workdir=wd)
    assert r.status == P.ERROR
    assert "refused" in r.summary


def test_process_reentrancy_suppressed(wd, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_VERIFICATION", "1")
    r = P.run_probe(P.ProbeSpec(kind=P.PROCESS, command="true", criterion_id="C1"), workdir=wd)
    assert r.status == P.UNAVAILABLE
    assert "nested probe suppressed" in r.summary


def test_process_captures_output(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.PROCESS, command="echo probe-marker", criterion_id="C1"), workdir=wd)
    assert r.passed
    assert r.observed.get("exit_code") == 0


# --------------------------------------------------------------------------- #
# artifact probe                                                               #
# --------------------------------------------------------------------------- #
def test_artifact_exists_and_contains(wd):
    with open(os.path.join(wd, "out.txt"), "w") as fh:
        fh.write("hello world")
    r = P.run_probe(P.ProbeSpec(kind=P.ARTIFACT, path="out.txt", must_contain="hello", criterion_id="C1"), workdir=wd)
    assert r.passed


def test_artifact_missing_fails(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.ARTIFACT, path="nope.txt", criterion_id="C1"), workdir=wd)
    assert r.status == P.FAIL


def test_artifact_empty_fails(wd):
    open(os.path.join(wd, "empty.txt"), "w").close()
    r = P.run_probe(P.ProbeSpec(kind=P.ARTIFACT, path="empty.txt", must_be_nonempty=True, criterion_id="C1"), workdir=wd)
    assert r.status == P.FAIL


def test_artifact_missing_content_fails(wd):
    with open(os.path.join(wd, "out.txt"), "w") as fh:
        fh.write("something else")
    r = P.run_probe(P.ProbeSpec(kind=P.ARTIFACT, path="out.txt", must_contain="REQUIRED", criterion_id="C1"), workdir=wd)
    assert r.status == P.FAIL


# --------------------------------------------------------------------------- #
# http probe — graceful, never raises                                          #
# --------------------------------------------------------------------------- #
def test_http_unreachable_fails_gracefully(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.HTTP, url="http://127.0.0.1:59999/nope", criterion_id="C1"),
                    timeout=2, workdir=wd)
    assert r.status == P.FAIL  # not an exception


def test_http_missing_url_errors(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.HTTP, criterion_id="C1"), workdir=wd)
    assert r.status == P.ERROR


# --------------------------------------------------------------------------- #
# unobservable — the downgrade signal                                          #
# --------------------------------------------------------------------------- #
def test_unobservable_is_downgrade(wd):
    r = P.run_probe(P.ProbeSpec(kind=P.UNOBSERVABLE, criterion_id="C1",
                                why_unobservable="pure design doc"), workdir=wd)
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True
    assert not r.passed


def test_unavailable_is_downgrade():
    # a receipt with UNAVAILABLE status also signals downgrade-to-council
    r = P.ProbeReceipt(P.BROWSER, P.UNAVAILABLE, "C1", summary="no browser")
    assert r.is_downgrade is True


# --------------------------------------------------------------------------- #
# unknown kind / crash safety                                                  #
# --------------------------------------------------------------------------- #
def test_unknown_kind_downgrades(wd):
    # OPEN REGISTRY (§3): an unknown modality is NOT an error — it downgrades to
    # unobservable (with the kind named) so a new/unsupported modality can never crash
    # the gate. (Superseded the old ERROR semantics in the modal refactor, turn 1203.)
    r = P.run_probe(P.ProbeSpec(kind="bogus", criterion_id="C1"), workdir=wd)
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True
    assert "bogus" in r.summary


def test_run_probes_empty():
    assert P.run_probes([]) == []


def test_format_probe_block_empty():
    assert P.format_probe_block([]) == ""


# --------------------------------------------------------------------------- #
# browser probe — the headline capability (observe DOM/console/screenshot)      #
# --------------------------------------------------------------------------- #
def _counter_spec(target):
    return P.ProbeSpec(
        kind=P.BROWSER, criterion_id="C1", target=target,
        click_selector="#inc", click_times=3,
        expect_text_in="#count", expect_text_equals="3",
        require_no_console_errors=True, screenshot=True,
    )


@_browser_only
def test_browser_observes_broken_as_fail():
    sc = dr.BrokenCounterScenario()
    try:
        r = P.run_probe(_counter_spec(sc.broken_path), workdir=sc.workdir)
        assert r.status == P.FAIL
        assert r.observed.get("observed_text") == "0"
        assert "incrementCounterTypoFn" in r.detail
        assert r.screenshot_path and os.path.exists(r.screenshot_path)   # screenshot for the Council
    finally:
        sc.cleanup()


@_browser_only
def test_browser_observes_real_fix_as_pass():
    sc = dr.BrokenCounterScenario()
    try:
        sc.apply_real_fix()
        r = P.run_probe(_counter_spec(sc.broken_path), workdir=sc.workdir)
        assert r.passed
        assert r.observed.get("observed_text") == "3"
    finally:
        sc.cleanup()


@_browser_only
def test_browser_unavailable_downgrades(monkeypatch):
    # when the browser tool isn't found, the probe DOWNGRADES (never crashes)
    monkeypatch.setattr(P, "_node_bin", lambda: None)
    r = P.run_probe(_counter_spec("/tmp/whatever.html"))
    assert r.status == P.UNAVAILABLE
    assert r.is_downgrade is True


@_browser_only
def test_browser_evidence_block_shows_observation():
    sc = dr.BrokenCounterScenario()
    try:
        r = P.run_probe(_counter_spec(sc.broken_path), workdir=sc.workdir)
        block = P.format_probe_block([r])
        assert "ENGINE PROBE" in block and "FAIL" in block
        assert "screenshot:" in block
    finally:
        sc.cleanup()


# =========================================================================== #
# MODAL senses — the open registry + cmx_provenance + media/document           #
# =========================================================================== #
import sqlite3  # noqa: E402


def _make_store(path, rows):
    """Build a state.db/cmx.db/lcm.db-shaped messages store and seed (role, content)."""
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, "
        "turn_index INTEGER, role TEXT, content TEXT);"
    )
    for i, (role, content) in enumerate(rows):
        c.execute("INSERT INTO messages(session_id,turn_index,role,content) VALUES('s',?,?,?)",
                  (i, role, content))
    c.commit()
    c.close()


@pytest.fixture()
def _isolate_ladder(monkeypatch, wd):
    """Point the whole provenance ladder at a controlled tmp dir; default all absent."""
    monkeypatch.setenv("CMX_DB_PATH", os.path.join(wd, "cmx.db"))
    monkeypatch.setenv("LCM_DB_PATH", os.path.join(wd, "lcm.db"))
    monkeypatch.setenv("HERMES_STATE_DB", os.path.join(wd, "state.db"))
    return wd


# --- open registry --------------------------------------------------------- #
def test_unknown_modality_downgrades_not_crash():
    # an unregistered modality must downgrade to unobservable, never raise / never ERROR.
    r = P.run_probe(P.ProbeSpec(kind="hologram", criterion_id="C9"))
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True
    assert "hologram" in r.summary


def test_known_kinds_registry_complete():
    for k in (P.PROCESS, P.BROWSER, P.HTTP, P.ARTIFACT, P.CMX_PROVENANCE,
              P.IMAGE, P.AUDIO, P.VIDEO, P.DOCUMENT, P.UNOBSERVABLE):
        assert k in P.KNOWN_KINDS


# --- cmx_provenance: the bypass-killer ------------------------------------- #
def test_provenance_prose_only_fails(_isolate_ladder):
    # the manufactured-independence bypass: assistant CLAIMS confirmation, no tool row.
    _make_store(os.path.join(_isolate_ladder, "state.db"),
                [("user", "fix it"), ("assistant", "a subagent confirmed the page works")])
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE,
                                evidence_terms=["confirmed", "page", "works"], criterion_id="C1"))
    assert r.status == P.FAIL
    assert "unsupported" in r.summary.lower()
    assert r.observed["backend"] == "state"


def test_provenance_real_tool_row_passes(_isolate_ladder):
    # a genuine role='tool' confirmation grounds the claim → PASS (not always-deny).
    _make_store(os.path.join(_isolate_ladder, "state.db"),
                [("user", "fix it"),
                 ("assistant", "claims done"),
                 ("tool", "subagent result: page works, confirmed, counter=3")])
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE,
                                evidence_terms=["confirmed", "page", "works"], criterion_id="C1"))
    assert r.status == P.PASS
    assert r.observed["role"] == "tool"


def test_provenance_prefers_cmx_over_state(_isolate_ladder):
    # when cmx is present it answers first (best rung); the row lives only in cmx here.
    _make_store(os.path.join(_isolate_ladder, "cmx.db"),
                [("tool", "verified: the endpoint returns 200, payload confirmed correct")])
    _make_store(os.path.join(_isolate_ladder, "state.db"),
                [("assistant", "I think it's fine")])
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE,
                                evidence_terms=["verified", "200", "confirmed"], criterion_id="C1"))
    assert r.status == P.PASS
    assert r.observed["backend"] == "cmx"


def test_provenance_no_store_downgrades(_isolate_ladder):
    # FLOOR case: no cmx, no lcm, no state → unobservable → downgrade (never crash).
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE, evidence_terms=["x"], criterion_id="C1"))
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True


def test_provenance_missing_terms_errors(_isolate_ladder):
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE, evidence_terms=[], criterion_id="C1"))
    assert r.status == P.ERROR


def test_provenance_reads_readonly_never_writes(_isolate_ladder):
    # the sense must never mutate the user's live store.
    db = os.path.join(_isolate_ladder, "state.db")
    _make_store(db, [("tool", "confirmed ok")])
    before = os.path.getmtime(db)
    import time
    time.sleep(0.02)
    P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE, evidence_terms=["nope"], criterion_id="C1"))
    assert os.path.getmtime(db) == before  # untouched


# --- image: vision required, OCR insufficient ------------------------------ #
def test_image_missing_file_errors():
    r = P.run_probe(P.ProbeSpec(kind=P.IMAGE, media_path="/nope/x.png", criterion_id="C1"))
    assert r.status == P.ERROR


def test_image_uses_registered_vision_backend(monkeypatch, wd):
    img = os.path.join(wd, "x.png")
    open(img, "wb").close()
    monkeypatch.setattr(P, "_VISION_BACKEND",
                        lambda path, q: "The chart shows a rising bar from 1 to 3.")
    r = P.run_probe(P.ProbeSpec(kind=P.IMAGE, media_path=img,
                                expect_visual="rising bar", criterion_id="C1"))
    assert r.status == P.PASS
    assert r.observed["source"] == "vision"


def test_image_ocr_structural_miss_downgrades(monkeypatch, wd):
    # OCR can't judge structure: a structural expectation it can't confirm → UNOBSERVABLE,
    # NOT a false FAIL (invariant §3.2 — vision is required for accuracy).
    img = os.path.join(wd, "x.png")
    open(img, "wb").close()
    monkeypatch.setattr(P, "_VISION_BACKEND", None)                 # no registered vision
    monkeypatch.setattr(P, "_hermes_vision", lambda p, q: None)     # no native vision
    monkeypatch.setattr(P, "_ocr_text", lambda p, timeout=30: "some unrelated glyphs")
    r = P.run_probe(P.ProbeSpec(kind=P.IMAGE, media_path=img,
                                expect_visual="rising bar chart", criterion_id="C1"))
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True


def test_image_no_backend_unavailable(monkeypatch, wd):
    img = os.path.join(wd, "x.png")
    open(img, "wb").close()
    monkeypatch.setattr(P, "_VISION_BACKEND", None)
    monkeypatch.setattr(P, "_hermes_vision", lambda p, q: None)     # no native vision
    monkeypatch.setattr(P, "_ocr_text", lambda p, timeout=30: None)  # no OCR either
    r = P.run_probe(P.ProbeSpec(kind=P.IMAGE, media_path=img, expect_visual="x", criterion_id="C1"))
    assert r.status == P.UNAVAILABLE


# --- audio ----------------------------------------------------------------- #
def test_audio_transcript_assert(monkeypatch, wd):
    aud = os.path.join(wd, "a.wav")
    open(aud, "wb").close()
    monkeypatch.setattr(P, "_TRANSCRIBE_BACKEND", lambda p: "the secret word is rosebud")
    ok = P.run_probe(P.ProbeSpec(kind=P.AUDIO, media_path=aud,
                                 transcript_contains="rosebud", criterion_id="C1"))
    assert ok.status == P.PASS
    no = P.run_probe(P.ProbeSpec(kind=P.AUDIO, media_path=aud,
                                 transcript_contains="citizen kane", criterion_id="C1"))
    assert no.status == P.FAIL


def test_audio_no_transcriber_unavailable(monkeypatch, wd):
    aud = os.path.join(wd, "a.wav")
    open(aud, "wb").close()
    monkeypatch.setattr(P, "_TRANSCRIBE_BACKEND", None)
    monkeypatch.setattr(P, "_hermes_transcribe", lambda p: None)   # no native STT
    r = P.run_probe(P.ProbeSpec(kind=P.AUDIO, media_path=aud,
                                transcript_contains="x", criterion_id="C1"))
    assert r.status == P.UNAVAILABLE


def test_audio_uses_hermes_native_stt(monkeypatch, wd):
    # when no backend is registered, the probe delegates to Hermes' OWN transcribe_audio
    # (no model download / no re-integration) — proven via the native-STT bridge.
    aud = os.path.join(wd, "a.wav")
    open(aud, "wb").close()
    monkeypatch.setattr(P, "_TRANSCRIBE_BACKEND", None)
    monkeypatch.setattr(P, "_hermes_transcribe",
                        lambda p: ("the deploy is healthy", "hermes:local"))
    r = P.run_probe(P.ProbeSpec(kind=P.AUDIO, media_path=aud,
                                transcript_contains="healthy", criterion_id="C1"))
    assert r.status == P.PASS
    assert r.observed["source"] == "hermes:local"


# --- document -------------------------------------------------------------- #
def test_document_unknown_format_downgrades(wd):
    # docx/latex with no converter present → unobservable downgrade, not a crash.
    doc = os.path.join(wd, "paper.docx")
    open(doc, "wb").close()
    r = P.run_probe(P.ProbeSpec(kind=P.DOCUMENT, media_path=doc,
                                must_contain="Abstract", criterion_id="C1"))
    assert r.status == P.UNOBSERVABLE_STATUS
    assert r.is_downgrade is True


def test_document_missing_file_errors():
    r = P.run_probe(P.ProbeSpec(kind=P.DOCUMENT, media_path="/nope/x.pdf", criterion_id="C1"))
    assert r.status == P.ERROR


@pytest.mark.skipif(not __import__("shutil").which("pdftotext"),
                    reason="pdftotext not present")
def test_document_pdf_text_assertion(wd):
    # build a tiny real PDF with embedded text via a minimal PDF (no external lib).
    pdf = os.path.join(wd, "doc.pdf")
    _write_minimal_pdf(pdf, "HELLO_PROBE_TOKEN")
    hit = P.run_probe(P.ProbeSpec(kind=P.DOCUMENT, media_path=pdf,
                                  must_contain="HELLO_PROBE_TOKEN", criterion_id="C1"),
                      workdir=wd)
    # pdftotext should extract the token → PASS; if the minimal PDF lacks a text layer
    # the probe still must not crash (PASS or FAIL, never ERROR/exception).
    assert hit.status in (P.PASS, P.FAIL)


def _write_minimal_pdf(path, text):
    """Write a minimal single-page PDF containing `text` (enough for pdftotext)."""
    content = (
        f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET"
    ).encode()
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, o)
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
            % (len(objs) + 1, xref))
    with open(path, "wb") as fh:
        fh.write(out)


# --- the four-environment degradation matrix (§3.5) ------------------------ #
def test_degradation_floor_observes_a_receipt_not_prose(_isolate_ladder):
    # NEITHER cmx nor council: the engine still produces a structured RECEIPT judged on
    # the verbatim record (state.db floor), strictly better than a text-only gate.
    _make_store(os.path.join(_isolate_ladder, "state.db"),
                [("assistant", "trust me, the subagent said it's complete")])
    r = P.run_probe(P.ProbeSpec(kind=P.CMX_PROVENANCE,
                                evidence_terms=["complete", "subagent"], criterion_id="C1"))
    # the floor falsifies the prose claim with a real receipt
    assert r.status == P.FAIL
    block = P.format_probe_block([r])
    assert "ENGINE PROBE" in block and "cmx_provenance" in block.lower()
