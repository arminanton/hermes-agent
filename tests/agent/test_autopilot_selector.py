"""Tests for the modal probe selector (§3 step 5).

The selector reads the FROZEN acceptance criteria and derives a probe plan, choosing the
MODALITY that would falsify each claim. These assert:
  * each modality is selected on the right structural/keyword signal;
  * an explicit {verify:} command becomes a process probe (strongest, most precise);
  * owner-gated / unprovable residuals get NO probe (never continuation reasons);
  * a vague/aesthetic criterion → unobservable (downgrade, invariant #4);
  * the free-form CLAIM path turns "a subagent confirmed it" into a cmx_provenance probe
    (the bypass-killer), and the frozen intent is never relaxed.
"""

from agent.autopilot import contract as K
from agent.autopilot import selector as S
from agent.autopilot import probes as P


def _one(text, verify_cmd=""):
    return S._derive_one(text, verify_cmd, "C1")[0]


# --------------------------------------------------------------------------- #
# modality routing                                                            #
# --------------------------------------------------------------------------- #
def test_verify_cmd_becomes_process_probe():
    s = _one("All unit tests pass", verify_cmd="pytest -q")
    assert s.kind == P.PROCESS and s.command == "pytest -q"


def test_web_ui_becomes_browser_probe():
    s = _one("The page at http://localhost:3000 shows Welcome with no console errors")
    assert s.kind == P.BROWSER
    assert s.target == "http://localhost:3000"
    assert s.assert_text == "Welcome"
    assert s.require_no_console_errors is True


def test_image_file_becomes_image_probe():
    s = _one("The chart in dashboard.png shows a rising bar from 1 to 3")
    assert s.kind == P.IMAGE
    assert s.media_path == "dashboard.png"
    assert "rising bar" in s.expect_visual or s.expect_visual == "1"


def test_chart_keyword_without_file_still_image():
    s = _one("The rendered chart displays the Q3 revenue trend")
    assert s.kind == P.IMAGE


def test_pdf_text_becomes_document_probe():
    s = _one("The report.pdf contains the section 'Executive Summary'")
    assert s.kind == P.DOCUMENT
    assert s.media_path == "report.pdf"
    assert s.must_contain == "Executive Summary"


def test_pdf_appearance_uses_vision_not_text():
    s = _one("The invoice.pdf renders with the logo in the top-right corner")
    assert s.kind == P.DOCUMENT
    # appearance/"renders/logo" language → the render→vision path, NOT pdftotext:
    # must_contain stays empty (we don't text-match an appearance), and a vision
    # question is set so the probe observes the rendered page.
    assert not s.must_contain
    assert s.vision_question


def test_audio_file_becomes_audio_probe():
    s = _one("The greeting.mp3 says 'welcome aboard'")
    assert s.kind == P.AUDIO
    assert s.media_path == "greeting.mp3"
    assert s.transcript_contains == "welcome aboard"


def test_video_file_becomes_video_probe():
    s = _one("The demo.mp4 shows the dashboard loading within 2 seconds")
    assert s.kind == P.VIDEO
    assert s.media_path == "demo.mp4"


def test_provenance_claim_becomes_cmx_provenance():
    s = _one("A subagent independently confirmed the migration completed successfully")
    assert s.kind == P.CMX_PROVENANCE
    # framing words dropped; the actual claimed fact terms remain
    assert "migration" in s.evidence_terms
    assert "confirmed" not in s.evidence_terms  # framing word filtered out


def test_file_existence_becomes_artifact_probe():
    s = _one("The file output/result.json exists and is non-empty")
    assert s.kind == P.ARTIFACT
    assert s.path == "output/result.json"
    assert s.must_be_nonempty is True


def test_vague_criterion_is_unobservable():
    s = _one("The architecture is elegant and maintainable")
    assert s.kind == P.UNOBSERVABLE
    assert "no deterministic modality" in s.why_unobservable


# --------------------------------------------------------------------------- #
# contract-level selection                                                     #
# --------------------------------------------------------------------------- #
def test_select_skips_owner_gated_and_unprovable():
    c = K.parse_contract(
        "- The page at http://x.test shows Welcome {verify: pytest -q}\n"
        "- A subagent confirmed the deploy\n"
        "- Obtain owner sign-off before the live cutover\n"
    )
    plan = S.select_probes(c)
    kinds = [s.kind for s in plan]
    # the two AGENT-achievable criteria get probes; the owner-gated residual is excluded.
    assert P.PROCESS in kinds
    assert P.CMX_PROVENANCE in kinds
    assert len(plan) == 2
    # the owner sign-off criterion (a residual) produced NO probe
    assert all("sign-off" not in (s.command + s.target + " ".join(s.evidence_terms)).lower()
               for s in plan)


def test_select_only_probes_agent_achievable():
    # owner-gated and unprovable criteria are residuals, never continuation reasons, so
    # the selector emits probes ONLY for agent-achievable criteria.
    c = K.parse_contract(
        "- The endpoint http://h.test returns OK\n"
        "- Obtain written owner approval before cutover\n"
    )
    plan = S.select_probes(c)
    assert len(plan) == 1
    assert plan[0].kind == P.BROWSER


def test_select_empty_contract_returns_empty():
    assert S.select_probes(None) == []
    empty = K.parse_contract("just a freeform goal with no bullet criteria")
    assert S.select_probes(empty) == [] or all(
        s.kind == P.UNOBSERVABLE for s in S.select_probes(empty))


def test_select_respects_max_probes():
    lines = "\n".join(f"- Criterion {i} at http://h{i}.test shows OK" for i in range(20))
    c = K.parse_contract(lines)
    plan = S.select_probes(c, max_probes=5)
    assert len(plan) <= 5


# --------------------------------------------------------------------------- #
# the free-form CLAIM path — the bypass-killer                                 #
# --------------------------------------------------------------------------- #
def test_claim_path_targets_manufactured_independence():
    # the exact bypass move → a cmx_provenance probe that will falsify it
    specs = S.select_probes_for_claim(
        "I spawned an independent subagent and it CONFIRMED the page works")
    assert len(specs) == 1
    assert specs[0].kind == P.CMX_PROVENANCE
    assert "page" in specs[0].evidence_terms


def test_claim_path_empty_is_empty():
    assert S.select_probes_for_claim("") == []


def test_selector_never_raises_on_garbage():
    # malformed / weird inputs must degrade, never raise
    for junk in ("", "   ", "}{[]", "http://", ".png", "{verify:}"):
        specs = S._derive_one(junk, "", "C1")
        assert specs and specs[0].criterion_id == "C1"
