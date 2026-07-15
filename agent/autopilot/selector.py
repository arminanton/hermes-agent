"""Modal probe selector — derive WHICH observations would break a claim (§3 step 5).

This is the piece that makes verification DYNAMIC per turn. Given the FROZEN acceptance
criteria (invariant #1: the goal text is frozen at run start — the selector reads it but
never redefines it) it derives a probe plan: a list of ``ProbeSpec`` the engine runs to
OBSERVE whether each criterion is actually met, choosing the MODALITY that would falsify
the claim (browser for a web UI, image/vision for a chart, document for a PDF, audio for
a sound file, cmx_provenance for an "a subagent confirmed it" style claim, process for an
executable check, …).

Two layers (owner-approved "try BOTH"):
  1. DETERMINISTIC mapper (no LLM, always on): a frozen criterion → probe specs by
     structure + keyword. Cheap, reproducible, and the floor that always runs.
  2. OPTIONAL LLM enricher (off by default): when a deterministic spec is thin/ambiguous,
     an aux model proposes a sharper probe (e.g. the exact selector+expected value). It
     can only ADD/REFINE within the frozen intent; it can never relax the criterion.

The selector NEVER executes anything — it returns the plan; the loop (step 6) runs it via
the engine (invariant #2: engine-run, never the model). A criterion with no observable
modality yields an ``unobservable`` spec → the loop downgrades (invariant #4).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from agent.autopilot import probes as _probes
from agent.autopilot.probes import ProbeSpec

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# deterministic signal tables (structure + keyword → modality)                  #
# --------------------------------------------------------------------------- #
# Each entry: (compiled pattern, modality, a builder that fills the modal fields).
# Order matters — the FIRST match wins, most-specific first.

_URL_RE = re.compile(r"https?://[^\s)]+")
_FILE_RE = re.compile(r"(?:[\w./~-]+/)?[\w.-]+\.(\w{1,5})\b")

# keyword → modality buckets (lowercased substring match on the criterion text).
_IMAGE_WORDS = ("screenshot", "image", "chart", "graph", "diagram", "renders", "rendered",
                "looks like", "appearance", "visual", "displayed", "logo", "icon", "thumbnail")
_AUDIO_WORDS = ("audio", "sound", "speech", "voice", "transcript", "mp3", "wav", "says ",
                "spoken", "pronounce")
_VIDEO_WORDS = ("video", "clip", "animation", "frame ", "mp4", "playback", "plays")
_DOCUMENT_WORDS = ("pdf", "docx", "document", "report", "latex", "compiled", "page ",
                   "paper", "invoice", "slide")
_BROWSER_WORDS = ("page", "button", "click", "dom", "console", "browser", "web app",
                  "renders in the browser", "ui", "front-end", "frontend", "html",
                  "load", "navigate", "element", "no console errors")
_PROVENANCE_WORDS = ("subagent", "independently confirmed", "independent verification",
                     "a tool confirmed", "verified by", "according to the run",
                     "as confirmed", "i already verified", "owner-gated", "sign-off")
_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "svg", "tiff"}
_AUDIO_EXT = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}
_VIDEO_EXT = {"mp4", "mov", "mkv", "webm", "avi"}
_DOC_EXT = {"pdf", "docx", "odt", "tex", "rtf", "pptx"}


def _has_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _find_file(text: str, exts: set) -> str:
    """Return the first file path in text whose extension is in ``exts``, else ''."""
    for m in _FILE_RE.finditer(text):
        if m.group(1).lower() in exts:
            return m.group(0)
    return ""


def _derive_one(text: str, verify_cmd: str, crit_id: str) -> list:
    """Deterministically map a single criterion to probe spec(s).

    Precedence (most-specific / most-direct observation first):
      1. explicit {verify:} command          → process (strongest, precise)
      2. a NAMED media file (.png/.pdf/.mp3…) → that modality (directly observable)
      3. provenance framing ("a subagent      → cmx_provenance (epistemic claim — beats
         confirmed it", "independently …")       the surface nouns it may also contain)
      4. a URL                                → browser (rendered DOM, not source)
      5. modality KEYWORDS (no file named)    → image/doc/audio/video/browser
      6. a file path + existence words        → artifact
      7. nothing observable                   → unobservable (downgrade)
    """
    low = text.lower()

    # 1) explicit executable check — the strongest, most precise observation.
    if verify_cmd:
        return [ProbeSpec(kind=_probes.PROCESS, command=verify_cmd, criterion_id=crit_id)]

    # 2) a NAMED media file wins regardless of stray keywords (a ".pdf renders with the
    #    logo" is a DOCUMENT, not an image — the file extension is the ground truth).
    img = _find_file(text, _IMAGE_EXT)
    if img:
        return [ProbeSpec(kind=_probes.IMAGE, criterion_id=crit_id, media_path=img,
                          vision_question=f"Does this image satisfy: {text.strip()[:200]}?",
                          expect_visual=_expectation_phrase(text))]
    doc = _find_file(text, _DOC_EXT)
    if doc:
        spec = ProbeSpec(kind=_probes.DOCUMENT, criterion_id=crit_id, media_path=doc,
                         vision_question=f"Does this document satisfy: {text.strip()[:200]}?")
        exp = _expectation_phrase(text)
        if _has_any(low, ("looks", "renders", "appearance", "layout", "chart", "diagram", "logo")):
            spec.expect_visual = exp
        else:
            spec.must_contain = exp
        return [spec]
    aud = _find_file(text, _AUDIO_EXT)
    if aud:
        return [ProbeSpec(kind=_probes.AUDIO, criterion_id=crit_id, media_path=aud,
                          transcript_contains=_expectation_phrase(text))]
    vid = _find_file(text, _VIDEO_EXT)
    if vid:
        return [ProbeSpec(kind=_probes.VIDEO, criterion_id=crit_id, media_path=vid,
                          vision_question=f"Does this video satisfy: {text.strip()[:200]}?",
                          expect_visual=_expectation_phrase(text))]

    # 3) provenance framing is an EPISTEMIC claim ("how we know it") and must be checked
    #    before the generic modality nouns it usually also contains ("…confirmed the PAGE
    #    works"): the bypass-killer takes priority over the surface modality.
    if _has_any(low, _PROVENANCE_WORDS):
        return [ProbeSpec(kind=_probes.CMX_PROVENANCE, criterion_id=crit_id,
                          evidence_terms=_provenance_terms(text))]

    # 4) a URL → browser observation (rendered DOM/console/screenshot, not source).
    url = _URL_RE.search(text)
    if url:
        spec = ProbeSpec(kind=_probes.BROWSER, criterion_id=crit_id, target=url.group(0),
                         require_no_console_errors=("console" in low or "error" in low))
        assertion = _expectation_phrase(text)
        if assertion:
            spec.assert_text = assertion
        return [spec]

    # 5) modality KEYWORDS when no file/URL is named (image before browser so "chart"
    #    routes to vision, then document, audio, video, then generic web-UI language).
    if _has_any(low, _IMAGE_WORDS):
        return [ProbeSpec(kind=_probes.IMAGE, criterion_id=crit_id,
                          vision_question=f"Does this image satisfy: {text.strip()[:200]}?",
                          expect_visual=_expectation_phrase(text))]
    if _has_any(low, _DOCUMENT_WORDS):
        return [ProbeSpec(kind=_probes.DOCUMENT, criterion_id=crit_id,
                          vision_question=f"Does this document satisfy: {text.strip()[:200]}?",
                          must_contain=_expectation_phrase(text))]
    if _has_any(low, _AUDIO_WORDS):
        return [ProbeSpec(kind=_probes.AUDIO, criterion_id=crit_id,
                          transcript_contains=_expectation_phrase(text))]
    if _has_any(low, _VIDEO_WORDS):
        return [ProbeSpec(kind=_probes.VIDEO, criterion_id=crit_id,
                          vision_question=f"Does this video satisfy: {text.strip()[:200]}?",
                          expect_visual=_expectation_phrase(text))]
    if _has_any(low, _BROWSER_WORDS):
        spec = ProbeSpec(kind=_probes.BROWSER, criterion_id=crit_id,
                         require_no_console_errors=("console" in low or "error" in low))
        assertion = _expectation_phrase(text)
        if assertion:
            spec.assert_text = assertion
        return [spec]

    # 6) a file path + existence/content language → artifact probe.
    anyfile = _FILE_RE.search(text)
    if anyfile and ("exist" in low or "create" in low or "write" in low or "file" in low):
        return [ProbeSpec(kind=_probes.ARTIFACT, path=anyfile.group(0), criterion_id=crit_id,
                          must_exist=True, must_be_nonempty=("non-empty" in low or "content" in low))]

    # 7) nothing observable → explicit downgrade signal (invariant #4).
    return [ProbeSpec(kind=_probes.UNOBSERVABLE, criterion_id=crit_id,
                      why_unobservable=f"no deterministic modality maps to: {text.strip()[:160]}")]


# phrases that introduce the expected/observable value in a criterion line.
_EXPECT_MARKERS = ("shows", "displays", "reads", "contains", "says", "equals", "is ",
                   "must show", "should show", "renders", "to ")


def _expectation_phrase(text: str) -> str:
    """Best-effort extract the concrete expected token/phrase from a criterion line.

    e.g. "the counter shows 3" → "3"; "the page contains 'Welcome'" → "Welcome".
    Returns '' when no concrete expectation is recoverable (the probe then observes
    without a strict assertion and hands the observation to the Council).
    """
    # quoted value wins
    q = re.search(r"['\"\u201c\u2018]([^'\"\u201d\u2019]{1,80})['\"\u201d\u2019]", text)
    if q:
        return q.group(1).strip()
    # a number is a strong concrete expectation
    n = re.search(r"\b(\d[\d,.]*)\b", text)
    low = text.lower()
    for mk in _EXPECT_MARKERS:
        idx = low.find(mk)
        if idx >= 0:
            tail = text[idx + len(mk):].strip(" .:-")
            # trim the tail at a clause boundary so the assertion stays tight: stop
            # before a URL, a trailing prepositional clause, or the next conjunction.
            tail = _URL_RE.split(tail)[0].strip()
            tail = re.split(r"\b(?:at|with|on|in|and|but|so that|when)\b", tail, maxsplit=1)[0].strip(" .:-")
            if tail:
                return tail[:80]
    if n:
        return n.group(1)
    return ""


def _provenance_terms(text: str) -> list:
    """Pull the salient nouns/values a provenance claim must be grounded by.

    Keeps alphanumeric tokens length>=4 (drops the boilerplate 'confirmed/subagent/
    independent' framing words so the search targets the actual claimed FACT).
    """
    framing = {"subagent", "independently", "independent", "verification", "confirmed",
               "verified", "according", "already", "owner", "gated", "sign", "off", "the",
               "and", "that", "this", "with", "from", "claim", "claims"}
    toks = re.findall(r"[A-Za-z0-9_]{4,}", text.lower())
    terms = [t for t in toks if t not in framing][:6]
    return terms or ["confirmed"]


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def select_probes(contract, *, llm_enrich: bool = False, max_probes: int = 12) -> list:
    """Derive a probe plan (list[ProbeSpec]) from a FROZEN AcceptanceContract.

    Deterministic-first (always). Only AGENT-achievable criteria get probes — owner-gated
    / unprovable residuals are NEVER continuation reasons (they're handled by the terminus
    logic), so the selector skips them here. With ``llm_enrich=True`` each thin spec is
    offered to an aux model to sharpen (frozen-intent-preserving). Returns at most
    ``max_probes`` specs. Never raises.
    """
    if contract is None or getattr(contract, "is_empty", True):
        return []
    plan: list = []
    try:
        agent_criteria = contract.agent_criteria()
    except Exception:  # noqa: BLE001
        agent_criteria = list(getattr(contract, "criteria", ()))
    for c in agent_criteria:
        try:
            specs = _derive_one(c.text, getattr(c, "verify_cmd", "") or "", c.id)
        except Exception as exc:  # noqa: BLE001 — selection must never crash the gate
            logger.debug("autopilot: selector failed on %s (%s)", getattr(c, "id", "?"), exc)
            specs = [ProbeSpec(kind=_probes.UNOBSERVABLE, criterion_id=getattr(c, "id", ""),
                               why_unobservable="selector error")]
        if llm_enrich:
            specs = [_maybe_enrich(s, c.text) for s in specs]
        plan.extend(specs)
        if len(plan) >= max_probes:
            break
    return plan[:max_probes]


def select_probes_for_claim(claim_text: str, *, criterion_id: str = "claim") -> list:
    """Derive observation probes for a free-form COMPLETION CLAIM the model just made
    (not a contract criterion). This is what catches the bypass face directly.

    A completion claim is PROSE, so the only universally-runnable observation of it is
    against the verbatim record: a provenance-framed claim ("a subagent independently
    confirmed the page works") yields a cmx_provenance probe that falsifies it. Other
    modalities need a concrete artifact/URL the claim text rarely supplies, and a
    target-less browser/image probe would only ERROR — so we emit a probe for a claim
    ONLY when it is provenance-framed OR it names a concrete file/URL we can observe.
    Returns [] for an ordinary claim (nothing falsifiable without a target).
    """
    if not (claim_text or "").strip():
        return []
    derived = _derive_one(claim_text, "", criterion_id)
    out = []
    for s in derived:
        if s.kind == _probes.CMX_PROVENANCE:
            out.append(s)
        elif s.kind in (_probes.IMAGE, _probes.AUDIO, _probes.VIDEO, _probes.DOCUMENT) and s.media_path:
            out.append(s)
        elif s.kind == _probes.BROWSER and s.target:
            out.append(s)
        elif s.kind == _probes.ARTIFACT and s.path:
            out.append(s)
        # a target-less browser/keyword guess or an UNOBSERVABLE is dropped (no probe)
    return out


# --------------------------------------------------------------------------- #
# optional LLM enricher (off by default) — refine within the frozen intent      #
# --------------------------------------------------------------------------- #
def _maybe_enrich(spec: ProbeSpec, criterion_text: str) -> ProbeSpec:
    """Offer a thin spec to an aux model to sharpen it. Frozen-intent-preserving:
    the enricher may fill in a selector / expected value / vision question, but it can
    NEVER change the criterion or relax the assertion. Fails soft to the original spec.
    """
    # only enrich specs that are observably thin (no concrete assertion yet)
    thin = spec.kind in (_probes.BROWSER, _probes.IMAGE, _probes.DOCUMENT, _probes.VIDEO) and not (
        spec.assert_text or spec.expect_visual or spec.expect_text_equals or spec.must_contain
    )
    if not thin:
        return spec
    try:
        from agent.autopilot import council_gate as _cg

        enricher = getattr(_cg, "enrich_probe_spec", None)
        if callable(enricher):
            refined = enricher(spec, criterion_text)
            if isinstance(refined, ProbeSpec):
                return refined
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: probe enrich skipped (%s)", exc)
    return spec
