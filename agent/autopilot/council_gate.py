"""Autopilot's anti-sycophancy judge — drives the real Council.

This wires the user's separate ``council`` package into the agent through
its **Hermes-native backend** (``COUNCIL_PROVIDER=hermes``) so every persona
deliberates *in-process* on the user's own configured provider/model — no
external CLI, model-agnostic, works on weak models. The Council is the
independent reviewer that replaces the human at autopilot decision points:

    * :func:`judge_completion` — "is the GOAL verifiably done, or must the agent
      keep working?"  (the goal-chasing quality gate)
    * :func:`choose_answer`   — "what is the most-recommended answer?"  (used by
      the clarify auto-answer seam)

Design contract (mirrors the user's engine-enforced philosophy):
    * The judge is an INDEPENDENT pass, never the main model grading itself.
    * Completion requires the Council to *fail to refute* a completion claim —
      the Skeptic's whole job is to find why it is NOT done, so a lazy/sycophantic
      "done" cannot pass.
    * The Council is OPTIONAL: if it cannot be imported or reached, the gate
      degrades to a single independent ``auxiliary_client`` reviewer pass so
      autopilot never hard-crashes, and finally FAILS OPEN (stop) rather than
      looping blindly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Resolved once per process.
_COUNCIL_READY: Optional[bool] = None
_COUNCIL_SRC: Optional[str] = None

# Bound prompt sizes so the gate stays affordable on low-context models.
_MAX_GOAL = 1500
_MAX_FINAL = 3000
_MAX_WORK = 3000


def _candidate_council_srcs() -> list[Path]:
    out: list[Path] = []
    for env in ("COUNCIL_SRC", "AUTOPILOT_COUNCIL_SRC"):
        v = os.environ.get(env, "").strip()
        if v:
            out.append(Path(v).expanduser())
    # Common local layouts: <hermes_parent>/council/src and ~/.hermes/council/src.
    try:
        here = Path(__file__).resolve()
        # .../<checkout>/agent/autopilot/council_gate.py -> parents[3] == <parent of checkout>
        parent_of_checkout = here.parents[3]
        out.append(parent_of_checkout / "council" / "src")
    except Exception:
        pass
    out.append(Path.home() / ".hermes" / "council" / "src")
    return out


def ensure_council_importable(council_model: str = "") -> bool:
    """Locate the Council package, add it to ``sys.path``, select the hermes lane.

    Returns True if ``council`` is importable afterwards. Idempotent.
    """
    global _COUNCIL_READY, _COUNCIL_SRC
    if _COUNCIL_READY is not None:
        return _COUNCIL_READY

    # Operator/test escape hatch: force the single-aux-reviewer fallback lane instead of
    # the full Council (e.g. to pin one explicit reviewer model, or in CI).
    if os.environ.get("AUTOPILOT_DISABLE_COUNCIL", "").strip().lower() in {"1", "true", "yes", "on"}:
        _COUNCIL_READY = False
        return False

    # Default the Council backend to the Hermes-native lane unless the operator
    # has explicitly pointed it somewhere else (e.g. a CLI provider).
    os.environ.setdefault("COUNCIL_PROVIDER", "hermes")
    if council_model:
        os.environ.setdefault("COUNCIL_HERMES_MODEL", council_model)

    try:
        import council.deliberation  # noqa: F401  (already on path)
        _COUNCIL_READY = True
        return True
    except Exception:
        pass

    for src in _candidate_council_srcs():
        try:
            libs = src / "libs"
            if (libs / "council" / "deliberation.py").exists():
                if str(libs) not in sys.path:
                    sys.path.insert(0, str(libs))
                os.environ.setdefault("COUNCIL_SRC", str(src))
                import council.deliberation  # noqa: F401
                _COUNCIL_SRC = str(src)
                _COUNCIL_READY = True
                logger.info("autopilot: Council loaded from %s", src)
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("autopilot: council candidate %s failed: %s", src, exc)
            continue

    _COUNCIL_READY = False
    logger.info("autopilot: Hermes Council not available; using auxiliary reviewer fallback")
    return False


def _trunc(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]"


@dataclass
class CompletionVerdict:
    """Result of the goal-completion quality gate."""

    complete: bool
    directive: str = ""          # the next-action directive when not complete
    confidence: float = 0.0
    verdict: str = ""            # raw council verdict: allow|deny|conditional
    source: str = "council"      # council | aux | fallback
    summary: str = ""            # short human-readable rationale
    raw: dict[str, Any] = field(default_factory=dict)


_COMPLETION_FRAME = (
    "You are deciding whether an autonomous coding agent may STOP now because its "
    "GOAL is fully and verifiably COMPLETE. Apply these verdict semantics strictly:\n"
    "- 'allow'       = STOP: the goal is genuinely, verifiably complete; no substantive work remains.\n"
    "- 'deny'        = DO NOT STOP: substantive required work is missing, wrong, or only promised.\n"
    "- 'conditional' = NOT YET: only specific verification or checks remain before it can be called done.\n"
    "Judge by evidence in the agent's ACTUAL result, not by how confident it sounds. "
    "A polished claim of completion with no verification is NOT complete. Promises of "
    "future work ('I will…', 'next I'd…') are NOT completion."
)


def _completion_question(goal: str, work_summary: str, final_response: str) -> str:
    return (
        f"{_COMPLETION_FRAME}\n\n"
        f"GOAL:\n{_trunc(goal, _MAX_GOAL)}\n\n"
        f"AGENT'S LATEST RESULT (what it would deliver as final):\n"
        f"{_trunc(final_response, _MAX_FINAL)}\n\n"
        f"WORK CONTEXT (recent steps):\n{_trunc(work_summary, _MAX_WORK)}"
    )


def _compose_directive(arbiter: dict[str, Any], deliberations: list[dict[str, Any]]) -> str:
    """Turn a 'not complete' council result into a concrete next-step directive."""
    bits: list[str] = []
    wrong = str(arbiter.get("most_likely_wrong_point", "") or "").strip()
    if wrong:
        bits.append(f"Gap found by independent review: {wrong}")
    checks = [str(c).strip() for c in (arbiter.get("required_checks") or []) if str(c).strip()]
    fastest = str(arbiter.get("fastest_uncertainty_reducing_check", "") or "").strip()
    if checks:
        bits.append("Do these next: " + "; ".join(checks[:4]) + ".")
    elif fastest:
        bits.append(f"Do this next: {fastest}.")
    else:
        # Fall back to the sharpest critic's key point.
        for d in deliberations:
            kp = d.get("key_points") or []
            claim = str(d.get("claim", "") or "").strip()
            if kp:
                bits.append("Address: " + "; ".join(str(k) for k in kp[:3]) + ".")
                break
            if claim:
                bits.append(f"Address: {claim}")
                break
    safe = str(arbiter.get("safest_reversible_path", "") or "").strip()
    if safe:
        bits.append(f"Safest path: {safe}")
    if not bits:
        bits.append("The goal is not yet verifiably complete; identify and finish the remaining work.")
    return " ".join(bits)


def _gate_panel() -> str:
    """The Council panel/preset the autopilot completion gate should use.

    Historically the gate hardcoded ``mode="fast"``, which selects the
    manufactured-disagreement ``fast`` panel: a single adversarial critic tuned
    to MANUFACTURE an objection every round. On a completion claim that is
    already substantively done, that panel never converges — it invents a fresh,
    ever-smaller PRESENTATION ask each tick (`conditional 0.55–0.62` forever),
    so the run only ends via the refinement-churn safety valve after burning
    several dead rounds. (Observed live on the NuData run, 2026-08-05.)

    The operator can point the gate at a convergent, audited panel instead
    (e.g. ``ship_gate_audited`` = Security + Compliance + Pre-mortem + Oracle +
    Evidence-Auditor + Sycophancy-Auditor), which weighs the deterministic
    receipts rather than reflexively objecting. Read from
    ``COUNCIL_GATE_PANEL`` (already set in the shipped config's council MCP env);
    empty falls back to the legacy fast-mode behaviour so nothing changes for
    operators who haven't opted in.
    """
    return os.environ.get("COUNCIL_GATE_PANEL", "").strip()


def _council_run(question: str, *, mode: str, max_tokens: int,
                 evidence_receipts: Optional[list] = None) -> dict[str, Any]:
    """Seam: run the real Council (kept separate so tests can stub it).

    When ``evidence_receipts`` are supplied (deterministic engine verification
    receipts), they are passed to the engine as first-class evidence so the
    deliberators reason against the engine's actual re-run, not the agent's prose.
    Falls back gracefully if the installed engine predates the receipts parameter.

    When ``COUNCIL_GATE_PANEL`` is set it is passed as the explicit ``panel`` so
    the completion gate uses a convergent audited panel instead of the
    never-settling ``fast`` critic (see :func:`_gate_panel`). An engine too old
    to accept ``panel`` degrades cleanly back to ``mode``.
    """
    from council.deliberation import run_council

    kwargs: dict[str, Any] = dict(mode=mode, evidence_search=False, max_tokens=max_tokens)
    panel = _gate_panel()
    if panel:
        kwargs["panel"] = panel

    def _invoke(**extra: Any) -> dict[str, Any]:
        call_kwargs = dict(kwargs, **extra)
        try:
            return run_council(question, **call_kwargs)
        except TypeError:
            # An older engine may lack `panel` and/or `evidence_receipts`. Retry
            # without the optional kwargs rather than failing the whole gate
            # (which would force the receipts-only fallback every tick).
            call_kwargs.pop("panel", None)
            call_kwargs.pop("evidence_receipts", None)
            if "panel" in kwargs:
                logger.debug("autopilot: council engine rejected optional kwargs; degrading")
            return run_council(question, **call_kwargs)

    if evidence_receipts:
        return _invoke(evidence_receipts=evidence_receipts)
    return _invoke()


def judge_completion(
    goal: str,
    work_summary: str,
    final_response: str,
    *,
    mode: str = "fast",
    council_model: str = "",
    max_tokens: int = 1200,
) -> CompletionVerdict:
    """Decide whether the goal is verifiably complete (the goal-chasing gate).

    Uses the real Hermes Council when available; otherwise a single independent
    auxiliary reviewer pass. Never raises — on total failure it FAILS OPEN
    (``complete=True``) so a broken judge cannot trap the user in a loop.
    """
    question = _completion_question(goal, work_summary, final_response)

    if ensure_council_importable(council_model):
        try:
            res = _council_run(question, mode=mode, max_tokens=max_tokens)
            verdict = str(res.get("verdict", "")).strip().lower()
            confidence = float(res.get("confidence", 0.0) or 0.0)
            arbiter = res.get("arbiter", {}) or {}
            deliberations = res.get("deliberations", []) or []
            complete = verdict == "allow"
            directive = "" if complete else _compose_directive(arbiter, deliberations)
            syco = res.get("sycophancy", {}) or {}
            summary = (
                f"council verdict={verdict or '?'} confidence={confidence:.2f} "
                f"panel={res.get('meta', {}).get('panel', '?')} "
                f"sycophancy={syco.get('overall', 0.0)}"
            )
            return CompletionVerdict(
                complete=complete,
                directive=directive,
                confidence=confidence,
                verdict=verdict,
                source="council",
                summary=summary,
                raw=res,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: council judge failed (%s); falling back to aux reviewer", exc)

    return _aux_completion(goal, work_summary, final_response, council_model=council_model)


# --------------------------------------------------------------------------- #
# Structured per-criterion satisfaction (autopilot health, "Ask 1").           #
#                                                                              #
# The textual heuristic in driver._update_satisfied_criteria infers which      #
# acceptance criteria are met from keyword overlap. That under-counts by design #
# but is fuzzy. When a frozen contract exists we can do better: ask the Council #
# to return an EXPLICIT per-criterion verdict, grounded in the engine's         #
# verification receipts (deterministic re-run) when present. The receipts make  #
# this exact rather than vibes: "C03 satisfied BECAUSE pytest exited 0".        #
# --------------------------------------------------------------------------- #
@dataclass
class CriterionVerdict:
    """Per-criterion satisfaction decision from the structured judge."""

    criterion_id: str
    satisfied: bool
    confidence: float = 0.0
    evidence: str = ""
    grounded_in_receipt: bool = False


@dataclass
class StructuredCompletion:
    """judge_criteria result: the overall verdict + a per-criterion map."""

    criteria: dict = field(default_factory=dict)  # id -> CriterionVerdict
    source: str = "council"
    raw: dict = field(default_factory=dict)

    def satisfied_ids(self) -> set:
        return {cid for cid, v in self.criteria.items() if v.satisfied}


_CRITERIA_FRAME = (
    "You are an adversarial completion auditor for an autonomous agent. For EACH "
    "listed acceptance criterion, decide whether it is VERIFIABLY satisfied by the "
    "agent's actual result and the ENGINE VERIFICATION RECEIPTS below. The receipts "
    "are a deterministic re-run performed by the engine OUTSIDE the agent's control "
    "— trust them over the agent's prose. A criterion backed by a PASS receipt is "
    "satisfied; one backed by a FAIL/REFUSED/absent receipt is NOT satisfied unless "
    "the result itself carries unambiguous proof. Do not accept a polished claim "
    "with no evidence. Respond with ONLY one JSON object: "
    '{"criteria": [{"id": "C01", "satisfied": true|false, "confidence": 0..1, '
    '"evidence": "one phrase: what proves/disproves it"}]}.'
)


def _criteria_question(goal: str, final_response: str, criteria: list, receipts_block: str) -> str:
    lines = "\n".join(f"- {c.id}: {c.text}" for c in criteria)
    rec = f"\n\nENGINE VERIFICATION RECEIPTS:\n{receipts_block}" if receipts_block else (
        "\n\nENGINE VERIFICATION RECEIPTS: (none — no executable checks ran this turn)"
    )
    return (
        f"{_CRITERIA_FRAME}\n\n"
        f"GOAL:\n{_trunc(goal, _MAX_GOAL)}\n\n"
        f"ACCEPTANCE CRITERIA TO AUDIT:\n{lines}\n\n"
        f"AGENT'S LATEST RESULT:\n{_trunc(final_response, _MAX_FINAL)}"
        f"{_trunc(rec, _MAX_WORK)}"
    )


def judge_criteria(
    goal: str,
    final_response: str,
    criteria: list,
    *,
    receipts_block: str = "",
    receipt_satisfied_ids: Optional[set] = None,
    receipt_dicts: Optional[list] = None,
    mode: str = "fast",
    council_model: str = "",
    max_tokens: int = 1200,
) -> StructuredCompletion:
    """Return an explicit per-criterion satisfaction map.

    Deterministic receipts win: any id in ``receipt_satisfied_ids`` (a check that
    actually exited 0) is marked satisfied and ``grounded_in_receipt=True`` REGARDLESS
    of the Council's text opinion — a real PASS is ground truth. For the remaining
    criteria the Council's structured verdict is used. Never raises; on judge failure
    it returns only the receipt-grounded ids (fail-closed: unproven => unsatisfied).
    """
    receipt_satisfied_ids = receipt_satisfied_ids or set()
    out = StructuredCompletion()
    by_id = {c.id: c for c in criteria}

    # 1) receipts are ground truth — seed them first.
    for cid in receipt_satisfied_ids:
        if cid in by_id:
            out.criteria[cid] = CriterionVerdict(
                criterion_id=cid, satisfied=True, confidence=1.0,
                evidence="engine verification receipt: check exited 0", grounded_in_receipt=True,
            )

    remaining = [c for c in criteria if c.id not in out.criteria]
    if not remaining:
        out.source = "receipts"
        return out

    question = _criteria_question(goal, final_response, remaining, receipts_block)
    try:
        if ensure_council_importable(council_model):
            res = _council_run(question, mode=mode, max_tokens=max_tokens,
                               evidence_receipts=receipt_dicts)
            out.raw = res
            # The structured per-criterion payload rides in the arbiter's free-form
            # answer; parse it out of the raw council result's text fields.
            parsed = _parse_criteria_payload(res)
            out.source = "council"
        else:
            parsed = _aux_judge_criteria(question, council_model=council_model)
            out.source = "aux"
        for item in parsed:
            cid = str(item.get("id", "")).strip()
            if cid in by_id and cid not in out.criteria:
                out.criteria[cid] = CriterionVerdict(
                    criterion_id=cid,
                    satisfied=bool(item.get("satisfied", False)),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    evidence=str(item.get("evidence", "") or "")[:200],
                    grounded_in_receipt=False,
                )
    except Exception as exc:  # noqa: BLE001 — structured judge must never crash the gate
        logger.warning("autopilot: judge_criteria failed (%s); receipts-only", exc)
        out.source = out.source or "fallback"
    return out


def _parse_criteria_payload(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the per-criterion list out of a council result. Looks in the arbiter
    answer text and the top-level fields, tolerating either {"criteria":[…]} or a
    bare list."""
    candidates: list[Any] = []
    arb = res.get("arbiter", {}) or {}
    for blob in (arb.get("answer"), arb.get("summary"), res.get("backend_text"), res.get("answer")):
        data = _extract_json(blob if isinstance(blob, str) else "")
        if data:
            candidates.append(data)
    if isinstance(res.get("criteria"), list):
        candidates.append({"criteria": res["criteria"]})
    for data in candidates:
        crits = data.get("criteria") if isinstance(data, dict) else None
        if isinstance(crits, list) and crits:
            return crits
    return []


def _aux_judge_criteria(question: str, *, council_model: str = "") -> list[dict[str, Any]]:
    """Single auxiliary-reviewer fallback for the structured per-criterion judge."""
    content = _aux_call(
        [
            {"role": "system", "content": "You are an adversarial completion auditor. Respond with ONLY the requested JSON object."},
            {"role": "user", "content": question},
        ],
        model=council_model, max_tokens=900, timeout=90,
    )
    data = _extract_json(content) or {}
    crits = data.get("criteria")
    return crits if isinstance(crits, list) else []


# --------------------------------------------------------------------------- #
# Fallback: a single independent auxiliary-model reviewer pass.                #
# Still NOT the main model grading itself — call_llm resolves an aux backend.  #
# --------------------------------------------------------------------------- #

_AUX_SYSTEM = (
    "You are an adversarial completion reviewer for an autonomous coding agent. "
    "You are NOT the agent and you do not trust it. Decide whether its GOAL is "
    "fully and verifiably complete. Reject lazy or sycophantic 'done' claims and "
    "promises of future work. Respond with ONLY one JSON object: "
    '{"complete": true|false, "confidence": 0..1, "next_action": "single concrete '
    'next step if not complete, else empty", "reason": "one sentence"}.'
)


def _aux_call(messages: list[dict[str, Any]], *, model: str, max_tokens: int, timeout: float) -> str:
    """Seam: single auxiliary-model call (kept separate so tests can stub it).

    ``model`` may be a bare model id ("claude-opus-4.8-fast") or a ``provider/model``
    form ("copilot/claude-opus-4.8-fast") — the latter pins the aux call to a specific
    provider instead of auxiliary_client's auto-detect (needed when the only authed lane
    isn't the auto-detect default, e.g. copilot). An explicit provider via
    ``AUTOPILOT_COUNCIL_PROVIDER`` / ``autopilot.council_provider`` takes precedence.
    """
    from agent.auxiliary_client import call_llm

    provider = os.environ.get("AUTOPILOT_COUNCIL_PROVIDER", "").strip()
    model_id = model
    if model and "/" in model and not provider:
        head, _, tail = model.partition("/")
        if head and tail:
            provider, model_id = head, tail

    # "-fast" is NOT a model id — it's the Anthropic Fast Mode KNOB (extra_body
    # {"speed":"fast"} + the fast beta header, ~2.5x output throughput). The wire model is
    # the base id; sending the literal "...-fast" string 400s model_not_supported. So
    # strip the suffix and translate it into the speed override. [grounded:
    # anthropic_adapter _supports_fast_mode + build_anthropic_kwargs extra_body["speed"]]
    fast_mode = False
    if model_id and model_id.endswith("-fast"):
        model_id = model_id[: -len("-fast")]
        fast_mode = True

    # Claude-on-Copilot MUST ride the Anthropic /v1/messages transport, not the OpenAI
    # /chat/completions path (which clamps to 168k AND rejects opus with a 400
    # model_not_supported / assistant-prefill error). call_llm doesn't expose api_mode,
    # so resolve the Anthropic-wire client directly (the agent_init copilot+claude
    # override, mirrored). [grounded: auxiliary_client _force_copilot_claude + 1M fix]
    if provider == "copilot" and "claude" in (model_id or "").lower():
        try:
            from agent.auxiliary_client import resolve_provider_client

            client, final_model, *_ = resolve_provider_client(
                "copilot", model=model_id, api_mode="anthropic_messages")
            if client is not None:
                create_kwargs: dict[str, Any] = {
                    "model": final_model, "messages": messages, "max_tokens": max_tokens}
                if fast_mode:
                    create_kwargs["extra_body"] = {"speed": "fast"}
                resp = client.chat.completions.create(**create_kwargs)
                return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — fall back to the generic path
            logger.debug("autopilot: copilot anthropic_messages route failed (%s)", exc)

    kwargs: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "timeout": timeout}
    if model_id:
        kwargs["model"] = model_id
    if provider:
        kwargs["provider"] = provider
    if fast_mode:
        kwargs["extra_body"] = {"speed": "fast"}
    resp = call_llm(**kwargs)
    return resp.choices[0].message.content or ""


def _aux_completion(
    goal: str, work_summary: str, final_response: str, *, council_model: str = ""
) -> CompletionVerdict:
    try:
        user = (
            f"GOAL:\n{_trunc(goal, _MAX_GOAL)}\n\n"
            f"AGENT'S LATEST RESULT:\n{_trunc(final_response, _MAX_FINAL)}\n\n"
            f"WORK CONTEXT:\n{_trunc(work_summary, _MAX_WORK)}"
        )
        content = _aux_call(
            [
                {"role": "system", "content": _AUX_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=council_model,
            max_tokens=600,
            timeout=90,
        )
        data = _extract_json(content) or {}
        complete = bool(data.get("complete", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        next_action = str(data.get("next_action", "") or "").strip()
        reason = str(data.get("reason", "") or "").strip()
        directive = "" if complete else (next_action or "Finish the remaining work toward the goal.")
        return CompletionVerdict(
            complete=complete,
            directive=directive,
            confidence=confidence,
            verdict="allow" if complete else "deny",
            source="aux",
            summary=f"aux reviewer complete={complete} ({reason[:80]})",
            raw=data,
        )
    except Exception as exc:  # noqa: BLE001
        # Total judge failure: fail OPEN (stop) so we never loop blindly.
        logger.warning("autopilot: aux reviewer failed (%s); failing open (stop)", exc)
        return CompletionVerdict(
            complete=True,
            directive="",
            confidence=0.0,
            verdict="allow",
            source="fallback",
            summary=f"judge unavailable ({exc}); delivered result",
        )


def _council_decision(options: list[str], decision_context: str) -> dict[str, Any]:
    """Seam: council multi-option decision (separate so tests can stub it)."""
    from council.deliberation import decision as _decision

    return _decision(options, decision_context=decision_context)


def _match_option(text: str, options: list[str]) -> str:
    """Return the single option that clearly appears in ``text``, else ''."""
    low = (text or "").lower()
    hits = [o for o in options if o and o.lower() in low]
    return hits[0] if len(hits) == 1 else ""


_PICK_SYSTEM = (
    "You stand in for an absent user and must choose the MOST DEFENSIBLE answer "
    "to a question an autonomous agent asked — not the most agreeable or easiest. "
    "Judge by evidence and consequences. Respond with ONLY one JSON object: "
    '{"choice": "the exact chosen option text, or your concise answer if open-ended", '
    '"rationale": "one sentence"}.'
)


def _aux_pick(question: str, options: list[str], *, context: str = "", council_model: str = "") -> str:
    """Independent reviewer that picks the most-defensible answer."""
    opt_block = ("\nOPTIONS (choose exactly one, return its exact text):\n- " + "\n- ".join(options)) if options else ""
    ctx_block = f"\nIndependent review notes:\n{_trunc(context, 1200)}" if context else ""
    user = f"QUESTION:\n{_trunc(question, _MAX_GOAL)}{opt_block}{ctx_block}"
    try:
        content = _aux_call(
            [{"role": "system", "content": _PICK_SYSTEM}, {"role": "user", "content": user}],
            model=council_model, max_tokens=400, timeout=90,
        )
        data = _extract_json(content) or {}
        choice = str(data.get("choice", "") or "").strip()
        if options:
            exact = _match_option(choice, options) or _match_option(question + " " + choice, options)
            return exact or choice or options[0]
        return choice
    except Exception as exc:  # noqa: BLE001
        logger.warning("autopilot: aux pick failed (%s)", exc)
        return options[0] if options else ""


@dataclass
class ClarifyDecision:
    """Result of the clarify human-surrogate, with enough context for the ADR."""

    answer: str = ""
    options: list[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "aux"          # council | aux | fallback


def choose_answer_detailed(
    question: str,
    options: Optional[list[str]] = None,
    *,
    council_model: str = "",
    max_tokens: int = 1200,
) -> ClarifyDecision:
    """Most-recommended answer to a clarify question, plus the decision context.

    Multi-option questions go through the Council's adversarial ``decision`` so
    the pick is anti-sycophantic; open-ended questions get the Council's safest
    recommended path. Falls back to a single independent reviewer pass, then to
    the first option, so a clarify call in autopilot always resolves. Returns the
    chosen answer alongside the options it weighed, a one-line rationale, and
    which reviewer produced it — the autopilot ADR records all of it.
    """
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]

    if ensure_council_importable(council_model):
        try:
            if opts:
                res = _council_decision(opts, question)
            else:
                res = _council_run(
                    f"What is the single most defensible answer to this question, and why?\n{question}",
                    mode="fast", max_tokens=max_tokens,
                )
            arb = res.get("arbiter", {}) or {}
            notes = " ".join(
                str(arb.get(k, "") or "")
                for k in ("safest_reversible_path", "what_evidence_supports", "most_likely_wrong_point")
            )
            rationale = str(arb.get("safest_reversible_path", "") or "").strip()
            if opts:
                direct = _match_option(notes, opts)
                if direct:
                    return ClarifyDecision(answer=direct, options=opts, rationale=rationale, source="council")
                picked = _aux_pick(question, opts, context=notes, council_model=council_model)
                return ClarifyDecision(answer=picked, options=opts, rationale=rationale, source="council")
            ans = rationale
            if ans:
                return ClarifyDecision(answer=ans, options=opts, rationale=rationale, source="council")
        except Exception as exc:  # noqa: BLE001
            logger.warning("autopilot: council choose_answer failed (%s); aux fallback", exc)

    picked = _aux_pick(question, opts, council_model=council_model)
    return ClarifyDecision(answer=picked, options=opts, rationale="", source="aux")


def choose_answer(
    question: str,
    options: Optional[list[str]] = None,
    *,
    council_model: str = "",
    max_tokens: int = 1200,
) -> str:
    """Most-recommended answer to a clarify question — the human-surrogate.

    Thin string wrapper over :func:`choose_answer_detailed` so existing callers
    that only need the answer text are unaffected.
    """
    return choose_answer_detailed(
        question, options, council_model=council_model, max_tokens=max_tokens
    ).answer


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Minimal JSON-object extraction (fenced or inline)."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        s = s.split("\n", 1)[-1] if "\n" in s else s
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None
