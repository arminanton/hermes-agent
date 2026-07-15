"""Probe loop wiring — select → engine-run → receipts → judge (§3 step 6).

This is the seam that makes the senses ACTIVE in the autopilot gate. Each turn, when the
goal is not yet self-evidently complete, the engine:

  1. SELECTS a modal probe plan from the FROZEN contract (selector.select_probes) AND from
     the model's own completion CLAIM (selector.select_probes_for_claim) — the latter is
     what catches the bypass face ("a subagent confirmed it" → a cmx_provenance probe);
  2. RUNS each probe itself (probes.run_probe — engine-run, invariant #2), bounded by a
     per-turn wall-clock budget;
  3. converts each ``ProbeReceipt`` into the SAME ``verification.Receipt`` shape the
     existing gap-closure machinery already consumes, so the Council judges on the
     OBSERVATION, not the prose, with no change to the downstream judge;
  4. records an ADR ``could-not-observe`` note for every probe that DOWNGRADED
     (unobservable/unavailable) — invariant #4 — so the run is auditable.

It is fail-soft end-to-end: any failure leaves the existing verification report untouched
(the probe layer is ADDITIVE, never a regression). Bounded by ``autopilot.probe_budget``
(default 120s/turn) and gated by ``autopilot.probes`` (default ON; the probes themselves
no-op cleanly when their backends are absent, so "on" is safe).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agent.autopilot import probes as _probes
from agent.autopilot import selector as _selector
from agent.autopilot import verification as _verification

logger = logging.getLogger(__name__)

# map a ProbeReceipt status → the verification.Receipt status vocabulary the
# gap-closure machinery understands (pass | fail | error | skipped). A DOWNGRADE
# (unobservable/unavailable) is recorded as 'skipped' so it NEVER counts as either
# satisfied or failed — it routes the criterion to the Council's text judgment.
_STATUS_MAP = {
    _probes.PASS: "pass",
    _probes.FAIL: "fail",
    _probes.ERROR: "error",
    _probes.UNAVAILABLE: "skipped",
    _probes.UNOBSERVABLE_STATUS: "skipped",
}


def probes_enabled(agent: Any) -> bool:
    """Whether the modal probe layer runs. Default ON; disable with
    autopilot.probes=false / AUTOPILOT_PROBES=0. (Probes no-op when backends are
    absent, so ON is safe — it simply adds observation when it's possible.)"""
    val = getattr(agent, "_autopilot_probes", None)
    if val is not None:
        return bool(val)
    env = os.environ.get("AUTOPILOT_PROBES", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _probe_budget(agent: Any) -> float:
    val = getattr(agent, "_autopilot_probe_budget", None)
    if val is not None:
        try:
            return float(val)
        except Exception:  # noqa: BLE001
            pass
    env = os.environ.get("AUTOPILOT_PROBE_BUDGET", "").strip()
    if env:
        try:
            return float(env)
        except Exception:  # noqa: BLE001
            pass
    return 120.0


def _per_probe_timeout(agent: Any) -> float:
    val = getattr(agent, "_autopilot_probe_timeout", None)
    if val is not None:
        try:
            return float(val)
        except Exception:  # noqa: BLE001
            pass
    return 60.0


def _to_verification_receipt(pr: "_probes.ProbeReceipt") -> "_verification.Receipt":
    """Convert a ProbeReceipt into the verification.Receipt shape (so the existing
    judge/gap-closure path consumes it unchanged)."""
    status = _STATUS_MAP.get(pr.status, "error")
    detail = (pr.detail or "")[:400]
    return _verification.Receipt(
        criterion_id=pr.criterion_id,
        command=f"probe:{pr.kind} {pr.summary}"[:200],
        status=status,
        exit_code=0 if status == "pass" else (1 if status == "fail" else None),
        stdout_tail=pr.summary[:400],
        stderr_tail=detail if status in ("fail", "error") else "",
        duration_s=0.0,
        ran_at=pr.ran_at or datetime.now(timezone.utc).isoformat(),
    )


def run_probe_plan(
    agent: Any,
    contract: Any,
    final_response: str,
    *,
    workdir: Optional[str] = None,
) -> tuple:
    """Select + run the modal probe plan; return (probe_receipts, downgrades).

    ``probe_receipts`` are verification.Receipt objects (mergeable into the existing
    VerificationReport). ``downgrades`` is a list of (criterion_id, kind, reason) for the
    ADR could-not-observe note. Never raises — returns ([], []) on any failure or when
    disabled.
    """
    if not probes_enabled(agent):
        return ([], [])
    try:
        plan = list(_selector.select_probes(contract))
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: probe selection failed (%s)", exc)
        plan = []
    # ALSO probe the model's own completion claim (the bypass-killer): a claim like
    # "a subagent independently confirmed it" becomes a cmx_provenance probe.
    try:
        plan.extend(_selector.select_probes_for_claim(final_response, criterion_id="_claim"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("autopilot: claim-probe selection failed (%s)", exc)
    if not plan:
        return ([], [])

    # The {verify:} COMMAND domain belongs to the verification harness (which enforces
    # the operator's execution opt-in + allowlist). The modal probe layer owns
    # OBSERVATION only, so drop process-kind specs here to avoid (a) double-running a
    # check the harness already runs and (b) bypassing the verification_exec gate.
    plan = [s for s in plan if s.kind != _probes.PROCESS]
    if not plan:
        return ([], [])

    wd = workdir or _verification._workdir(agent)  # reuse the harness workdir resolution
    per_timeout = _per_probe_timeout(agent)
    budget = _probe_budget(agent)
    spent = 0.0
    receipts: list = []
    downgrades: list = []
    # re-entrancy: a probe that shells out inherits AUTOPILOT_VERIFICATION=1, so nested
    # probe runs suppress themselves (same guard as the verification harness).
    nested = os.environ.get("AUTOPILOT_VERIFICATION", "").strip() == "1"
    if nested:
        return ([], [])
    for spec in plan:
        if spent >= budget:
            downgrades.append((spec.criterion_id, spec.kind, "probe budget exhausted"))
            continue
        started = time.monotonic()
        remaining = max(1.0, min(per_timeout, budget - spent))
        pr = _probes.run_probe(spec, timeout=remaining, workdir=wd)
        spent += time.monotonic() - started
        if pr.is_downgrade:
            downgrades.append((spec.criterion_id, spec.kind, pr.summary))
            # still record it as a 'skipped' receipt so the block shows what was attempted
        receipts.append(_to_verification_receipt(pr))
        logger.info("autopilot: probe %s[%s] -> %s", spec.kind, spec.criterion_id, pr.status)
    return (receipts, downgrades)


def merge_into_report(report: Any, probe_receipts: list) -> None:
    """Append probe receipts to an existing VerificationReport in place, and mark it
    enabled if probes produced anything (so the downstream 'use_structured' path fires
    on observation even when there were no {verify:} commands)."""
    if not probe_receipts:
        return
    report.receipts.extend(probe_receipts)
    report.enabled = True
    extra = f"; +{len(probe_receipts)} modal probe(s)"
    report.note = (report.note or "") + extra
