"""Frozen acceptance contract + achievable-bar terminus (autopilot health Fix 1+2).

The autopilot non-termination pathology (NuData "Gate A", 2026-06-23, ~30h) had a
structural root, proven from the run's own ledger:

  * the completion gate contained a sub-condition unsatisfiable BY THE AGENT
    (prove the *independence* of a verifier the agent spawned itself), so
    "keep going until the gate is satisfied" was a provably non-terminating loop; and
  * "done" was a re-readable text the agent re-litigated every turn (the ledger
    held four mutually-contradictory "authoritative terminus" blocks, each
    retracting the last), so the terminus was a thing to *argue*, not a fact.

This module removes both. At run start the goal contract is parsed ONCE into an
immutable, content-hashed list of acceptance criteria, each tagged with WHO can
satisfy it:

    agent_achievable   — the agent can do it and verify it (write code, run tests).
    owner_gated        — only a human grants it (sign-off, business cutover, accept).
    unprovable_by_agent— no fixed point from inside the run (verifier independence,
                          distinct-trust-domain attestation the agent cannot make).

The terminus is then a COMPUTED FACT, not a reading:

    DONE  ⟺  every agent_achievable criterion is satisfied
             AND every remaining criterion is owner_gated or unprovable_by_agent.

On DONE the engine HALTS with a NAMED-RESIDUAL report ("complete except: <items>")
instead of looping. A criterion tagged owner_gated / unprovable_by_agent can NEVER
be a continuation reason; it is a residual from parse time. That single rule kills
the unsatisfiable-gate loop: "prove your own independence" halts as a named residual
rather than spinning forever.

The contract is frozen (content-hashed). The agent cannot re-derive a different
terminus mid-run because the terminus is computed from the frozen contract, not
re-parsed from the goal text each turn. This kills the goalpost-moving.

Self-contained (no imports from the rest of autopilot) so it unit-tests in isolation.
Fail-soft: any parse error yields an empty contract, which disables the terminus
gate (the run falls back to the pre-existing Council-only behavior) rather than
crashing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Satisfiability tags.
AGENT_ACHIEVABLE = "agent_achievable"
OWNER_GATED = "owner_gated"
UNPROVABLE = "unprovable_by_agent"

# Phrase banks used to TAG a criterion's satisfiability at parse time. Deliberately
# conservative: a criterion is agent_achievable UNLESS its text clearly names an
# owner-gate or an unprovable-by-agent condition. (Over-tagging agent_achievable is
# safe: it just means the bar stays high and the Council still governs. Over-tagging
# owner_gated would be an escape hatch, so these banks are tight and specific.)

_OWNER_GATED_MARKERS = (
    "owner acceptance", "owner-acceptance", "owner sign-off", "owner signoff",
    "owner approval", "business sign-off", "business signoff", "business approval",
    "live cutover", "live-cutover", "production cutover", "go-live sign-off",
    "stakeholder sign-off", "manager approval", "human approval", "human sign-off",
    "requires sign-off", "awaiting sign-off", "pending sign-off", "owner-gated",
    "owner attestation", "owner-attestation", "user acceptance", "uat sign-off",
)

_UNPROVABLE_MARKERS = (
    "independent verifier", "independent verification", "verifier independence",
    "distinct trust domain", "distinct-trust-domain", "separate trust domain",
    "distinct iam principal", "distinct-iam-principal", "non-collusive",
    "non-agent verifier", "external attestation", "third-party attestation",
    "out-of-band attestation", "principal-independence", "principal independence",
    "prove independence", "prove its independence", "independently reproduced by a",
    "a party other than the executor", "non-agent-invoked",
)

# Lines that look like acceptance criteria. We scan the goal text for bullet/numbered
# items and for explicit "Verify:" / "Success criteria" style lines. This is a
# best-effort structural parse: a goal with no recognizable criteria yields an empty
# contract (terminus gate disabled, Council-only behavior preserved).
_CRITERION_LINE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)]|\(\d+\)|[A-Z]?\d+[.)]?\s*[—:-])\s+(.*\S)\s*$"
)
_VERIFY_LINE = re.compile(r"^\s*(?:verify|success criteri|acceptance|must be true|done when)\b.*?:\s*(.*\S)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion, with a frozen satisfiability tag.

    ``verify_cmd`` is an OPTIONAL deterministic check command the engine can run
    (outside the agent's control) to produce a real exit-code/output receipt that
    GROUNDS this criterion's satisfaction — instead of inferring it from prose.
    Parsed from an inline ``{verify: <cmd>}`` token in the criterion text. Empty
    when the criterion carries no executable check (the textual path still works).
    """

    id: str
    text: str
    satisfiability: str  # AGENT_ACHIEVABLE | OWNER_GATED | UNPROVABLE
    verify_cmd: str = ""


@dataclass
class AcceptanceContract:
    """Parsed-once, content-hashed acceptance contract for an autopilot run."""

    criteria: tuple = ()
    content_hash: str = ""
    source_len: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.criteria

    def agent_criteria(self) -> list:
        return [c for c in self.criteria if c.satisfiability == AGENT_ACHIEVABLE]

    def residual_criteria(self) -> list:
        """owner_gated + unprovable_by_agent — the things that can only ever be
        named residuals, never continuation reasons."""
        return [c for c in self.criteria if c.satisfiability in (OWNER_GATED, UNPROVABLE)]

    def verifiable_criteria(self) -> list:
        """agent_achievable criteria that carry an executable verify_cmd — the ones
        the deterministic harness can produce a real receipt for."""
        return [c for c in self.criteria if c.satisfiability == AGENT_ACHIEVABLE and c.verify_cmd]


# Inline executable-check token: ``{verify: <command>}`` (or ``{verify:cmd: …}``).
# Extracted from a criterion line so the prose stays readable while the command is
# captured for the deterministic harness. Kept deliberately explicit (a literal
# token, not "any backticked text") so we never execute something the author did
# not clearly mark as the check command.
_VERIFY_CMD_RE = re.compile(r"\{verify(?:[_:]cmd)?:\s*(.+?)\}", re.IGNORECASE | re.DOTALL)


def _extract_verify_cmd(text: str) -> tuple[str, str]:
    """Split a criterion line into ``(clean_text, verify_cmd)``.

    Pulls the ``{verify: <cmd>}`` token out of the prose (so the displayed text is
    clean) and returns the command separately. Returns ``("…", "")`` when absent.
    """
    m = _VERIFY_CMD_RE.search(text)
    if not m:
        return text, ""
    cmd = m.group(1).strip()
    clean = _VERIFY_CMD_RE.sub("", text).strip()
    # collapse any double-space left by the removal
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean, cmd


def _classify(text: str) -> str:
    low = text.lower()
    # unprovable takes priority over owner_gated (verifier-independence is the worst
    # case and must never be treated as merely owner-grantable).
    if any(m in low for m in _UNPROVABLE_MARKERS):
        return UNPROVABLE
    if any(m in low for m in _OWNER_GATED_MARKERS):
        return OWNER_GATED
    return AGENT_ACHIEVABLE


def parse_contract(goal_text: str) -> AcceptanceContract:
    """Parse the goal contract text ONCE into a frozen AcceptanceContract.

    Best-effort + fail-soft: returns an empty contract (which disables the terminus
    gate) on any error or when no criteria are recognizable.
    """
    try:
        if not goal_text or not goal_text.strip():
            return AcceptanceContract()
        seen: set = set()
        crits: list = []
        for raw in goal_text.splitlines():
            line = raw.rstrip()
            m = _CRITERION_LINE.match(line) or _VERIFY_LINE.match(line)
            if not m:
                continue
            text = m.group(1).strip()
            # pull any inline {verify: <cmd>} BEFORE dedup so the clean prose is the
            # dedup/display key and the command travels with the criterion.
            text, verify_cmd = _extract_verify_cmd(text)
            # skip trivially short or duplicate criteria
            if len(text) < 8 or text.lower() in seen:
                continue
            seen.add(text.lower())
            cid = f"C{len(crits) + 1:02d}"
            crits.append(Criterion(
                id=cid, text=text[:500], satisfiability=_classify(text),
                verify_cmd=verify_cmd[:1000],
            ))
        if not crits:
            return AcceptanceContract()
        h = hashlib.sha256(
            "\n".join(f"{c.id}:{c.satisfiability}:{c.verify_cmd}:{c.text}" for c in crits).encode("utf-8")
        ).hexdigest()[:16]
        return AcceptanceContract(criteria=tuple(crits), content_hash=h, source_len=len(goal_text))
    except Exception as exc:  # noqa: BLE001 — parsing must never crash the run
        logger.debug("autopilot: contract parse failed (%s)", exc)
        return AcceptanceContract()


def synthesize_floor_enabled(agent: Any = None) -> bool:
    """Whether to synthesize a minimal 1-criterion contract for a bare goal that
    produced none (the naive-user floor). Default ON; disable with
    autopilot.synthesize_contract_floor=false / AUTOPILOT_SYNTH_CONTRACT_FLOOR=0.
    """
    if agent is not None:
        v = getattr(agent, "_autopilot_synthesize_floor", None)
        if v is not None:
            return bool(v)
    env = os.environ.get("AUTOPILOT_SYNTH_CONTRACT_FLOOR", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def synthesize_minimal_contract(goal_text: str) -> AcceptanceContract:
    """Build a 1-criterion frozen contract for a bare goal that parsed to nothing.

    The single criterion is AGENT_ACHIEVABLE with NO verify_cmd: there is no project
    check to bind, so the Council judges it on evidence. Its purpose is structural —
    it makes the contract non-empty so the achievable-bar and refinement-churn
    terminus have a deliverable to bind to, giving a naive one-sentence goal the same
    self-termination floor a hand-authored REBORN contract gets. Fail-soft: returns
    an empty contract on any error (preserving prior behavior).
    """
    try:
        g = " ".join((goal_text or "").split())
        if not g:
            return AcceptanceContract()
        text = f"The goal «{g[:200]}» is substantively complete and the work is genuinely done"
        crit = Criterion(id="C01", text=text[:500], satisfiability=AGENT_ACHIEVABLE, verify_cmd="")
        h = hashlib.sha256(
            f"{crit.id}:{crit.satisfiability}:{crit.verify_cmd}:{crit.text}".encode("utf-8")
        ).hexdigest()[:16]
        return AcceptanceContract(criteria=(crit,), content_hash=h, source_len=len(goal_text or ""))
    except Exception as exc:  # noqa: BLE001 — synthesis must never crash the run
        logger.debug("autopilot: minimal-contract synthesis failed (%s)", exc)
        return AcceptanceContract()


def get_or_parse(agent: Any, goal_text: str) -> AcceptanceContract:
    """Return the agent's frozen contract, parsing+freezing it on first call.

    The contract is cached on the agent and NEVER re-parsed (the terminus is a fact,
    not a re-reading). If a later call presents a *different* goal text, the frozen
    contract is kept and a warning is logged — the agent does not get to redefine the
    criteria mid-run.
    """
    cached: Optional[AcceptanceContract] = getattr(agent, "_autopilot_contract", None)
    if cached is not None:
        new_hash = parse_contract(goal_text).content_hash
        if new_hash and cached.content_hash and new_hash != cached.content_hash:
            logger.warning(
                "autopilot: goal text changed mid-run (frozen=%s new=%s) — keeping the "
                "frozen contract; the agent does not redefine criteria mid-run.",
                cached.content_hash, new_hash,
            )
        return cached
    contract = parse_contract(goal_text)
    # AUTO-DRAFT verification gates from the project's own checks so a plain
    # `/autopilot goal "<text>"` gets grounded gates with no {verify:} tokens and no
    # author effort. Fail-soft + lazy import (autocheck imports from this module).
    try:
        from agent.autopilot import autocheck as _autocheck

        if _autocheck.autodraft_enabled(agent):
            root = _verification_workdir(agent)
            use_llm = _autodraft_llm_enabled(agent)
            council_model = getattr(agent, "_autopilot_council_model", "") or ""
            contract = _autocheck.autodraft_contract(
                contract, goal_text=goal_text, root=root,
                use_llm=use_llm, council_model=council_model,
            )
    except Exception as exc:  # noqa: BLE001 — auto-draft must never break freeze
        logger.debug("autopilot: auto-draft skipped (%s)", exc)
    # NAIVE-USER FLOOR: if the goal produced NO criteria (a bare one-sentence goal
    # like "make the homepage faster" or "write a blog post") AND auto-draft found
    # no project checks to synthesize from, the contract is still empty — which
    # disables BOTH the achievable-bar and refinement-churn terminus, leaving an
    # average user's run with no self-termination floor. Synthesize a single
    # minimal agent-achievable criterion so every autopilot goal gets the same
    # structural protection the REBORN contracts get by hand. It has no verify_cmd
    # (no project check), so the Council judges it on evidence — but its mere
    # presence makes `deliverable_present` true once real work exists, so the churn
    # terminus can conclude a polish spiral on a contractless goal too.
    if contract.is_empty and synthesize_floor_enabled(agent):
        contract = synthesize_minimal_contract(goal_text)
    try:
        agent._autopilot_contract = contract
    except Exception:  # noqa: BLE001
        pass
    return contract


def _verification_workdir(agent: Any = None) -> str:
    """Resolve the project root for check detection (mirrors verification._workdir
    without importing it at module load)."""
    wd = None
    if agent is not None:
        wd = getattr(agent, "_autopilot_verification_workdir", None)
    wd = wd or os.environ.get("AUTOPILOT_VERIFICATION_WORKDIR", "")
    if wd and os.path.isdir(str(wd)):
        return str(wd)
    return os.getcwd()


def _autodraft_llm_enabled(agent: Any = None) -> bool:
    """Whether to use the optional LLM check-mapper. Default OFF (the deterministic
    keyword mapper is reliable and free); enable with autopilot.autodraft_llm=true /
    AUTOPILOT_AUTODRAFT_LLM=1."""
    if agent is not None:
        val = getattr(agent, "_autopilot_autodraft_llm", None)
        if val is not None:
            return bool(val)
    return os.environ.get("AUTOPILOT_AUTODRAFT_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def contract_enabled(agent: Any = None) -> bool:
    """Whether the frozen-contract terminus gate is active. Default ON; disable with
    autopilot.contract_terminus=false / AUTOPILOT_CONTRACT_TERMINUS=0."""
    if agent is not None:
        val = getattr(agent, "_autopilot_contract_terminus", None)
        if val is not None:
            return bool(val)
    env = os.environ.get("AUTOPILOT_CONTRACT_TERMINUS", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


@dataclass
class TerminusResult:
    """Outcome of the achievable-bar terminus check."""

    halt: bool = False
    reason: str = ""          # "achievable-bar" when halting; "" otherwise
    residual_text: str = ""   # human-readable named residual for the ADR + report


def achievable_bar_halt(
    contract: AcceptanceContract,
    *,
    satisfied_ids: set,
    council_denial_reason: str = "",
) -> TerminusResult:
    """The terminus rule. HALT iff every agent_achievable criterion is satisfied AND
    the only remaining (unsatisfied) criteria are owner_gated / unprovable_by_agent.

    ``satisfied_ids`` is the set of Criterion.id values the run has verified satisfied
    (the engine's gap-closure ledger). ``council_denial_reason`` is recorded for the
    residual note only; it does NOT keep the run going when the achievable bar is met
    — that is the whole point (an unprovable independence denial must not loop).
    """
    if contract.is_empty:
        return TerminusResult(halt=False)
    agent_open = [c for c in contract.agent_criteria() if c.id not in satisfied_ids]
    if agent_open:
        # Real agent-achievable work remains — the bar is NOT met. Keep going.
        return TerminusResult(halt=False)
    residuals = contract.residual_criteria()
    if not residuals:
        # Everything agent-achievable is done and there are no owner/unprovable
        # residuals — the run is genuinely complete; let the normal complete path
        # handle it (return halt=False here so we don't double-emit).
        return TerminusResult(halt=False)
    lines = []
    for c in residuals:
        kind = "owner sign-off required" if c.satisfiability == OWNER_GATED else "not provable by the agent"
        lines.append(f"  - [{c.id}] ({kind}) {c.text}")
    note = (
        "All agent-achievable acceptance criteria are satisfied. The run halts at the "
        "achievable bar; the only remaining items can be satisfied solely by the owner "
        "or are not provable from inside the run:\n" + "\n".join(lines)
    )
    if council_denial_reason:
        note += f"\n(Last Council denial reason, recorded for context: {council_denial_reason[:300]})"
    return TerminusResult(halt=True, reason="achievable-bar", residual_text=note)


# --------------------------------------------------------------------------- #
# Fix 2 — reject self-spawned verifiers as independence evidence               #
#                                                                              #
# The NuData producer repeatedly manufactured its own "independent" confirmation:#
# it spawned delegate_task subagents IN ITS OWN LINEAGE and cited their         #
# "confirmation" as independent verification (ledger L488-501, L727 concedes    #
# "NOT a separate trust domain"). An agent cannot verify its own independence;   #
# the request is the unwinnable recursion. This detects a completion/independence #
# claim that rests on a self-lineage verifier so the engine can refuse to treat  #
# it as satisfying an `unprovable_by_agent` criterion.                          #
# --------------------------------------------------------------------------- #

_SELF_VERIFIER_MARKERS = (
    "i spawned", "i delegated", "a subagent i", "subagent i spawned",
    "my subagent", "my delegate", "delegate_task subagent", "a delegate_task",
    "spawned a subagent", "spawned a verifier", "my own subagent", "my own verifier",
    "a different-model subagent", "i ran an independent subagent",
    "an independent subagent (separate", "neutral subagent", "spot-check subagent",
    "my fan-out worker", "my fan-out subagent", "i fanned out",
)

# Phrases that legitimately denote a TRULY external signal (allowed).
_EXTERNAL_VERIFIER_MARKERS = (
    "the council", "hermes council", "mcp_council", "a separate run",
    "a different session", "a separate trust domain run", "the audit run",
    "a foreign-session", "the owner confirmed", "owner-confirmed",
)


def claims_self_spawned_independence(text: str) -> bool:
    """True when ``text`` cites a SELF-lineage subagent as independent verification.

    Used to reject independence theater: a claim that an `unprovable_by_agent`
    criterion is "met" because a subagent the agent spawned confirmed it is NOT
    accepted. A claim resting on the Council or a genuinely separate run is fine.
    """
    if not text:
        return False
    low = text.lower()
    cites_independence = any(
        w in low for w in ("independent", "independence", "verifier", "verified by", "cross-confirm")
    )
    if not cites_independence:
        return False
    if any(m in low for m in _SELF_VERIFIER_MARKERS):
        # It is claiming independence via a self-spawned worker — theater.
        # Allow only if it ALSO clearly rests on a truly-external signal.
        return not any(x in low for x in _EXTERNAL_VERIFIER_MARKERS)
    return False


# --------------------------------------------------------------------------- #
# Fix 4 — REFINEMENT-CHURN TERMINUS (diminishing-returns / "polish" loop)       #
#                                                                              #
# The NuData LOCAL-PROOF run exposed a failure the other terminus checks miss:  #
# the deliverable (GO-NO-GO.md + real benches) was DONE, but the Council kept    #
# returning `conditional` with a DIFFERENTLY-WORDED ask each round —            #
#   round N   : "show a gate-by-gate table"                                     #
#   round N+1 : "show faster/cheaper DELTAS not absolute numbers"               #
#   round N+2 : "cite the cost levers on a line like the latency gates"         #
#   round N+3 : "enumerate the PRIMARY bench files, not the attestation"        #
# Each ask is real but SMALLER, and each is PRESENTATION of already-produced     #
# evidence, not a missing measurement. The two existing breakers don't fire:    #
#   * achievable_bar_halt: never triggers because the Council never marks the    #
#     criteria "satisfied" — it keeps saying "conditional, one more re-cite".    #
#   * semantic-circuit-breaker: keys on the denial reason being IDENTICAL N      #
#     turns running, but the wording rotates every round, resetting it.         #
#                                                                              #
# This detector is WORDING-INDEPENDENT. It concludes the run when, for K         #
# consecutive judged rounds, ALL of these structural facts hold:                #
#   (1) every round was a REAL adjudication (verdict.source in {council, aux}) — #
#       never a fail-open/unjudged turn;                                         #
#   (2) NO round returned `deny` — i.e. nothing is FAILING, the work is sound;   #
#       a single `deny` (real missing/wrong work) resets the tracker to 0;       #
#   (3) ZERO acceptance criteria closed across the window — no new ground was    #
#       gained, consistent with "the substance is already done";                #
#   (4) confidence is NOT climbing toward acceptance (max confidence in the      #
#       window stays below an accept threshold) — if the Council were trending   #
#       toward `allow`, that's real convergence, NOT churn, so keep going.       #
# When all hold for K rounds the run CONCLUDES (delivers the standing            #
# deliverable) with an honest note: "substantive work complete; remaining        #
# Council asks were presentation refinements on already-produced evidence."      #
#                                                                              #
# This is the achievable-bar terminus applied to the REFINEMENT axis instead of  #
# the new-measurement axis. It does NOT need a max-continuations cap: it stops    #
# on the SHAPE of the loop (polish with no new substance), not a turn count, so   #
# a genuinely-progressing run (criteria closing, or any `deny`, or rising         #
# confidence) is never cut short.                                                #
# --------------------------------------------------------------------------- #

# A `conditional`/`allow` verdict at or above this confidence is treated as real
# convergence-in-progress, not churn — so the churn window will not accumulate
# while the Council is trending toward acceptance. Tunable via env.
def _churn_accept_confidence() -> float:
    try:
        return float(os.environ.get("AUTOPILOT_CHURN_ACCEPT_CONF", "0.85"))
    except (TypeError, ValueError):
        return 0.85


def churn_window_k(agent: Any = None) -> int:
    """How many consecutive presentation-only judged rounds conclude the run.

    Default 4. Disable the refinement-churn terminus entirely with 0
    (cfg autopilot.refinement_churn_k / env AUTOPILOT_REFINEMENT_CHURN_K).
    """
    default = 4
    if agent is not None:
        v = getattr(agent, "_autopilot_refinement_churn_k", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return default
    try:
        return int(os.environ.get("AUTOPILOT_REFINEMENT_CHURN_K", default))
    except (TypeError, ValueError):
        return default


@dataclass
class RefinementChurnTracker:
    """Per-run accumulator of the structural churn signal (no wording involved).

    The driver calls ``record(...)`` once per not-complete judged round with the
    facts of that round; ``rounds`` counts consecutive presentation-only rounds.
    Any round that breaks the pattern (a ``deny``, a closed criterion, an
    unjudged/fail-open turn, or confidence at/above the accept threshold) resets
    ``rounds`` to 0 — so the window only fills on genuine diminishing-returns spin.
    """

    rounds: int = 0
    max_conf_in_window: float = 0.0

    def record(
        self,
        *,
        verdict_label: str,
        source: str,
        confidence: float,
        criteria_closed_this_round: int,
        deliverable_present: bool,
    ) -> None:
        judged = source in ("council", "aux")
        is_deny = (verdict_label or "").strip().lower() == "deny"
        converging = confidence >= _churn_accept_confidence()
        # The pattern holds ONLY if: a real judge ran, it did NOT fail the work,
        # no criterion closed (no new substance), confidence isn't trending to
        # accept, AND a standing deliverable actually exists to conclude on.
        if (
            judged
            and not is_deny
            and criteria_closed_this_round == 0
            and not converging
            and deliverable_present
        ):
            self.rounds += 1
            self.max_conf_in_window = max(self.max_conf_in_window, confidence)
        else:
            self.rounds = 0
            self.max_conf_in_window = 0.0


@dataclass
class ChurnResult:
    conclude: bool = False
    reason: str = ""
    note: str = ""


def refinement_churn_conclude(
    tracker: RefinementChurnTracker,
    *,
    k: int,
    deliverable_hint: str = "",
) -> ChurnResult:
    """Decide whether the run should CONCLUDE on diminishing-returns refinement.

    Concludes iff ``k > 0`` and the tracker has accumulated ``k`` consecutive
    presentation-only judged rounds. The note is honest about WHY it concluded
    (substantive work done; remaining asks were presentation refinements).
    """
    if k <= 0:
        return ChurnResult(conclude=False)
    if tracker.rounds < k:
        return ChurnResult(conclude=False)
    note = (
        f"Refinement-churn terminus: {tracker.rounds} consecutive judged rounds were "
        "presentation-only — a real reviewer adjudicated each, none returned 'deny' "
        "(nothing is failing), no acceptance criterion newly closed, and confidence "
        "never trended toward acceptance. The substantive deliverable is complete; the "
        "remaining Council asks are re-presentation / re-formatting / additional "
        "citation of evidence already produced, not missing measurements. Concluding "
        "the run with the standing deliverable instead of polishing indefinitely."
    )
    if deliverable_hint:
        note += f"\nStanding deliverable: {deliverable_hint}"
    return ChurnResult(conclude=True, reason="refinement-churn", note=note)
