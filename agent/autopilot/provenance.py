"""Provenance backend ladder — the verbatim-record sense (cmx → lcm → state.db).

This is the engine-run backend for the ``cmx_provenance`` probe (design §3.4). It
answers ONE question deterministically: *does the verbatim conversation record actually
support a claim, or does the claim rest only on the model's own prose?*

The bypass face of the autopilot derail is manufactured independence — "a subagent I
spawned CONFIRMED it", "senior-owner determination: COMPLETE" — which a text-only judge
cannot refute. But it IS falsifiable against the record: a real tool/subagent
confirmation appears as a ``role='tool'`` turn containing that result. No such row ⇒ the
claim is unsupported. That is a database lookup, not an opinion. This module performs it.

THE LADDER (owner-specified true fallbacks; the sense is never fully dark):
  1. cmx     — ~/.hermes/cmx.db          (hermes-cmx VerbatimStore; the best record)
  2. lcm     — ~/.hermes/lcm.db          (hermes-lcm MessageStore; when LCM is live)
  3. state   — ~/.hermes/state.db        (SessionDB; ALWAYS exists — the universal floor)

All three share a ``messages(role, content, ...)`` table, so one role-filtered content
query is portable across every rung. We read READ-ONLY (``mode=ro`` URI) and never write.

Design invariants honored:
  * the ENGINE runs this, never the model — a receipt can't be fabricated by the agent;
  * fail-soft: an unreadable/locked/absent DB is skipped; if the WHOLE ladder is
    unreachable the probe returns ``unavailable`` → DOWNGRADE (never a crash);
  * a claim supported only by ``assistant``/``user`` prose is NOT supported — only
    evidence-bearing roles (tool/subagent output) can ground a claim.
"""

from __future__ import annotations

import logging
import os
# Use the SAME sqlite3 driver hermes_state / kanban_db use: pysqlite3 when available,
# stdlib otherwise. This is REQUIRED, not cosmetic — the live cmx.db / state.db carry
# FTS5 *trigram* virtual tables, and the host's stdlib sqlite3 (3.26 on Oracle Linux 8,
# older glibc) LACKS the trigram tokenizer, so opening/querying those DBs through stdlib
# can raise "no such tokenizer: trigram". pysqlite3 (3.53.x) ships the modern engine.
# Honors HERMES_SQLITE_DRIVER (auto|pysqlite3|stdlib) exactly like the rest of Hermes.
_sqlite_driver_pref = os.environ.get("HERMES_SQLITE_DRIVER", "auto").strip().lower()
if _sqlite_driver_pref in {"stdlib", "sqlite3"}:
    import sqlite3
else:
    try:
        import pysqlite3 as sqlite3  # type: ignore[no-redef]
    except ImportError:
        if _sqlite_driver_pref in {"pysqlite3", "modern"}:
            raise
        import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Connection type alias (pysqlite3 has no type stubs; its runtime API == stdlib's).
_Connection = Any

# Evidence-bearing roles: only a row in one of these can GROUND a claim. Prose roles
# (assistant/user) are the claim itself, never its own evidence.
EVIDENCE_ROLES = ("tool", "tool_result", "function", "observation", "tool_call_result")

# The ladder, best→floor. Each entry: (name, default path). Env overrides allow the
# probe to be pointed at a specific store (e.g. cmx-PG-backed deployments expose a
# mirror, or a test fixture DB).
_LADDER = (
    ("cmx", "~/.hermes/cmx.db", "CMX_DB_PATH"),
    ("lcm", "~/.hermes/lcm.db", "LCM_DB_PATH"),
    ("state", "~/.hermes/state.db", "HERMES_STATE_DB"),
)


@dataclass
class ProvenanceResult:
    """Whether the verbatim record supports a claim, and which rung answered."""

    supported: bool                       # an evidence-role row actually contains the claim
    reachable: bool                       # at least one ladder rung was queryable
    backend: str = ""                     # which rung answered: cmx|lcm|state|""
    matched_role: str = ""                # role of the supporting row, if any
    matched_excerpt: str = ""             # short excerpt of the supporting content
    rows_searched: int = 0
    backends_tried: list = field(default_factory=list)

    @property
    def is_downgrade(self) -> bool:
        """No rung was reachable → the sense can't observe → DOWNGRADE (not a fail)."""
        return not self.reachable


def _resolve_rung(default_path: str, env_var: str) -> Optional[str]:
    """Resolve a ladder rung's DB path (env override wins), or None if it's absent."""
    path = os.environ.get(env_var, "").strip() or os.path.expanduser(default_path)
    return path if path and os.path.exists(path) else None


def _open_ro(path: str) -> Optional[_Connection]:
    """Open a SQLite DB strictly READ-ONLY. Returns None on any failure."""
    try:
        # file: URI with mode=ro guarantees we can never write the user's live store.
        uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.execute("SELECT 1")  # probe it's actually usable
        return conn
    except Exception as exc:  # noqa: BLE001
        logger.debug("provenance: cannot open %s read-only (%s)", path, exc)
        return None


def _has_messages_table(conn: _Connection) -> bool:
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchall()
        if not rows:
            return False
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        return {"role", "content"} <= cols
    except Exception:  # noqa: BLE001
        return False


def _search_rung(
    conn: _Connection,
    evidence_terms: list,
    *,
    evidence_roles: tuple,
    session_id: str = "",
    scan_limit: int = 20000,
) -> tuple[bool, str, str, int]:
    """Search one rung's messages for an evidence-role row containing ALL terms.

    Returns (supported, matched_role, matched_excerpt, rows_searched). Uses the FTS
    mirror to NARROW when present (fast on the 1GB+ cmx.db), then verifies role +
    full-term containment on the candidate rows. Falls back to a bounded table scan
    when FTS is absent or errors.
    """
    terms = [t.strip().lower() for t in evidence_terms if t and t.strip()]
    if not terms:
        return (False, "", "", 0)

    role_ph = ",".join("?" for _ in evidence_roles)
    sess_clause = " AND m.session_id = ?" if session_id else ""
    candidates = []
    rows_searched = 0

    # 1) FTS-narrowed path: match the longest term (most selective) to shrink the scan.
    longest = max(terms, key=len)
    try:
        fts_q = '"' + longest.replace('"', '""') + '"'
        params = [fts_q, *evidence_roles]
        if session_id:
            params.append(session_id)
        sql = (
            "SELECT m.role, m.content FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid "
            f"WHERE messages_fts MATCH ? AND m.role IN ({role_ph}){sess_clause} "
            "LIMIT 5000"
        )
        candidates = conn.execute(sql, params).fetchall()
    except Exception:  # noqa: BLE001 — FTS may be absent/locked; fall back to scan
        candidates = []

    # 2) Fallback bounded scan over evidence-role rows only.
    if not candidates:
        try:
            params = list(evidence_roles)
            if session_id:
                params.append(session_id)
            sql = (
                f"SELECT role, content FROM messages WHERE role IN ({role_ph})"
                f"{(' AND session_id = ?' if session_id else '')} LIMIT ?"
            )
            params.append(scan_limit)
            candidates = conn.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("provenance: rung scan failed (%s)", exc)
            return (False, "", "", 0)

    for role, content in candidates:
        rows_searched += 1
        low = (content or "").lower()
        if all(term in low for term in terms):
            excerpt = (content or "").strip().replace("\n", " ")[:200]
            return (True, role, excerpt, rows_searched)
    return (False, "", "", rows_searched)


def supports_claim(
    evidence_terms: list,
    *,
    evidence_roles: tuple = EVIDENCE_ROLES,
    session_id: str = "",
) -> ProvenanceResult:
    """Walk the cmx→lcm→state ladder; return whether the record supports the claim.

    Stops at the FIRST rung that (a) is reachable AND (b) supports the claim. If a rung
    is reachable but does NOT support the claim it still counts as 'reachable' (we have a
    real record to judge against) and the walk continues to give a better rung a chance,
    but a 'reachable + unsupported' result is what falsifies the bypass. Only when NO
    rung is reachable do we return ``reachable=False`` → DOWNGRADE.
    """
    tried: list = []
    reachable = False
    best_unsupported: Optional[ProvenanceResult] = None

    for name, default_path, env_var in _LADDER:
        path = _resolve_rung(default_path, env_var)
        if not path:
            continue
        conn = _open_ro(path)
        if conn is None:
            continue
        try:
            if not _has_messages_table(conn):
                continue
            tried.append(name)
            reachable = True
            ok, role, excerpt, n = _search_rung(
                conn, evidence_terms, evidence_roles=evidence_roles, session_id=session_id
            )
            if ok:
                return ProvenanceResult(
                    supported=True, reachable=True, backend=name,
                    matched_role=role, matched_excerpt=excerpt,
                    rows_searched=n, backends_tried=tried,
                )
            # remember the first reachable-but-unsupported rung as the falsifier
            if best_unsupported is None:
                best_unsupported = ProvenanceResult(
                    supported=False, reachable=True, backend=name,
                    rows_searched=n, backends_tried=list(tried),
                )
        finally:
            conn.close()

    if best_unsupported is not None:
        best_unsupported.backends_tried = tried
        return best_unsupported
    return ProvenanceResult(supported=False, reachable=reachable, backends_tried=tried)
