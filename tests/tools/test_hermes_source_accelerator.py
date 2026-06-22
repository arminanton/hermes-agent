from __future__ import annotations

import json
import sys
from pathlib import Path

def _resolve_accel_root():
    """Find the source_accelerator package — tolerate test-runs from both the
    live tree (src is sibling of workspace) AND any candidate/quarantine tree
    (where src may have no sibling workspace). Fall back to the canonical live
    workspace path so tests pass when the candidate src is run in isolation."""
    # First: sibling-of-src convention (parents[3] is the repo root: <root>/src/tests/tools/file.py)
    candidate = Path(__file__).resolve().parents[3] / "workspace" / "scripts" / "source_accelerator"
    if candidate.exists():
        return candidate
    # Fallback: canonical live workspace on this host
    live = Path("/mnt/devvm/custom/hermes/workspace/scripts/source_accelerator")
    if live.exists():
        return live
    # Last resort: env var override
    import os as _os
    env = _os.environ.get("HERMES_SOURCE_ACCELERATOR_PATH")
    if env and Path(env).exists():
        return Path(env)
    return candidate  # original (will fail later with a clear ModuleNotFoundError)


ACCEL_ROOT = _resolve_accel_root()
if str(ACCEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCEL_ROOT))


import pytest


def _list_accel_hash_schemas(conn) -> set[str]:
    """Return source_accel_<hash> schema names (never the bare production one).
    Tolerates psycopg dict_row (default in this package) or plain-tuple rows."""
    rows = conn.execute(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name LIKE 'source_accel\\_%'"
    ).fetchall()
    out: set[str] = set()
    for r in rows:
        name = r["schema_name"] if isinstance(r, dict) else r[0]
        out.add(name)
    return out


@pytest.fixture(autouse=True)
def _drop_leaked_test_schemas():
    """Drop any per-tmp source_accel_<hash> schema this test creates so the
    shared cmx_dev Postgres never accumulates orphaned test schemas. Hard guard:
    NEVER touches the production "source_accel" schema (no hash suffix) or any
    non source_accel_* schema. Best-effort; failures here never fail the test.
    """
    before: set[str] = set()
    try:
        from source_accelerator import db as _sadb

        with _sadb._raw_connect(autocommit=True) as _c:
            before = _list_accel_hash_schemas(_c)
    except Exception:
        before = set()
    yield
    try:
        from source_accelerator import db as _sadb

        with _sadb._raw_connect(autocommit=True) as _c:
            after = _list_accel_hash_schemas(_c)
            for sch in after - before:
                # Only drop hash-suffixed schemas; never the bare production one.
                if sch != "source_accel" and sch.startswith("source_accel_"):
                    _c.execute(f'DROP SCHEMA IF EXISTS "{sch}" CASCADE')
    except Exception:
        pass



def _fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "src"
    workspace = tmp_path / "workspace"
    (source / "tools").mkdir(parents=True)
    (workspace / "skills" / "devops" / "sample").mkdir(parents=True)
    (workspace / "logs").mkdir(parents=True)
    (workspace / "memories").mkdir(parents=True)
    (workspace / "sessions").mkdir(parents=True)
    (workspace / "obsidian" / "Hermes").mkdir(parents=True)
    (source / "tools" / "sample_tool.py").write_text(
        "from tools.registry import registry\n\n"
        "def sample_handler(args, **kw):\n    return {'ok': True}\n\n"
        "registry.register(name='sample_lookup', toolset='sample', schema={}, handler=sample_handler)\n",
        encoding="utf-8",
    )
    (workspace / "skills" / "devops" / "sample" / "SKILL.md").write_text(
        "---\nname: sample\ndescription: Sample source accelerator skill\n---\n# Sample Skill\nUse source accelerator first.\n",
        encoding="utf-8",
    )
    (workspace / "config.yaml").write_text("providers:\n  openai:\n    model: gpt-test\napi_key: should-redact\n", encoding="utf-8")
    (workspace / "logs" / "agent.log").write_text("2026-01-01 ERROR sample failure token=secretvalue\n", encoding="utf-8")
    (workspace / "memories" / "MEMORY.md").write_text("Source accelerator memory fact\n", encoding="utf-8")
    (workspace / "sessions" / "s1.json").write_text(json.dumps({"title": "Sample session", "messages": ["used sample_lookup"]}), encoding="utf-8")
    db = workspace / "indexes" / "hsa.sqlite"
    monkeypatch.setenv("HERMES_SOURCE_ROOT", str(source))
    monkeypatch.setenv("HERMES_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("HERMES_SOURCE_INDEX_DB", str(db))
    # The Postgres backend freezes its "default index path" at import time
    # (db._DEFAULT_INDEX_DB). When that freeze captures THIS test's tmp path
    # (import order varies in a full-suite run), _schema_for() would collapse
    # the tmp path onto the shared production "source_accel" schema and the
    # test would read/write live production rows. Realign the frozen default to
    # db.py's own no-env-override default so this unique tmp db is always
    # treated as non-default and gets its own isolated source_accel_<hash>
    # schema. Touches only the test; never mutates the live backend or cmx_dev.
    from source_accelerator import db as _sadb

    monkeypatch.setattr(
        _sadb,
        "_DEFAULT_INDEX_DB",
        Path("/mnt/devvm/custom/hermes/workspace/indexes/hermes-source-index.sqlite"),
        raising=False,
    )
    return source, workspace, db


def test_refresh_search_open_status(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    from source_accelerator.config import get_config
    from source_accelerator.indexer import refresh_fast
    from source_accelerator.query import open_result, search
    from source_accelerator.status import status

    cfg = get_config()
    result = refresh_fast(cfg=cfg, write_obsidian=True)
    assert result["status"] == "ok"
    assert result["files_seen"] >= 3
    st = status(cfg)
    assert st["exists"] is True
    assert st["trigram_available"] is True
    assert st["counts"]["tools"] >= 1

    found = search("sample_lookup", scope="src", limit=5, cfg=cfg)
    assert found["fallback"] is False
    assert found["results"]
    assert any(r["kind"] in {"tool", "content"} for r in found["results"])

    opened = open_result(found["results"][0]["result_id"], context_lines=5, cfg=cfg)
    assert opened["path"].endswith("sample_tool.py")
    assert "sample_lookup" in opened["content"]
    assert "Opened current file from disk" in opened["verify_note"]


def test_missing_index_uses_deterministic_fallback(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    from source_accelerator.config import get_config
    from source_accelerator.query import search

    cfg = get_config()
    result = search("sample_lookup", scope="src", limit=5, cfg=cfg)
    assert result["fallback"] is True
    assert result["fallback_reason"] == "missing_or_uninitialized_index"
    assert result["results"]


def test_relationship_query_routes_to_gitnexus(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    from source_accelerator.config import get_config
    from source_accelerator.query import search

    result = search("what breaks if I change sample_lookup", cfg=get_config())
    assert result["mode"] == "relationship"
    assert "mcp_gitnexus_context" in result["suggested_next_tools"]


def test_workspace_default_ignores_hermes_home(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/tmp/this-should-not-be-used")
    monkeypatch.delenv("HERMES_WORKSPACE_ROOT", raising=False)

    import importlib
    import source_accelerator.config as cfgmod  # pyright: ignore[reportMissingImports]

    cfgmod = importlib.reload(cfgmod)
    cfg = cfgmod.get_config()
    assert str(cfg.workspace_root) == "/mnt/devvm/custom/hermes/workspace"
