# Forward-compat: applying onto v0.17.0

This PR diffs against v0.16.0 (`3c231eb`). Between v0.16.0 and v0.17.0
(`2bd1977d8`) upstream drifted the import block of
`tests/hermes_cli/test_kanban_db.py` (added `import subprocess`), which collides
with this PR's removal of the bare `import sqlite3` (the test now uses
`hermes_state.sqlite3` alias, see line ~24).

It is a single one-line import-ordering conflict. The conflict-free,
v0.17.0-ready variant of that file is committed under
[`v0.17.0-ready/`](./v0.17.0-ready/): it keeps upstream's `import subprocess`
and honors this PR's `sqlite3`-alias change. Resolved file has 0 conflict
markers and compiles; all `sqlite3.` usages resolve through the
`sqlite3 = _hermes_state.sqlite3` alias.

Every other file in this PR (`hermes_cli/kanban_db.py`, `hermes_state.py`,
`tests/test_hermes_state.py`, `tests/test_hermes_state_wal_fallback.py`) applies
cleanly on v0.17.0.
