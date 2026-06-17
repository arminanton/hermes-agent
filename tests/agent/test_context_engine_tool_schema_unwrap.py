"""Regression tests for the context-engine tool-schema double-wrap guard.

A context engine's ``get_tool_schemas()`` is contracted to return BARE schemas
(``{name, description, parameters}``). The host (``agent_init``) wraps each one
in the OpenAI envelope (``{"type": "function", "function": {...}}``) before
adding it to ``agent.tools``. If an engine mistakenly pre-wraps its schemas,
wrapping again produces ``{"function": {"function": {...}}}`` whose OUTER
``function.name`` is empty, so the provider rejects the whole request with
HTTP 400 ``tools[N].function.name: empty string`` — breaking every turn.

``_normalize_context_engine_schema`` unwraps an already-enveloped schema so
both bare and pre-wrapped inputs register with a correct name. These tests
exercise the REAL helper from ``agent_init`` (not a re-implementation) and the
registration shape it produces.

Run:
    python -m pytest tests/agent/test_context_engine_tool_schema_unwrap.py -q
"""

from __future__ import annotations

from agent.agent_init import _normalize_context_engine_schema


def _register(schemas):
    """Mirror the host wrap that follows normalization in agent_init.

    Returns (tools, valid_names) the way the context-engine collection block
    builds them, so we can assert no tool reaches the provider unnamed.
    """
    tools = []
    valid_names = set()
    existing = set()
    for raw in schemas:
        schema = _normalize_context_engine_schema(raw)
        name = schema.get("name", "") if isinstance(schema, dict) else ""
        if name and name in existing:
            continue
        tools.append({"type": "function", "function": schema})
        if name:
            valid_names.add(name)
            existing.add(name)
    return tools, valid_names


# --- the helper in isolation -------------------------------------------------


def test_bare_schema_passes_through_unchanged():
    bare = {"name": "grep", "description": "d", "parameters": {"type": "object", "properties": {}}}
    assert _normalize_context_engine_schema(bare) is bare


def test_enveloped_schema_is_unwrapped_to_inner():
    inner = {"name": "grep", "description": "d", "parameters": {"type": "object", "properties": {}}}
    enveloped = {"type": "function", "function": inner}
    assert _normalize_context_engine_schema(enveloped) is inner


def test_non_dict_input_returned_unchanged():
    assert _normalize_context_engine_schema(None) is None
    assert _normalize_context_engine_schema("not-a-schema") == "not-a-schema"


def test_envelope_with_top_level_name_is_not_unwrapped():
    # A bare schema that legitimately has both a name and a "type" key must not
    # be mistaken for the envelope (the guard requires a missing top-level name).
    bare_with_type = {"name": "grep", "type": "function", "parameters": {}}
    assert _normalize_context_engine_schema(bare_with_type) is bare_with_type


# --- the registration shape it produces -------------------------------------


def test_bare_schema_wraps_with_outer_name():
    tools, names = _register([{"name": "grep", "description": "d", "parameters": {"type": "object", "properties": {}}}])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "grep"          # outer name present
    assert "function" not in tools[0]["function"]          # not double-wrapped
    assert names == {"grep"}


def test_prewrapped_schema_is_not_double_wrapped():
    prewrapped = {"type": "function", "function": {
        "name": "grep", "description": "d",
        "parameters": {"type": "object", "properties": {}}}}
    tools, names = _register([prewrapped])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "grep"          # real name, not empty
    assert "function" not in tools[0]["function"], "schema was double-wrapped"
    assert names == {"grep"}


def test_mixed_bare_and_prewrapped():
    schemas = [
        {"name": "bare_tool", "description": "d", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "function": {
            "name": "wrapped_tool", "description": "d",
            "parameters": {"type": "object", "properties": {}}}},
    ]
    tools, names = _register(schemas)
    assert names == {"bare_tool", "wrapped_tool"}
    for t in tools:
        assert t["function"].get("name"), "every tool must have a non-empty name"
        assert "function" not in t["function"], "no double-wrap"


def test_no_empty_names_emitted():
    """The exact failure signature: no tool may reach the provider unnamed."""
    prewrapped = [
        {"type": "function", "function": {"name": "alpha", "description": "d", "parameters": {}}},
        {"type": "function", "function": {"name": "beta", "description": "d", "parameters": {}}},
        {"type": "function", "function": {"name": "gamma", "description": "d", "parameters": {}}},
    ]
    tools, _ = _register(prewrapped)
    for i, t in enumerate(tools):
        nm = t.get("function", {}).get("name", "")
        assert nm and nm.strip(), f"tools[{i}].function.name is empty -> would HTTP 400"
