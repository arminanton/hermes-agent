"""Copilot returns Gemini reasoning under ``reasoning_text`` — capture it.

Second, distinct root cause behind zero-reasoning `-z` runs. Wiring
``reasoning_config`` through oneshot fixed the Anthropic lane (Claude), but
Gemini rides chat_completions and stayed at zero. Live probing of
api.githubcopilot.com showed why: it answers with

    message keys: ['content', 'reasoning_opaque', 'reasoning_text', 'role']
    usage: {..., 'reasoning_tokens': 455}

and streams ``reasoning_text`` deltas. The OpenAI SDK has no such field, so it
lands in ``model_extra`` and every Hermes reader (which looks only at
``reasoning`` / ``reasoning_content`` / ``reasoning_details``) dropped it. The
model thought, the account was billed for the tokens, and nothing was persisted.

Run: python -m pytest tests/agent/test_copilot_reasoning_text_capture.py -q
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import extract_reasoning


class _SDKMessage:
    """Mimics an OpenAI SDK message: unknown wire fields go to model_extra."""

    def __init__(self, content="hi", model_extra=None, **known):
        self.content = content
        self.tool_calls = None
        self.model_extra = model_extra or {}
        for key, value in known.items():
            setattr(self, key, value)


# ── extract_reasoning ──────────────────────────────────────────────────


class TestExtractReasoningReadsReasoningText:
    def test_reasoning_text_in_model_extra_is_captured(self):
        """The exact shape Copilot returns for Gemini."""
        msg = _SDKMessage(model_extra={
            "reasoning_text": "**Exploring Multiplication Strategies**\n\n47 x 53 = 2491.",
            "reasoning_opaque": "opaque-blob",
        })
        got = extract_reasoning(None, msg)
        assert got is not None, "Copilot's reasoning_text was dropped again"
        assert "Exploring Multiplication Strategies" in got

    def test_reasoning_text_as_plain_attribute_is_captured(self):
        msg = _SDKMessage(reasoning_text="thinking out loud")
        assert extract_reasoning(None, msg) == "thinking out loud"

    def test_standard_reasoning_field_still_wins_and_is_not_duplicated(self):
        """Providers on the normal field must be unaffected."""
        msg = _SDKMessage(
            reasoning="canonical reasoning",
            model_extra={"reasoning_text": "canonical reasoning"},
        )
        got = extract_reasoning(None, msg)
        assert got == "canonical reasoning"
        assert got.count("canonical reasoning") == 1

    def test_reasoning_content_provider_unaffected(self):
        msg = _SDKMessage(reasoning_content="deepseek style")
        assert extract_reasoning(None, msg) == "deepseek style"

    @pytest.mark.parametrize("value", ["", "   ", None, 123, {"a": 1}])
    def test_junk_reasoning_text_ignored(self, value):
        msg = _SDKMessage(model_extra={"reasoning_text": value})
        assert extract_reasoning(None, msg) is None

    def test_no_reasoning_anywhere_returns_none(self):
        assert extract_reasoning(None, _SDKMessage()) is None

    def test_opaque_blob_is_never_treated_as_reasoning(self):
        """reasoning_opaque is an encrypted handle, not displayable text."""
        msg = _SDKMessage(model_extra={"reasoning_opaque": "AAAA-encrypted"})
        assert extract_reasoning(None, msg) is None


# ── transport normalization ────────────────────────────────────────────


def _normalize(msg, finish_reason="stop"):
    """Run the chat_completions transport's response normalizer."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    raw = SimpleNamespace(choices=[choice], usage=None, model="gemini-3.5-flash")
    transport = ChatCompletionsTransport.__new__(ChatCompletionsTransport)
    return ChatCompletionsTransport.normalize_response(transport, raw)


class TestTransportMapsReasoningTextOntoReasoning:
    def test_reasoning_text_becomes_canonical_reasoning(self):
        msg = _SDKMessage(model_extra={"reasoning_text": "step by step: 2491"})
        assert _normalize(msg).reasoning == "step by step: 2491"

    def test_existing_reasoning_is_not_clobbered(self):
        msg = _SDKMessage(
            reasoning="from the standard field",
            model_extra={"reasoning_text": "from copilot"},
        )
        assert _normalize(msg).reasoning == "from the standard field"

    def test_reasoning_content_still_lands_in_provider_data(self):
        msg = _SDKMessage(model_extra={"reasoning_content": "moonshot style"})
        assert _normalize(msg).reasoning_content == "moonshot style"

    def test_absent_reasoning_stays_none(self):
        assert _normalize(_SDKMessage()).reasoning is None

    def test_blank_reasoning_text_stays_none(self):
        msg = _SDKMessage(model_extra={"reasoning_text": "   "})
        assert _normalize(msg).reasoning is None

    def test_blank_standard_field_does_not_suppress_the_fallback(self):
        """A placeholder ``reasoning: ""`` must not hide the real text."""
        msg = _SDKMessage(
            reasoning="",
            model_extra={"reasoning_text": "the actual thinking"},
        )
        assert _normalize(msg).reasoning == "the actual thinking"


class TestCoexistenceNoDuplicationNoLoss:
    """Both standard and non-standard fields populated with DIFFERENT text.

    Copilot has not been observed doing this, but the fallback must degrade
    predictably if it starts: keep the standard field, never concatenate the
    two, and never emit the same text twice.
    """

    DIFFERENT = {"reasoning_text": "copilot-specific text"}

    def test_transport_keeps_standard_and_drops_the_duplicate(self):
        result = _normalize(_SDKMessage(reasoning="standard text", model_extra=dict(self.DIFFERENT)))
        assert result.reasoning == "standard text"
        assert "copilot-specific text" not in (result.reasoning or "")

    def test_transport_preserves_reasoning_content_separately(self):
        extra = dict(self.DIFFERENT)
        extra["reasoning_content"] = "content-field text"
        result = _normalize(_SDKMessage(model_extra=extra))
        # reasoning_content keeps its own slot; reasoning_text fills `reasoning`
        assert result.reasoning_content == "content-field text"
        assert result.reasoning == "copilot-specific text"

    def test_extract_reasoning_emits_each_distinct_text_once(self):
        msg = _SDKMessage(reasoning="alpha", model_extra={"reasoning_text": "beta"})
        got = extract_reasoning(None, msg)
        assert got.count("alpha") == 1
        assert got.count("beta") == 1

    def test_extract_reasoning_does_not_duplicate_identical_text(self):
        msg = _SDKMessage(
            reasoning_content="same text",
            model_extra={"reasoning_text": "same text"},
        )
        assert extract_reasoning(None, msg).count("same text") == 1

    def test_streaming_never_concatenates_both_fields(self):
        delta = SimpleNamespace(
            reasoning="standard delta",
            model_extra={"reasoning_text": "copilot delta"},
        )
        picked = TestStreamingReasoningTextDeltas._pick(delta)
        assert picked == "standard delta"
        assert "copilot delta" not in picked


# ── streaming deltas ───────────────────────────────────────────────────


class TestStreamingReasoningTextDeltas:
    """Copilot streams thinking as ``reasoning_text`` delta fields."""

    @staticmethod
    def _pick(delta):
        """Mirror the accumulator's field-selection precedence."""
        reasoning_text = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
        )
        if not reasoning_text:
            extra = getattr(delta, "model_extra", None) or {}
            if isinstance(extra, dict):
                candidate = extra.get("reasoning_text")
                if isinstance(candidate, str) and candidate:
                    reasoning_text = candidate
        return reasoning_text

    def test_delta_reasoning_text_is_picked_up(self):
        delta = SimpleNamespace(model_extra={"reasoning_text": "chunk one"})
        assert self._pick(delta) == "chunk one"

    def test_standard_delta_fields_take_precedence(self):
        delta = SimpleNamespace(
            reasoning_content="standard",
            model_extra={"reasoning_text": "copilot"},
        )
        assert self._pick(delta) == "standard"

    def test_content_only_delta_yields_nothing(self):
        assert self._pick(SimpleNamespace(content="hello", model_extra={})) is None

    def test_accumulator_uses_the_same_selection(self):
        """Keep this test honest: assert the production source really does it."""
        import inspect

        from agent import chat_completion_helpers

        source = inspect.getsource(chat_completion_helpers)
        assert 'reasoning_text' in source
        assert '_delta_extra.get("reasoning_text")' in source


# ── recorded live fixtures ─────────────────────────────────────────────


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "copilot"
RESPONSE_FIXTURE = FIXTURE_DIR / "gemini_reasoning_text_response.json"
STREAM_FIXTURE = FIXTURE_DIR / "gemini_reasoning_text_stream.json"


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _as_sdk_message(raw_message):
    """Rebuild the OpenAI SDK's shape: unknown wire keys land in model_extra."""
    known = {"content", "role", "tool_calls", "refusal", "reasoning", "reasoning_content"}
    extra = {k: v for k, v in raw_message.items() if k not in known}
    msg = _SDKMessage(content=raw_message.get("content"), model_extra=extra)
    msg.tool_calls = raw_message.get("tool_calls")
    for key in ("reasoning", "reasoning_content"):
        if key in raw_message:
            setattr(msg, key, raw_message[key])
    return msg


class TestRecordedCopilotFixtures:
    """Drive the extractors from responses actually recorded off Copilot.

    These are verbatim captures from api.githubcopilot.com (gemini-3.5-flash,
    reasoning effort high), not hand-written shapes, so they pin the real wire
    contract rather than the author's idea of it.
    """

    def test_canary_recorded_response_still_carries_reasoning_text(self):
        """Fails LOUDLY if the recorded contract stops matching expectations.

        If Copilot renames or drops the field, the production fallback degrades
        silently (back to no reasoning, no crash). This canary is the alarm: it
        asserts the recorded shape the fallback depends on actually exists, so a
        contract change surfaces as a red test rather than silent data loss.
        """
        fixtures = _load(RESPONSE_FIXTURE)
        assert set(fixtures) >= {"non_streaming", "non_streaming_tool_call"}
        for name, payload in fixtures.items():
            message = payload["choices"][0]["message"]
            assert "reasoning_text" in message, (
                f"recorded Copilot response '{name}' no longer has reasoning_text; "
                "the capture path in agent/transports/chat_completions.py and "
                "agent/agent_runtime_helpers.py needs updating to the new key"
            )
            assert isinstance(message["reasoning_text"], str)
            assert message["reasoning_text"].strip()
            # the standard fields are genuinely absent, which is the whole bug
            assert "reasoning" not in message
            assert "reasoning_content" not in message

    def test_canary_recorded_stream_still_carries_reasoning_text_deltas(self):
        stream = _load(STREAM_FIXTURE)
        assert "reasoning_text" in stream["delta_keys_observed"], (
            "recorded Copilot stream no longer emits reasoning_text deltas"
        )
        assert "reasoning" not in stream["delta_keys_observed"]
        assert "reasoning_content" not in stream["delta_keys_observed"]

    def test_recorded_non_streaming_response_yields_reasoning(self):
        message = _load(RESPONSE_FIXTURE)["non_streaming"]["choices"][0]["message"]
        result = _normalize(_as_sdk_message(message))
        assert result.reasoning == message["reasoning_text"]
        assert len(result.reasoning) > 100

    def test_recorded_tool_call_response_yields_reasoning_and_keeps_tool_calls(self):
        """The tool-call shape, not just the plain-answer shape."""
        message = _load(RESPONSE_FIXTURE)["non_streaming_tool_call"]["choices"][0]["message"]
        assert message["tool_calls"], "fixture should exercise the tool-call path"
        sdk_message = _as_sdk_message(message)
        assert extract_reasoning(None, sdk_message) == message["reasoning_text"]

    def test_recorded_opaque_field_is_not_mistaken_for_reasoning(self):
        message = _load(RESPONSE_FIXTURE)["non_streaming"]["choices"][0]["message"]
        assert "reasoning_opaque" in message
        got = extract_reasoning(None, _as_sdk_message(message))
        assert got == message["reasoning_text"]
        assert message["reasoning_opaque"] not in got

    def test_recorded_stream_deltas_accumulate(self):
        """Replay the recorded deltas through the accumulator's selection."""
        stream = _load(STREAM_FIXTURE)
        collected = []
        for chunk in stream["chunks"]:
            for choice in chunk.get("choices", []):
                raw_delta = choice.get("delta") or {}
                known = {"content", "role", "tool_calls"}
                delta = SimpleNamespace(
                    model_extra={k: v for k, v in raw_delta.items() if k not in known}
                )
                picked = TestStreamingReasoningTextDeltas._pick(delta)
                if picked:
                    collected.append(picked)
        assert collected, "no reasoning was recovered from the recorded stream"
        assert "".join(collected).strip()

    def test_fixtures_look_like_verbatim_copilot_captures(self):
        """Provenance guard: these must stay real recordings, not hand edits.

        Asserts server-assigned fields no author would invent by hand (opaque
        response ids, epoch timestamps, Copilot's non-OpenAI ``copilot_usage``
        block, and the ``usage.reasoning_tokens`` counter that corroborates the
        model really did reason).
        """
        for name, payload in _load(RESPONSE_FIXTURE).items():
            assert payload["model"] == "gemini-3.5-flash", name
            # server-assigned, opaque, not a round number
            assert isinstance(payload["id"], str) and len(payload["id"]) > 10, name
            assert isinstance(payload["created"], int) and payload["created"] > 1_700_000_000, name
            # Copilot-specific envelope key absent from the OpenAI schema
            assert "copilot_usage" in payload, name
            # the model was billed for reasoning it produced
            assert payload["usage"]["reasoning_tokens"] > 0, name

    def test_documented_drift_limitation_is_recorded(self):
        """The gap the fixtures cannot cover must stay written down."""
        readme = FIXTURE_DIR / "README.md"
        assert readme.exists(), "fixture provenance/limitations README is missing"
        text = readme.read_text(encoding="utf-8")
        assert "reasoning_text" in text
        assert "live drift probe" in text.lower()
