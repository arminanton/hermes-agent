"""Tests for the canonical reasoning-effort vocabulary and clamping.

Ported alongside agent/reasoning_effort.py (upstream f7d90c9410). Asserts the
clamp POLICY as invariants (nearest-weaker, never escalate, monotonic, floor,
none-never-a-target) and pins the live-verified Copilot/Codex/grok wire sets we
depend on, rather than snapshotting every vendor list.
"""

from agent.reasoning_effort import (
    EFFORT_LADDER,
    CODEX_GPT56_EFFORTS,
    CODEX_LEGACY_EFFORTS,
    XAI_GROK46_EFFORTS,
    XAI_LEGACY_EFFORTS,
    clamp_effort,
    codex_supported_efforts,
    grok_supported_efforts,
    kimi_supported_efforts,
    requested_effort,
)


class TestClampPolicy:
    def test_supported_passes_verbatim(self):
        assert clamp_effort("high", ("low", "medium", "high")) == "high"

    def test_unknown_supported_set_passes_through(self):
        assert clamp_effort("ultra", None) == "ultra"
        assert clamp_effort("ultra", ()) == "ultra"

    def test_bespoke_name_passes_through(self):
        # A custom provider name that isn't a ladder level is not guessed at.
        assert clamp_effort("turbo", ("low", "high")) == "turbo"

    def test_nearest_weaker_when_unsupported(self):
        # xhigh not supported -> nearest weaker is high (never escalate to max).
        assert clamp_effort("xhigh", ("low", "medium", "high")) == "high"

    def test_never_escalates(self):
        # max requested, only low/high supported -> high, not something above.
        assert clamp_effort("max", ("low", "high")) == "high"

    def test_floor_when_nothing_weaker(self):
        # low requested but provider floor is high (GLM-5.2 shape) -> high.
        assert clamp_effort("low", ("high", "max")) == "high"

    def test_none_never_a_degradation_target(self):
        # minimal unsupported, only none available -> do NOT switch thinking off.
        assert clamp_effort("minimal", ("none",)) == "minimal"

    def test_none_passes_verbatim_when_supported(self):
        assert clamp_effort("none", ("none", "low", "high")) == "none"

    def test_overrides_win_first(self):
        # Kimi K3: medium documented to round UP to high (its middle/default).
        assert clamp_effort("medium", ("low", "high", "max"), {"medium": "high"}) == "high"

    def test_monotonic_ladder(self):
        supported = ("low", "high")
        low = EFFORT_LADDER.index(clamp_effort("low", supported))
        med = EFFORT_LADDER.index(clamp_effort("medium", supported))
        high = EFFORT_LADDER.index(clamp_effort("high", supported))
        assert low <= med <= high  # a stronger ask never resolves weaker


class TestCodexEfforts:
    def test_gpt56_has_max(self):
        assert codex_supported_efforts("gpt-5.6-sol") == CODEX_GPT56_EFFORTS
        assert "max" in CODEX_GPT56_EFFORTS

    def test_legacy_gpt5_rejects_minimal_and_max(self):
        # gpt-5.4/5.5 floor is none, top is xhigh (live-verified).
        assert codex_supported_efforts("gpt-5.4") == CODEX_LEGACY_EFFORTS
        assert codex_supported_efforts("gpt-5.5") == CODEX_LEGACY_EFFORTS
        assert "minimal" not in CODEX_LEGACY_EFFORTS
        assert "max" not in CODEX_LEGACY_EFFORTS
        assert CODEX_LEGACY_EFFORTS[0] == "none"

    def test_minimal_clamps_to_low_on_codex(self):
        # gpt-5.4 rejects minimal; nearest weaker supported is low
        # (none is not a degradation target).
        assert clamp_effort("minimal", CODEX_LEGACY_EFFORTS) == "low"

    def test_max_clamps_to_xhigh_on_gpt55(self):
        assert clamp_effort("max", CODEX_LEGACY_EFFORTS) == "xhigh"


class TestGrokEfforts:
    def test_grok46_adds_xhigh(self):
        assert grok_supported_efforts("grok-4.6") == XAI_GROK46_EFFORTS
        assert "xhigh" in XAI_GROK46_EFFORTS

    def test_grok45_tops_at_high(self):
        assert grok_supported_efforts("grok-4.5") == XAI_LEGACY_EFFORTS
        assert "xhigh" not in XAI_LEGACY_EFFORTS

    def test_grok_aggregator_prefix(self):
        assert grok_supported_efforts("x-ai/grok-4.6") == XAI_GROK46_EFFORTS

    def test_unknown_grok_no_dial(self):
        assert grok_supported_efforts("grok-3") == ()

    def test_xhigh_clamps_to_high_on_grok45(self):
        assert clamp_effort("xhigh", XAI_LEGACY_EFFORTS) == "high"


class TestKimiEfforts:
    def test_k3_slug_detection(self):
        from agent.reasoning_effort import KIMI_K3_EFFORTS, KIMI_K2_EFFORTS

        assert kimi_supported_efforts("k3") == KIMI_K3_EFFORTS
        assert kimi_supported_efforts("kimi-k3-256k") == KIMI_K3_EFFORTS
        assert kimi_supported_efforts("kimi-k2.6") == KIMI_K2_EFFORTS


class TestRequestedEffort:
    def test_extracts_effort(self):
        assert requested_effort({"effort": "high"}) == "high"

    def test_none_when_disabled(self):
        assert requested_effort({"enabled": False, "effort": "high"}) is None

    def test_none_when_absent(self):
        assert requested_effort(None) is None
        assert requested_effort({}) is None
