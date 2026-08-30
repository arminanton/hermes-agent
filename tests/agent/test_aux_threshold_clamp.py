"""The auxiliary-model threshold clamp must be releasable.

When the auxiliary compression model's context window is smaller than the main
model's compression threshold, Hermes lowers the live threshold so compression
can still run. That auto-correct is right; making it permanent was not.

The old code overwrote ``threshold_percent`` outright::

    agent.context_compressor.threshold_percent = new_threshold / main_ctx

so a single transient event, an aux outage, a provider hiccup, a momentarily
mis-resolved window, rewrote the ratio for the rest of the session. Every later
``update_model`` re-derived from the shrunken value, and a 922k main model that
touched an 80k aux once kept compressing at roughly 8.7% of its window long
after the aux recovered.

The fix remembers the pre-clamp ratio and restores it as soon as the aux model
can cover the original threshold again.
"""


class _Compressor:
    """Minimal stand-in for ContextCompressor's threshold surface."""

    def __init__(self, context_length=922_000, threshold_percent=0.75):
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.threshold_tokens = int(context_length * threshold_percent)


def _clamp(cc, aux_context):
    """The clamp/release logic as wired into check_compression_feasibility."""
    prior = getattr(cc, "_aux_clamped_from_percent", None)
    if prior is not None:
        orig = int((cc.context_length or 0) * prior)
        if aux_context >= orig:
            # full recovery: drop the clamp
            cc.threshold_percent = prior
            cc.threshold_tokens = orig
            cc._aux_clamped_from_percent = None
            cc._aux_clamp_tokens = None
        else:
            # partial recovery: track current aux capacity, keep the baseline
            cc.threshold_tokens = aux_context
            if cc.context_length:
                cc.threshold_percent = aux_context / cc.context_length
            cc._aux_clamp_tokens = aux_context
    if aux_context < cc.threshold_tokens:
        if getattr(cc, "_aux_clamped_from_percent", None) is None:
            cc._aux_clamped_from_percent = cc.threshold_percent
        cc._aux_clamp_tokens = aux_context
        cc.threshold_tokens = aux_context
        if cc.context_length:
            cc.threshold_percent = aux_context / cc.context_length
    return cc


# --------------------------------------------------------------------------

def test_healthy_aux_leaves_threshold_alone():
    cc = _clamp(_Compressor(), aux_context=922_000)
    assert cc.threshold_tokens == 691_500
    assert cc.threshold_percent == 0.75
    assert getattr(cc, "_aux_clamped_from_percent", None) is None


def test_small_aux_lowers_the_threshold():
    """The auto-correct itself is correct and must keep working."""
    cc = _clamp(_Compressor(), aux_context=80_000)
    assert cc.threshold_tokens == 80_000
    assert cc.threshold_percent < 0.75


def test_original_percent_is_remembered():
    cc = _clamp(_Compressor(), aux_context=80_000)
    assert cc._aux_clamped_from_percent == 0.75


def test_clamp_is_released_when_aux_recovers():
    """The regression: a transient outage must not permanently degrade."""
    cc = _clamp(_Compressor(), aux_context=80_000)
    assert cc.threshold_tokens == 80_000          # degraded
    cc = _clamp(cc, aux_context=922_000)          # aux comes back
    assert cc.threshold_tokens == 691_500         # restored
    assert cc.threshold_percent == 0.75
    assert cc._aux_clamped_from_percent is None


def test_repeated_outages_do_not_ratchet_downward():
    """Each clamp must measure against the ORIGINAL ratio, not the last one.

    Without the remembered baseline the threshold ratchets: 691,500 -> 80,000
    -> 64,000 -> ..., each outage compounding the previous degradation.
    """
    cc = _Compressor()
    for aux in (80_000, 922_000, 64_000, 922_000):
        cc = _clamp(cc, aux)
    assert cc.threshold_tokens == 691_500
    assert cc.threshold_percent == 0.75


def test_partial_recovery_stays_clamped():
    """An aux that recovers but still cannot cover the original stays clamped."""
    cc = _clamp(_Compressor(), aux_context=80_000)
    cc = _clamp(cc, aux_context=200_000)
    assert cc.threshold_tokens == 200_000
    assert cc._aux_clamped_from_percent == 0.75   # baseline still remembered


def test_the_922k_to_80k_scenario_from_the_logs():
    """The exact live case: a 922k model must not stay pinned at 80k."""
    cc = _Compressor(context_length=922_000, threshold_percent=0.75)
    cc = _clamp(cc, 80_000)
    assert cc.threshold_percent < 0.09            # ~8.7%, the reported damage
    cc = _clamp(cc, 922_000)
    assert cc.threshold_percent == 0.75
