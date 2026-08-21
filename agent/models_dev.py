"""Models.dev registry integration — primary database for providers and models.

Fetches from https://models.dev/api.json — a community-maintained database
of 4000+ models across 109+ providers.  Provides:

- **Provider metadata**: name, base URL, env vars, documentation link
- **Model metadata**: context window, max output, cost/M tokens, capabilities
  (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
  open-weights flag, family grouping, deprecation status

Data resolution order (like TypeScript OpenCode):
  1. Bundled snapshot (ships with the package — offline-first)
  2. Disk cache (~/.hermes/models_dev_cache.json)
  3. Network fetch (https://models.dev/api.json)
  4. Background refresh every 60 minutes

Other modules should import the dataclasses and query functions from here
rather than parsing the raw JSON themselves.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_json_write

import requests

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 3600  # 1 hour in-memory

# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0


# ---------------------------------------------------------------------------
# Dataclasses — rich metadata for providers and models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Full metadata for a single model from models.dev."""

    id: str
    name: str
    family: str
    provider_id: str        # models.dev provider ID (e.g. "anthropic")

    # Capabilities
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False       # supports image/file attachments (vision)
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False

    # Modalities
    input_modalities: Tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: Tuple[str, ...] = ()

    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None

    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None

    # Metadata
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""          # "alpha", "beta", "deprecated", or ""
    interleaved: Any = False  # True or {"field": "reasoning_content"}

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_cost(self) -> str:
        """Human-readable cost string, e.g. '$3.00/M in, $15.00/M out'."""
        if not self.has_cost_data():
            return "unknown"
        parts = [f"${self.cost_input:.2f}/M in", f"${self.cost_output:.2f}/M out"]
        if self.cost_cache_read is not None:
            parts.append(f"cache read ${self.cost_cache_read:.2f}/M")
        return ", ".join(parts)

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        caps = []
        if self.reasoning:
            caps.append("reasoning")
        if self.tool_call:
            caps.append("tools")
        if self.supports_vision():
            caps.append("vision")
        if self.supports_pdf():
            caps.append("PDF")
        if self.supports_audio_input():
            caps.append("audio")
        if self.structured_output:
            caps.append("structured output")
        if self.open_weights:
            caps.append("open weights")
        return ", ".join(caps) if caps else "basic"


@dataclass
class ProviderInfo:
    """Full metadata for a provider from models.dev."""

    id: str                         # models.dev provider ID
    name: str                       # display name
    env: Tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


# ---------------------------------------------------------------------------
# Provider ID mapping: Hermes ↔ models.dev
# ---------------------------------------------------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter",
    "novita": "novita-ai",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "zai": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "stepfun": "stepfun",
    "kimi-coding-cn": "kimi-for-coding",
    "minimax": "minimax",
    "minimax-oauth": "minimax",
    "minimax-cn": "minimax-cn",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "qwen-oauth": "alibaba",
    "copilot": "github-copilot",
    "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "gemini": "google",
    "google": "google",
    "xai": "xai",
    # xAI OAuth is an authentication/transport path for the same xAI model
    # catalog, so model metadata should resolve through the xAI provider.
    "xai-oauth": "xai",
    "xiaomi": "xiaomi",
    "nvidia": "nvidia",
    "groq": "groq",
    "mistral": "mistral",
    "togetherai": "togetherai",
    "perplexity": "perplexity",
    "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}

# Reverse mapping: models.dev → Hermes (built lazily)
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, str]] = None



def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.json"


def _load_disk_cache() -> Dict[str, Any]:
    """Load models.dev data from disk cache."""
    try:
        cache_path = _get_cache_path()
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("Failed to load models.dev disk cache: %s", e)
    return {}


def _disk_cache_age_seconds() -> Optional[float]:
    """Return age (in seconds) of the disk cache file, or None if missing.

    Used by ``fetch_models_dev`` to short-circuit the network probe when
    a recent on-disk cache exists. Errors (missing file, permission
    denied, weird filesystem) all return None — callers fall through
    to the network fetch path.
    """
    try:
        cache_path = _get_cache_path()
        if not cache_path.exists():
            return None
        mtime = cache_path.stat().st_mtime
        age = time.time() - mtime
        # Negative age means the file's mtime is in the future (clock skew
        # or system clock reset). Treat as "unknown freshness" → fall
        # through to network so we don't serve potentially-bad data
        # forever.
        if age < 0:
            return None
        return age
    except Exception as e:
        logger.debug("Failed to stat models.dev disk cache: %s", e)
        return None


def _save_disk_cache(data: Dict[str, Any]) -> None:
    """Save models.dev data to disk cache atomically."""
    try:
        cache_path = _get_cache_path()
        atomic_json_write(cache_path, data, indent=None, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save models.dev disk cache: %s", e)


def fetch_models_dev(force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch models.dev registry. Cache hierarchy: in-mem → disk → network.

    Returns the full registry dict keyed by provider ID, or empty dict on failure.

    Cache hierarchy (when ``force_refresh=False``):
      1. In-memory cache, populated and < TTL old → return immediately.
      2. **Disk cache file < TTL old by mtime → load, populate in-mem, return.**
         No network call. Saves ~500 ms per cold-start agent construction;
         ``models.dev`` only changes when providers add new models, so a
         1 hour staleness window is acceptable (same TTL as in-mem cache).
      3. Network fetch → on success, save to disk + in-mem and return.
      4. Network fails → fall back to ANY available disk cache (even stale)
         with a short 5 min in-mem grace period before retrying network.

    When ``force_refresh=True`` (used by ``hermes config refresh``, the
    \"refresh model catalog\" code path), stages 1 and 2 are skipped. The
    function always hits the network and only falls back to disk if the
    network call fails.
    """
    global _models_dev_cache, _models_dev_cache_time

    # Stage 1: fresh in-memory cache wins. This is the hot path on
    # long-lived processes — no I/O, no system calls.
    if (
        not force_refresh
        and _models_dev_cache
        and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL
    ):
        return _models_dev_cache

    # Stage 2: fresh-by-mtime disk cache short-circuits the network call.
    # Only kicks in on cold-start processes (in-mem cache is empty or
    # expired) and only when the user hasn't asked for a forced refresh.
    # Skipped if the disk cache file is missing, unreadable, or older
    # than _MODELS_DEV_CACHE_TTL.
    if not force_refresh:
        disk_age = _disk_cache_age_seconds()
        if disk_age is not None and disk_age < _MODELS_DEV_CACHE_TTL:
            disk_data = _load_disk_cache()
            if disk_data:
                _models_dev_cache = disk_data
                # Anchor in-mem TTL to the disk file's age so we don't
                # extend an already-aging cache by another full hour.
                _models_dev_cache_time = time.time() - disk_age
                logger.debug(
                    "Loaded models.dev from fresh disk cache "
                    "(%d providers, age=%.0fs)", len(disk_data), disk_age,
                )
                return _models_dev_cache

    # Stage 3: network fetch.
    try:
        response = requests.get(MODELS_DEV_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data:
            _models_dev_cache = data
            _models_dev_cache_time = time.time()
            _save_disk_cache(data)
            logger.debug(
                "Fetched models.dev registry: %d providers, %d total models",
                len(data),
                sum(len(p.get("models", {})) for p in data.values() if isinstance(p, dict)),
            )
            return data
    except Exception as e:
        logger.debug("Failed to fetch models.dev: %s", e)

    # Stage 4: network failed — fall back to whatever disk cache exists,
    # even if it's stale. Give it a short 5 min in-mem TTL so we retry
    # the network soon instead of serving stale data for a full hour.
    if not _models_dev_cache:
        _models_dev_cache = _load_disk_cache()
        if _models_dev_cache:
            _models_dev_cache_time = time.time() - _MODELS_DEV_CACHE_TTL + 300
            logger.debug("Loaded models.dev from disk cache (%d providers)", len(_models_dev_cache))

    return _models_dev_cache


def lookup_models_dev_context(provider: str, model: str) -> Optional[int]:
    """Look up context_length for a provider+model combo in models.dev.

    Returns the context window in tokens, or None if not found.
    Handles case-insensitive matching and filters out context=0 entries.

    An EXPLICIT ``model_overrides`` config entry for this provider+model wins
    over the catalog value; ``_default`` entries fill the gap only when the
    catalog has no answer — the supported self-unblock path for models with
    wrong or missing context in models.dev (#84482).
    """
    # Explicit config override, checked before catalog so it always wins.
    override_ctx = _override_context_window(provider, model)
    if override_ctx is not None:
        return override_ctx

    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return _default_override_context(provider)

    data = fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return _default_override_context(provider)

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return _default_override_context(provider)

    # Exact match
    entry = models.get(model)
    if entry:
        ctx = _extract_context(entry)
        if ctx:
            return ctx

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower:
            ctx = _extract_context(mdata)
            if ctx:
                return ctx

    # Suffix-aware fallback: some providers (e.g. ollama-cloud) store
    # model IDs with :cloud / -cloud suffixes in models.dev while the
    # live API returns bare names.  Without this, kimi-k2.6 misses the
    # kimi-k2.6:cloud entry and falls through to stale OpenRouter metadata
    # reporting 32768 — tripping the 64k minimum-context guard.
    # The suffix-stripping in fetch_ollama_cloud_models() handles the
    # model-picker UX; this handles the context-length lookup path.
    for suffix in (":cloud", "-cloud"):
        suffixed_key = model + suffix
        entry = models.get(suffixed_key)
        if entry:
            ctx = _extract_context(entry)
            if ctx:
                return ctx
        # Also try case-insensitive
        suffixed_lower = model_lower + suffix
        for mid, mdata in models.items():
            if mid.lower() == suffixed_lower:
                ctx = _extract_context(mdata)
                if ctx:
                    return ctx

    # Catalog miss — a _default override may fill the gap (#84482).
    return _default_override_context(provider)


def _extract_context(entry: Dict[str, Any]) -> Optional[int]:
    """Extract context_length from a models.dev model entry.

    Returns None for invalid/zero values (some audio/image models have context=0).
    """
    if not isinstance(entry, dict):
        return None
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    ctx = limit.get("context")
    if isinstance(ctx, (int, float)) and ctx > 0:
        return int(ctx)
    return None


# ---------------------------------------------------------------------------
# Model capability metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelCapabilities:
    """Structured capability metadata for a model from models.dev."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    context_window: int = 200000
    max_output_tokens: int = 8192
    model_family: str = ""


# --------------------------------------------------------------------------- #
# Per-model metadata overrides (config.yaml → model_overrides)                 #
# --------------------------------------------------------------------------- #
#
# Canonical override schema (the ONLY key space consumers accept):
#   context_window, max_output_tokens, supports_tools, supports_vision,
#   supports_reasoning, model_family
#
# Resolution semantics:
#   1. ``model_overrides.<provider>.<model_id>`` — explicit override. Always
#      wins over the catalog (and over our probe-verified overrides) for the
#      fields it sets (partial patch). This is the user's manual declaration,
#      so it is the highest authority in the metadata stack.
#   2. ``model_overrides.<provider>._default`` / ``model_overrides._default``
#      — FILL-GAP defaults. They apply ONLY to models the catalog does not
#      know (the #8731/#84482 self-unblock path for custom/local/new models)
#      and never displace catalog data for known models. A
#      ``_default: {context_window: 128000}`` therefore cannot clamp every
#      catalog-known model of a provider.
#
# Provider keys accept the Hermes provider id (as used elsewhere in
# config.yaml) or the models.dev provider id (``copilot`` and
# ``github-copilot`` both work). Model ids match exactly, then
# case-insensitively (mirroring catalog lookup).
#
# Ported from upstream dafdba324a + de47d19f1f, adapted to compose with our
# existing ``_resolve_probe_override`` layer in ``get_model_info`` (user
# config wins over probe-verified overrides).

_OVERRIDE_CACHE: Optional[Dict[str, Any]] = None
_OVERRIDE_CACHE_CFG_ID: int = 0
_OVERRIDE_WARNED_KEYS: set = set()

#: The recognized canonical override fields (for validation / docs).
_OVERRIDE_FIELDS = frozenset({
    "context_window", "max_output_tokens", "supports_tools",
    "supports_vision", "supports_reasoning", "model_family",
})


def _load_model_overrides() -> Dict[str, Any]:
    """Load and cache the ``model_overrides`` config section.

    Caches by ``id(cfg)`` so a config reload (new dict identity) invalidates
    automatically. Returns empty dict on any failure.
    """
    global _OVERRIDE_CACHE, _OVERRIDE_CACHE_CFG_ID
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        cfg = load_config_readonly()
        cfg_id = id(cfg)
        if cfg_id == _OVERRIDE_CACHE_CFG_ID and _OVERRIDE_CACHE is not None:
            return _OVERRIDE_CACHE
        raw = cfg_get(cfg, "model_overrides", default={})
        overrides = raw if isinstance(raw, dict) else {}
        _OVERRIDE_CACHE = overrides
        _OVERRIDE_CACHE_CFG_ID = cfg_id
        return overrides
    except Exception:
        return {}


def _provider_override_section(provider: str) -> Optional[Dict[str, Any]]:
    """Return the override section for *provider*, or None.

    Accepts either the Hermes provider id or the models.dev provider id as the
    config key, so ``copilot`` and ``github-copilot`` both work regardless of
    which id space a caller passes in.
    """
    overrides = _load_model_overrides()
    if not overrides:
        return None
    provider_key = (provider or "").strip()
    if not provider_key:
        return None

    candidates = [provider_key]
    mapped = PROVIDER_TO_MODELS_DEV.get(provider_key)
    if mapped and mapped != provider_key:
        candidates.append(mapped)
    # Reverse: caller passed a models.dev id, config keyed by Hermes id.
    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        if mdev_id == provider_key and hermes_id != provider_key:
            candidates.append(hermes_id)

    for key in candidates:
        section = overrides.get(key)
        if isinstance(section, dict):
            return section
    return None


def _explicit_model_override(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Return the explicit per-provider+model override dict, or None.

    Model ids match exactly first, then case-insensitively (skipping the
    ``_default`` sentinel), mirroring catalog lookup behavior.
    """
    model_key = (model or "").strip()
    if not model_key:
        return None
    section = _provider_override_section(provider)
    if section is None:
        return None

    entry = section.get(model_key)
    if isinstance(entry, dict):
        return entry

    model_lower = model_key.lower()
    for mid, mdata in section.items():
        if mid == "_default":
            continue
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata
    return None


def _default_model_override(provider: str) -> Optional[Dict[str, Any]]:
    """Return the fill-gap ``_default`` override for *provider*, or None.

    Checks the per-provider ``_default`` first, then the global one. Only
    consulted for models the catalog does not know (see the block comment).
    """
    section = _provider_override_section(provider)
    if section is not None:
        default = section.get("_default")
        if isinstance(default, dict):
            return default
    overrides = _load_model_overrides()
    global_default = overrides.get("_default")
    if isinstance(global_default, dict):
        return global_default
    return None


def _override_for(
    provider: str, model: str, *, catalog_hit: bool
) -> Optional[Dict[str, Any]]:
    """Select the override dict for a lookup, honoring fill-gap semantics.

    Explicit per-provider+model overrides always apply. ``_default`` entries
    apply only when the catalog has no entry for the model.
    """
    explicit = _explicit_model_override(provider, model)
    if explicit is not None:
        return explicit
    if catalog_hit:
        return None
    return _default_model_override(provider)


def _override_int(override: Dict[str, Any], key: str) -> Optional[int]:
    """Coerce an override field to a positive int, warning once on garbage."""
    raw = override.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    warn_key = (key, repr(raw))
    if warn_key not in _OVERRIDE_WARNED_KEYS:
        _OVERRIDE_WARNED_KEYS.add(warn_key)
        logger.warning(
            "model_overrides: ignoring invalid %s value %r "
            "(expected a positive integer)", key, raw,
        )
    return None


def _override_context_window(provider: str, model: str) -> Optional[int]:
    """Return the EXPLICITLY overridden context_window, or None.

    Explicit-only on purpose: this runs early in the resolution chain
    (agent/model_metadata.py step 0b, before custom_providers and live
    probes), where a ``_default`` must not preempt more specific sources.
    Fill-gap defaults are applied later by ``lookup_models_dev_context`` once
    the catalog has actually missed.
    """
    ov = _explicit_model_override(provider, model)
    if ov is None:
        return None
    return _override_int(ov, "context_window")


def _default_override_context(provider: str) -> Optional[int]:
    """Fill-gap context from a ``_default`` override, for catalog misses."""
    default = _default_model_override(provider)
    if default is None:
        return None
    return _override_int(default, "context_window")


def _override_to_catalog_shape(override: Dict[str, Any]) -> Dict[str, Any]:
    """Translate canonical override keys into a models.dev-shaped patch.

    ``get_model_info``/``_parse_model_info`` consume the raw catalog shape
    (``limit.context``, ``tool_call``, ...). All override consumers accept ONE
    canonical schema (the documented ``context_window``/``supports_*`` keys),
    so this boundary translates rather than forcing users to know the internal
    catalog shape.
    """
    patch: Dict[str, Any] = {}
    limit: Dict[str, Any] = {}
    ctx = _override_int(override, "context_window")
    if ctx is not None:
        limit["context"] = ctx
    out = _override_int(override, "max_output_tokens")
    if out is not None:
        limit["output"] = out
    if limit:
        patch["limit"] = limit
    if "supports_tools" in override:
        patch["tool_call"] = bool(override["supports_tools"])
    if "supports_reasoning" in override:
        patch["reasoning"] = bool(override["supports_reasoning"])
    if "supports_vision" in override:
        patch["attachment"] = bool(override["supports_vision"])
        patch["_vision_override"] = bool(override["supports_vision"])
    if "model_family" in override:
        patch["family"] = str(override["model_family"] or "")
    return patch


def _merge_catalog_entry_with_override(
    raw: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Patch a catalog entry with a canonical-schema override.

    Sub-dicts (``limit``, ``modalities``) are merged, not clobbered — an
    override setting only ``context_window`` must not wipe the catalog's
    ``limit.output``.
    """
    shaped = _override_to_catalog_shape(override)
    merged = dict(raw)
    limit_patch = shaped.pop("limit", None)
    if limit_patch:
        base_limit = raw.get("limit")
        base_limit = dict(base_limit) if isinstance(base_limit, dict) else {}
        base_limit.update(limit_patch)
        merged["limit"] = base_limit
    vision_override = shaped.pop("_vision_override", None)
    if vision_override is not None:
        base_mods = raw.get("modalities")
        base_mods = dict(base_mods) if isinstance(base_mods, dict) else {}
        input_mods = base_mods.get("input")
        input_mods = list(input_mods) if isinstance(input_mods, list) else []
        if vision_override and "image" not in input_mods:
            input_mods.append("image")
        elif not vision_override and "image" in input_mods:
            input_mods.remove("image")
        base_mods["input"] = input_mods
        merged["modalities"] = base_mods
    merged.update(shaped)
    return merged


def _get_provider_models(provider: str) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict from models.dev.

    Returns the models dict or None if the provider is unknown or has no data.
    """
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None

    data = fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    return models


def _find_model_entry(models: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Find a model entry by exact match, then case-insensitive fallback."""
    # Exact match
    entry = models.get(model)
    if isinstance(entry, dict):
        return entry

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata

    return None


def get_model_capabilities(provider: str, model: str) -> Optional[ModelCapabilities]:
    """Look up full capability metadata from models.dev cache.

    Uses the existing fetch_models_dev() and PROVIDER_TO_MODELS_DEV mapping.
    Returns None if model not found.

    EXPLICIT ``model_overrides`` entries (per-provider+model) win over catalog
    values for the fields they set. ``_default`` entries fill the gap only for
    models the catalog does not know — the supported self-unblock path for
    custom/local models (#8731) and for models with wrong metadata in
    models.dev (#84482). An override may set any subset of fields; unspecified
    fields fall through to the catalog value (or sensible defaults when the
    model is absent from the catalog).

    Extracts from model entry fields:
      - reasoning  (bool)  → supports_reasoning
      - tool_call  (bool)  → supports_tools
      - attachment (bool)  → supports_vision
      - limit.context (int) → context_window
      - limit.output  (int) → max_output_tokens
      - family     (str)   → model_family
    """
    models = _get_provider_models(provider)
    entry = _find_model_entry(models, model) if models is not None else None

    # Select the override AFTER the catalog lookup: explicit overrides always
    # apply; _default entries only fill gaps for catalog misses.
    override = _override_for(provider, model, catalog_hit=entry is not None)

    # If no catalog entry and no override, we can't resolve capabilities.
    if entry is None and override is None:
        return None

    if entry is not None:
        # Extract capability flags (default to False if missing)
        supports_tools = bool(entry.get("tool_call", False))
        # Vision: prefer explicit `modalities.input` when models.dev provides it.
        # The older `attachment` flag can be stale or too broad for image routing;
        # fall back to it only when the input modalities are absent/invalid.
        input_mods = entry.get("modalities", {})
        if isinstance(input_mods, dict):
            input_mods = input_mods.get("input")
        else:
            input_mods = None
        if isinstance(input_mods, list):
            supports_vision = "image" in input_mods
        else:
            supports_vision = bool(entry.get("attachment", False))
        supports_reasoning = bool(entry.get("reasoning", False))

        # Extract limits
        limit = entry.get("limit", {})
        if not isinstance(limit, dict):
            limit = {}

        ctx = limit.get("context")
        context_window = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 200000

        out = limit.get("output")
        max_output_tokens = int(out) if isinstance(out, (int, float)) and out > 0 else 8192

        model_family = entry.get("family", "") or ""
    else:
        # Unknown model — derive sensible defaults. The override patches
        # whichever fields it specifies; the rest stay at defaults safe for
        # agentic use (tools on, vision/reasoning off).
        supports_tools = True
        supports_vision = False
        supports_reasoning = False
        context_window = 200000
        max_output_tokens = 8192
        model_family = ""

    # Apply override patches (each field is optional in the override dict).
    if override is not None:
        if "supports_tools" in override:
            supports_tools = bool(override["supports_tools"])
        if "supports_vision" in override:
            supports_vision = bool(override["supports_vision"])
        if "supports_reasoning" in override:
            supports_reasoning = bool(override["supports_reasoning"])
        ctx_ov = _override_int(override, "context_window")
        if ctx_ov is not None:
            context_window = ctx_ov
        out_ov = _override_int(override, "max_output_tokens")
        if out_ov is not None:
            max_output_tokens = out_ov
        if "model_family" in override:
            model_family = str(override["model_family"] or "")

    return ModelCapabilities(
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        model_family=model_family,
    )


def list_provider_models(provider: str) -> List[str]:
    """Return all model IDs for a provider from models.dev.

    Returns an empty list if the provider is unknown or has no data.
    """
    from hermes_cli.models import normalize_provider
    provider = normalize_provider(provider) or provider
    
    models = _get_provider_models(provider)
    if models is None:
        return []
    return [
        mid for mid in models.keys()
        if not _should_hide_from_provider_catalog(provider, mid)
    ]


# Patterns that indicate non-agentic or noise models (TTS, embedding,
# dated preview snapshots, live/streaming-only, image-only).
import re
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|"
    r"-image\b|-image-preview\b|-customtools\b",
    re.IGNORECASE,
)

# Google's live Gemini catalogs currently include a mix of stale slugs and
# Gemma models whose TPM quotas are too small for normal Hermes agent traffic.
# Keep capability metadata available for direct/manual use, but hide these from
# the Gemini model catalogs we surface in setup and model selection.
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under
    # agent-style traffic despite advertising large context windows.
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "gemma-4-26b-a4b-it",
    "gemma-3-1b",
    "gemma-3-1b-it",
    "gemma-3-2b",
    "gemma-3-2b-it",
    "gemma-3-4b",
    "gemma-3-4b-it",
    "gemma-3-12b",
    "gemma-3-12b-it",
    "gemma-3-27b",
    "gemma-3-27b-it",
    # Stale/retired Google slugs that still surface through models.dev-backed
    # Gemini selection but 404 on the current Google endpoints.
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    provider_lower = (provider or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    if provider_lower in {"gemini", "google"} and model_lower in _GOOGLE_HIDDEN_MODELS:
        return True
    return False


def list_agentic_models(provider: str) -> List[str]:
    """Return model IDs suitable for agentic use from models.dev.

    Filters for tool_call=True and excludes noise (TTS, embedding,
    dated preview snapshots, live/streaming, image-only models).
    Returns an empty list on any failure.
    """
    models = _get_provider_models(provider)
    if models is None:
        return []

    result = []
    for mid, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if _should_hide_from_provider_catalog(provider, mid):
            continue
        if not entry.get("tool_call", False):
            continue
        if _NOISE_PATTERNS.search(mid):
            continue
        result.append(mid)
    return result



# ---------------------------------------------------------------------------
# Rich dataclass constructors — parse raw models.dev JSON into dataclasses
# ---------------------------------------------------------------------------

def _parse_model_info(model_id: str, raw: Dict[str, Any], provider_id: str) -> ModelInfo:
    """Convert a raw models.dev model entry dict into a ModelInfo dataclass."""
    limit = raw.get("limit") or {}
    if not isinstance(limit, dict):
        limit = {}

    cost = raw.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}

    modalities = raw.get("modalities") or {}
    if not isinstance(modalities, dict):
        modalities = {}

    input_mods = modalities.get("input") or []
    output_mods = modalities.get("output") or []

    ctx = limit.get("context")
    ctx_int = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 0
    out = limit.get("output")
    out_int = int(out) if isinstance(out, (int, float)) and out > 0 else 0
    inp = limit.get("input")
    inp_int = int(inp) if isinstance(inp, (int, float)) and inp > 0 else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        reasoning=bool(raw.get("reasoning", False)),
        tool_call=bool(raw.get("tool_call", False)),
        attachment=bool(raw.get("attachment", False)),
        temperature=bool(raw.get("temperature", False)),
        structured_output=bool(raw.get("structured_output", False)),
        open_weights=bool(raw.get("open_weights", False)),
        input_modalities=tuple(input_mods) if isinstance(input_mods, list) else (),
        output_modalities=tuple(output_mods) if isinstance(output_mods, list) else (),
        context_window=ctx_int,
        max_output=out_int,
        max_input=inp_int,
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=float(cost["cache_read"]) if "cache_read" in cost and cost["cache_read"] is not None else None,
        cost_cache_write=float(cost["cache_write"]) if "cache_write" in cost and cost["cache_write"] is not None else None,
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def _parse_provider_info(provider_id: str, raw: Dict[str, Any]) -> ProviderInfo:
    """Convert a raw models.dev provider entry dict into a ProviderInfo."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


# ---------------------------------------------------------------------------
# Provider-level queries
# ---------------------------------------------------------------------------

def get_provider_info(provider_id: str) -> Optional[ProviderInfo]:
    """Get full provider metadata from models.dev.

    Accepts either a Hermes provider ID (e.g. "kilocode") or a models.dev
    ID (e.g. "kilo").  Returns None if the provider is not in the catalog.
    """
    # Resolve Hermes ID → models.dev ID
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)

    data = fetch_models_dev()
    raw = data.get(mdev_id)
    if not isinstance(raw, dict):
        return None

    return _parse_provider_info(mdev_id, raw)


# ---------------------------------------------------------------------------
# Probe-verified overrides: authoritative (Phase A8, 2026-06-04)
# ---------------------------------------------------------------------------
#
# models.dev is a community catalog that consistently UNDER-reports limits
# for the github-copilot provider section (e.g. opus-4.8 listed as 200k/64k
# but live probe + adapter wire path is 999,968/128,000). Trusting it in
# get_model_info makes `hermes /models` display lie even when the wire path
# uses correct numbers.
#
# This override table is the single source of truth for per-(provider, model)
# context_window / max_output / supported reasoning_effort whenever the data
# is provably wrong upstream. Source of truth: empirically probed per-model
# context/output limits (documented in the PR description).
#
# Matching rules:
#   1. provider_id is normalized lowercase + first segment (e.g. "github-copilot",
#      "google", "anthropic").
#   2. model_id is normalized: lower-case, strip any "anthropic/" prefix,
#      apply dot↔dash family equivalence so claude-opus-4.7 == claude-opus-4-7.
#   3. exact match wins; substring family match (claude-opus-4.7 → claude-opus-4)
#      provides a graceful fallback for un-versioned aliases like "claude-opus-4".
#
# Adding a model here OVERRIDES models.dev for that provider+model (context
# window, max output, and (optionally) reasoning_effort. Other ModelInfo fields
# (modalities, cost, etc.) still come from models.dev when available.
#
# Update when probes change. Do NOT edit ~/.hermes/models_dev_cache.json; it
# gets clobbered every TTL refresh from the upstream community catalog.

# Per-(provider, model_canonical) override entries.
# Keys are TUPLES so dict lookup is O(1) regardless of provider.
# Values are dicts merged onto the parsed ModelInfo via dataclasses.replace().
_PROBE_VERIFIED_OVERRIDES: Dict[Tuple[str, str], Dict[str, Any]] = {
    # ─── provider=github-copilot, claude family ─────────────────────────────
    # /v1/messages, beta triplet + X-Copilot-Agent-Slug: copilot-1m-context.
    # Probe V18.1 reached 999,968 input tokens (1M − 32 system overhead) on
    # opus-4.8. We use the round 1,000,000 here to match how the user thinks
    # about the cap; the conversation_loop's response-error path will adopt
    # the precise server cap (~999,968) on first turn if it matters.
    # Keyed on dot form; the resolver also tries the dash variant on lookup.
    ("github-copilot", "claude-opus-4.8"):     {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-opus-4.7"):     {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-opus-4.6"):     {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-opus-4.5"):     {"context_window":   200_000, "max_output":  64_000},
    ("github-copilot", "claude-sonnet-4.6"):   {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-sonnet-4.5"):   {"context_window":   200_000, "max_output":  64_000},
    ("github-copilot", "claude-sonnet-4.7"):   {"context_window": 1_000_000, "max_output":  64_000},
    ("github-copilot", "claude-sonnet-4.8"):   {"context_window": 1_000_000, "max_output":  64_000},
    ("github-copilot", "claude-haiku-4.5"):    {"context_window":   200_000, "max_output": 200_000},
    # mythos* aliases all map to the underlying opus-4.7 deployment (Phase D).
    ("github-copilot", "claude-mythos"):       {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-mythos-1"):     {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-mythos-preview"):       {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-mythos-1-preview"):     {"context_window": 1_000_000, "max_output": 128_000},
    # claude-sonnet-5 (new Copilot default, 2026-07) + claude-opus-4.8-fast (the
    # "(fast mode)" preview variant). Live /models catalog with
    # X-GitHub-Api-Version: 2026-07-01 (probe-verified 2026-07-13, account
    # e126380_magh): ctx window 1,000,000, max_prompt 936,000, max_output 64,000,
    # efforts [low,medium,high,xhigh,max]. Picker shows "1M" (=window) + "Max".
    # context_window kept at the round 1,000,000 to match the opus-4.8 convention
    # above; the wire cap self-corrects to the exact ~936k prompt budget on the
    # first response-error turn if it ever matters.
    ("github-copilot", "claude-sonnet-5"):      {"context_window": 1_000_000, "max_output": 128_000},
    ("github-copilot", "claude-opus-4.8-fast"): {"context_window": 1_000_000, "max_output": 128_000},
    # claude-opus-5 (Copilot CLI 1.0.79 catalog, 2026-08). Live /models reports
    # ctx window 1,000,000, max_prompt 936,000, max_output 64,000, efforts
    # [low,medium,high,xhigh,max] -- but the 64,000 is the SAME Claude-family
    # under-report opus-4.8/sonnet-5 carry. Probe-verified live 2026-08-04
    # (account e126380_magh, /v1/messages, beta triplet + copilot-1m-context):
    # max_tokens=128,000 -> 200 OK; max_tokens=200,000 -> 400
    # "128000, which is the maximum allowed number of output tokens". Real
    # output cap is 128,000, matching opus-4.8. context_window kept at the round
    # 1,000,000 convention (self-corrects to ~936k prompt budget on first turn).
    ("github-copilot", "claude-opus-5"):        {"context_window": 1_000_000, "max_output": 128_000},

    # ─── provider=github-copilot, gpt-5 family ──────────────────────────────
    # /responses, input:string schema. gpt-5.5 marketed at 1.05M total window.
    # The 900k probe number was a soft-throttle artifact; the live ./src/
    # _ANTHROPIC_OUTPUT_LIMITS / model_metadata table uses 1,050,000 which
    # matches OpenAI's documented total window (input+output combined).
    # gpt-5.5 / gpt-5.4 explicitly tested 512k output → SUCCESS.
    # NOTE: 2026-06-04 the Codex `/models` endpoint reports 272k for gpt-5.5
    # that's a conservative routing default, NOT the real cap. The probe
    # V18.1 evidence stands.
    ("github-copilot", "gpt-5.5"):             {"context_window": 1_050_000, "max_output": 512_000},
    # gpt-5.6 Sol/Terra/Luna (GA 2026-07). Live /models catalog with
    # X-GitHub-Api-Version: 2026-07-01 (probe-verified 2026-07-13): ctx window
    # 1,050,000, max_prompt 922,000, max_output 128,000, efforts
    # [none,low,medium,high,xhigh,max]. Picker shows "1.1M" (=rounded window) +
    # "Max". Same family caps as gpt-5.5 but with the extra "max" effort tier.
    ("github-copilot", "gpt-5.6-sol"):         {"context_window": 1_050_000, "max_output": 512_000},
    ("github-copilot", "gpt-5.6-terra"):       {"context_window": 1_050_000, "max_output": 512_000},
    ("github-copilot", "gpt-5.6-luna"):        {"context_window": 1_050_000, "max_output": 512_000},
    # gpt-5.4: Copilot CLI 1.0.79 unified it up to the gpt-5.5/5.6 tier. Live
    # /models (2026-08-04) reports window 1,050,000 / max_prompt 922,000, and
    # the wire CONFIRMS it: a >922k prompt on /responses returns
    # 400 "prompt token count ... exceeds the limit of 922000" (probe-verified,
    # account e126380_magh). The old 750,000 here was from a pre-1.0.79 catalog
    # and under-reported. Picker shows "1.1M". max_output 512k retained
    # (probe-verified for the gpt-5.x family; catalog's 128k is the under-report).
    ("github-copilot", "gpt-5.4"):             {"context_window": 1_050_000, "max_output": 512_000},
    ("github-copilot", "gpt-5.4-mini"):        {"context_window":   400_000, "max_output": 400_000},
    # gpt-5.3-codex was renamed → gpt-5.3-codex-spark in the 2026-06-04 Codex
    # catalog refresh. Keep both keys so old configs still resolve.
    ("github-copilot", "gpt-5.3-codex"):       {"context_window":   272_000, "max_output": 128_000},
    ("github-copilot", "gpt-5.3-codex-spark"): {"context_window":   128_000, "max_output": 128_000},
    ("github-copilot", "gpt-5.2"):             {"context_window":   272_000, "max_output": 128_000},
    ("github-copilot", "gpt-5.2-codex"):       {"context_window":   272_000, "max_output": 128_000},
    ("github-copilot", "gpt-5-mini"):          {"context_window":   128_000, "max_output": 128_000},
    ("github-copilot", "gpt-4.1"):             {"context_window":    64_000, "max_output":  64_000},

    # ─── provider=github-copilot, xAI Grok ─────────────────────────────────
    # grok-4.5: TWO real numbers, kept in their correct roles.
    #   * 500k = the enforced INPUT cap = long_context billing tier max_prompt.
    #     This tier is unlocked by X-GitHub-Api-Version: 2026-07-01 (which Hermes
    #     sends on every Copilot call), so it's the cap actually granted on the
    #     wire — the same tier the Copilot CLI runs, which is why the CLI reaches
    #     ~500k input (confirmed live 2026-08-04, account e126380_magh). The 372k
    #     from earlier probes was the DEFAULT tier (no 2026-07-01 header) — a
    #     floor, not the ceiling. This 500k is the ACCOUNTING budget that drives
    #     compression; it lives in model_metadata.get_model_context_length.
    #   * 628k = the DISPLAY window the CLI /context meter shows
    #     ("grok-4.5 · 45k/628k tokens") = 500k input + 128k output.
    # This ModelInfo.context_window is a DISPLAY-side value (feeds /models
    # listings and the banner fallback), so it carries 628k to mirror the CLI
    # picker. Token accounting does NOT read this — it uses the 500k input cap.
    # max_output 128k per the catalog.
    ("github-copilot", "grok-4.5"):            {"context_window":   628_000, "max_output": 128_000},

    # ─── provider=openai-codex (ChatGPT Codex backend, NOT Copilot proxy) ──
    # chatgpt.com/backend-api/codex/responses. Slug universe for this account
    # captured 2026-06-04 via /codex/models endpoint:
    #   gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.3-codex-spark, codex-auto-review
    # The numeric limits below are the probe-verified empirical caps, NOT the
    # catalog's conservative defaults. Codex's /models reports 272k for
    # gpt-5.5 but the actual /responses endpoint accepts ~1M (the same way
    # Copilot's /models lies about Claude). All 4 visible models support
    # reasoning_effort low/medium/high/xhigh.
    ("openai-codex", "gpt-5.5"):               {"context_window": 1_050_000, "max_output": 512_000},
    ("openai-codex", "gpt-5.4"):               {"context_window":   750_000, "max_output": 512_000},
    ("openai-codex", "gpt-5.4-mini"):          {"context_window":   400_000, "max_output": 400_000},
    ("openai-codex", "gpt-5.3-codex-spark"):   {"context_window":   128_000, "max_output": 128_000},
    # codex-auto-review is hidden in catalog (visibility=hide) but reachable;
    # only Codex spawns it internally for auto-review.
    ("openai-codex", "codex-auto-review"):     {"context_window":   272_000, "max_output": 128_000},
    # Hermes legacy alias: points to the renamed -spark on this account.
    ("openai-codex", "gpt-5.3-codex"):         {"context_window":   128_000, "max_output": 128_000},

    # ─── provider=github-copilot, gemini family ────────────────────────────
    # gemini-3.1-pro-preview is integrator-blocked on Copilot's `copilot-4-cli`.
    # When users request gemini via provider=copilot, fall through to
    # `gemini-2.5-pro` proxy clamp limits (128k/65k). For the unlocked path
    # see provider=google entries below.
    ("github-copilot", "gemini-2.5-pro"):      {"context_window":   128_000, "max_output":  65_536},
    ("github-copilot", "gemini-3-flash-preview"): {"context_window":   128_000, "max_output":  65_536},
    # gemini-3.1-pro-preview unreachable through Copilot proxy. Set to 0 so
    # the UI shows "n/a"; users should pick provider=google instead, which
    # unlocks the model via cloudcode-pa OAuth (Phase A9, 2026-06-04).
    ("github-copilot", "gemini-3.1-pro-preview"): {"context_window":         0, "max_output":       0},
    # gemini-3.5-flash on Copilot: live /models 1.0.79 reports window 1,000,000
    # / max_prompt 936,000 / long_context 936,000 (endpoint /chat/completions).
    # The CLI picker shows "1M" (= the same catalog the image renders). The old
    # 200,000 here was a stale proxy-clamp value from when gemini was
    # integrator-gated; 1.0.79 exposes the full window on this account.
    ("github-copilot", "gemini-3.5-flash"):    {"context_window": 1_000_000, "max_output":  65_536},
    # gemini-3.6-flash (new in 1.0.79): same 1M window / 936k prompt tier.
    ("github-copilot", "gemini-3.6-flash"):    {"context_window": 1_000_000, "max_output":  65_536},

    # ─── provider=google (cloudcode-pa OAuth, the "alternative token") ──
    # Phase A9 unlock 2026-06-04: opus/sonnet/gemini-3.x reachable via the
    # cloudcode-pa.googleapis.com Code Assist endpoint when we DON'T inject
    # the Antigravity-internal X-Goog-User-Project header. Vendor-doc context
    # caps used for context_window; output ceiling is the documented 65,536
    # for all gemini-3.x families (per haimaker.ai/blog/best-gemini-models-for-openclaw
    # and Google Code Assist docs).
    ("google", "gemini-2.5-pro"):              {"context_window": 1_048_576, "max_output":  65_536},
    ("google", "gemini-2.5-flash"):            {"context_window": 1_048_576, "max_output":  65_536},
    ("google", "gemini-3-pro-preview"):        {"context_window": 1_000_000, "max_output":  65_536},
    ("google", "gemini-3.1-pro-preview"):      {"context_window": 1_000_000, "max_output":  65_536},
    ("google", "gemini-3-flash-preview"):      {"context_window": 1_000_000, "max_output":  65_536},
    ("google", "gemini-3.1-flash-lite-preview"): {"context_window": 1_000_000, "max_output":  65_536},

    # ─── provider=anthropic (vendor-direct) ─────────────────────────────────
    # Pro/Max subscription via api.anthropic.com.
    ("anthropic", "claude-opus-4.8"):          {"context_window": 1_000_000, "max_output": 128_000},
    ("anthropic", "claude-opus-4.7"):          {"context_window": 1_000_000, "max_output": 128_000},
    ("anthropic", "claude-opus-4.6"):          {"context_window": 1_000_000, "max_output": 128_000},
    ("anthropic", "claude-opus-4.5"):          {"context_window":   200_000, "max_output": 128_000},
    ("anthropic", "claude-sonnet-4.6"):        {"context_window": 1_000_000, "max_output":  64_000},
    ("anthropic", "claude-sonnet-4.5"):        {"context_window":   200_000, "max_output":  64_000},
    ("anthropic", "claude-haiku-4.5"):         {"context_window":   200_000, "max_output":  64_000},
}


# Hermes provider id ↔ override-key normalization. We pin to the
# models.dev-style id (left side of PROVIDER_TO_MODELS_DEV) so override
# matching tracks the same provider taxonomy as the rest of this module.
_OVERRIDE_PROVIDER_ALIASES = {
    "copilot": "github-copilot",
    "github-copilot": "github-copilot",
    "github-models": "github-copilot",
    "github-model": "github-copilot",
    "github": "github-copilot",
    "google": "google",
    "gemini": "google",
    "google-code-assist": "google",
    "google-gemini-cli": "google",
    "google-vertex": "google",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def _canonicalize_model_id(model_id: str) -> str:
    """Normalize a model id for override-table lookup.

    - lowercased
    - strip ``vendor/`` prefix (e.g. ``anthropic/claude-opus-4.7`` → ``claude-opus-4.7``)
    - strip date stamps (``-20250929``)

    Dots are PRESERVED. Hermes config / catalog ids are dot-form
    (``gpt-5.5``, ``gemini-2.5-pro``, ``claude-opus-4.7``). The override
    table is keyed on dot form to match. The lookup function additionally
    tries the dash variant (``claude-opus-4-7``) so the Anthropic SDK shape
    works too.
    """
    import re as _re
    m = (model_id or "").strip().lower()
    if "/" in m:
        m = m.split("/", 1)[1]
    # date stamps at the end (-YYYYMMDD)
    m = _re.sub(r"-(\d{8})$", "", m)
    return m


def _resolve_probe_override(provider_id: str, model_id: str) -> Optional[Dict[str, Any]]:
    """Return override dict for this (provider, model), or None.

    Tries:
      1. Exact canonical match (dot form: ``claude-opus-4.7``, ``gpt-5.5``).
      2. Dash↔dot variant on the version suffix (``claude-opus-4-7`` ↔ ``claude-opus-4.7``).
      3. Progressive family-prefix shrink so ``claude-opus-4-7-20251101`` →
         ``claude-opus-4-7`` → ``claude-opus-4`` if needed.
    """
    import re as _re
    norm_provider = _OVERRIDE_PROVIDER_ALIASES.get(
        (provider_id or "").strip().lower(),
        (provider_id or "").strip().lower(),
    )
    canonical = _canonicalize_model_id(model_id)
    if not norm_provider or not canonical:
        return None

    # Build the set of canonical variants to try.
    variants = [canonical]
    # If canonical has a dash-version suffix like ``-4-7`` or ``-4-8``, also
    # try the dot form (``-4.7`` / ``-4.8``).
    dot_variant = _re.sub(r"-(\d+)-(\d+)(?=-|$)", r"-\1.\2", canonical)
    if dot_variant != canonical:
        variants.append(dot_variant)
    # Trailing single ``-N`` (e.g. ``claude-haiku-4-5`` → ``claude-haiku-4.5``):
    # the regex above already handles ``-4-5`` since it matches non-final too,
    # but be explicit for the final-token case.
    dot_variant_tail = _re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", canonical)
    if dot_variant_tail not in variants:
        variants.append(dot_variant_tail)
    # Pure dash → dot for the last hyphen-digit run only (covers ``gpt-5-5`` →
    # ``gpt-5.5``). Keep it conservative; don't touch non-numeric segments.
    dash_to_dot_last = _re.sub(r"-(\d+)$", r".\1", canonical)
    if dash_to_dot_last not in variants:
        variants.append(dash_to_dot_last)
    # Dot variant: covers Anthropic SDK shape (claude-opus-4-7 ↔ claude-opus-4.7)
    # when canonical IS the dot form.
    if "." in canonical:
        dash_form = canonical.replace(".", "-")
        if dash_form not in variants:
            variants.append(dash_form)

    # 1+2: exact + dash↔dot variants.
    for v in variants:
        hit = _PROBE_VERIFIED_OVERRIDES.get((norm_provider, v))
        if hit is not None:
            return hit

    # 3: progressive family-prefix shrink across all variants.
    for v in variants:
        parts = v.split("-")
        while len(parts) > 1:
            parts.pop()
            candidate = "-".join(parts)
            hit = _PROBE_VERIFIED_OVERRIDES.get((norm_provider, candidate))
            if hit is not None:
                return hit
    return None


# ---------------------------------------------------------------------------
# Model-level queries (rich ModelInfo)
# ---------------------------------------------------------------------------

def get_model_info(
    provider_id: str, model_id: str
) -> Optional[ModelInfo]:
    """Get full model metadata from models.dev, with probe-verified overrides
    applied for providers where models.dev is known to lie (notably
    github-copilot, see _PROBE_VERIFIED_OVERRIDES above).

    Accepts Hermes or models.dev provider ID.  Tries exact match then
    case-insensitive fallback.  Returns None if not found in models.dev AND
    not in the override table.

    The override layers (lowest → highest authority):
      1. models.dev catalog entry (base).
      2. probe-verified override (``_PROBE_VERIFIED_OVERRIDES``): replaces
         numeric limits where models.dev is known to lie.
      3. USER ``model_overrides`` config: the user's manual declaration, the
         highest authority — wins over both catalog and probe overrides.
         Explicit per-provider+model entries patch known models; ``_default``
         entries fill the gap only for models the catalog does not know
         (#8731/#84482 self-unblock). Canonical schema, sub-dicts merged.
    """
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)
    probe_override = _resolve_probe_override(provider_id, model_id)

    data = fetch_models_dev()
    pdata = data.get(mdev_id)

    base: Optional[ModelInfo] = None
    matched_id = model_id
    if isinstance(pdata, dict):
        models = pdata.get("models", {})
        if isinstance(models, dict):
            # Exact match
            raw = models.get(model_id)
            if isinstance(raw, dict):
                base = _parse_model_info(model_id, raw, mdev_id)
            else:
                # Case-insensitive fallback
                model_lower = model_id.lower()
                for mid, mdata in models.items():
                    if mid.lower() == model_lower and isinstance(mdata, dict):
                        base = _parse_model_info(mid, mdata, mdev_id)
                        matched_id = mid
                        break

    # Layer 2: probe-verified override on top of the catalog base.
    if probe_override:
        if base is not None:
            # Apply override on top: replace numeric limits, keep everything else.
            from dataclasses import replace as _replace
            base = _replace(base, **{k: v for k, v in probe_override.items() if hasattr(base, k)})
        else:
            # No models.dev entry: synthesize a minimal honest one.
            base = ModelInfo(
                id=model_id,
                name=model_id,
                family=_canonicalize_model_id(model_id).rsplit("-", 1)[0] or model_id,
                provider_id=mdev_id,
                context_window=int(probe_override.get("context_window", 0) or 0),
                max_output=int(probe_override.get("max_output", 0) or 0),
            )

    # Layer 3: USER model_overrides config — highest authority. Explicit
    # entries win over catalog + probe for known models; _default fills the gap
    # only when the model is entirely unknown (no catalog and no probe).
    user_override = _override_for(
        provider_id, model_id, catalog_hit=base is not None
    )
    if user_override is not None:
        from dataclasses import replace as _replace
        patch = _model_info_patch_from_override(user_override)
        if base is not None:
            if patch:
                base = _replace(base, **patch)
        else:
            # Override-only synthesis: canonical fields → ModelInfo.
            base = ModelInfo(
                id=model_id,
                name=model_id,
                family=str(user_override.get("model_family") or "")
                or (_canonicalize_model_id(model_id).rsplit("-", 1)[0] or model_id),
                provider_id=mdev_id,
                context_window=patch.get("context_window", 0),
                max_output=patch.get("max_output", 0),
                reasoning=patch.get("reasoning", False),
                tool_call=patch.get("tool_call", True),
                attachment=patch.get("attachment", False),
            )

    return base


def _model_info_patch_from_override(override: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a canonical model_overrides dict into ModelInfo field kwargs.

    Only the fields the override actually sets are returned, so a partial
    override patches just those fields on the base ModelInfo (dataclass.replace).
    """
    patch: Dict[str, Any] = {}
    ctx = _override_int(override, "context_window")
    if ctx is not None:
        patch["context_window"] = ctx
    out = _override_int(override, "max_output_tokens")
    if out is not None:
        patch["max_output"] = out
    if "supports_tools" in override:
        patch["tool_call"] = bool(override["supports_tools"])
    if "supports_reasoning" in override:
        patch["reasoning"] = bool(override["supports_reasoning"])
    if "supports_vision" in override:
        patch["attachment"] = bool(override["supports_vision"])
    if "model_family" in override:
        patch["family"] = str(override["model_family"] or "")
    return patch
