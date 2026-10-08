"""Worker registry for the Unified Router.

Single source of truth for the worker pool: canonical model IDs, provider
routing info, and pricing. Every worker is text-only -- a model that also
happens to accept pictures is still used through a text-only interface, and
the router has no way to express anything else. Costs are USD per 1M tokens
and are provider-specific (do not mix prices across providers). A cost
field set to None means the price is unknown or not offered by the
provider -- never encode "unknown" as 0.

Lookup here is case-insensitive exact matching by design -- no aliases and no
substring matching, unlike the legacy check_llm_name().
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WorkerSpec:
    model_id: str
    provider: str
    api_model_name: str
    input_cost_per_1m_tokens: Optional[float]
    output_cost_per_1m_tokens: Optional[float]
    # Price for input tokens the provider reports as cached in ordinary usage
    # (usage.cached_tokens / prompt_cache_hit_tokens). This is the one the cost
    # path applies, because it is the one the trajectory data can evidence.
    cache_hit_input_cost_per_1m_tokens: Optional[float]
    pricing_source: str
    pricing_snapshot_date: str
    description: str
    # Price for a hit on an EXPLICIT cache the caller created through a cache
    # API. Recorded for completeness and deliberately NOT applied to
    # usage.cached_tokens: nothing in this project creates explicit caches, and
    # billing implicit hits at the explicit rate would understate cost.
    explicit_cache_hit_cost_per_1m_tokens: Optional[float] = None


_WORKER_SPECS: Tuple[WorkerSpec, ...] = (
    # ---- hosted workers -------------------------------------
    # This registry, not ACTIVE_WORKER_POOL, is what validate_route consults:
    # route_grammar.validate_route -> get_worker(model_id) -> WORKER_REGISTRY.
    # A worker whose id is in the pool and in the rendered rules but NOT here
    # is rejected as unknown_model and the trajectory scores -1, which looks
    # like a policy failure and is a configuration one. A worker needs an entry
    # in ALL THREE places to exist: this registry (validation),
    # route_grammar.ACTIVE_WORKER_POOL (single/multi prompt) and the channel
    # config (dispatch).
    #
    # Prices are per 1M tokens, read from the OpenRouter endpoints API for
    # the PINNED provider of each worker. cost_lambda is 0, so
    # nothing here reaches the reward; they are recorded so a cost term is a
    # configuration change rather than a data-collection exercise.
    WorkerSpec(
        model_id="gemma3_27b_it",
        provider="openrouter",
        api_model_name="google/gemma-3-27b-it",
        input_cost_per_1m_tokens=0.119,
        output_cost_per_1m_tokens=0.20,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="OpenRouter /models/<id>/endpoints, provider Novita (bf16)",
        pricing_snapshot_date="2026-09-03",
        description=(
            "27B instruction-tuned; strong on text understanding, summarization and "
            "open-ended answers, weaker on mathematics and code."
        ),
    ),
    WorkerSpec(
        model_id="llama33_70b",
        provider="openrouter",
        api_model_name="meta-llama/llama-3.3-70b-instruct",
        input_cost_per_1m_tokens=0.25,
        output_cost_per_1m_tokens=0.75,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="OpenRouter /models/<id>/endpoints, provider Crusoe (bf16)",
        pricing_snapshot_date="2026-09-03",
        description=(
            "70B dense open-weight; broad knowledge and steady multi-step reasoning "
            "across knowledge, mathematics and code."
        ),
    ),
    WorkerSpec(
        model_id="gemini25_flash_lite",
        provider="openrouter",
        api_model_name="google/gemini-2.5-flash-lite",
        input_cost_per_1m_tokens=0.10,
        output_cost_per_1m_tokens=0.40,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="OpenRouter /models/<id>/endpoints, provider Google AI Studio",
        pricing_snapshot_date="2026-09-03",
        description=(
            "Consistent on everyday knowledge, instruction following and structured "
            "output, shallow on long derivations."
        ),
    ),
    # ---------------------------------------------------------- fast-local --
    #
    # Three of the six are served locally (vLLM) rather than through a paid
    # API. Swapping who answers a <route> changes nothing else: grammar,
    # paradigms, roles, refs and the reward are all independent of it.
    #
    # PRICES ARE None ON PURPOSE, and that is a stronger statement than 0.
    # A locally served model has no vendor price, and inventing one would put
    # a fabricated number into the only signal cost-aware training would ever
    # read. None means "unknown", the registry docstring's own rule, and
    # compute_trajectory_cost() still refuses to run on it -- which is why the
    # cost path is now skipped outright while cost_lambda is 0 (route_reward
    # .compute_trajectory_cost(with_prices=...)). Turning cost_lambda up with
    # this pool active will fail loudly at the first successful dispatch,
    # which is the correct outcome: the prices have to be decided first.
    #
    # qwen3_8b was an OpenRouter spec until now. The canonical id is the same
    # model, so it is RE-POINTED rather than duplicated -- _build_registry
    # rejects a duplicate model_id, and a second id for one model would break
    # the "no aliases" rule the whole lookup rests on.
    WorkerSpec(
        model_id="qwen3_8b",
        provider="local_vllm",
        api_model_name="qwen3-8b",
        input_cost_per_1m_tokens=None,
        output_cost_per_1m_tokens=None,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="local vLLM; no vendor price applies (see the note above)",
        pricing_snapshot_date="2026-08-22",
        description=(
            "8B dense; reliable on direct facts, short reasoning and instruction "
            "following, weak on long derivations and specialist domains."
        ),
    ),
    WorkerSpec(
        model_id="llama31_8b",
        provider="local_vllm",
        api_model_name="llama3.1-8b",
        input_cost_per_1m_tokens=None,
        output_cost_per_1m_tokens=None,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="local vLLM; no vendor price applies (see the note above)",
        pricing_snapshot_date="2026-08-22",
        description=(
            "8B dense from a different model family; solid on everyday knowledge, "
            "text understanding and open-ended answers, weak on long derivations "
            "and precise technical work."
        ),
    ),
    WorkerSpec(
        model_id="qwen3_30b_a3b",
        # METADATA ONLY. The channel config serves
        # this worker from OpenRouter, not from local vLLM --
        # the spec still said local_vllm with no price, which is the one thing
        # a registry must not do: describe a paid endpoint as free. Provider,
        # api_model_name and price are corrected to what dispatch actually
        # uses; nothing cost-aware is enabled by this (reward_manager
        # scores correctness only, and compute_trajectory_cost still reads
        # prices only when cost_lambda > 0, which the main run leaves at 0).
        provider="openrouter",
        api_model_name="qwen/qwen3-30b-a3b-instruct-2507",
        input_cost_per_1m_tokens=0.10,
        output_cost_per_1m_tokens=0.30,
        cache_hit_input_cost_per_1m_tokens=None,
        pricing_source="OpenRouter /models/<id>/endpoints, provider Nebius (fp8)",
        pricing_snapshot_date="2026-09-05",
        description=(
            "Mixture-of-experts; holds up on multi-step reasoning, mathematics, "
            "code and long derivations."
        ),
    ),
)


def _build_registry(specs: Tuple[WorkerSpec, ...]) -> Dict[str, WorkerSpec]:
    registry: Dict[str, WorkerSpec] = {}
    for spec in specs:
        if spec.model_id != spec.model_id.strip().lower():
            raise ValueError(f"Canonical model_id must be lowercase: {spec.model_id!r}")
        if spec.model_id in registry:
            raise ValueError(f"Duplicate model_id in worker registry: {spec.model_id!r}")
        registry[spec.model_id] = spec
    return registry


# ANONYMOUS DIRECTORY. ROUTEWEAVER_WORKER_ALIAS=1 adds one spec per
# alias so validate_route()/get_worker() resolve worker_1 .. worker_6 without a
# translation layer. Each alias spec COPIES its backend's spec -- provider,
# api_model_name and every price field are taken from the real entry with
# dataclasses.replace, so they cannot drift -- and overrides only model_id and
# description (the operator-supplied capability hint the prompt renders).
# The real specs are kept, so a run with the flag off resolves the same way.
# Default off: without the flag this tuple and the registry are unchanged.
def _alias_specs(specs: Tuple[WorkerSpec, ...]) -> Tuple[WorkerSpec, ...]:
    import dataclasses
    import worker_alias
    if not worker_alias.enabled():
        return ()
    by_id = {sp.model_id: sp for sp in specs}
    out = []
    for alias, real in worker_alias.ALIAS_TO_REAL.items():
        base = by_id.get(real)
        if base is None:
            raise ValueError(
                f"worker_alias maps {alias!r} to {real!r}, which is not in the "
                f"worker registry; the alias cannot inherit a backend")
        out.append(dataclasses.replace(
            base, model_id=alias,
            description=worker_alias.DESCRIPTIONS[alias]))
    return tuple(out)


_WORKER_SPECS = _WORKER_SPECS + _alias_specs(_WORKER_SPECS)

WORKER_REGISTRY: Dict[str, WorkerSpec] = _build_registry(_WORKER_SPECS)


# Fields of a WorkerSpec that a run's WORKER/REWARD semantics actually depend
# on. Everything else on the spec is provenance: true, worth recording, and
# incapable of changing what the router does or what it is paid.
_SEMANTIC_FIELDS = (
    'model_id',                              # canonical identity, matched by the grammar
    'api_model_name',                        # what is asked of the provider
    'input_cost_per_1m_tokens',              # reward, via compute_trajectory_cost
    'output_cost_per_1m_tokens',             # reward
    'cache_hit_input_cost_per_1m_tokens',    # reward
)
# Recorded, never hashed into the gate: pricing_source, pricing_snapshot_date,
# description (the PROMPT reads prompt_pool.WORKER_DESCRIPTIONS, not this),
# provider, explicit_cache_hit_cost_per_1m_tokens (deliberately unconsumed).
# Rewording any of them must not invalidate a dataset.
_PROVENANCE_FIELDS = ('provider', 'pricing_source', 'pricing_snapshot_date',
                      'description', 'explicit_cache_hit_cost_per_1m_tokens')


def _digest(payload) -> str:
    import hashlib
    return hashlib.sha256(repr(payload).encode('utf-8')).hexdigest()[:16]


def registry_fingerprint(specs: Tuple[WorkerSpec, ...] = None) -> str:
    """Worker and reward semantics of the ACTIVE pool.

    This freezes what a run actually trained against: which workers existed,
    what was asked of the provider, and the prices that enter the
    reward. It goes into the run manifest, the rollout dump and wandb, so two
    runs can be shown to have used the same worker/reward configuration.

    It is deliberately NOT a hash of every field of every spec. Doing that made
    a reworded pricing_source invalidate six datasets, while leaving two real
    holes -- editing a worker DESCRIPTION or dropping a worker from the active
    pool both change every baked prompt and neither moved the hash, because the
    prompt is built from prompt_pool, not from these specs. Prompt
    compatibility is prompt_pool_fingerprint's job.

    Legacy specs that are not in the active pool are excluded: they exist so
    a run with the alias off resolves, and they cannot affect one with it on.
    """
    from route_grammar import ACTIVE_WORKER_POOL
    specs = _WORKER_SPECS if specs is None else specs
    active = {s.model_id: s for s in specs}
    payload = [tuple(getattr(active[w], f) for f in _SEMANTIC_FIELDS)
               for w in ACTIVE_WORKER_POOL if w in active]
    return _digest(('worker-reward-v1', tuple(ACTIVE_WORKER_POOL), payload,
                    _thinking_freeze()))


def _thinking_freeze():
    """The frozen direct-worker setting, as the config actually serves it.

    Part of worker semantics: the same worker with thinking on is a different
    worker in every way that matters to the reward. Read lazily and tolerantly
    -- an unreadable config must not make the fingerprint uncomputable, only
    unfrozen, and that shows up as the literal below.

    TWO spellings, because the switch is not the same field on every backend
    and reading only one of them would let the fingerprint keep saying
    "frozen" about a setting it no longer sees:

      * `thinking: {type: disabled}` -- the hosted providers' own top-level
        field;
      * `chat_template_kwargs: {enable_thinking: false}` -- the vLLM
        convention, which is what a LOCALLY served hybrid model such as
        Qwen3-8B actually obeys. vLLM silently IGNORES unknown top-level
        fields, so sending the providers' spelling to a local server would
        leave thinking ON with no error anywhere.

    Both are hashed, so a pool that switches backends moves the fingerprint
    rather than quietly losing the guarantee.

    Reads the DEFAULT config path, not the one a given run was launched with
    (the launcher passes that as +router.channel_config_path, which this
    function cannot see). Pre-existing behaviour, left alone deliberately:
    a run whose channel file is elsewhere gets empty entries here, and its
    real topology is recorded by channel_config_fingerprint instead.
    """
    import os
    import yaml
    path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'route_channels.yaml'))
    try:
        cfg = yaml.safe_load(open(path))
    except Exception:
        return 'channel-config-unavailable'
    from route_grammar import ACTIVE_WORKER_POOL
    out = []
    for worker_id in ACTIVE_WORKER_POOL:
        entries = (cfg.get('workers') or {}).get(worker_id) or []
        thinking = {repr(((e.get('extra_body') or {}).get('thinking'),
                          ((e.get('extra_body') or {}).get('chat_template_kwargs')
                           or {}).get('enable_thinking')))
                    for e in entries}
        out.append((worker_id, tuple(sorted(thinking))))
    return tuple(out)


def registry_provenance() -> dict:
    """The descriptive fields, recorded alongside a run and hashed into
    nothing. Changing any of these is a documentation edit."""
    return {s.model_id: {f: getattr(s, f) for f in _PROVENANCE_FIELDS}
            for s in _WORKER_SPECS}


def channel_config_fingerprint() -> str:
    """Resolved channel chain, per-entry model, thinking, timeout and order.

    Recorded with a run so a provider-topology change is visible after the
    fact. NOT a dataset gate: the chain can change between two runs of the same
    parquet without making that parquet wrong.
    """
    import os
    import yaml
    path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'route_channels.yaml'))
    try:
        cfg = yaml.safe_load(open(path))
    except Exception:
        return 'unavailable'
    workers = {}
    for worker_id, entries in sorted((cfg.get('workers') or {}).items()):
        workers[worker_id] = [
            (e.get('channel'), e.get('model'), e.get('max_tokens'),
             e.get('timeout'), repr((e.get('extra_body') or {}).get('thinking')))
            for e in entries]
    return _digest(('channel-v1', workers))


def get_worker(model_id: str) -> Optional[WorkerSpec]:
    """Case-insensitive exact-match lookup after strip(). No aliases, no substring matching."""
    return WORKER_REGISTRY.get(model_id.strip().lower())


def is_valid_worker(model_id: str) -> bool:
    return get_worker(model_id) is not None


def list_worker_ids() -> List[str]:
    return list(WORKER_REGISTRY.keys())
