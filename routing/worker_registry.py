"""Worker registry for the Unified Router."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WorkerSpec:
    model_id: str
    provider: str
    api_model_name: str
    input_cost_per_1m_tokens: Optional[float]
    output_cost_per_1m_tokens: Optional[float]
    cache_hit_input_cost_per_1m_tokens: Optional[float]
    pricing_source: str
    pricing_snapshot_date: str
    description: str
    explicit_cache_hit_cost_per_1m_tokens: Optional[float] = None


_WORKER_SPECS: Tuple[WorkerSpec, ...] = (
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


_SEMANTIC_FIELDS = (
    'model_id',                              # canonical identity, matched by the grammar
    'api_model_name',                        # what is asked of the provider
    'input_cost_per_1m_tokens',              # reward, via compute_trajectory_cost
    'output_cost_per_1m_tokens',             # reward
    'cache_hit_input_cost_per_1m_tokens',    # reward
)
_PROVENANCE_FIELDS = ('provider', 'pricing_source', 'pricing_snapshot_date',
                      'description', 'explicit_cache_hit_cost_per_1m_tokens')


def _digest(payload) -> str:
    import hashlib
    return hashlib.sha256(repr(payload).encode('utf-8')).hexdigest()[:16]


def registry_fingerprint(specs: Tuple[WorkerSpec, ...] = None) -> str:
    """Worker and reward semantics of the ACTIVE pool."""
    from route_grammar import ACTIVE_WORKER_POOL
    specs = _WORKER_SPECS if specs is None else specs
    active = {s.model_id: s for s in specs}
    payload = [tuple(getattr(active[w], f) for f in _SEMANTIC_FIELDS)
               for w in ACTIVE_WORKER_POOL if w in active]
    return _digest(('worker-reward-v1', tuple(ACTIVE_WORKER_POOL), payload,
                    _thinking_freeze()))


def _thinking_freeze():
    """The frozen direct-worker setting, as the config actually serves it."""
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
    nothing. Changing any of these is a documentation edit.
    """
    return {s.model_id: {f: getattr(s, f) for f in _PROVENANCE_FIELDS}
            for s in _WORKER_SPECS}


def channel_config_fingerprint() -> str:
    """Resolved channel chain, per-entry model, thinking, timeout and order."""
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
    """Case-insensitive exact-match lookup after strip()."""
    return WORKER_REGISTRY.get(model_id.strip().lower())


def is_valid_worker(model_id: str) -> bool:
    return get_worker(model_id) is not None


def list_worker_ids() -> List[str]:
    return list(WORKER_REGISTRY.keys())
