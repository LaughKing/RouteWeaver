"""Route grammar for the Unified Router.

Lightweight parsing for the router's structured output tags:

    <mode>single|multi|agentic</mode>
    <route model="MODEL_ID">QUERY</route>                      (single/multi)

    <paradigm>refine|ensemble|debate|plan</paradigm>           (agentic only)
    <layer index="1">
      <route id="1-1" model="MODEL_ID" role="ROLE" refs="1-1,1-2">TASK</route>
      ...
    </layer>

parse_mode / parse_route check structure only (tags, required
attributes, non-empty query). validate_route checks semantics against the
worker registry (known model).

AGENTIC MODE. single/multi stay exactly as they were -- one route per turn,
parsed by parse_route, which still returns only the FIRST route because the
single/multi call site depends on that contract. Agentic
adds a second, additive surface: parse_paradigm / parse_layer /
parse_routes, used only by the agent loop and the reward analyzer.

The per-paradigm structure (how many layers, how many routes each, which role,
which refs) is defined ONCE in PARADIGM_SPECS and enforced ONCE in
validate_agentic_layer. The rollout loop and the reward analyzer both call
that function rather than re-deriving the rules -- the same single-definition
discipline the tag-casing helpers below follow, and for the same reason: two
copies of a protocol rule drift, and the drift shows up as an unexplainable
-1 reward on trajectories the rollout accepted.

Deliberately not a general XML parser: router output is free text with
embedded tags, not a well-formed document, and lenient parsers would
silently repair the very mistakes RL needs to observe and penalize.

GrammarError.detail is for logging and tests only; the first version
rollout integration must not feed it back to the policy as observations.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from worker_registry import WorkerSpec, get_worker

# Version of the output grammar/protocol this module implements. Baked into
# every dataset row (route_protocol_version) and checked at training load time
# so a dataset built for an older grammar can never silently train a new one.
#
# v2 drops the `image` attribute -- the router is text-only, so a route has
# nothing left to say about pictures. The version bump is what stops an older
# dataset, whose baked prompts still teach image="true|false", from silently
# training against a parser that now rejects that attribute as unknown.
ROUTE_PROTOCOL_VERSION = 'unified-router-v2'

# v3 adds the agentic surface (<paradigm>, <layer>, route id/role/refs) on top
# of v2. It is a SUPERSET: a v3-aware parser reads a v2 single/multi dataset
# unchanged, which is why both versions stay accepted at training load time
# (dataset_protocol.assert_dataset_protocol) instead of v3 invalidating every
# parquet built so far. Datasets that actually teach agentic must be stamped
# v3; a v2 dataset simply never emits the new tags.
# v4 = the v3 grammar with a changed worker pool. The grammar itself is
# unchanged; the version moves because the ACTION SPACE did, and a dataset
# built against an older pool offers a worker that no longer exists. The
# registry fingerprint would catch it too -- this makes it legible.
AGENTIC_PROTOCOL_VERSION = 'unified-router-v4'
# v3 stays readable so an older dataset still loads; new
# datasets stamp v4. The guard against training on a stale pool is the registry
# fingerprint, checked by the training entry -- not this tuple.
SUPPORTED_PROTOCOL_VERSIONS = (ROUTE_PROTOCOL_VERSION, 'unified-router-v3',
                               AGENTIC_PROTOCOL_VERSION)


class ErrorCode(Enum):
    MODE_MISSING = "mode_missing"
    MODE_DUPLICATE = "mode_duplicate"
    MODE_INVALID = "mode_invalid"
    ROUTE_MISSING = "route_missing"
    ROUTE_MALFORMED = "route_malformed"
    ATTR_MISSING = "attr_missing"
    ATTR_UNKNOWN = "attr_unknown"
    ATTR_DUPLICATE = "attr_duplicate"
    QUERY_EMPTY = "query_empty"
    PLACEHOLDER = "placeholder"
    UNKNOWN_MODEL = "unknown_model"
    # curriculum: the mode PARSED fine, it is simply outside the action space
    # this stage opens. Distinct from MODE_INVALID (not a mode at all) so a
    # stage-1 policy reaching for <mode>multi</mode> is legible as curriculum
    # pressure rather than as a grammar failure.
    MODE_NOT_ALLOWED = "mode_not_allowed"
    # balanced stage-3: the paradigm PARSED fine and is a real
    # paradigm, it is simply not the one this sample was conditioned on.
    # Mirror of MODE_NOT_ALLOWED one level down, and distinct from
    # PARADIGM_INVALID for the same legibility reason.
    PARADIGM_NOT_ALLOWED = "paradigm_not_allowed"
    # --- agentic mode (v3). Codes are stable strings: they become reward
    # metric column names (rewards/reward_manager.FIXED_VIOLATION_COLUMNS).
    PARADIGM_MISSING = "paradigm_missing"
    PARADIGM_DUPLICATE = "paradigm_duplicate"
    PARADIGM_INVALID = "paradigm_invalid"
    PARADIGM_UNEXPECTED = "paradigm_unexpected"
    LAYER_MISSING = "layer_missing"
    LAYER_MALFORMED = "layer_malformed"
    LAYER_DUPLICATE = "layer_duplicate"
    LAYER_UNEXPECTED = "layer_unexpected"
    LAYER_INDEX_INVALID = "layer_index_invalid"
    LAYER_INDEX_NONCONSECUTIVE = "layer_index_nonconsecutive"
    LAYER_ROUTE_COUNT_INVALID = "layer_route_count_invalid"
    ROUTE_ID_MISSING = "route_id_missing"
    ROUTE_ID_MALFORMED = "route_id_malformed"
    ROUTE_ID_DUPLICATE = "route_id_duplicate"
    ROLE_MISSING = "role_missing"
    ROLE_INVALID = "role_invalid"
    REFS_INVALID = "refs_invalid"
    REFS_UNEXPECTED = "refs_unexpected"
    REFS_INCOMPLETE = "refs_incomplete"


@dataclass(frozen=True)
class GrammarError:
    code: ErrorCode
    detail: str


@dataclass(frozen=True)
class RouteCall:
    model_id: str  # normalized with strip().lower(), same shape as registry canonical IDs
    query: str
    # Agentic-only fields. They default to the single/multi shape so every
    # positional construction -- RouteCall("qwen3_8b", "q") -- and every
    # existing equality assertion keeps working untouched.
    route_id: Optional[str] = None
    role: Optional[str] = None
    refs: Tuple[str, ...] = ()
    layer: Optional[int] = None

    @property
    def is_agentic(self) -> bool:
        return self.route_id is not None


# ---------------------------------------------------------------- tag casing
#
# Protocol tag NAMES are matched case-insensitively; everything else stays
# exactly as strict as it was.
#
# WHY: tag case carries no routing semantics, but it used to decide the whole
# reward. A model writing <Route ...>...</Route> parsed as ROUTE_MISSING, so
# the rollout never dispatched, the analyzer saw zero dispatched routes, and
# the trajectory scored -1. A whole group of those is a group with no reward
# spread, and with the group-std floor in place that produces NO gradient --
# which is how a cold start fails to start at all.
#
# Deliberately NOT relaxed by this: attribute names and values, worker ids,
# mode/route-count rules, observation pairing, and the strict/final_valid split.
# Everything WE emit -- prompt text, injected observations -- stays lowercase;
# this is a parser relaxation, not a change to the canonical protocol.
#
# These helpers are the single definition of the rule. route_reward and
# generation import them rather than re-deriving the patterns, so the rollout
# gate and the reward analyzer cannot drift apart on casing.

# design decision: <think> is no longer a protocol tag. The router
# is a non-thinking policy (Qwen3-4B-Instruct-2507); any <think> in a
# trajectory is illegal grammar (route_reward: 'think_unexpected'), never
# stripped or tolerated.
PROTOCOL_TAGS = ('mode', 'paradigm', 'route', 'layer', 'observation', 'answer')

# Tags whose opening form may carry attributes. `route` always could; `layer`
# carries index=; `observation` carries id= in agentic mode. Everything else
# stays attribute-free -- allowing attributes there would silently widen the
# grammar. Note that widening `observation` costs one fabrication check: a
# stray attribute no longer breaks the open/close balance by itself, so the
# analyzer checks observation ids against the environment's route records
# instead (route_reward._analyze_agentic).
_ATTR_TAGS = ('route', 'layer', 'observation')

_TAG_FLAGS = re.IGNORECASE | re.DOTALL


def block_pattern(tag: str) -> "re.Pattern":
    """<tag ...>body</tag>, body captured as group 1. Attributes are tolerated
    in the opening tag only for the tags in _ATTR_TAGS."""
    opening = rf"<{tag}\b[^>]*>" if tag in _ATTR_TAGS else rf"<{tag}\s*>"
    return re.compile(rf"{opening}(.*?)</{tag}\s*>", _TAG_FLAGS)


def _open_pattern(tag: str) -> "re.Pattern":
    return re.compile(rf"<{tag}\b" if tag in _ATTR_TAGS else rf"<{tag}\s*>", re.IGNORECASE)


def close_pattern(tag: str) -> "re.Pattern":
    return re.compile(rf"</{tag}\s*>", re.IGNORECASE)


_OPEN_PATTERNS = {tag: _open_pattern(tag) for tag in PROTOCOL_TAGS}
_CLOSE_PATTERNS = {tag: close_pattern(tag) for tag in PROTOCOL_TAGS}


def count_open(tag: str, text: str) -> int:
    return len(_OPEN_PATTERNS[tag].findall(text))


def count_close(tag: str, text: str) -> int:
    return len(_CLOSE_PATTERNS[tag].findall(text))


def find_open(tag: str, text: str) -> int:
    """Offset of the first opening tag, or -1 -- same contract as str.find, so
    callers comparing positions keep working unchanged."""
    match = _OPEN_PATTERNS[tag].search(text)
    return -1 if match is None else match.start()


_END_ANCHORED = {tag: re.compile(rf"</{tag}\s*>\s*$", re.IGNORECASE) for tag in PROTOCOL_TAGS}


def ends_with_close(tag: str, text: str) -> bool:
    """Anchored at the end, not 'the first closing tag happens to end the
    string': a trajectory that answers early and then corrects itself has
    several </answer> tags and only the LAST one may close the trajectory."""
    return _END_ANCHORED[tag].search(text) is not None


_MODE_PATTERN = block_pattern('mode')
_PARADIGM_PATTERN = block_pattern('paradigm')
_ROUTE_OPEN_PATTERN = _OPEN_PATTERNS['route']
_LAYER_OPEN_PATTERN = _OPEN_PATTERNS['layer']
_ROUTE_PATTERN = re.compile(r"<route\b([^>]*)>(.*?)</route\s*>", _TAG_FLAGS)
_LAYER_PATTERN = re.compile(r"<layer\b([^>]*)>(.*?)</layer\s*>", _TAG_FLAGS)
# attribute NAMES are lowercased at use (see parse_route); this pattern is
# unchanged -- unknown and duplicate attributes are still rejected
_ATTR_PATTERN = re.compile(r"""\s*(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
# Agentic attributes accept DOUBLE QUOTES ONLY. The legacy pattern above keeps
# tolerating single quotes because an older dataset was built against that
# tolerance; the new surface has no such history and starts strict.
_ATTR_PATTERN_STRICT = re.compile(r"""\s*(\w+)\s*=\s*"([^"]*)\"""")

_VALID_MODES = ("single", "multi", "agentic")
AGENTIC_MODE = "agentic"

# ------------------------------------------------------------- curriculum --
#
# The curriculum opens the ACTION SPACE step by step; it does not reorder the
# data. All four stages draw on the same Main-6 query pool and differ only in
# which modes the policy may use, so a stage is fully described by an ordered
# subset of _VALID_MODES.
#
# THE ONLY LEGAL SUBSETS ARE PREFIXES. single -> single,multi ->
# single,multi,agentic is a nesting, and every later stage must contain every
# earlier one: a policy that lost `single` at stage 2 would be relearning the
# cheap action instead of building on it. So 'single,agentic' is rejected, not
# because the parser could not represent it, but because it is not a curriculum.
#
# ONE parse entry (parse_allowed_modes) is used by the prompt renderer, the
# rollout gate, the reward and the launcher. Four copies of "which modes count"
# is exactly how a stage-1 dataset ends up trained with a stage-3 reward.
CURRICULUM_STAGES: Dict[str, Tuple[str, ...]] = {
    'stage1': ('single',),
    'stage2': ('single', 'multi'),
    # multi and agentic in isolation -- see ALLOWED_MODE_SETS
    'stage2_multi': ('multi',),
    'stage3_agentic': ('agentic',),
    'stage3': _VALID_MODES,
    # Stage 4 is stage 3's action space on a different schedule, not a
    # different grammar.
    'stage4': _VALID_MODES,
}
# The curriculum opens the action space progressively, so the legal sets are
# PREFIXES of _VALID_MODES -- with one deliberate exception. ('agentic',) is an
# isolation set, not a prefix. Inside the full space agentic is attempted only
# a handful of times per step, which leaves "the policy cannot write the
# grammar" and "the policy never gets enough agentic samples to learn it"
# confounded. A stage on agentic alone separates them. It is a diagnostic and a
# possible curriculum stage, not a relaxation: the grammar and the validator
# are untouched.
# ('multi',) is the second isolation set, for the same reason
# as ('agentic',) and against the same confound. Stage 2 opens single+multi
# together, and single is already learned, so a policy that never emits multi
# is equally consistent with "multi is not taught" and "multi never wins the
# competition against single". Only a stage where multi is the sole option
# separates them.
ALLOWED_MODE_SETS: Tuple[Tuple[str, ...], ...] = (
    ('single',), ('single', 'multi'), ('multi',), ('agentic',), _VALID_MODES)

FULL_ACTION_SPACE = _VALID_MODES


def parse_allowed_modes(spec) -> Tuple[str, ...]:
    """THE allowed-modes entry point. Returns an ordered tuple of modes.

    Accepts None or '' (the full action space, i.e. what every pre-curriculum
    caller already does), a stage name, or an explicit comma-separated list.
    Deliberately strict: no aliases, no case folding, no whitespace-tolerant
    reordering, and no subset that is not one of ALLOWED_MODE_SETS. A typo has
    to fail loudly at launch, because the alternative is a run that trains a
    different action space than its dataset was built for and still looks
    completely normal.
    """
    if spec is None:
        return FULL_ACTION_SPACE
    if isinstance(spec, (list, tuple)):
        modes = tuple(spec)
    else:
        text = str(spec).strip()
        if not text:
            return FULL_ACTION_SPACE
        if text in CURRICULUM_STAGES:
            return CURRICULUM_STAGES[text]
        # no empty-part filtering: 'single,' is a typo, and silently repairing
        # it is the same class of leniency this parser exists to refuse
        modes = tuple(text.split(','))
    if modes not in ALLOWED_MODE_SETS:
        raise ValueError(
            f"allowed_modes must be exactly one of "
            f"{[','.join(s) for s in ALLOWED_MODE_SETS]} (or a stage name in "
            f"{sorted(CURRICULUM_STAGES)}), got {spec!r}. Subsets are prefixes "
            f"of {_VALID_MODES}; order and spelling are significant.")
    return modes


def stage_for_allowed_modes(modes: Sequence[str]) -> str:
    """Canonical stage label for a mode set, for dataset metadata."""
    modes = tuple(modes)
    for name in ('stage1', 'stage2', 'stage3'):
        if CURRICULUM_STAGES[name] == modes:
            return name
    raise ValueError(f"no stage matches {modes!r}")
_REQUIRED_ATTRS = ("model",)
# Rejected only when the whole field (after strip+lower) equals one of
# these template placeholders; substrings inside a real query are fine.
_MODEL_PLACEHOLDERS = ("model_id",)
_QUERY_PLACEHOLDERS = ("query", "your_query", "concrete_subquery")


def parse_mode(text: str) -> Tuple[Optional[str], Optional[GrammarError]]:
    matches = _MODE_PATTERN.findall(text)
    if len(matches) == 0:
        return None, GrammarError(ErrorCode.MODE_MISSING, "no <mode> tag found")
    if len(matches) > 1:
        return None, GrammarError(
            ErrorCode.MODE_DUPLICATE, f"found {len(matches)} <mode> tags, expected exactly one")
    mode = matches[0].strip().lower()
    if mode not in _VALID_MODES:
        return None, GrammarError(
            ErrorCode.MODE_INVALID,
            f"mode must be one of {_VALID_MODES}, got {matches[0].strip()!r}")
    return mode, None


def parse_route(text: str) -> Tuple[Optional[RouteCall], Optional[GrammarError]]:
    """Parse the first <route ...>...</route> in text. Structure checks only:
    the model id is returned normalized but NOT resolved against the registry
    (that is validate_route's job)."""
    match = _ROUTE_PATTERN.search(text)
    if match is None:
        if _ROUTE_OPEN_PATTERN.search(text):
            return None, GrammarError(ErrorCode.ROUTE_MALFORMED, "<route> tag is not properly closed")
        return None, GrammarError(ErrorCode.ROUTE_MISSING, "no <route> tag found")

    attr_str, body = match.group(1), match.group(2)

    # Same scanner the agentic parser uses, with the legacy quoting rule and
    # the legacy whitelist -- so `id`/`role`/`refs`/`layer` remain ATTR_UNKNOWN
    # here, which is what keeps them out of single/multi (protocol rule 13).
    attrs, err = _parse_attrs(attr_str, _REQUIRED_ATTRS, _ATTR_PATTERN)
    if err is not None:
        return None, err

    for name in _REQUIRED_ATTRS:
        if name not in attrs:
            return None, GrammarError(ErrorCode.ATTR_MISSING, f"missing required attribute {name!r}")

    model_id = attrs["model"].strip().lower()
    if not model_id:
        return None, GrammarError(ErrorCode.ATTR_MISSING, "model attribute is empty")
    if model_id in _MODEL_PLACEHOLDERS:
        return None, GrammarError(
            ErrorCode.PLACEHOLDER, f"model attribute is the template placeholder {attrs['model'].strip()!r}")

    query = body.strip()
    if not query:
        return None, GrammarError(ErrorCode.QUERY_EMPTY, "route query is empty")
    if query.lower() in _QUERY_PLACEHOLDERS:
        return None, GrammarError(
            ErrorCode.PLACEHOLDER, f"route query is the template placeholder {query!r}")

    return RouteCall(model_id=model_id, query=query), None


def validate_route(route: RouteCall) -> Tuple[Optional[WorkerSpec], Optional[GrammarError]]:
    """Semantic checks against the worker registry. Returns the resolved
    WorkerSpec on success so callers can dispatch without a second lookup.

    ANONYMOUS DIRECTORY. The registry keeps the real ids so a
    real id still resolves, but under ROUTEWEAVER_WORKER_ALIAS=1 the ONLINE
    candidate set is the offered pool and nothing else. Without this second
    check a route naming `qwen3_8b` resolved and DISPATCHED during an alias run
    -- the anonymity would have been a prompt-side convention that the validator
    let the policy walk straight around. agentic was already safe (the V2 path
    passes router.worker_ids into actions.validate_layer as available_models);
    single/multi had only the registry lookup, so this is where the hole was.

    Deliberately the SAME ErrorCode as an unknown id: to the policy an id outside
    the offered list is simply not a worker, and the message repeats only what the
    policy itself wrote, so nothing about the backend is disclosed.
    """
    spec = get_worker(route.model_id)
    if spec is None:
        return None, GrammarError(ErrorCode.UNKNOWN_MODEL, f"unknown model id {route.model_id!r}")
    if (_worker_alias.enabled()
            and route.model_id.strip().lower() not in ACTIVE_WORKER_POOL):
        return None, GrammarError(ErrorCode.UNKNOWN_MODEL,
                                  f"unknown model id {route.model_id!r}")
    return spec, None


# ============================================================ agentic mode ==
#
# Everything below is additive: single/multi never reaches it, and nothing
# above changed behaviour for them.

# The active pool the PROMPT offers, in every mode. Deliberately a subset of
# the registry (qwen3_8b is excluded) while the registry itself stays untouched
# -- a route naming a registry worker outside this pool is still dispatchable,
# so this is a prompt-side curation knob, not a second registry.
#
# ONE definition: routing/prompt_pool.py renders its worker list and every
# worked example from this tuple, so the ids the policy is shown and the ids
# the tests check are the same object, not two copies that can drift.
# A worker that changes backend gets its own canonical id rather than being
# swapped in behind an existing one: a different model is a different action,
# and reusing the id would make two action spaces indistinguishable after the
# fact.
#
# Ordered by size, the
# same ascending-capability convention the API pool used -- worker order is
# rendered verbatim into the prompt, so it is a real (if small) treatment
# variable and is not to be permuted casually.
#
# The pool is now THREE workers rather than four. That is the one thing about
# this swap that is not a like-for-like substitution, and it is deliberate:
# there is no fourth local model in the agreed set. It shortens the prompt's
# worker table and narrows the choice, so fast-local numbers are not directly
# comparable in absolute terms with the four-worker API line.
# MAIN-EXPERIMENT POOL, organised in GraphPlanner's three tiers.
# This tuple is what single/multi render -- prompt_pool._render_workers reads it
# directly, so a runtime router.worker_ids does NOT reach those two modes (only
# agentic takes the list at run time, via mode_selector.mode_rules). Pool size
# and the agentic worker BUDGET are independent: max_workers_v2 caps calls per
# trajectory, not distinct models, and actions.validate_layer keys duplicates on
# (role, model, instruction), so one model may serve several roles.
ACTIVE_WORKER_POOL = (
    # small
    "qwen3_8b",
    "llama31_8b",
    # medium
    "qwen3_30b_a3b",
    "gemma3_27b_it",
    "llama33_70b",
    # large
    "gemini25_flash_lite",
)

# ANONYMOUS DIRECTORY. ROUTEWEAVER_WORKER_ALIAS=1 replaces the pool the
# PROMPT offers with worker_1 .. worker_6 (worker_alias.ALIAS_POOL); the tuple
# above stays as the real-id pool and is what every existing launcher still
# gets, because the flag defaults to off. The alias ids are registered as real
# WorkerSpecs (worker_registry) and as channel entries (a channel config whose
# `workers:` keys are the aliases), so parse -> validate_route -> dispatch all
# resolve without a translation layer anywhere in the hot path.
import worker_alias as _worker_alias  # noqa: E402

if _worker_alias.enabled():
    REAL_WORKER_POOL = ACTIVE_WORKER_POOL
    ACTIVE_WORKER_POOL = _worker_alias.ALIAS_POOL
else:
    REAL_WORKER_POOL = ACTIVE_WORKER_POOL


@dataclass(frozen=True)
class LayerSpec:
    """One layer of a paradigm: which role, how many routes, and which earlier
    layers every route in it must reference."""
    role: str
    min_routes: int
    max_routes: int
    refs_from: Tuple[int, ...] = ()   # 1-based layer indices; () means refs forbidden


# THE single definition of paradigm structure. The rollout loop and the reward
# analyzer both read this table through validate_agentic_layer(); neither is
# allowed its own copy.
PARADIGM_SPECS: Dict[str, Tuple[LayerSpec, ...]] = {
    'refine': (
        LayerSpec('solver', 1, 1),
        LayerSpec('critic', 1, 1, (1,)),
        LayerSpec('refiner', 1, 1, (1, 2)),
    ),
    'ensemble': (
        LayerSpec('solver', 2, 3),
        LayerSpec('aggregator', 1, 1, (1,)),
    ),
    'debate': (
        LayerSpec('proposer', 2, 2),
        LayerSpec('rebutter', 2, 2, (1,)),
        LayerSpec('judge', 1, 1, (1, 2)),
    ),
    'plan': (
        LayerSpec('planner', 1, 1),
        LayerSpec('executor', 1, 3, (1,)),
        LayerSpec('aggregator', 1, 1, (1, 2)),
    ),
}

_VALID_PARADIGMS = tuple(PARADIGM_SPECS)

_ROUTE_ID_PATTERN = re.compile(r"^(\d+)-(\d+)$")
_AGENTIC_REQUIRED_ATTRS = ("id", "model", "role")
_AGENTIC_OPTIONAL_ATTRS = ("refs",)
_LAYER_ATTRS = ("index",)


@dataclass(frozen=True)
class ParsedLayer:
    index: int
    routes: Tuple[RouteCall, ...] = ()


def paradigm_layer_count(paradigm: str) -> Optional[int]:
    specs = PARADIGM_SPECS.get(paradigm)
    return None if specs is None else len(specs)


def parse_paradigm(text: str) -> Tuple[Optional[str], Optional[GrammarError]]:
    """<paradigm>...</paradigm>, same exactly-once contract as parse_mode."""
    matches = _PARADIGM_PATTERN.findall(text)
    if len(matches) == 0:
        return None, GrammarError(ErrorCode.PARADIGM_MISSING, "no <paradigm> tag found")
    if len(matches) > 1:
        return None, GrammarError(
            ErrorCode.PARADIGM_DUPLICATE,
            f"found {len(matches)} <paradigm> tags, expected exactly one")
    paradigm = matches[0].strip().lower()
    if paradigm not in _VALID_PARADIGMS:
        return None, GrammarError(
            ErrorCode.PARADIGM_INVALID,
            f"paradigm must be one of {_VALID_PARADIGMS}, got {matches[0].strip()!r}")
    return paradigm, None


def _parse_attrs(attr_str: str, allowed: Sequence[str],
                 pattern: "re.Pattern") -> Tuple[Optional[Dict[str, str]], Optional[GrammarError]]:
    """Shared attribute scanner: rejects unparseable padding, duplicates and
    names outside `allowed`. `pattern` picks the quoting rule."""
    attrs: Dict[str, str] = {}
    pos = 0
    for match in pattern.finditer(attr_str):
        if attr_str[pos:match.start()].strip():
            return None, GrammarError(
                ErrorCode.ROUTE_MALFORMED,
                f"unparseable content in attributes: {attr_str[pos:match.start()].strip()!r}")
        name = match.group(1).lower()
        if name in attrs:
            return None, GrammarError(ErrorCode.ATTR_DUPLICATE, f"duplicate attribute {name!r}")
        if name not in allowed:
            return None, GrammarError(ErrorCode.ATTR_UNKNOWN, f"unknown attribute {name!r}")
        # the legacy pattern has one value group per quote style, the strict one
        # has a single group; take whichever matched
        attrs[name] = next((g for g in match.groups()[1:] if g is not None), '')
        pos = match.end()
    if attr_str[pos:].strip():
        return None, GrammarError(
            ErrorCode.ROUTE_MALFORMED,
            f"unparseable content in attributes: {attr_str[pos:].strip()!r}")
    return attrs, None


def _split_refs(raw: str) -> Tuple[str, ...]:
    """refs="1-1,1-2" (commas and/or whitespace). Order preserved, blanks dropped."""
    return tuple(part for part in re.split(r"[,\s]+", raw.strip()) if part)


def parse_routes(text: str) -> Tuple[List[RouteCall], Optional[GrammarError]]:
    """Parse EVERY <route> in text under the agentic attribute grammar
    (id/model/role required, refs optional, double quotes only).

    Returns ([], error) on the first malformed route -- a layer is accepted or
    rejected as a whole, never partially dispatched.
    """
    routes: List[RouteCall] = []
    for match in _ROUTE_PATTERN.finditer(text):
        attr_str, body = match.group(1), match.group(2)
        attrs, err = _parse_attrs(
            attr_str, _AGENTIC_REQUIRED_ATTRS + _AGENTIC_OPTIONAL_ATTRS, _ATTR_PATTERN_STRICT)
        if err is not None:
            return [], err

        for name in _AGENTIC_REQUIRED_ATTRS:
            if name not in attrs or not attrs[name].strip():
                code = {'id': ErrorCode.ROUTE_ID_MISSING,
                        'role': ErrorCode.ROLE_MISSING}.get(name, ErrorCode.ATTR_MISSING)
                return [], GrammarError(code, f"missing required attribute {name!r}")

        route_id = attrs['id'].strip()
        id_match = _ROUTE_ID_PATTERN.match(route_id)
        if id_match is None:
            return [], GrammarError(
                ErrorCode.ROUTE_ID_MALFORMED,
                f"route id must be '<layer>-<index>', got {route_id!r}")

        model_id = attrs['model'].strip().lower()
        if model_id in _MODEL_PLACEHOLDERS:
            return [], GrammarError(
                ErrorCode.PLACEHOLDER,
                f"model attribute is the template placeholder {attrs['model'].strip()!r}")

        query = body.strip()
        if not query:
            return [], GrammarError(ErrorCode.QUERY_EMPTY, f"route {route_id} has an empty task")
        if query.lower() in _QUERY_PLACEHOLDERS:
            return [], GrammarError(
                ErrorCode.PLACEHOLDER, f"route task is the template placeholder {query!r}")

        routes.append(RouteCall(
            model_id=model_id,
            query=query,
            route_id=route_id,
            role=attrs['role'].strip().lower(),
            refs=_split_refs(attrs.get('refs', '')),
            layer=int(id_match.group(1)),
        ))
    if not routes and _ROUTE_OPEN_PATTERN.search(text):
        return [], GrammarError(ErrorCode.ROUTE_MALFORMED, "<route> tag is not properly closed")
    return routes, None


def parse_layer(text: str) -> Tuple[Optional[ParsedLayer], Optional[GrammarError]]:
    """Parse the ONE <layer index="N"> block an agentic turn must contain.

    Structure only: index is an integer and the routes parse. Whether the layer
    fits the declared paradigm is validate_agentic_layer()'s job.
    """
    matches = list(_LAYER_PATTERN.finditer(text))
    if not matches:
        if _LAYER_OPEN_PATTERN.search(text):
            return None, GrammarError(ErrorCode.LAYER_MALFORMED, "<layer> tag is not properly closed")
        return None, GrammarError(ErrorCode.LAYER_MISSING, "no <layer> tag found")
    if len(matches) > 1:
        return None, GrammarError(
            ErrorCode.LAYER_DUPLICATE,
            f"found {len(matches)} <layer> blocks, expected exactly one per turn")

    attr_str, body = matches[0].group(1), matches[0].group(2)
    attrs, err = _parse_attrs(attr_str, _LAYER_ATTRS, _ATTR_PATTERN_STRICT)
    if err is not None:
        return None, err
    if 'index' not in attrs:
        return None, GrammarError(ErrorCode.LAYER_INDEX_INVALID, "missing required attribute 'index'")
    raw_index = attrs['index'].strip()
    if not raw_index.isdigit() or int(raw_index) < 1:
        return None, GrammarError(
            ErrorCode.LAYER_INDEX_INVALID, f"layer index must be a positive integer, got {raw_index!r}")

    routes, err = parse_routes(body)
    if err is not None:
        return None, err
    return ParsedLayer(index=int(raw_index), routes=tuple(routes)), None


def validate_agentic_layer(paradigm: str, layer: ParsedLayer,
                           completed_layers: Sequence[Sequence[str]]) -> List[GrammarError]:
    """THE agentic protocol check, shared by the rollout loop and the reward.

    ``completed_layers`` is the ordered list of already-executed layers, each
    the ordered route ids of that layer. Returns every violation found (empty
    list = the layer may be dispatched).
    """
    specs = PARADIGM_SPECS.get(paradigm)
    if specs is None:
        return [GrammarError(ErrorCode.PARADIGM_INVALID, f"unknown paradigm {paradigm!r}")]

    expected_index = len(completed_layers) + 1
    if layer.index != expected_index:
        return [GrammarError(
            ErrorCode.LAYER_INDEX_NONCONSECUTIVE,
            f"expected layer index {expected_index}, got {layer.index}")]
    if layer.index > len(specs):
        return [GrammarError(
            ErrorCode.LAYER_INDEX_INVALID,
            f"paradigm {paradigm!r} has {len(specs)} layers, got index {layer.index}")]

    spec = specs[layer.index - 1]
    errors: List[GrammarError] = []
    if not spec.min_routes <= len(layer.routes) <= spec.max_routes:
        errors.append(GrammarError(
            ErrorCode.LAYER_ROUTE_COUNT_INVALID,
            f"layer {layer.index} of {paradigm!r} takes {spec.min_routes}-{spec.max_routes} "
            f"routes, got {len(layer.routes)}"))

    completed_ids = [rid for ids in completed_layers for rid in ids]
    required_refs = {rid for i in spec.refs_from for rid in completed_layers[i - 1]}
    seen = set(completed_ids)

    for position, route in enumerate(layer.routes, start=1):
        expected_id = f"{layer.index}-{position}"
        if route.route_id != expected_id:
            errors.append(GrammarError(
                ErrorCode.ROUTE_ID_MALFORMED,
                f"expected route id {expected_id!r}, got {route.route_id!r}"))
        if route.route_id in seen:
            errors.append(GrammarError(
                ErrorCode.ROUTE_ID_DUPLICATE, f"route id {route.route_id!r} already used"))
        seen.add(route.route_id)

        if route.role != spec.role:
            errors.append(GrammarError(
                ErrorCode.ROLE_INVALID,
                f"layer {layer.index} of {paradigm!r} requires role {spec.role!r}, "
                f"got {route.role!r}"))

        if not spec.refs_from:
            if route.refs:
                errors.append(GrammarError(
                    ErrorCode.REFS_UNEXPECTED,
                    f"route {route.route_id} is in the first layer and must not carry refs"))
            continue
        # Forward refs, self refs and refs to ids that never existed all land
        # here: only ids from ALREADY COMPLETED layers are referenceable.
        unknown = [ref for ref in route.refs if ref not in completed_ids]
        if unknown:
            errors.append(GrammarError(
                ErrorCode.REFS_INVALID,
                f"route {route.route_id} references unavailable ids {unknown}"))
        if set(route.refs) != required_refs:
            errors.append(GrammarError(
                ErrorCode.REFS_INCOMPLETE,
                f"route {route.route_id} must reference exactly "
                f"{sorted(required_refs)}, got {sorted(set(route.refs))}"))
    return errors


_OBSERVATION_PATTERN = re.compile(r"<observation\b([^>]*)>(.*?)</observation\s*>", _TAG_FLAGS)
_OBSERVATION_ATTRS = ("id",)


def parse_observations(text: str) -> List[Tuple[Optional[str], str]]:
    """Every <observation> block as (id, body), document order.

    Observations are written by the ENVIRONMENT, so this never returns an
    error: its job is to report what is in the text so the analyzer can
    compare that against what the environment actually dispatched. A block
    whose id is absent or unparseable yields id=None, which the analyzer then
    fails to match against any route.
    """
    found: List[Tuple[Optional[str], str]] = []
    for match in _OBSERVATION_PATTERN.finditer(text):
        attrs, err = _parse_attrs(match.group(1), _OBSERVATION_ATTRS, _ATTR_PATTERN_STRICT)
        obs_id = None if err is not None else (attrs.get('id') or None)
        found.append((obs_id, match.group(2)))
    return found


def question_of_prompt(messages) -> str:
    """The question this sample is about, recovered from the rendered prompt.

    The template is <rules> ... "Question: {question}", so everything after the
    LAST such marker is the question and nothing else. Recovered rather than
    threaded through because verl hands the agent loop the rendered chat, not
    the row it came from.

    Empty is a hard error, not a silently empty header: a multi worker that is
    told "Original question:" followed by nothing has been given a false
    signal, which is worse than the bare query this replaced.
    """
    body = str((messages[-1] or {}).get("content", "")) if messages else ""
    question = body.rsplit("Question:", 1)[-1].strip() if "Question:" in body else ""
    if not question:
        raise ValueError(
            "cannot recover the original question from the prompt; multi "
            "context injection would send an empty goal")
    return question


def build_multi_worker_query(question: str, task: str) -> str:
    """The text a MULTI worker actually receives (Uno-style, frozen).

    A worker handed `task` alone cannot notice that the task is wrong. Asked
    for "the birthdate of the Count of Luxembourg who fathered Bonne de
    Luxembourg", a router may decompose correctly, have hop 1 name the WRONG
    person, and hand hop 2 only "what is the birthdate of Charles IV" -- which
    it answers correctly about the wrong someone. Both hops succeed and the
    trajectory is wrong. Across 448
    valid multi trajectories the gold answer never reached ANY observation in
    16.5% of the cases where forced-single got it right.

    So the worker is told what the whole question is. It is NOT told what the
    earlier hops returned: those go back to the ROUTER, which reads them and
    writes the next sub-task. That division is the point --

      the router plans, reads the history and decides what to ask next;
      the worker knows the goal and answers exactly one sub-task.

    Injecting every prior observation into the worker instead (see the
    multi-context-wrapper branch) also carries every prior MISTAKE into it,
    inflates the payload with text the sub-task does not need, and blurs which
    component is responsible for planning. Kept as an appendix variant, not as
    the default.

    agentic keeps its explicit refs: its layers run in parallel, so a route
    there genuinely chooses its evidence and cannot infer it from order.
    """
    if not task or not task.strip():
        raise ValueError("a multi sub-task cannot be empty")
    if not question or not question.strip():
        raise ValueError("a multi worker query needs the original question")
    return f"Original question:\n{question.strip()}\n\nCurrent sub-task:\n{task.strip()}"


def build_worker_query(route: RouteCall, observations_by_id: Dict[str, str]) -> str:
    """The text an agentic worker actually receives.

    A bare route body is not self-contained: a critic asked to "find the flaw"
    with no access to what it is critiquing produces noise. The environment --
    which already holds every upstream observation -- splices the referenced
    outputs in, so the router never has to copy an upstream answer forward
    just to make the next call executable.
    """
    if not route.is_agentic:
        return route.query
    sections = [f"Role: {route.role}"]
    if route.refs:
        block = "\n\n".join(
            f"[{ref}]\n{observations_by_id.get(ref, '')}".rstrip() for ref in route.refs)
        sections.append(f"Referenced outputs:\n{block}")
    sections.append(f"Assigned task:\n{route.query}")
    return "\n\n".join(sections)
