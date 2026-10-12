"""Route grammar for the Unified Router."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from worker_registry import WorkerSpec, get_worker

ROUTE_PROTOCOL_VERSION = 'unified-router-v2'

AGENTIC_PROTOCOL_VERSION = 'unified-router-v4'
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
    MODE_NOT_ALLOWED = "mode_not_allowed"
    PARADIGM_NOT_ALLOWED = "paradigm_not_allowed"
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
    route_id: Optional[str] = None
    role: Optional[str] = None
    refs: Tuple[str, ...] = ()
    layer: Optional[int] = None

    @property
    def is_agentic(self) -> bool:
        return self.route_id is not None


PROTOCOL_TAGS = ('mode', 'paradigm', 'route', 'layer', 'observation', 'answer')

_ATTR_TAGS = ('route', 'layer', 'observation')

_TAG_FLAGS = re.IGNORECASE | re.DOTALL


def block_pattern(tag: str) -> "re.Pattern":
    """<tag ...>body</tag>, body captured as group 1. Attributes are tolerated
    in the opening tag only for the tags in _ATTR_TAGS.
    """
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
    callers comparing positions keep working unchanged.
    """
    match = _OPEN_PATTERNS[tag].search(text)
    return -1 if match is None else match.start()


_END_ANCHORED = {tag: re.compile(rf"</{tag}\s*>\s*$", re.IGNORECASE) for tag in PROTOCOL_TAGS}


def ends_with_close(tag: str, text: str) -> bool:
    """True if the text ends with the closing tag (the last one counts)."""
    return _END_ANCHORED[tag].search(text) is not None


_MODE_PATTERN = block_pattern('mode')
_PARADIGM_PATTERN = block_pattern('paradigm')
_ROUTE_OPEN_PATTERN = _OPEN_PATTERNS['route']
_LAYER_OPEN_PATTERN = _OPEN_PATTERNS['layer']
_ROUTE_PATTERN = re.compile(r"<route\b([^>]*)>(.*?)</route\s*>", _TAG_FLAGS)
_LAYER_PATTERN = re.compile(r"<layer\b([^>]*)>(.*?)</layer\s*>", _TAG_FLAGS)
_ATTR_PATTERN = re.compile(r"""\s*(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_ATTR_PATTERN_STRICT = re.compile(r"""\s*(\w+)\s*=\s*"([^"]*)\"""")

_VALID_MODES = ("single", "multi", "agentic")
AGENTIC_MODE = "agentic"

CURRICULUM_STAGES: Dict[str, Tuple[str, ...]] = {
    'stage1': ('single',),
    'stage2': ('single', 'multi'),
    'stage2_multi': ('multi',),
    'stage3_agentic': ('agentic',),
    'stage3': _VALID_MODES,
    'stage4': _VALID_MODES,
}
ALLOWED_MODE_SETS: Tuple[Tuple[str, ...], ...] = (
    ('single',), ('single', 'multi'), ('multi',), ('agentic',), _VALID_MODES)

FULL_ACTION_SPACE = _VALID_MODES


def parse_allowed_modes(spec) -> Tuple[str, ...]:
    """THE allowed-modes entry point. Returns an ordered tuple of modes."""
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
    """Parse the first <route ...>...</route> in text (structure checks only)."""
    match = _ROUTE_PATTERN.search(text)
    if match is None:
        if _ROUTE_OPEN_PATTERN.search(text):
            return None, GrammarError(ErrorCode.ROUTE_MALFORMED, "<route> tag is not properly closed")
        return None, GrammarError(ErrorCode.ROUTE_MISSING, "no <route> tag found")

    attr_str, body = match.group(1), match.group(2)

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
    """
    spec = get_worker(route.model_id)
    if spec is None:
        return None, GrammarError(ErrorCode.UNKNOWN_MODEL, f"unknown model id {route.model_id!r}")
    if (_worker_alias.enabled()
            and route.model_id.strip().lower() not in ACTIVE_WORKER_POOL):
        return None, GrammarError(ErrorCode.UNKNOWN_MODEL,
                                  f"unknown model id {route.model_id!r}")
    return spec, None


ACTIVE_WORKER_POOL = (
    "qwen3_8b",
    "llama31_8b",
    "qwen3_30b_a3b",
    "gemma3_27b_it",
    "llama33_70b",
    "gemini25_flash_lite",
)

import worker_alias as _worker_alias  # noqa: E402

if _worker_alias.enabled():
    REAL_WORKER_POOL = ACTIVE_WORKER_POOL
    ACTIVE_WORKER_POOL = _worker_alias.ALIAS_POOL
else:
    REAL_WORKER_POOL = ACTIVE_WORKER_POOL


@dataclass(frozen=True)
class LayerSpec:
    """One layer of a paradigm: which role, how many routes, and which earlier
    layers every route in it must reference.
    """
    role: str
    min_routes: int
    max_routes: int
    refs_from: Tuple[int, ...] = ()   # 1-based layer indices; () means refs forbidden


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
    names outside `allowed`. `pattern` picks the quoting rule.
    """
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
    """Parse the ONE <layer index="N"> block an agentic turn must contain."""
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
    """THE agentic protocol check, shared by the rollout loop and the reward."""
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
    """Every <observation> block as (id, body), document order."""
    found: List[Tuple[Optional[str], str]] = []
    for match in _OBSERVATION_PATTERN.finditer(text):
        attrs, err = _parse_attrs(match.group(1), _OBSERVATION_ATTRS, _ATTR_PATTERN_STRICT)
        obs_id = None if err is not None else (attrs.get('id') or None)
        found.append((obs_id, match.group(2)))
    return found


def question_of_prompt(messages) -> str:
    """The question this sample is about, recovered from the rendered prompt."""
    body = str((messages[-1] or {}).get("content", "")) if messages else ""
    question = body.rsplit("Question:", 1)[-1].strip() if "Question:" in body else ""
    if not question:
        raise ValueError(
            "cannot recover the original question from the prompt; multi "
            "context injection would send an empty goal")
    return question


def build_multi_worker_query(question: str, task: str) -> str:
    """The text a MULTI worker actually receives (Uno-style, frozen)."""
    if not task or not task.strip():
        raise ValueError("a multi sub-task cannot be empty")
    if not question or not question.strip():
        raise ValueError("a multi worker query needs the original question")
    return f"Original question:\n{question.strip()}\n\nCurrent sub-task:\n{task.strip()}"


def build_worker_query(route: RouteCall, observations_by_id: Dict[str, str]) -> str:
    """The text an agentic worker actually receives."""
    if not route.is_agentic:
        return route.query
    sections = [f"Role: {route.role}"]
    if route.refs:
        block = "\n\n".join(
            f"[{ref}]\n{observations_by_id.get(ref, '')}".rstrip() for ref in route.refs)
        sections.append(f"Referenced outputs:\n{block}")
    sections.append(f"Assigned task:\n{route.query}")
    return "\n\n".join(sections)
