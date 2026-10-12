"""The joint free-routing mode selector: turn 1 of a single trajectory."""
import re

MODES = ("single", "multi", "agentic")

MODE_LINES = {
    "single": "single \u2014 one worker handles the task.",
    "multi": "multi \u2014 workers handle the task sequentially.",
    "agentic": ("agentic \u2014 role-assigned workers handle the task in "
                "adaptive layers."),
}

SELECTOR_TEMPLATE = """You are a router. Choose how worker models should handle the question.

## Modes

No mode is preferred by default.

{menu}

## Response

Reply with exactly one mode declaration and nothing else:
{declaration}

You will receive the corresponding execution rules after choosing.

Question: {question}
"""
SELECTOR_VERSION = "v2-2026-08-19"

SELECTOR_V1_MODE_LINES = {
    "single": "single -- one worker call, then you answer.",
    "multi": ("multi -- worker calls in sequence, each written after reading "
              "the previous reply."),
    "agentic": ("agentic -- worker calls in parallel layers, each worker given "
                "an assigned role."),
}
SELECTOR_V1_TEMPLATE = """You are a router. Workers do the work; you decide how the work is organised.

## Modes

Choose exactly ONE mode. No mode is better by default.

{menu}

## Procedure

- Your entire reply now is the mode declaration and nothing else: {declaration}
- After you declare it, you will be given the execution rules for the mode you chose, and you will carry the task out under them.
- Do not solve the question yourself and do not call a worker yet.

Question: {question}
"""


def build(question, order=MODES, version="v2"):
    """The selector prompt."""
    order = tuple(order)
    if sorted(order) != sorted(MODES):
        raise ValueError(f"order must be a permutation of {MODES}, got {order}")
    if version == "v1":
        menu = "\n".join(SELECTOR_V1_MODE_LINES[m] for m in order)
        declaration = " or ".join(f"<mode>{m}</mode>" for m in order)
        return SELECTOR_V1_TEMPLATE.format(menu=menu, declaration=declaration,
                                           question=str(question).strip())
    if version != "v2":
        raise ValueError(f"unknown selector version {version!r}")
    menu = "\n".join(MODE_LINES[m] for m in order)
    declaration = "\n".join(f"<mode>{m}</mode>" for m in order)
    return SELECTOR_TEMPLATE.format(menu=menu, declaration=declaration,
                                    question=str(question).strip())


PERMUTATIONS = tuple(__import__("itertools").permutations(MODES))
assert len(PERMUTATIONS) == 6


def order_for(sample_id, seed=0):
    """The menu order for one QUERY, a pure function of (sample_id, seed)."""
    import hashlib
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode()).hexdigest()
    return PERMUTATIONS[int(digest[:8], 16) % len(PERMUTATIONS)]


_MODE_TAG = re.compile(r"<mode>\s*([a-zA-Z_]+)\s*</mode>", re.I)
_FORBIDDEN_IN_SELECTOR = {
    "route_in_selector": re.compile(r"<route\b", re.I),
    "layer_in_selector": re.compile(r"<layer\b", re.I),
    "action_in_selector": re.compile(r"<action\b", re.I),
    "paradigm_in_selector": re.compile(r"<paradigm\b", re.I),
    "observation_in_selector": re.compile(r"<observation\b", re.I),
    "answer_in_selector": re.compile(r"<answer\b", re.I),
}


def parse_selection(text):
    """(mode|None, [violation codes])."""
    text = str(text or "")
    codes = []
    hits = _MODE_TAG.findall(text)
    for code, pattern in _FORBIDDEN_IN_SELECTOR.items():
        if pattern.search(text):
            codes.append(code)
    if not hits:
        codes.append("selector_mode_missing")
        return None, sorted(set(codes))
    if len(hits) > 1:
        codes.append("selector_mode_duplicate")
        return None, sorted(set(codes))
    mode = hits[0].strip().lower()
    if mode not in MODES:
        codes.append("selector_mode_invalid")
        return None, sorted(set(codes))
    if codes:
        return None, sorted(set(codes))
    return mode, []


SELECTOR_RETRY_CUE = (
    "That reply was not a mode declaration. Reply with exactly one of "
    "{declaration} and nothing else.")


def retry_cue(order=MODES):
    return SELECTOR_RETRY_CUE.format(
        declaration=" or ".join(f"<mode>{m}</mode>" for m in order))


_QUESTION_ANCHOR = "Question:"
_SENTINEL = "@@RFM_QUESTION_SLOT@@"


def _rules_only(render, what):
    """`render` is a callable question -> full prompt."""
    full = render(_SENTINEL)
    if _SENTINEL not in full:
        raise ValueError(
            f"{what} did not interpolate the question; the rules/question slice "
            f"is unsafe")
    head = full.split(_SENTINEL)[0]
    if _QUESTION_ANCHOR not in head:
        raise ValueError(
            f"{what} no longer places {_QUESTION_ANCHOR!r} before the question; "
            f"the slice is unsafe")
    return head[:head.rindex(_QUESTION_ANCHOR)].rstrip()


_RULES_HEAD = {
    "single": "Your mode is single. These are its execution rules.",
    "multi": "Your mode is multi. These are its execution rules.",
    "agentic": "Your mode is agentic. These are its execution rules.",
}
PREDECLARED_SENTENCE = ("The execution mode has already been declared. "
                        "Do not declare it again.")
_RULES_TAIL = (PREDECLARED_SENTENCE + "\n"
                 "Continue now, under these rules, on the question above.")

_DECL_BULLET = re.compile(r"^\s*\+\s*<mode.*EXACTLY ONCE", re.I)
_DECL_INSTRUCTION = re.compile(
    r"^\s*(One routing mode is available; declare it\.|"
    r"Declare exactly one of these (two|three) modes\.|"
    r"First,? declare|Choose exactly one mode)", re.I)
_STANDALONE_MODE_TAG = re.compile(r"^\s*<mode>\s*\w+\s*</mode>\s*$", re.I)
_RESIDUAL_MODE = re.compile(r"<mode", re.I)
_RESIDUAL_DECLARE = re.compile(r"\bdeclare\b", re.I)


def strip_mode_declaration_rules(body):
    """(cleaned_body, [removed lines])."""
    kept, removed = [], []
    for line in str(body).splitlines():
        if _DECL_BULLET.match(line) or _DECL_INSTRUCTION.match(line) \
                or _STANDALONE_MODE_TAG.match(line):
            removed.append(line)
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    leftovers = [l for l in cleaned.splitlines()
                 if _RESIDUAL_MODE.search(l) or _RESIDUAL_DECLARE.search(l)]
    if leftovers:
        raise AssertionError(
            "mode-declaration text survived the predeclared strip: "
            f"{leftovers[:3]}")
    return cleaned, removed


def mode_rules(mode, question, worker_ids=None, sm_prompt_builder=None,
               predeclared=True, return_removed=False, routing_state=False,
               max_workers=None, forced_worker=None):
    """The environment's turn-2 injection for `mode`."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "agentic":
        import route_syntax
        ids = list(worker_ids or ())
        if not ids:
            raise ValueError("agentic rules need the worker id list")
        body = _rules_only(
            lambda q: route_syntax.build(q, ids,
                                         routing_state_rules=routing_state,
                                         max_workers=max_workers),
            "route_syntax.build")
    else:
        if sm_prompt_builder is None:
            raise ValueError("single/multi rules need mode_prompts.build_sm_prompt")
        forced_mode = f"forced_{mode}"
        body = _rules_only(lambda q: sm_prompt_builder(q, forced_mode, worker_ids),
                           f"build_sm_prompt({forced_mode})")
    removed = []
    if predeclared:
        body, removed = strip_mode_declaration_rules(body)
    if forced_worker:
        from worker_explore_schedule import FORCED_LINE   # noqa: PLC0415
        body = f"{body}\n\n{FORCED_LINE.format(worker=forced_worker)}"
    text = f"{_RULES_HEAD[mode]}\n\n{body}\n\n{_RULES_TAIL}"
    return (text, removed) if return_removed else text


def assert_no_leakage(prompt_text, sample):
    """The selector prompt may contain the question and nothing else about the row."""
    text = str(prompt_text or "")
    question = str(sample.get("raw_question") or "").strip()
    if question:
        text = text.replace(question, " ")
    low = text.lower()
    leaks = []
    for key in ("dataset", "scorer_dataset", "domain", "pool", "scorer"):
        value = sample.get(key)
        if value and str(value).lower() in low:
            leaks.append(f"{key}={value}")
    for gold in (sample.get("gold") or []):
        g = str(gold).strip()
        if len(g) >= 4 and g.lower() in low:
            leaks.append(f"gold={g[:40]}")
    meta = sample.get("meta") or {}
    for key in ("rfm_dataset", "rfm_domain", "rfm_pool", "tier", "difficulty",
                "task", "discipline", "topic"):
        value = meta.get(key)
        if value and len(str(value)) >= 4 and str(value).lower() in low:
            leaks.append(f"meta.{key}={value}")
    return leaks


for _m, _line in MODE_LINES.items():
    assert _line.startswith(f"{_m} \u2014 "), _m
    assert _line.endswith("."), _m
_WORDS = {m: len(l.split()) for m, l in MODE_LINES.items()}
assert max(_WORDS.values()) - min(_WORDS.values()) <= 5, \
    f"mode menu lines are not comparable in length: {_WORDS}"
for _tag in ("<route", "<layer", "<action", "<paradigm", "<observation", "<answer"):
    assert _tag not in SELECTOR_TEMPLATE, \
        f"selector template must contain no {_tag} -- it would be a worked example"
_NO_DEFAULT_SENTENCE = "No mode is preferred by default."
assert _NO_DEFAULT_SENTENCE in SELECTOR_TEMPLATE
_body_without_denial = SELECTOR_TEMPLATE.replace(_NO_DEFAULT_SENTENCE, " ").lower()
for _bad in ("simple", "cheap", "expensive", "powerful", "prefer", "usually",
             "recommended", "default", "best for", "complex", "advanced"):
    assert _bad not in _body_without_denial, \
        f"selector template hints a default via {_bad!r}"
assert SELECTOR_TEMPLATE.count("{declaration}") == 1
assert "Choose exactly ONE mode" not in SELECTOR_TEMPLATE
