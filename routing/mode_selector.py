"""The joint free-routing mode selector: turn 1 of a single trajectory.

WHY THIS EXISTS. The reported runs has three validated execution chains and three
FORCED entry points. Free routing needs a fourth thing that none of them
provide: a first turn in which the router sees only the choice, decides it, and
nothing else. This module is that turn, and only that turn -- it owns no
execution and no rules of its own.

TWO PIECES:

  SELECTOR_TEMPLATE   the short, symmetric, example-free prompt for turn 1.
  mode_rules(mode)    the chosen mode's execution rules, sliced out of the
                      EXISTING forced-mode prompt for that mode. Not rewritten:
                      the rules an executor sees under free routing are the same
                      bytes it sees under forced routing, so a free run and a
                      forced run differ in how the mode was chosen and in
                      nothing else.

SELECTOR DESIGN CONSTRAINTS, and how each is met:

  symmetric length/tone   the three menu lines are 9 / 13 / 12 tokens under the
                          Qwen tokenizer (see tests); each is one clause naming
                          how many workers and how they relate. Asserted at
                          import so an edit cannot quietly unbalance them.
  no worked example       there is no <mode> ... </mode> filled in anywhere, no
                          route, no layer, no trajectory. The declaration form
                          appears once per mode, inside the enumeration the
                          router must choose from, which is the minimum needed
                          to state the output format at all.
  no default              "No mode is better by default." plus alphabetical-free
                          ordering handled by the caller: build takes an
                          `order` argument so a run can permute the menu, and
                          nothing in the text calls any mode simple, cheap,
                          powerful or preferred.
  no leakage              the template interpolates the question and nothing
                          else. No benchmark name, no dataset/domain label, no
                          gold, no scorer, no difficulty. Asserted by
                          assert_no_leakage against the row's own metadata.
  declaration only        the procedure says the rules for the chosen mode
                          arrive afterwards, so turn 1 has nothing else it could
                          usefully emit. This is a prompt-level constraint, NOT
                          constrained decoding: sampling is untouched, and a
                          turn that emits something else is recorded as a
                          selector violation rather than prevented.

The worker table is deliberately ABSENT here. Picking a topology does not need
the model list, and both mode rule bodies already carry it, so leaving it out
keeps turn 1 short and keeps the three options symmetric in what they are
allowed to reference.
"""
import re

MODES = ("single", "multi", "agentic")

# One clause each: what the workers do and how they relate. No adjective that
# could rank them, and no count that makes one sound cheaper than another.
# Frozen (v2 selector, user-supplied wording).
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
# The declaration block is one line per mode, in the SAME order as the menu, so a
# permutation moves both together and neither list can hint a default by sitting
# in a different order from the other.
SELECTOR_VERSION = "v2-2026-08-19"

# The previous wording, kept so the A/B control can render it without a git
# checkout. Frozen: it is a measurement instrument now, not a live prompt.
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
    """The selector prompt.

    `order` permutes the menu AND the declaration list together, and never
    touches the per-mode wording -- so varying it varies position only.
    """
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
    """The menu order for one QUERY, a pure function of (sample_id, seed).

    Because it depends on the sample_id and nothing else, every one of that
    query's K rollouts gets the SAME order -- which is what makes a group's mode
    spread a property of the policy rather than of the menu. Balanced across the
    six permutations by hashing, so a 300-query sweep lands ~50 per permutation
    without anyone assigning them.
    """
    import hashlib
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode()).hexdigest()
    return PERMUTATIONS[int(digest[:8], 16) % len(PERMUTATIONS)]


# --------------------------------------------------------------- validation --

_MODE_TAG = re.compile(r"<mode>\s*([a-zA-Z_]+)\s*</mode>", re.I)
# Anything the first turn must not contain: an action, a layer, a paradigm, a
# fabricated observation, or a final answer.
_FORBIDDEN_IN_SELECTOR = {
    "route_in_selector": re.compile(r"<route\b", re.I),
    "layer_in_selector": re.compile(r"<layer\b", re.I),
    "action_in_selector": re.compile(r"<action\b", re.I),
    "paradigm_in_selector": re.compile(r"<paradigm\b", re.I),
    "observation_in_selector": re.compile(r"<observation\b", re.I),
    "answer_in_selector": re.compile(r"<answer\b", re.I),
}


def parse_selection(text):
    """(mode|None, [violation codes]).

    Exactly one legal <mode> tag is required. Zero, two, an unknown value, or any
    other protocol tag in the same turn is a violation, and the mode is refused
    -- there is no repair and no default, because a default would be the thing
    this whole design is trying not to have.
    """
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
        # a legal declaration that also carried an action is still refused: the
        # trajectory would then contain an un-executed action the executor never
        # sees, and the reward analyser would read it as the router's own move
        return None, sorted(set(codes))
    return mode, []


SELECTOR_RETRY_CUE = (
    "That reply was not a mode declaration. Reply with exactly one of "
    "{declaration} and nothing else.")


def retry_cue(order=MODES):
    return SELECTOR_RETRY_CUE.format(
        declaration=" or ".join(f"<mode>{m}</mode>" for m in order))


# ----------------------------------------------------- mode execution rules --
#
# Sliced from the existing forced prompts. Both templates end with
# "Question: {question}", so everything before that anchor is the rulebook and
# nothing else -- no question, no answer-format terminator, no benchmark text.
# The slice is asserted rather than trusted: a template that stops ending with
# the anchor must break the build, not silently inject a truncated rulebook.

_QUESTION_ANCHOR = "Question:"
# The slice is taken against a SENTINEL question, not the real one. Splitting on
# "Question:" directly is unsafe: a BBEH or LongBench problem statement can itself
# contain that word, and rpartition would then cut inside the question and inject
# half of it into the rulebook. Rendering the builder with a sentinel makes the
# boundary unambiguous whatever the row says.
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
    # drop the now-empty "Question:" header that preceded the sentinel
    return head[:head.rindex(_QUESTION_ANCHOR)].rstrip()


_RULES_HEAD = {
    "single": "Your mode is single. These are its execution rules.",
    "multi": "Your mode is multi. These are its execution rules.",
    "agentic": "Your mode is agentic. These are its execution rules.",
}
# The ONE sentence the injected rulebook may say about declaring a mode.
PREDECLARED_SENTENCE = ("The execution mode has already been declared. "
                        "Do not declare it again.")
_RULES_TAIL = (PREDECLARED_SENTENCE + "\n"
                 "Continue now, under these rules, on the question above.")

# ------------------------------------------- stripping the declaration rules --
#
# THE PROBLEM. mode_rules reuses the forced prompt verbatim, and a forced prompt
# tells the router to declare its mode -- because under forced routing it must.
# Injected after the mode is already declared, those lines contradict it.
# Measured in
# the rendered bodies:
#
#   single   L16 "+ <mode> appears EXACTLY ONCE, in your first response, ..."
#            L24 "One routing mode is available; declare it."
#            L27 "<mode>single</mode>"      (the mode-rule template slot)
#            L32 "<mode>single</mode>"      (inside the worked example)
#   multi    same three shapes, three <mode>multi</mode> lines
#   agentic  L18 "+ <mode>agentic</mode> appears EXACTLY ONCE, at the start ..."
#            L42 "<mode>agentic</mode>"     (inside the worked continuation)
#
# THE FIX IS LINE-ORIENTED AND CLOSED. Three patterns, each removing whole lines,
# and then an assertion that no line mentioning <mode> and no declare-instruction
# survives. A regex that rewrote sentences would be unauditable; deleting whole
# lines leaves a diff a human can read, and the removed lines are returned so the
# rollout row can record exactly what the router was NOT shown.
#
# ONLY the predeclared path calls this. Forced single/multi/agentic keep their own
# prompt byte-for-byte, so their grammar, executor and reward are untouched.
_DECL_BULLET = re.compile(r"^\s*\+\s*<mode.*EXACTLY ONCE", re.I)
_DECL_INSTRUCTION = re.compile(
    r"^\s*(One routing mode is available; declare it\.|"
    r"Declare exactly one of these (two|three) modes\.|"
    r"First,? declare|Choose exactly one mode)", re.I)
_STANDALONE_MODE_TAG = re.compile(r"^\s*<mode>\s*\w+\s*</mode>\s*$", re.I)
# what must NOT survive
_RESIDUAL_MODE = re.compile(r"<mode", re.I)
_RESIDUAL_DECLARE = re.compile(r"\bdeclare\b", re.I)


def strip_mode_declaration_rules(body):
    """(cleaned_body, [removed lines]) -- see the block comment above."""
    kept, removed = [], []
    for line in str(body).splitlines():
        if _DECL_BULLET.match(line) or _DECL_INSTRUCTION.match(line) \
                or _STANDALONE_MODE_TAG.match(line):
            removed.append(line)
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    # collapse the blank-line pairs the deletions leave behind, so the rulebook
    # does not acquire ragged spacing that a diff would flag as a wording change
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
    """The environment's turn-2 injection for `mode`.

    `predeclared=True` (the free-routing default) also deletes every line that
    tells the router to declare a mode -- see strip_mode_declaration_rules. Pass
    False to get the untouched forced rulebook.

    single/multi come from mode_prompts.build_sm_prompt (render_prompt with that one
    mode's action space, explicit_v2 with the compact fallback, plus ROUTER_RULES
    and the mode's suffix). agentic comes from route_syntax.build (the V2 unified
    route grammar with the four roles and MAX_WORKERS=6). Both are the callers'
    own builders, imported rather than reimplemented, so this function adds
    exactly one framing sentence at each end and no rule of its own.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "agentic":
        import route_syntax
        ids = list(worker_ids or ())
        if not ids:
            raise ValueError("agentic rules need the worker id list")
        # routing_state=True also renders the three clauses the static text only
        # implies; one of them names the "Valid refs" line of the dynamic header,
        # so the two are switched together and never separately.
        # The worked example's model id is taken from the LIVE pool. Left at a
        # fixed default it can name a worker with no enabled channel -- the
        # route resolves to NO_ENABLED_CHANNEL, becomes an empty observation,
        # and the example is the strongest prior the policy has. Only the id
        # moves; the example text, the roles and the grammar are untouched.
        # max_workers=None -> route_syntax's own default, so the rendered
        # text is unchanged for every caller that does not reconfigure the
        # budget. Only agentic has a worker budget to state.
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
        # Prompt-level ONLY. The router still writes the whole <route model="...">
        # itself and the reward is defined over the tokens it emitted; nothing
        # rewrites an action afterwards (contrast ROUTER_FORCE_WORKERS, which
        # base_loop refuses under training for exactly that reason).
        # Sitting inside `body` puts it in the same mask-0 observation channel as
        # the rest of the rules and inside the span the reward manager strips.
        from worker_explore_schedule import FORCED_LINE   # noqa: PLC0415
        body = f"{body}\n\n{FORCED_LINE.format(worker=forced_worker)}"
    text = f"{_RULES_HEAD[mode]}\n\n{body}\n\n{_RULES_TAIL}"
    return (text, removed) if return_removed else text


def assert_no_leakage(prompt_text, sample):
    """The selector prompt may contain the question and nothing else about the row.

    Checks the row's own labels do not appear: dataset/domain/pool names, the
    gold strings, the scorer name, and any difficulty/topic metadata value. This
    is what stops a free-routing run from becoming a dataset-identity classifier
    by accident.
    """
    # The QUESTION is removed before checking. It is the one thing the selector
    # is allowed to contain, and leaving it in produces false positives that would
    # kill legitimate rows: meta.rfm_domain "MATH" is a substring of
    # "mathematics", "task" of "tasks", "topic" of "topical". What is under test
    # is the text THIS module wrote around the question.
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
        # a one/two-character gold ("B", "42") can occur inside a legitimate
        # question; only a gold long enough to be identifying is checked
        if len(g) >= 4 and g.lower() in low:
            leaks.append(f"gold={g[:40]}")
    meta = sample.get("meta") or {}
    for key in ("rfm_dataset", "rfm_domain", "rfm_pool", "tier", "difficulty",
                "task", "discipline", "topic"):
        value = meta.get(key)
        if value and len(str(value)) >= 4 and str(value).lower() in low:
            leaks.append(f"meta.{key}={value}")
    return leaks


# ------------------------------------------------------------ import guards --
#
# Symmetry and the absence of an example are properties this file must keep, so
# they are checked here rather than left to a reviewer.
for _m, _line in MODE_LINES.items():
    assert _line.startswith(f"{_m} \u2014 "), _m
    assert _line.endswith("."), _m
_WORDS = {m: len(l.split()) for m, l in MODE_LINES.items()}
assert max(_WORDS.values()) - min(_WORDS.values()) <= 5, \
    f"mode menu lines are not comparable in length: {_WORDS}"
for _tag in ("<route", "<layer", "<action", "<paradigm", "<observation", "<answer"):
    assert _tag not in SELECTOR_TEMPLATE, \
        f"selector template must contain no {_tag} -- it would be a worked example"
# "prefer" is allowed in exactly one place: the sentence that DENIES a default.
# Everything else on this list would smuggle a ranking into a symmetric menu.
_NO_DEFAULT_SENTENCE = "No mode is preferred by default."
assert _NO_DEFAULT_SENTENCE in SELECTOR_TEMPLATE
_body_without_denial = SELECTOR_TEMPLATE.replace(_NO_DEFAULT_SENTENCE, " ").lower()
for _bad in ("simple", "cheap", "expensive", "powerful", "prefer", "usually",
             "recommended", "default", "best for", "complex", "advanced"):
    assert _bad not in _body_without_denial, \
        f"selector template hints a default via {_bad!r}"
# exactly one filled <mode> per mode, inside the declaration enumeration
assert SELECTOR_TEMPLATE.count("{declaration}") == 1
# and the template must not carry the earlier "Choose exactly ONE mode"
# wording, which read as an instruction about the CHOICE rather than the reply
assert "Choose exactly ONE mode" not in SELECTOR_TEMPLATE
