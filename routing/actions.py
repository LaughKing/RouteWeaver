"""The router's action grammar, and every rule the environment enforces.

One or more actions per turn. Every <action> in a message forms ONE LAYER whose
members run blind to each other, so a parallel fan-out is expressible:

    <action>
      <role>executor</role>
      <model>worker_3</model>
      <refs>n1,n2</refs>
      <instruction>...</instruction>
    </action>

The environment answers with

    <observation id="n3" role="executor">...</observation>

This surface states only role, model, refs and instruction. It names no
paradigm: the topology is whatever the layers the router writes turn out to be,
not one of a fixed list it is handed.

RULES, and why each is a rule rather than a hope:

  R1  role must be one of the four
  R2  model must be a worker id that has an enabled channel -- otherwise the call
      returns an empty observation and the trajectory silently degrades
  R3  refs must name nodes that already exist
  R4  at most MAX_WORKERS worker calls
  R5  the trajectory must END with summarizer
  R6  planner and verifier may not be the final node (implied by R5, checked
      separately so a violation is legible in the record)
  R7  (retired) the revision-loop count cap. max_workers and
      max_action_turns already bound how much re-checking fits.
  R8  the last remaining call must be the summarizer, so R5 stays satisfiable
  R9  a layer's refs may only name nodes from EARLIER layers. Members of one
      layer are independent by definition; letting them reference each other
      would make the layer a chain wearing a layer's name.
  R10 a summarizer is always a layer of one -- it terminates the trajectory, so
      there is nothing for a sibling to return into.

A violation is NOT executed. The router is told which rule it broke, by name,
and re-prompted. A generic "invalid action" at temperature 0 makes the model
re-emit the same text and burn the turn budget.
"""
import re

MAX_WORKERS = 5
MAX_REVISION_LOOPS = 1

_ACTION = re.compile(r"<action>(.*?)</action>", re.S | re.I)
_FIELD = {name: re.compile(rf"<{name}>(.*?)</{name}>", re.S | re.I)
          for name in ("role", "model", "refs", "instruction")}

VIOLATIONS = {
    "action_missing": "your message contained no <action> block",
    "action_malformed": "the <action> block was not closed or could not be read",
    "role_missing": "<role> was absent",
    "role_invalid": "<role> must be exactly one of planner, executor, verifier, summarizer",
    "model_missing": "<model> was absent",
    "model_invalid": "<model> must be copied character for character from the "
                     "list of available worker ids",
    "instruction_missing": "<instruction> was absent or empty",
    "ref_unknown": "<refs> named a node that does not exist yet",
    "ref_self": "<refs> may only name EARLIER nodes",
    "budget_exhausted": "the worker-call budget is used up",
    "must_summarize_now": "only one call remains, so it must be the summarizer",
    "summarizer_needs_refs": "the summarizer must reference the earlier outputs "
                             "it is reconciling",
    "verifier_needs_refs": "a verifier must reference the output it is auditing",
    "already_finished": "the summarizer has already run; the trajectory is over",
    "layer_exceeds_budget": "that layer would need more calls than remain",
    # This is the router's commonest instinct -- writing the whole plan as four
    # consecutive actions -- so the message has to teach the distinction, not just
    # refuse. Under the one-action grammar the first block was executed and the
    # rest dropped; under layers that same message is a fan-out whose members
    # reference each other, which is a chain wearing a layer's name.
    "ref_same_layer": "actions sent in ONE message run AT THE SAME TIME, so they "
                      "cannot reference each other. If you meant them as "
                      "consecutive steps, send only the FIRST one now and the "
                      "next after you see its observation",
    "summarizer_not_alone": "the summarizer must be sent on its own, not alongside "
                            "other actions",
    "duplicate_in_layer": "two actions in the layer are identical; that is one "
                          "call issued twice, not a fan-out",
}

_CUE_HEAD = "Your previous action was not executed: "
_CUE_TAIL = (" Emit one corrected <action> block. Do not repeat an action you "
             "have already sent.")


def violation_cue(codes, state=None, repeated=False, max_workers=MAX_WORKERS):
    """Name the broken rule, and state the facts needed to fix it.

    Naming the rule is not enough on its own. At temperature 0 the router
    re-emitted an identical rejected action on every one of 12 turns, because
    "your refs do not exist" does not say WHICH ids do. So the cue carries the
    node list -- or its absence -- and on a verbatim repeat it spells out the
    legal move instead of restating the complaint.
    """
    reasons = []
    for code in codes:
        text = VIOLATIONS.get(code)
        if text and text not in reasons:
            reasons.append(text)
    if not reasons:
        reasons = ["it did not follow the action format"]
    cue = _CUE_HEAD + "; ".join(reasons) + "."

    if state is not None:
        nodes = state.get("nodes") or []
        if nodes:
            listed = ", ".join(f'{n["id"]} ({n["role"]})' for n in nodes)
            cue += f" Existing nodes you may reference: {listed}."
        else:
            cue += (" No worker has run yet, so NO node ids exist: your next "
                    "action must leave <refs> empty.")
        left = max_workers - len(nodes)
        cue += f" Calls remaining: {left}."
        if left == 1:
            cue += " That last call must be the summarizer."

    if repeated:
        cue += (" You have now sent this same action twice; it will keep being "
                "rejected. Send a DIFFERENT action that satisfies the rule above.")
    return cue + _CUE_TAIL


def parse_actions(text):
    """([action, ...], [codes]) -- every <action> block in the message is ONE LAYER.

    Several actions in one message run as a parallel layer whose members cannot
    see each other's output, and that expressiveness is the whole point of this
    version. The previous grammar read one action per turn, so the router could
    only build CHAINS -- and the one topology family that has ever shown a
    positive effect in this project, independent solvers plus an aggregation step,
    was not expressible at all. The consequence was measurable: 71% of chain-only
    trajectories collapsed onto a single four-role template, the verifier never
    once returned FAIL, and the summarizer reproduced its executor's answer
    verbatim in 12 of 14 cases. A chain of opinions adds no information.

    A layer of one is the ordinary sequential case, so nothing is given up.
    """
    blocks = _ACTION.findall(str(text or ""))
    if not blocks:
        if re.search(r"<action>", str(text or ""), re.I):
            return [], ["action_malformed"]
        return [], ["action_missing"]
    layer, errors = [], []
    for body in blocks:
        action, errs = _parse_one(body)
        layer.append(action)
        errors.extend(errs)
    return layer, sorted(set(errors))


def parse_action(text):
    """The first action only; the single-action rule tests use this."""
    layer, errors = parse_actions(text)
    return (layer[0] if layer else None), errors


def _parse_one(body):
    """One <action> body -> (action, codes).

    """
    got = {}
    for name, pattern in _FIELD.items():
        found = pattern.findall(body)
        got[name] = found[-1].strip() if found else None
    errors = []
    role = (got["role"] or "").lower().strip()
    if not got["role"]:
        errors.append("role_missing")
    if not got["model"]:
        errors.append("model_missing")
    if not (got["instruction"] or "").strip():
        errors.append("instruction_missing")
    refs = [r.strip() for r in (got["refs"] or "").replace(";", ",").split(",")
            if r.strip()]
    return {"role": role, "model": (got["model"] or "").strip(),
            "refs": refs, "instruction": (got["instruction"] or "").strip()}, errors


def validate(action, state, available_models, max_workers=MAX_WORKERS):
    """Rule check against the trajectory so far. Returns violation codes."""
    from roles import NON_TERMINAL_ROLES, ROLES, TERMINAL_ROLE  # noqa: PLC0415
    errors = []
    if state["finished"]:
        return ["already_finished"]
    if action["role"] not in ROLES:
        errors.append("role_invalid")
    if action["model"] not in available_models:
        errors.append("model_invalid")

    used = len(state["nodes"])
    remaining = max_workers - used
    if remaining <= 0:
        errors.append("budget_exhausted")
    elif remaining == 1 and action["role"] != TERMINAL_ROLE:
        # R8: the summarizer is mandatory, so the last slot is reserved for it.
        errors.append("must_summarize_now")

    known = {node["id"] for node in state["nodes"]}
    for ref in action["refs"]:
        if ref not in known:
            errors.append("ref_unknown")
            break

    if action["role"] == TERMINAL_ROLE and state["nodes"] and not action["refs"]:
        errors.append("summarizer_needs_refs")
    if action["role"] == "verifier" and not action["refs"]:
        errors.append("verifier_needs_refs")
    # planner_has_refs REMOVED. Its premise -- "the planner runs
    # first" -- was never enforced by this grammar: nothing orders the roles, so
    # a planner may legally sit in any layer, and 7 of 39 agentic rejections on
    # the max_workers=8 run were mid-trajectory planners re-planning off verifier
    # feedback while referencing only completed nodes. Ref VALIDITY is unchanged:
    # ref_unknown and ref_same_layer below still reject unknown, future and
    # same-layer ids for every role, planner included.

    # revision_loop_exceeded REMOVED. Total complexity is bounded by
    # max_workers (8, last slot reserved for the summarizer) and max_action_turns
    # (10): a revise cycle costs >=2 calls, so at most 3 can fit before
    # layer_exceeds_budget / must_summarize_now fires. The separate count cap was
    # a second bound on an already-bounded quantity and cost 11 of 39 rejections,
    # all of them structurally sound execute->verify->revise loops with budget to
    # spare. state["revision_loops"] is still COUNTED and reported; only the cap
    # is gone.

    if action["role"] in NON_TERMINAL_ROLES and remaining == 1:
        errors.append("must_summarize_now")
    return sorted(set(errors))


def _is_revision(action, state):
    """An executor that references a verifier is the revision loop."""
    by_id = {node["id"]: node for node in state["nodes"]}
    return any(by_id.get(ref, {}).get("role") == "verifier"
               for ref in action["refs"])


def is_revision(action, state):
    return _is_revision(action, state)


def validate_layer(layer, state, available_models, max_workers=MAX_WORKERS):
    """Whole-layer rule check. Returns (codes, per_action_codes).

    A layer is accepted or rejected as a unit: executing half a fan-out would
    give the aggregator a lopsided view that the router never asked for.
    """
    from roles import TERMINAL_ROLE                             # noqa: PLC0415
    if state["finished"]:
        return ["already_finished"], []
    if not layer:
        return ["action_missing"], []

    codes, per = [], []
    for action in layer:
        errs = validate(action, state, available_models, max_workers=max_workers)
        # budget/terminal rules are layer-level; drop the per-action verdicts that
        # only make sense for a layer of one and re-derive them below
        errs = [e for e in errs if e not in
                ("budget_exhausted", "must_summarize_now")]
        per.append(errs)
        codes.extend(errs)

    used = len(state["nodes"])
    remaining = max_workers - used
    has_terminal = any(a["role"] == TERMINAL_ROLE for a in layer)
    if len(layer) > remaining:
        codes.append("layer_exceeds_budget")
    elif not has_terminal and remaining - len(layer) < 1:
        # R8 in layer form: something must be left for the mandatory summarizer
        codes.append("must_summarize_now")
    if has_terminal and len(layer) > 1:
        codes.append("summarizer_not_alone")

    known = {n["id"] for n in state["nodes"]}
    planned = {f"n{used + i + 1}" for i in range(len(layer))}
    for action in layer:
        if any(ref in planned for ref in action["refs"]):
            codes.append("ref_same_layer")
        elif any(ref not in known for ref in action["refs"]):
            codes.append("ref_unknown")

    seen = set()
    for action in layer:
        key = (action["role"], action["model"], action["instruction"])
        if key in seen:
            codes.append("duplicate_in_layer")
        seen.add(key)
    return sorted(set(codes)), per
