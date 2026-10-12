"""The router's action grammar, and every rule the environment enforces."""
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
    """Name the broken rule, and state the facts needed to fix it."""
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
    """([action, ...], [codes]) -- every <action> block in the message is ONE LAYER."""
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
    """One <action> body -> (action, codes)."""
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
    """Whole-layer rule check. Returns (codes, per_action_codes)."""
    from roles import TERMINAL_ROLE                             # noqa: PLC0415
    if state["finished"]:
        return ["already_finished"], []
    if not layer:
        return ["action_missing"], []

    codes, per = [], []
    for action in layer:
        errs = validate(action, state, available_models, max_workers=max_workers)
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
