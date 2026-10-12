"""Route-family surface syntax for the adaptive-layer agentic grammar."""
import re

from actions import VIOLATIONS

MAX_WORKERS = 6

_ROUTE = re.compile(r"<route\b([^>]*)>(.*?)</route>", re.S | re.I)
_ATTR = re.compile(r"""(model|role|refs)\s*=\s*["']([^"']*)["']""", re.I)
_MODE = re.compile(r"<mode>\s*(.*?)\s*</mode>", re.S | re.I)

_SKIN = {
    "action_missing": "your message contained no <route> tag",
    "action_malformed": "a <route> tag was not closed or could not be read",
    "role_missing": 'the role="..." attribute was absent',
    "role_invalid": 'role="..." must be exactly one of planner, executor, '
                    "verifier, summarizer",
    "model_missing": 'the model="..." attribute was absent',
    "model_invalid": 'model="..." must be copied character for character from '
                     "the list of available worker ids",
    "instruction_missing": "the route body (the worker's task) was empty",
    "ref_unknown": 'refs="..." named a node that does not exist yet',
    "ref_same_layer": "routes sent in ONE message run AT THE SAME TIME, so they "
                      "cannot reference each other. If you meant them as "
                      "consecutive steps, send only the FIRST one now and the "
                      "next after you see its observation",
    "summarizer_not_alone": "the summarizer must be sent on its own, not "
                            "alongside other routes",
    "duplicate_in_layer": "two routes in the message are identical; that is one "
                          "call issued twice, not a fan-out",
    "mode_missing": "declare <mode>agentic</mode> once at the start of your "
                    "first message",
}

_CUE_HEAD = "Your previous message was not executed: "
_CUE_TAIL = (" Emit one corrected <route> tag (or several for a parallel "
             "layer). Do not repeat a route you have already sent.")


def parse_routes(text, expect_mode=False):
    """([action, ...], [codes]) -- every <route> in the message is ONE LAYER."""
    text = str(text or "")
    errors = []
    if expect_mode:
        m = _MODE.search(text)
        if not m or m.group(1).strip().lower() != "agentic":
            errors.append("mode_missing")
    matches = _ROUTE.findall(text)
    if not matches:
        if re.search(r"<route\b", text, re.I):
            return [], sorted(set(errors + ["action_malformed"]))
        return [], sorted(set(errors + ["action_missing"]))
    layer = []
    for attrs_text, body in matches:
        got = {name.lower(): value.strip()
               for name, value in _ATTR.findall(attrs_text)}
        if "role" not in got:
            errors.append("role_missing")
        if not got.get("model"):
            errors.append("model_missing")
        if not body.strip():
            errors.append("instruction_missing")
        refs = [r.strip() for r in got.get("refs", "").replace(";", ",").split(",")
                if r.strip()]
        layer.append({"role": got.get("role", "").lower(),
                      "model": got.get("model", ""),
                      "refs": refs, "instruction": body.strip()})
    return layer, sorted(set(errors))


def violation_cue(codes, state=None, repeated=False, max_workers=MAX_WORKERS):
    """Same state-carrying cue as actions.violation_cue, in route wording."""
    reasons = []
    for code in codes:
        text = _SKIN.get(code) or VIOLATIONS.get(code)
        if text and text not in reasons:
            reasons.append(text)
    if not reasons:
        reasons = ["it did not follow the route format"]
    cue = _CUE_HEAD + "; ".join(reasons) + "."
    if state is not None:
        nodes = state.get("nodes") or []
        if nodes:
            listed = ", ".join(f'{n["id"]} ({n["role"]})' for n in nodes)
            cue += f" Existing nodes you may reference: {listed}."
        else:
            cue += (" No worker has run yet, so NO node ids exist: your next "
                    'route must use refs="".')
        left = max_workers - len(nodes)
        cue += f" Calls remaining: {left}."
        if left == 1:
            cue += " That last call must be the summarizer."
    if repeated:
        cue += (" You have now sent this same route twice; it will keep being "
                "rejected. Send a DIFFERENT route that satisfies the rule above.")
    return cue + _CUE_TAIL


TEMPLATE = """Answer the question at the end by coordinating a team of workers. You do not solve any part of the question yourself: workers do the work, and your own reasoning stays out of the answer.

## Allowed model IDs

Copy exactly one model ID from this list. Do not invent, rename, combine, or normalize model IDs. Every worker call must name one id copied character for character, selected from [{model_list}].
{worker_catalog}
## Roles

Each call gives one worker model one role:
{roles}

## Global grammar

    + <mode>agentic</mode> appears EXACTLY ONCE, at the start of your first response, and never again.
    + A worker call is <route model="MODEL_ID" role="ROLE" refs="IDS">CONCRETE TASK</route>. The task must be concrete, non-empty, never a placeholder.
    + refs lists the ids of earlier outputs this worker needs (refs="" for the first layer). The environment splices the referenced outputs into the worker's message, so never copy an earlier output into your task text. A route may reference nodes from EARLIER layers only, never a sibling sent in the same message.
    + Several <route> tags in ONE message form one LAYER: they run at the same time and CANNOT see each other's output. Use a layer when you want independent work on the same thing. One route is simply a layer of one.
    + Worker replies come back as <observation id="n1" role="...">...</observation>, one per route. The ENVIRONMENT writes those. Never write, invent or paraphrase one yourself.
    + Every worker already receives the original question; your task text says what THIS worker should do with it.
    + After the summarizer's observation arrives, finish with <answer>...</answer> as the last thing you write.

## Rules

1. You may make at most {max_workers} worker calls in total.
2. The LAST call must be the summarizer.
3. planner and verifier may never be the last call.
4. A verifier must reference at least one completed worker output.
5. A summarizer must reference at least one completed worker output.
6. A summarizer must be sent alone, not alongside other routes.
7. Do not put your own answer in a task, and do not ask a worker merely to confirm or agree with an answer. A worker is there to do work.
8. Your final <answer> comes after the summarizer has run, and only then.
{state_rules}
You choose how many workers to use and in what arrangement. Using fewer is allowed if the question does not need more.

## Example -- structure only, not a recommendation

Your FIRST message contains ONLY your first layer of routes, e.g.:

<mode>agentic</mode>
<route model="{example_model}" role="executor" refs="">solve it your own way</route>
<route model="{second_model}" role="executor" refs="">solve it independently, from first principles</route>

Then STOP and wait. The environment will reply with one
<observation id="n1" role="executor">...</observation> per route; only after
reading them do you send your next layer (for example a summarizer with
refs="n1,n2"), and only after the summarizer's observation arrives do you
write <answer>...</answer>.

Question: {question}
"""

PROMPT_VERSION = "route_v2.1"


_STATE_RULES = """9. Each message contains exactly one layer.
10. A route may reference only node IDs listed in Valid refs. When Valid refs is none, every route must use refs="".
11. After sending a layer, stop and wait for the worker observations."""


def routing_state(action_state, max_workers=MAX_WORKERS):
    """The dynamic header the environment writes before each router action."""
    nodes = action_state.get("nodes") or []
    layers = action_state.get("layers") or []
    ids = [str(n["id"]) for n in nodes]
    listed = ", ".join(ids) if ids else "none"
    return ("Routing state:\n"
            f"Current layer: {len(layers) + 1}\n"
            f"Completed node IDs: {listed}\n"
            f"Valid refs: {listed}\n"
            f"Worker calls remaining: {max(0, max_workers - len(nodes))}\n"
            f"Summarizer completed: "
            f"{'yes' if action_state.get('finished') else 'no'}")


def build(question, worker_ids, example_model="qwen3_8b",
          routing_state_rules=False, max_workers=None):
    """Render the rulebook; `max_workers=None` uses the module default."""
    from roles import ROLE_MENU                                 # noqa: PLC0415
    second = "llama31_8b"
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import worker_alias as _wa      # noqa: PLC0415
    catalog = ""
    if _wa.enabled():
        example_model = _wa.EXAMPLE_PRIMARY
        second = _wa.EXAMPLE_SECONDARY
        catalog = "\n" + _wa.catalog_block(worker_ids) + "\n"
    roles = "\n".join(f"  {name}: {desc}" for name, desc in ROLE_MENU.items())
    return TEMPLATE.format(model_list=", ".join(worker_ids), roles=roles,
                           worker_catalog=catalog,
                           max_workers=(MAX_WORKERS if max_workers is None
                                        else int(max_workers)),
                           example_model=example_model, second_model=second,
                           state_rules=(_STATE_RULES if routing_state_rules
                                        else ""),
                           question=question.strip())
