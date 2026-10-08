"""Route-family surface syntax for the adaptive-layer agentic grammar.

SAME SEMANTICS AS actions.py, DIFFERENT SKIN. User-confirmed the
`<action>` block grammar diverged from the single/multi route-grammar family,
and a small model will later be trained on all modes at once -- a unified token
pattern trains better. So the agentic action space is re-expressed as:

    <mode>agentic</mode>                          (once, first message)
    <route model="ID" role="ROLE" refs="n1,n2">instruction</route>

Several <route> tags in one message form ONE LAYER, exactly as several
<action> blocks did. Parsing produces the SAME action dicts
({role, model, refs, instruction}), so actions.validate / validate_layer are
reused verbatim -- the rules live in one place regardless of skin.

Two settings apply to THIS grammar only, so the earlier grammar is
unaffected:
  * MAX_WORKERS = 6 (was 5)
  * after the summarizer's observation, the ROUTER writes its own final
    <answer>; a row scores correct if EITHER the summarizer's answer or the
    router's <answer> is correct (both are recorded separately)
"""
import re

from actions import VIOLATIONS

MAX_WORKERS = 6

_ROUTE = re.compile(r"<route\b([^>]*)>(.*?)</route>", re.S | re.I)
_ATTR = re.compile(r"""(model|role|refs)\s*=\s*["']([^"']*)["']""", re.I)
_MODE = re.compile(r"<mode>\s*(.*?)\s*</mode>", re.S | re.I)

# wording adjusted for the route skin; every code keeps its actions.py meaning
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

# v2.1: the v2.0 example was a full mock transcript -- routes,
# observations and answer inline, single/multi style. Most routers
# imitated it wholesale on turn 1, fabricating <observation> blocks and
# answering their own question; every such turn cost a violation cue. The old
# <action> prompt never did this because its observation syntax appeared under
# a "you will receive" narration, not inside an imitable conversation. This
# example now shows only the first MESSAGE and narrates the rest.
PROMPT_VERSION = "route_v2.1"


# The clauses that the static rules only imply. Added ONLY when the dynamic
# routing-state header is on, because clause 9 refers to it by name; with the
# header off, build() renders exactly the text it always did.
#
# Audit of the six candidate clauses against the existing text:
#   "routes in one turn are parallel and cannot reference one another"
#       -> already explicit, Global grammar bullet 2. Not repeated.
#   "a summarizer must be the only route in its layer"
#       -> already explicit, Rule 5. Not repeated.
#   "exactly one layer per turn"      -> only implied  -> clause 8
#   "refs must come from Valid refs"  -> new, names the header -> clause 9
#   "after a layer, stop and wait"    -> only in the example  -> clause 10
_STATE_RULES = """9. Each message contains exactly one layer.
10. A route may reference only node IDs listed in Valid refs. When Valid refs is none, every route must use refs="".
11. After sending a layer, stop and wait for the worker observations."""


def routing_state(action_state, max_workers=MAX_WORKERS):
    """The dynamic header the environment writes before each router action.

    Derived from the SAME action_state dict validate_layer is given, so what it
    calls a valid ref and what the validator accepts cannot drift. Nodes are
    appended to that dict only after their layer has actually run, which is what
    keeps the routes about to be written out of `Valid refs`.

    It reports state and nothing else: no question, no grammar, no worker
    output, no dataset or reward, no suggestion about which role or shape to
    use next, and nothing that depends on what the previous action got wrong.
    """
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
    """`max_workers=None` keeps the module default, so every existing caller
    renders byte-for-byte the same text. It exists because rule 1 of the
    rulebook states the budget as a NUMBER: when the runtime budget is
    reconfigured (router.max_workers_v2) and this stays pinned to the module
    constant, the rulebook tells the policy "at most 6" while the validator and
    the routing-state header both work to 8. That silent disagreement is worse
    than either budget on its own, so the number follows the configuration."""
    from roles import ROLE_MENU                                 # noqa: PLC0415
    # DECOUPLED, REAL IDS RESTORED. `second` used to be
    # drawn from worker_ids, so with a per-query display order the worked example
    # moved with the candidate order and demo anchoring could not be told apart
    # from position. The anchors stay FIXED (that decoupling is kept) but they are
    # real pool members again.
    #
    # WHY a PLACEHOLDER id here was wrong: every id demonstrated in a prompt
    # has to be one the validator accepts, because the policy copies its
    # examples. With MODEL_ID_A/B in the template the agentic rulebook carried
    # NO legal complete-route example while telling the policy "Copy exactly
    # one model ID from this list. Do not invent...", and the agentic grammar
    # violation rate roughly doubles against the same pool named for real.
    second = "llama31_8b"
    # ANONYMOUS DIRECTORY. Two changes, both gated:
    #   * the worked route's anchors become the fixed aliases, so the rulebook's
    #     only complete worked route names ids that are in the allowed list;
    #   * agentic renders THE SAME worker catalog single/multi render, from the
    #     same worker_alias.catalog_block(). Before this, agentic saw the id list
    #     and no descriptions at all (measured: 0 description lines against 7 for
    #     single/multi), so "one directory for three modes" was true of the ids
    #     and false of the descriptions.
    # Nothing else moves: roles, the global grammar, rules 1-8, the refs
    # constraints, the worker budget, the summarizer requirement, the termination
    # rule and the example's task texts are untouched.
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    import worker_alias as _wa      # noqa: PLC0415
    catalog = ""
    if _wa.enabled():
        example_model = _wa.EXAMPLE_PRIMARY
        second = _wa.EXAMPLE_SECONDARY
        # worker_ids is the per-query DISPLAY ORDER, same as prompt_pool gets.
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
