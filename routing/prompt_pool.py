"""Prompt templates for the Unified Router grammar chain.

The prompt teaches THREE routing modes:

    single    <mode>single</mode> + exactly one <route model="...">
    multi     <mode>multi</mode>  + 2-4 sequential <route model="...">
    agentic   <mode>agentic</mode> + <paradigm> + one <layer> per turn, each
              layer holding one or more <route id= model= role= refs=>

Datasets must be rebuilt (data/build_*.py) after any edit here -- prompts are
baked into the parquet at build time, which is also
why PROMPT_PROTOCOL_VERSION below is stamped into every row the generators
write: a parquet whose baked prompt teaches an older grammar can then be
recognised instead of silently training against a parser that moved on.

SINGLE SOURCE OF TRUTH. Nothing about the protocol is retyped here:

  - the worker ids offered come from route_grammar.ACTIVE_WORKER_POOL;
  - the per-paradigm layer/role/refs rules are RENDERED from
    route_grammar.PARADIGM_SPECS by _render_paradigms(), the same table
    validate_agentic_layer() enforces at rollout and reward time.

A prompt that documented its own copy of those rules would drift from the
validator, and the drift arrives as an unexplainable -1 on trajectories the
policy was explicitly taught to produce.

Worker descriptions follow the upstream Router-R1 convention: model-card style
capability prose, no benchmark numbers and no prices. Cost is deliberately NOT
stated here -- the router is meant to learn the cost/quality tradeoff from the
reward's cost term, not from a prompt prior. (Reference prices still live in
routing/worker_registry.py, which is what the reward reads.)
Worker ids must stay in sync with that registry; naming a worker here that the
registry does not know turns into a format violation at rollout time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_grammar import (  # noqa: E402
    FULL_ACTION_SPACE, parse_allowed_modes,
    ACTIVE_WORKER_POOL, AGENTIC_PROTOCOL_VERSION, PARADIGM_SPECS)
# ANONYMOUS DIRECTORY. The single catalog source; inert when
# ROUTEWEAVER_WORKER_ALIAS is unset. Imported here rather than lower down because
# prompt_pool_fingerprint() reads CATALOG_VERSION.
import worker_alias as _worker_alias  # noqa: E402

# The grammar version these templates teach. Stamped into every generated row
# so a dataset and the parser that loads it can be checked against each other.
PROMPT_PROTOCOL_VERSION = AGENTIC_PROTOCOL_VERSION


def prompt_pool_fingerprint(allowed_modes=None, variant='compact',
                            allowed_paradigms=None) -> str:
    """Is a baked prompt still the prompt this code would render?

    This is the DATASET gate. It covers exactly what ends up in the row: the
    protocol version, the offered worker ids in order, the descriptions those
    ids are rendered with, the stage's action space, and the template text
    itself -- which carries the mode rules, the paradigm specs and the examples,
    so a change to any of them moves this hash without needing to be enumerated.

    What it deliberately does NOT cover is worker pricing, provider names or
    any prose that never reaches a prompt. Those cannot make a baked prompt
    wrong, and hashing them once meant a reworded pricing_source invalidated
    six datasets.

    Conversely it closes the two holes the registry hash had: rewording a
    WORKER_DESCRIPTION, or dropping a worker from ACTIVE_WORKER_POOL, both
    change every prompt and neither moved the old fingerprint.
    """
    import hashlib
    modes = tuple(parse_allowed_modes(allowed_modes))
    payload = (
        'prompt-pool-v1',
        PROMPT_PROTOCOL_VERSION,
        tuple(ACTIVE_WORKER_POOL),
        tuple((w, WORKER_DESCRIPTIONS[w]) for w in ACTIVE_WORKER_POOL),
        modes,
        # mode rules, paradigms and examples verbatim. A variant restates the
        # same rules differently, so it must move the hash -- but the DEFAULT
        # payload stays byte-identical to the pre-variant one, or adding this
        # parameter would have invalidated every frozen dataset built before it.
        _apply_variant(_stage_body(modes), variant, allowed_paradigms, modes),
    ) + ((variant,) if variant != 'compact' else ()) \
      + ((tuple(allowed_paradigms),) if allowed_paradigms else ()) \
      + ((_worker_alias.CATALOG_VERSION,) if _worker_alias.enabled() else ())
    # CATALOG VERSION, APPENDED not inserted, and only when the anonymous
    # directory is on. Without it a cached template or a built dataset could hash
    # identically across two different catalogs and be silently interchangeable.
    # Appending keeps the flag-off payload byte-identical to the pre-alias one --
    # inserting a None element instead moved the default hash (measured
    # 8b5cff4439dacbb5 -> 7dd0598739e99a1c) and would have invalidated every
    # frozen dataset's recorded fingerprint.
    return hashlib.sha256(repr(payload).encode('utf-8')).hexdigest()[:16]


def render_prompt(question, model='qwen', allowed_modes=None, variant='compact',
                  allowed_paradigms=None, worker_ids=None):
    """The single place a router prompt is produced.

    allowed_modes selects the curriculum stage's action space; None keeps the
    full space, so every pre-curriculum caller is unaffected.
    """
    modes = parse_allowed_modes(allowed_modes)
    if (modes == FULL_ACTION_SPACE and variant == 'compact'
            and allowed_paradigms is None and worker_ids is None):
        template = PROMPT_TEMPLATE_QWEN if model == 'qwen' else PROMPT_TEMPLATE_LLAMA
    else:
        template = _build_template(modes, variant, allowed_paradigms, worker_ids)
    return template.format_map({'question': question})


# ------------------------------------------------------- rendered protocol --

def _render_paradigms(allowed=None):
    """The four paradigms as ONE compact line each, straight out of
    PARADIGM_SPECS.

        ensemble: L1 solver x2-3 | L2 aggregator x1 refs=L1

    The prose form this replaced cost ~230 tokens for the same four facts per
    layer (role, route count, which layers to reference). The prompt budget is
    the binding constraint -- the router base takes 2048 -- so the protocol is
    stated in the densest form that is still unambiguous, and the legend line
    above the table carries the meaning of "xN" and "refs=".
    """
    lines = []
    for name, specs in PARADIGM_SPECS.items():
        if allowed is not None and name not in allowed:
            continue
        parts = []
        for index, spec in enumerate(specs, start=1):
            count = (str(spec.min_routes) if spec.min_routes == spec.max_routes
                     else f'{spec.min_routes}-{spec.max_routes}')
            part = f'L{index} {spec.role} x{count}'
            if spec.refs_from:
                part += ' refs=' + ','.join(f'L{i}' for i in spec.refs_from)
            parts.append(part)
        lines.append(f'{name}: ' + ' | '.join(parts))
    return '\n'.join(lines)


# One line per worker: enough to rank the workers against a sub-question, and
# no more. Under the anonymous directory these are replaced wholesale by
# worker_alias.DESCRIPTIONS, so the router reads capability text and never a
# model name.
WORKER_DESCRIPTIONS = {
    'qwen3_8b':
        "Qwen3-8B. Instruction-tuned text model for general text generation, "
        "question answering and instruction following.",
    'llama31_8b':
        "Llama-3.1-8B-Instruct. Instruction-tuned text model for general text "
        "generation, question answering and instruction following.",
    'qwen3_30b_a3b':
        "Qwen3-30B-A3B-Instruct. Instruction-tuned text model for general text "
        "generation, question answering and instruction following.",
    'gemma3_27b_it':
        "Gemma-3-27B-IT. Instruction-tuned text model for general text "
        "generation, question answering and instruction following.",
    'llama33_70b':
        "Llama-3.3-70B-Instruct. Instruction-tuned text model for general text "
        "generation, question answering and instruction following.",
    'gemini25_flash_lite':
        "Gemini-2.5-Flash-Lite. Instruction-tuned text model for general text "
        "generation, question answering and instruction following.",
}

# ANONYMOUS DIRECTORY. Merged into the SAME dict rather than read
# only by the renderer, because WORKER_DESCRIPTIONS has other consumers --
# prompt_pool_fingerprint hashes it, and tests assert every offered id has an
# entry. One merge covers all of them; a renderer-local lookup covered only one
# and left the fingerprint raising KeyError. Default off.
if _worker_alias.enabled():
    WORKER_DESCRIPTIONS.update(_worker_alias.DESCRIPTIONS)


def _render_workers(worker_ids=None):
    # worker_ids: per-query DISPLAY ORDER (agent_loops/worker_explore_schedule.display_order).
    # Rendering only -- validate_route resolves by registry lookup, so no parser,
    # dispatcher or reward path reads a position. None keeps the canonical order.
    #
    # ANONYMOUS DIRECTORY: under ROUTEWEAVER_WORKER_ALIAS=1 the descriptions come
    # from worker_alias, and one sentence states the output budget the
    # dispatcher actually enforces. It is stated ONCE rather than per worker
    # because the enforced cap is currently the same 4096 for all seven (see
    # worker_alias's docstring); repeating an identical number seven times would
    # add ~70 tokens of zero-information text to every single/multi prompt.
    ids = list(worker_ids or ACTIVE_WORKER_POOL)
    if _worker_alias.enabled():
        # ONE source for all three modes: route_syntax renders the same call for
        # agentic, so the catalog cannot drift into three maintained copies.
        return _worker_alias.catalog_block(ids)
    return '\n'.join(f'{worker_id}: {WORKER_DESCRIPTIONS[worker_id]}'
                     for worker_id in ids)


_TEMPLATE = """
Answer the question at the end.

## Allowed model IDs

Copy exactly one model ID from this list. Do not invent, rename, combine, or normalize model IDs. Every worker call must name one id copied character for character, selected from [WORKER_IDS].

WORKER_DESCRIPTIONS_BLOCK

## Global grammar

    + <mode> appears EXACTLY ONCE, in your first response, before your first worker call, and never again.
    + A worker call is <route ...>CONCRETE TASK</route>. The task must be concrete, non-empty, never a placeholder, and never a copied worker description.
    + Worker replies come back as <observation>...</observation>. The ENVIRONMENT writes those. Never write, invent or paraphrase one yourself.
    + XML attributes use plain double quotes.
    + Finish with <answer>...</answer> as the last thing you write, and never answer without routing first.

## Mode rules

Declare exactly one of these three modes.

single -- exactly ONE worker call, in your first response; answer once its observation arrives.
<mode>single</mode>
<route model="MODEL_ID">CONCRETE TASK</route>

multi -- 2 to 4 worker calls in total, at most one <route> per turn, each chosen after reading the previous observation.
<mode>multi</mode>
<route model="MODEL_ID">CONCRETE TASK</route>

agentic -- workers run in parallel LAYERS. Exactly ONE complete <layer> per turn, opened and closed in the same turn; a layer may hold several routes.
<mode>agentic</mode>
<paradigm>PARADIGM</paradigm>
<layer index="1">
<route id="1-1" model="MODEL_ID" role="ROLE">CONCRETE TASK</route>
</layer>
<layer index="2">
<route id="2-1" model="MODEL_ID" role="ROLE" refs="1-1">CONCRETE TASK</route>
</layer>

    + id, model and role are required on every agentic route; refs is required from layer 2 on.
    + Layer indices start at 1 and grow by exactly 1. Route ids are "LAYER-POSITION" and are never reused.
    + refs may name ONLY ids from layers already executed and observed -- never the same layer, never itself, never an id that does not exist. The environment feeds the referenced output to the worker, so do not retype it.
    + id, role, refs, <paradigm> and <layer> are FORBIDDEN in single and multi.
    + Answer only after the paradigm's LAST layer has run and its observations have arrived.

## Paradigms

Pick one and follow its layer plan exactly. "xN" is how many routes that layer takes; "refs=L1,L2" means every route there must reference EVERY route id of those layers.

PARADIGM_SPECS_BLOCK

## Example -- single

<mode>single</mode>
<route model="MODEL_ID_A">Who created the constructed language Esperanto?</route>
<observation>Esperanto was created by L. L. Zamenhof.</observation>
<answer>L. L. Zamenhof</answer>

## Example -- agentic

<mode>agentic</mode>
<paradigm>ensemble</paradigm>
<layer index="1">
<route id="1-1" model="MODEL_ID_A" role="solver">Who created the constructed language Esperanto?</route>
<route id="1-2" model="MODEL_ID_B" role="solver">In which year was Esperanto first published?</route>
</layer>
<observation id="1-1">Esperanto was created by L. L. Zamenhof.</observation>
<observation id="1-2">Esperanto was first published in 1887.</observation>
<layer index="2">
<route id="2-1" model="MODEL_ID_C" role="aggregator" refs="1-1,1-2">Combine the creator and the publication year into one sentence.</route>
</layer>
<observation id="2-1">Esperanto was created by L. L. Zamenhof and first published in 1887.</observation>
<answer>L. L. Zamenhof, 1887</answer>

The examples above are ILLUSTRATIONS of the required structure, not recommendations -- choose the workers and the mode that fit the question below, and do not reuse the example question or answer.
Question: {question}
"""


# ------------------------------------------------------ curriculum staging --
#
# A stage prompt is the full prompt MINUS the modes it does not open. It is
# built by deleting delimited blocks rather than by assembling a stage-specific
# body, for one reason: the full-action-space path then performs no
# transformation at all, so it is byte-identical to what is already baked into
# data/router_v1_main by construction, not by a test that happens to pass.
#
# Showing a mode the stage forbids would be the worst of both worlds -- the
# policy spends tokens learning a syntax whose every use is rejected before
# dispatch, and the reward it gets back cannot distinguish "wrong mode" from
# "bad routing".

def _drop(text: str, start: str, end: str) -> str:
    """Delete [start, end) -- both anchors must be unique in the template."""
    for anchor in (start, end):
        assert text.count(anchor) == 1, f"ambiguous anchor {anchor!r}"
    begin = text.index(start)
    return text[:begin] + text[text.index(end, begin):]


_MODE_COUNT_WORD = {1: None, 2: 'two', 3: 'three'}


# Prompt VARIANTS. A variant changes how the same rules are stated, never what
# they are -- the validator is the only definition and is untouched. Each
# variant gets its own prompt_pool_fingerprint, so a dataset built with one can
# never be trained under the other.
PROMPT_VARIANTS = ('compact', 'explicit', 'explicit_v2')


def _insert_multi_examples(body: str) -> str:
    """Put the worked multi examples where a reader would look for them.

    Ordering matters only for readability -- single, then multi, then agentic,
    matching the mode rules above them. The three branches are the three shapes
    a stage body can have: one with an agentic example, one with only the
    single example, and one (multi in isolation) with no example section at all.
    """
    from explicit_multi import explicit_multi_section
    block = explicit_multi_section() + "\n"
    if '## Example -- agentic' in body:
        cut = body.index('## Example -- agentic')
    elif 'The examples above' in body:
        cut = body.index('The examples above')
    else:
        end = body.index('Question: {question}')
        cut = body.rfind('\n\n', 0, end) + 2
    return body[:cut] + block + body[cut:]


def _apply_variant(body: str, variant: str, allowed_paradigms=None,
                   allowed_modes=None) -> str:
    if variant == 'compact':
        # A single forced paradigm must not ship the template's worked example:
        # that example is an ENSEMBLE trajectory, and next to a rulebook that
        # names a different paradigm it is a contradiction, not teaching
        # material. The explicit path drops it (see the twin comment below);
        # left in, a policy told to use plan copies the example's
        # <paradigm>ensemble</paradigm> label essentially every time. Dropped
        # for ALL four
        # targets -- ensemble included -- so the four forced prompts stay
        # isomorphic: no target gets a worked example the others lack. The
        # trailing "The example above ..." sentence goes with it (unlike the
        # explicit path there is no replacement example for it to describe),
        # so the drop runs through to the Question anchor.
        if allowed_paradigms and len(allowed_paradigms) == 1 \
                and '## Example -- agentic' in body:
            return _drop(body, '## Example -- agentic', 'Question:')
        return body
    if variant == 'explicit_v2':
        # Restates BOTH taught surfaces: the agentic paradigms (as 'explicit'
        # does) and multi, which has never had a worked example in any variant.
        # Unlike 'explicit' it does not insist on a paradigm section -- it
        # applies wherever at least one of the two surfaces exists, which is
        # what lets a multi-only stage and the full-space prompt share the
        # same teaching material rather than each stage owning its own copy.
        touched = False
        if '## Paradigms' in body:
            body = _apply_variant(body, 'explicit', allowed_paradigms)
            touched = True
        if 'multi -- 2 to 4 worker calls' in body:
            body = _insert_multi_examples(body)
            touched = True
        if not touched:
            raise ValueError(
                f"variant {variant!r} does not apply to this action space: it "
                f"has neither a paradigm section nor a multi rule to restate.")
        return body
    if variant != 'explicit':
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    from explicit_paradigms import explicit_section  # noqa: F401
    # The template's own '## Example -- agentic' is an ENSEMBLE trajectory. Left
    # in place while a single non-ensemble paradigm is forced, the prompt would
    # order the router to use plan and then show it a worked ensemble -- and the
    # measurement would be of that contradiction rather than of whether plan can
    # be learned. explicit_section already carries a complete legal example for
    # whichever paradigm is forced, so the base one is dropped rather than
    # duplicated.
    if '## Paradigms' not in body:
        # A stage without agentic has no paradigm section, so there is nothing
        # a variant could restate: 'single' and 'single,multi' render the same
        # bytes either way. Returning the body unchanged would be worse than
        # this error -- the dataset would carry an 'explicit' fingerprint over
        # text identical to the compact one, so two fingerprints would name the
        # same prompt and the gate would reject a dataset for no visible reason.
        #
        # It also means the curriculum's first two stages are unaffected by
        # the variant choice.
        raise ValueError(
            f"variant {variant!r} does not apply to this action space: it has "
            f"no '## Paradigms' section to restate. Only stages that include "
            f"agentic have one.")
    start = body.index('## Paradigms')
    end = body.index('## Example --', start)
    body = body[:start] + explicit_section(allowed_paradigms) + body[end:]

    # The template's own '## Example -- agentic' is an ENSEMBLE trajectory.
    # Left in place while a single non-ensemble paradigm is forced, the prompt
    # would order the router to use plan and then show it a worked ensemble,
    # and the measurement would be of that contradiction rather than of whether
    # plan can be learned. explicit_section has just supplied a complete legal
    # example for the forced paradigm, so the base one is dropped rather than
    # duplicated -- and only NOW, because the swap above needs
    # '## Example --' as its end anchor.
    #
    # The agentic-only stage rewrites the trailing sentence to the singular, so
    # both spellings are accepted; hard-coding one of them silently skipped
    # this and left every forced prompt carrying the example it was replacing.
    if allowed_paradigms and len(allowed_paradigms) == 1 and '## Example -- agentic' in body:
        for tail in ('The examples above', 'The example above'):
            if tail in body:
                return _drop(body, '## Example -- agentic', tail)
        raise ValueError("cannot locate the end of the agentic example block")
    return body


def _stage_body(allowed_modes) -> str:
    """The template body for one action space. Full space returns it verbatim."""
    modes = tuple(allowed_modes)
    if modes == FULL_ACTION_SPACE:
        return _TEMPLATE

    body = _TEMPLATE
    if modes == ('agentic',):
        # Agentic in isolation. Not a prefix set, so it deletes from the FRONT:
        # both simpler mode blocks and their examples go, the paradigm table and
        # the agentic example stay. The grammar itself is untouched -- this
        # removes options, it does not soften rules.
        body = _drop(body, 'single -- exactly ONE worker call', 'agentic -- workers run in parallel')
        body = _drop(body, '## Example -- single', '## Example -- agentic')
        body = body.replace('Declare exactly one of these three modes.',
                            'One routing mode is available; declare it.')
        body = body.replace(
            'The examples above are ILLUSTRATIONS of the required structure, not '
            'recommendations -- choose the workers and the mode that fit the question '
            'below, and do not reuse the example question or answer.',
            'The example above is an ILLUSTRATION of the required structure, not a '
            'recommendation -- choose the workers that fit the question below, and do '
            'not reuse the example question or answer.')
        return body
    if modes == ('multi',):
        # Multi in isolation, by the same argument as ('agentic',): it deletes
        # from BOTH ends, so single cannot be chosen and agentic does not exist.
        # What remains is the multi rule line -- and, in the compact variant,
        # NO worked example, because the template has never contained one
        # (explicit_multi.py is what adds them).
        body = _drop(body, 'single -- exactly ONE worker call', 'multi -- 2 to 4 worker calls')
        body = _drop(body, 'agentic -- workers run in parallel', '## Paradigms')
        body = _drop(body, '## Paradigms', '## Example -- single')
        body = _drop(body, '## Example -- single', 'The examples above')
        body = body.replace('Declare exactly one of these three modes.',
                            'One routing mode is available; declare it.')
        body = body.replace(
            'The examples above are ILLUSTRATIONS of the required structure, not '
            'recommendations -- choose the workers and the mode that fit the question '
            'below, and do not reuse the example question or answer.',
            'Choose the workers that fit the question below.')
        return body
    if 'agentic' not in modes:
        # the mode block and its rules, the paradigm table, the worked example
        body = _drop(body, 'agentic -- workers run in parallel', '## Paradigms')
        body = _drop(body, '## Paradigms', '## Example -- single')
        body = _drop(body, '## Example -- agentic', 'The examples above')
    if 'multi' not in modes:
        # the legal sets are PREFIXES of _VALID_MODES, so no multi implies no
        # agentic: the agentic block is already gone and the next surviving
        # heading is the single example
        body = _drop(body, 'multi -- 2 to 4 worker calls', '## Example -- single')

    word = _MODE_COUNT_WORD[len(modes)]
    body = body.replace(
        'Declare exactly one of these three modes.',
        f'Declare exactly one of these {word} modes.' if word
        else 'One routing mode is available; declare it.')
    if len(modes) == 1:
        body = body.replace(
            'The examples above are ILLUSTRATIONS of the required structure, not '
            'recommendations -- choose the workers and the mode that fit the question '
            'below, and do not reuse the example question or answer.',
            'The example above is an ILLUSTRATION of the required structure, not a '
            'recommendation -- choose the worker that fits the question below, and do '
            'not reuse the example question or answer.')
    return body


def _build_template(allowed_modes=FULL_ACTION_SPACE, variant='compact',
                    allowed_paradigms=None, worker_ids=None):
    """One body, rendered from the shared protocol tables."""
    body = (_apply_variant(_stage_body(allowed_modes), variant, allowed_paradigms,
                           tuple(allowed_modes))
            .replace('WORKER_IDS', ', '.join(worker_ids or ACTIVE_WORKER_POOL))
            .replace('PARADIGM_SPECS_BLOCK', _render_paradigms(allowed_paradigms))
            .replace('WORKER_DESCRIPTIONS_BLOCK', _render_workers(worker_ids)))
    # WORKED-EXAMPLE IDS. The template's single and agentic examples carry the
    # literals MODEL_ID_A/B/C. Under the anonymous directory they are bound to
    # fixed alias anchors, so the examples demonstrate ids that are in the
    # allowed list and that validate_route() accepts. Without the flag the
    # literals are left exactly as they are -- the MODEL_ID_A regression in the
    # real-id path is a SEPARATE decision and is not silently fixed here.
    if _worker_alias.enabled():
        body = (body.replace('MODEL_ID_A', _worker_alias.EXAMPLE_PRIMARY)
                    .replace('MODEL_ID_B', _worker_alias.EXAMPLE_SECONDARY)
                    .replace('MODEL_ID_C', _worker_alias.EXAMPLE_TERTIARY))
    return body


# The two families currently share one body -- they have been byte-identical
# since the grammar chain replaced the legacy <search> templates. They stay
# separate names because every generator selects by model family.
PROMPT_TEMPLATE_QWEN = _build_template()
PROMPT_TEMPLATE_LLAMA = _build_template()
