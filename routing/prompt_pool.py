"""Prompt templates for the Unified Router grammar chain."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_grammar import (  # noqa: E402
    FULL_ACTION_SPACE, parse_allowed_modes,
    ACTIVE_WORKER_POOL, AGENTIC_PROTOCOL_VERSION, PARADIGM_SPECS)
import worker_alias as _worker_alias  # noqa: E402

PROMPT_PROTOCOL_VERSION = AGENTIC_PROTOCOL_VERSION


def prompt_pool_fingerprint(allowed_modes=None, variant='compact',
                            allowed_paradigms=None) -> str:
    """Is a baked prompt still the prompt this code would render?"""
    import hashlib
    modes = tuple(parse_allowed_modes(allowed_modes))
    payload = (
        'prompt-pool-v1',
        PROMPT_PROTOCOL_VERSION,
        tuple(ACTIVE_WORKER_POOL),
        tuple((w, WORKER_DESCRIPTIONS[w]) for w in ACTIVE_WORKER_POOL),
        modes,
        _apply_variant(_stage_body(modes), variant, allowed_paradigms, modes),
    ) + ((variant,) if variant != 'compact' else ()) \
      + ((tuple(allowed_paradigms),) if allowed_paradigms else ()) \
      + ((_worker_alias.CATALOG_VERSION,) if _worker_alias.enabled() else ())
    return hashlib.sha256(repr(payload).encode('utf-8')).hexdigest()[:16]


def render_prompt(question, model='qwen', allowed_modes=None, variant='compact',
                  allowed_paradigms=None, worker_ids=None):
    """The single place a router prompt is produced."""
    modes = parse_allowed_modes(allowed_modes)
    if (modes == FULL_ACTION_SPACE and variant == 'compact'
            and allowed_paradigms is None and worker_ids is None):
        template = PROMPT_TEMPLATE_QWEN if model == 'qwen' else PROMPT_TEMPLATE_LLAMA
    else:
        template = _build_template(modes, variant, allowed_paradigms, worker_ids)
    return template.format_map({'question': question})


def _render_paradigms(allowed=None):
    """The four paradigms as ONE compact line each, straight out of
    PARADIGM_SPECS.
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

if _worker_alias.enabled():
    WORKER_DESCRIPTIONS.update(_worker_alias.DESCRIPTIONS)


def _render_workers(worker_ids=None):
    ids = list(worker_ids or ACTIVE_WORKER_POOL)
    if _worker_alias.enabled():
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


def _drop(text: str, start: str, end: str) -> str:
    """Delete [start, end) -- both anchors must be unique in the template."""
    for anchor in (start, end):
        assert text.count(anchor) == 1, f"ambiguous anchor {anchor!r}"
    begin = text.index(start)
    return text[:begin] + text[text.index(end, begin):]


_MODE_COUNT_WORD = {1: None, 2: 'two', 3: 'three'}


PROMPT_VARIANTS = ('compact', 'explicit', 'explicit_v2')


def _insert_multi_examples(body: str) -> str:
    """Put the worked multi examples where a reader would look for them."""
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
        if allowed_paradigms and len(allowed_paradigms) == 1 \
                and '## Example -- agentic' in body:
            return _drop(body, '## Example -- agentic', 'Question:')
        return body
    if variant == 'explicit_v2':
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
    if '## Paradigms' not in body:
        raise ValueError(
            f"variant {variant!r} does not apply to this action space: it has "
            f"no '## Paradigms' section to restate. Only stages that include "
            f"agentic have one.")
    start = body.index('## Paradigms')
    end = body.index('## Example --', start)
    body = body[:start] + explicit_section(allowed_paradigms) + body[end:]

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
        body = _drop(body, 'agentic -- workers run in parallel', '## Paradigms')
        body = _drop(body, '## Paradigms', '## Example -- single')
        body = _drop(body, '## Example -- agentic', 'The examples above')
    if 'multi' not in modes:
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
    if _worker_alias.enabled():
        body = (body.replace('MODEL_ID_A', _worker_alias.EXAMPLE_PRIMARY)
                    .replace('MODEL_ID_B', _worker_alias.EXAMPLE_SECONDARY)
                    .replace('MODEL_ID_C', _worker_alias.EXAMPLE_TERTIARY))
    return body


PROMPT_TEMPLATE_QWEN = _build_template()
PROMPT_TEMPLATE_LLAMA = _build_template()
