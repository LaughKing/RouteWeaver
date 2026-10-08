# Copyright 2026 RouteWeaver.
"""Stage 2: linear anneal from forced modes to free routing.

Registered as `routeweaver`. UnifiedAgentLoop with exactly one method
replaced -- `_declare_mode`. Everything below the declaration (rulebook,
four-role grammar, refs, summarizer contract, observation budget, in-loop code
execution) is inherited unchanged.

    alpha_t = 1 - t / FORCED_ANNEAL_STEPS        t = 0 .. FORCED_ANNEAL_STEPS-1

alpha is the probability a QUERY is forced. At t=0 every query is forced, so
the first Stage-2 step reproduces Stage-1 exactly; at the end essentially none
are and the run is free routing.

WHERE t COMES FROM, AND WHY NOT A COUNTER. The agent loop runs inside
AgentLoopWorker and is never told the trainer's global step. It does not need
to be: the parquet is written in DomainQuotaBatchSampler order and consumed by
a SequentialSampler with shuffle=False, so `extra_info["batch_index"]` IS the
step that row belongs to. t = batch_index - FORCED_START_BATCH. That makes the
schedule a pure function of the data, identical on every worker, reproducible
on replay, and correct across a resume -- none of which a process-local counter
would be.

WHY THE DECISION IS PER QUERY AND NOT PER ROLLOUT. GRPO's advantage is computed
within a group of K rollouts of one prompt. A group holding both forced and
free rollouts would be comparing a trajectory whose mode was given against one
that had to spend a decision on it, and the baseline would absorb the
difference as if it were policy quality. So the draw is a deterministic hash of
sample_id ALONE -- every rollout of a query gets the same u, and with the same
alpha (same batch_index) the same verdict. Nothing is sampled at runtime, so
there is no seed to thread and no way for two workers to disagree.
"""
from verl.experimental.agent_loop.agent_loop import register

from unified_loop import UnifiedAgentLoop

import mode_selector

# Schedule functions live in exploration_schedule.py (pure, no verl) so the dataset
# builder and the support hook share them. Re-exported here for the tests.
from exploration_schedule import (DEFAULT_START_BATCH, DEFAULT_STEPS,
                                  alpha_for, forced_for_row)


@register("routeweaver")
class RouteWeaverAgentLoop(UnifiedAgentLoop):
    """Forced with probability alpha(step), free otherwise -- per query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Read from the router config, not the environment: the loop runs inside
        # a ray AgentLoopWorker and an exported shell variable is not guaranteed
        # to reach it. The env values remain the defaults so a bare import and
        # the unit tests still work.
        cfg = self.config.get("router", {}) or {}
        self.forced_start_batch = int(
            cfg.get("forced_start_batch", DEFAULT_START_BATCH))
        self.forced_anneal_steps = int(cfg.get("forced_anneal_steps", DEFAULT_STEPS))

    async def _declare_mode(self, state, sampling_params, request_id, context,
                            response_ids, response_mask, response_logprobs,
                            logprobs_seen, extra):
        alpha = alpha_for(extra.get("batch_index") or 0,
                          self.forced_start_batch, self.forced_anneal_steps)
        sample_id = extra.get("sample_id") or ""
        forced = forced_for_row(extra, self.forced_start_batch,
                                self.forced_anneal_steps)
        state.stage2_alpha = alpha
        state.stage2_forced = int(forced)

        if not forced:
            # FREE: the policy writes <mode> and keeps the gradient. Unchanged
            # V2 behaviour, including its retry/violation handling.
            return await super()._declare_mode(
                state, sampling_params, request_id, context, response_ids,
                response_mask, response_logprobs, logprobs_seen, extra)

        # FORCED: byte-identical to the forced branch of RouteWeaverAgentLoop._declare_mode.
        mode = str(extra.get("forced_mode") or "").strip()
        if mode not in mode_selector.MODES:
            raise ValueError(
                f"routeweaver needs extra_info['forced_mode'] in "
                f"{mode_selector.MODES} for a forced row, got {mode!r} for "
                f"sample {sample_id!r}")
        declaration = f"<mode>{mode}</mode>"
        start = len(response_ids)
        self._inject_env(state, self._chat_delta_ids(declaration), context,
                         response_ids, response_mask, response_logprobs,
                         logprobs_seen)
        state.selector_span = (start, len(response_ids))
        state.mode_span_end = len(response_ids)
        state.selector_raw = declaration
        state.selector_attempts = 0
        state.selector_violations = []
        state.forced_mode = mode
        return mode, logprobs_seen

    def _finish(self, prompt_ids, response_ids, response_mask,
                response_logprobs, logprobs_seen, state, extra):
        output = super()._finish(prompt_ids, response_ids, response_mask,
                                 response_logprobs, logprobs_seen, state, extra)
        fields = output.extra_fields
        forced = int(getattr(state, "stage2_forced", 0))
        fields["stage2_alpha"] = float(getattr(state, "stage2_alpha", 1.0))
        fields["stage2_forced"] = forced
        fields["stage2_free"] = 1 - forced
        fields["forced_mode"] = str(extra.get("forced_mode") or "")
        fields["mode_forced"] = forced
        fields["mode_policy"] = "stage2_anneal"
        # Mask provenance, and the reason forced and free cannot share one field:
        # a forced declaration is environment text and must be all mask 0, a
        # free one is the policy's and must be mask 1. Reporting them under one
        # name would make a leak in either direction invisible.
        span = state.selector_span or ()
        values = (response_mask[span[0]:min(span[1], len(response_mask))]
                  if span else [])
        if forced:
            fields.pop("mode_span_mask_min", None)
            fields["forced_mode_span_mask_max"] = int(max(values)) if values else 0
            fields["forced_mode_span_tokens"] = len(values)
            fields["free_mode_span_mask_min"] = -1
        else:
            fields["forced_mode_span_mask_max"] = -1
            fields["forced_mode_span_tokens"] = 0
            fields["free_mode_span_mask_min"] = int(min(values)) if values else -1
        return output
