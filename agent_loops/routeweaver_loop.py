# Copyright 2026 RouteWeaver.
"""Stage 2: linear anneal from forced modes to free routing."""
from verl.experimental.agent_loop.agent_loop import register

from unified_loop import UnifiedAgentLoop

import mode_selector

from exploration_schedule import (DEFAULT_START_BATCH, DEFAULT_STEPS,
                                  alpha_for, forced_for_row)


@register("routeweaver")
class RouteWeaverAgentLoop(UnifiedAgentLoop):
    """Forced with probability alpha(step), free otherwise -- per query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            return await super()._declare_mode(
                state, sampling_params, request_id, context, response_ids,
                response_mask, response_logprobs, logprobs_seen, extra)

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
