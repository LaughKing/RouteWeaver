# Copyright 2026 RouteWeaver.
"""Task reward: {-1, 0, +1}. The benchmark verdict is untouched; a third bucket is
added for trajectories the POLICY terminally broke.
"""
import json

from reward_manager import (RouteWeaverRewardManager, SYSTEM_KINDS,
                                    PARSER, INCORRECT, WRAPPER)

TERMINAL_INVALID = "terminal_invalid"
POLICY_INVALID = "policy_invalid"


class TaskRewardManager(RouteWeaverRewardManager):

    async def run_single(self, data) -> dict:
        out = await super().run_single(data)
        info = out["reward_extra_info"]
        item = data[-1:][0]
        fields = dict(item.non_tensor_batch.get("tool_extra_fields") or {})

        infra = int(info.get("infra_failed") or 0)
        correct = int(info.get("correct") or 0)
        errors = list(fields.get("grammar_errors") or [])
        answered = bool(str(fields.get("router_answer") or "").strip()
                        or str(fields.get("summarizer_answer") or "").strip())

        recovery = int(fields.get("policy_violation_recovery", 1))
        policy_invalid = str(fields.get("policy_invalid") or "")

        if not recovery:
            terminal_invalid = int(not infra and bool(policy_invalid))
            first_pass_valid = int(not policy_invalid)
            eventual_valid = first_pass_valid
        else:
            terminal_invalid = int(not infra and not answered and bool(errors))
            first_pass_valid = int(not errors)
            eventual_valid = int(answered or correct)

        if infra:
            reward = 0.0                      # neutralised, then gated out
        elif terminal_invalid and not recovery:
            reward = -1.0
        elif correct:
            reward = 1.0
        elif terminal_invalid:
            reward = -1.0
        else:
            reward = 0.0

        info["reward_value"] = float(reward)
        info["terminal_invalid"] = terminal_invalid
        info["first_pass_valid"] = first_pass_valid
        info["eventual_valid"] = eventual_valid
        info["cue_count"] = len(errors)
        info["cue_messages"] = int(fields.get("cue_messages") or 0)
        info["state_headers"] = int(fields.get("state_headers") or 0)
        info["routing_state_enabled"] = int(
            fields.get("routing_state_enabled") or 0)
        info["policy_violation_recovery"] = recovery
        info["policy_invalid"] = policy_invalid
        info["answer_premature"] = int(
            any(str(c) == "answer_premature" for c in errors))
        info["refs_violation"] = int(
            any(str(c) in ("ref_same_layer", "ref_unknown", "ref_forward",
                           "ref_malformed") for c in errors))
        info["forced_mode"] = str(fields.get("forced_mode") or "")
        info["forced_mode_span_mask_max"] = int(
            fields.get("forced_mode_span_mask_max") or 0)
        info["route_signature"] = str(fields.get("layer_shape_str") or "")
        if not infra and terminal_invalid:
            info["failure_kind"] = POLICY_INVALID if not recovery else TERMINAL_INVALID
        out["reward_score"] = float(reward)
        return out
