# Copyright 2026 RouteWeaver.
"""Task reward: {-1, 0, +1}. The benchmark verdict is untouched; a third bucket is
added for trajectories the POLICY terminally broke.

    +1  the benchmark says the answer is correct
     0  the trajectory completed legally and the answer is wrong
    -1  the trajectory ended in a policy-caused grammar/protocol violation --
        no legal terminal answer was ever produced

WHY -1 EXISTS. With the mode given, the only thing left to learn is the
pipeline, and a 0 for "never produced anything" is indistinguishable from a 0
for "tried and was wrong". Under
a group baseline those two are the same event, so a policy that stops emitting
routes altogether is never punished relative to one that tries. -1 separates
them. It is a policy signal only: everything the environment broke stays out.

WHAT IS EMPHATICALLY NOT -1. A trajectory that violated the grammar, got a cue,
and then completed is scored by its FINAL task verdict (+1/0); its first-pass
violation is recorded separately (`first_pass_valid`, `cue_count`) so recovery is
measurable without being paid for. Provider 429/503/timeout, gateway failures and
sandbox faults remain `infra_failed` and leave the gradient entirely.
"""
import json

from reward_manager import (RouteWeaverRewardManager, SYSTEM_KINDS,
                                    PARSER, INCORRECT, WRAPPER)

TERMINAL_INVALID = "terminal_invalid"
# the no-cue branch's name for the same event: the policy's first
# illegal action ended the episode, and nothing after it ran.
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

        # NO-CUE BRANCH (router.policy_violation_recovery=False). The loop ends
        # the episode at the FIRST illegal action and reports which code did it.
        # That flag, not "grammar_errors is non-empty", is the terminal signal:
        # grammar_errors also collects non-terminal records such as
        # fabricated_observation_blocked.
        recovery = int(fields.get("policy_violation_recovery", 1))
        policy_invalid = str(fields.get("policy_invalid") or "")

        if not recovery:
            # -1 dominates a summarizer answer that arrived before the illegal
            # action: "first violation terminates" would mean nothing if a
            # trajectory could violate its way past an already-earned answer.
            terminal_invalid = int(not infra and bool(policy_invalid))
            # By construction there is no second pass, so these two are the
            # same number. Emitting both keeps the columns identical whether or
            # not recovery is enabled.
            first_pass_valid = int(not policy_invalid)
            eventual_valid = first_pass_valid
        else:
            # TERMINAL means: the episode is over and the policy never produced a
            # legal answer. A recovered violation is not terminal.
            terminal_invalid = int(not infra and not answered and bool(errors))
            first_pass_valid = int(not errors)
            eventual_valid = int(answered or correct)

        if infra:
            reward = 0.0                      # neutralised, then gated out
        elif terminal_invalid and not recovery:
            # -1 OUTRANKS a correct answer on the no-cue branch, and only there.
            # The case is real: agentic can run planner/executor/summarizer
            # legally, have the summarizer produce the right answer, and then
            # emit an illegal extra route instead of <answer>. If +1 won, the
            # policy could earn full reward while never honouring the terminal
            # contract -- exactly the violation this branch stops masking.
            # Under recovery=True the old ordering is untouched.
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
        # cue_count counts violation CODES, not messages -- one rejected turn
        # can carry several. `cue_messages` is the message count, and it is the
        # number that must read 0 on the no-cue branch.
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
