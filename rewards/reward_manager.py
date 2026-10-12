"""RouteWeaver V2 reward manager for verl's reward loop."""
import json
import os
import sys

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "routing"),
           os.path.join(_ROOT, "evaluation"),
           os.path.join(_ROOT, "rewards")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_scoring                                              # noqa: E402

CODE_SCORERS = ("taco_mm", "livecodebench", "apps_functional", "stdin")
TIMEOUT, PROVIDER, SANDBOX, PARSER, INCORRECT = (
    "execution_timeout", "provider_error", "sandbox_error",
    "parser_error", "genuine_incorrect")
WRAPPER = "wrapper_error"
SYSTEM_KINDS = (TIMEOUT, PROVIDER, SANDBOX, WRAPPER)


def _classify(fields, verdict, correct):
    recs = fields.get("worker_records") or []
    if int(fields.get("loop_errors") or 0) or int(fields.get("wrapper_errors") or 0):
        return WRAPPER
    if not int(fields.get("selector_valid") or 0):
        return PARSER
    if recs and all(not r.get("success") for r in recs):
        errs = " ".join(str(r.get("error") or "") for r in recs).lower()
        return TIMEOUT if ("timeout" in errs or "timed out" in errs) else PROVIDER
    if verdict.get("sandbox_error") or int(fields.get("exec_sandbox_errors") or 0):
        return SANDBOX
    if correct:
        return "correct"
    if fields.get("grammar_errors") and not str(
            fields.get("router_answer") or fields.get("summarizer_answer") or ""):
        return PARSER
    return INCORRECT


class RouteWeaverRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None,
                 reward_router_address=None, reward_model_tokenizer=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)

    async def run_single(self, data) -> dict:
        data = data[-1:]
        item = data[0]
        fields = dict(item.non_tensor_batch.get("tool_extra_fields") or {})
        extra = dict(item.non_tensor_batch.get("extra_info") or {})
        payload = json.loads(extra.get("payload_json") or "{}")
        scorer_dataset = (extra.get("scorer_dataset")
                          or item.non_tensor_batch.get("data_source"))
        gold = payload.get("gold") or []
        meta = payload.get("meta") or {}

        mode = str(fields.get("executed_mode") or "")
        router_answer = str(fields.get("router_answer") or "")
        summarizer_answer = str(fields.get("summarizer_answer") or "")
        router_turn = str(fields.get("router_final_turn") or "")
        if not router_turn and router_answer:
            router_turn = f"<answer>{router_answer}</answer>"
        terminal = (str(fields.get("summarizer_output") or "")
                    if mode == "agentic" else "")
        worker_replies = [str(r.get("output") or "")
                          for r in (fields.get("worker_records") or [])
                          if r.get("success")]
        try:
            verdict = bench_scoring.score_final_output(
                scorer_dataset, gold, meta, router_text=router_turn or None,
                terminal_reply=terminal or None, worker_replies=worker_replies)
        except Exception as exc:                                  # noqa: BLE001
            verdict = {"correct": False, "em": 0.0, "f1": 0.0,
                       "router_correct": False, "terminal_correct": False,
                       "sandbox_error": f"{type(exc).__name__}: {exc}"}
        v_router = {"correct": bool(verdict.get("router_correct"))}
        v_summ = {"correct": bool(verdict.get("terminal_correct"))}
        correct = bool(verdict.get("correct"))
        kind = _classify(fields, verdict, correct)
        infra_failed = int(kind in SYSTEM_KINDS)
        reward = 0.0 if infra_failed else (1.0 if correct else 0.0)

        recs = fields.get("worker_records") or []
        out = {
            "reward_score": float(reward),
            "reward_extra_info": {
                "infra_failed": infra_failed,
                "failure_kind": kind,
                "is_system_failure": int(infra_failed),
                "correct": int(correct),
                "em": float(verdict.get("em") or 0.0),
                "f1": float(verdict.get("f1") or 0.0),
                "router_answer_correct": int(bool(v_router.get("correct"))),
                "summarizer_answer_correct": int(bool(v_summ.get("correct"))),
                "judged_router_text": router_turn,
                "judged_terminal_text": terminal,
                "judged_source": str(verdict.get("source") or ""),
                "mode": mode,
                "mode_single": int(mode == "single"),
                "mode_multi": int(mode == "multi"),
                "mode_agentic": int(mode == "agentic"),
                "selector_valid": int(fields.get("selector_valid") or 0),
                "selector_attempts": int(fields.get("selector_attempts") or 0),
                "grammar_invalid": int(bool(fields.get("grammar_errors"))),
                "grammar_error_count": len(fields.get("grammar_errors") or []),
                "finished_with_summarizer": int(
                    bool(fields.get("finished_with_summarizer"))),
                "revision_loops": int(fields.get("revision_loops") or 0),
                "max_layer_width": int(fields.get("max_layer_width") or 0),
                "rfm_sample_id": str(fields.get("sample_id") or ""),
                "rfm_domain": str(fields.get("domain") or ""),
                "rfm_pool": str(fields.get("pool") or ""),
                "rfm_dataset": str(fields.get("rfm_dataset") or ""),
                "scorer_dataset": str(scorer_dataset or ""),
                "loop_errors": int(fields.get("loop_errors") or 0),
                "wrapper_errors": int(fields.get("wrapper_errors") or 0),
                "worker_calls": len(recs),
                "failed_worker_calls": sum(
                    1 for r in recs if not r.get("success")),
                "worker_output_tokens": sum(
                    int(r.get("output_tokens") or 0) for r in recs),
                "obs_blocks": int(fields.get("obs_blocks") or 0),
                "obs_truncated": int(fields.get("obs_truncated") or 0),
                "obs_truncated_any": int(fields.get("obs_truncated_any") or 0),
                "obs_dropped_tokens": int(fields.get("obs_dropped_tokens") or 0),
                "obs_budget_min": int(fields.get("obs_budget_min", -1)),
                "worker_input_tokens": sum(
                    int(r.get("input_tokens") or 0) for r in recs),
                "turns": int(fields.get("turns") or 0),
                "route_count": int(fields.get("route_count") or 0),
                "injected_rules_chars": int(
                    fields.get("injected_rules_chars") or 0),
                "worker_menu_order": str(fields.get("worker_menu_order") or ""),
                "forced_worker": str(fields.get("forced_worker") or ""),
                "rollout_n": int(fields.get("rollout_n") if fields.get("rollout_n") is not None else -1),
                "worker_spans": str(fields.get("worker_spans") or ""),
                "explore_beta": float(fields.get("explore_beta") or 0.0),
                "role_sequence_str": ",".join(fields.get("role_sequence") or []),
                "layer_shape_str": "|".join(fields.get("layer_shape") or []),
                "policy_token_count": int(fields.get("policy_token_count") or 0),
                "env_token_count": int(fields.get("env_token_count") or 0),
                "policy_span_mask_min": int(fields.get("policy_span_mask_min") or 0),
                "env_span_mask_max": int(fields.get("env_span_mask_max") or 0),
                "mode_span_mask_min": int(fields.get("mode_span_mask_min") or 0),
                "mode_span_tokens": int(fields.get("mode_span_tokens") or 0),
                "mode_span_start": int(fields.get("mode_span_start", -1)),
                "mode_span_end": int(fields.get("mode_span_end", -1)),
            },
        }
        return out
