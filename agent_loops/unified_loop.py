"""RouteWeaver V2 agent loop for verl: local Qwen3-4B policy, grammar, 4 roles."""
import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register

from route_grammar import parse_route, validate_route

from base_loop import (RoutingAgentLoopBase, _OPEN_OBSERVATION,
                               _TURN_STOPS, _MODE_STOPS)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys                                                        # noqa: E402
for _p in (os.path.join(_ROOT, "routing"),
           os.path.join(_ROOT, "evaluation"),
           os.path.join(_ROOT, "rewards")):
    if _p not in sys.path:
        sys.path.append(_p)

import actions as v2_actions                                      # noqa: E402
import mode_selector
from worker_span import worker_spans                             # noqa: E402
from rollout_ordinal_hook import KEY as ORDINAL_KEY               # noqa: E402
from worker_explore_schedule import (  # noqa: E402
    beta_for as we_beta_for, display_order as we_display_order,
    forced_worker as we_forced_worker, enabled as we_enabled,
    worker_for_rollout as we_worker_for_rollout)                                              # noqa: E402
import roles as v2_roles                                          # noqa: E402
import route_syntax                                               # noqa: E402

logger = logging.getLogger(__name__)

_AGENTIC_V2_STOPS = ["</answer>", "<observation"]

MAX_WORKERS_V2 = route_syntax.MAX_WORKERS          # 6
DEFAULT_WORKER_IDS = ("worker_1", "worker_2", "worker_3")
CODE_EXEC_SCORERS = ("apps_functional", "stdin", "livecodebench", "taco_mm")
CALC_EXEC_SCORERS = ("math",)
CALC_TIMEOUT_S = 10
CALC_STDOUT_CHARS = 600


_POOL_LOCK = threading.Lock()
_POOL = {"pool": None, "key": None, "factory": None}
_SEMAPHORES: "dict" = {}


def set_pool_factory(factory):
    """Test seam: install a fake WorkerPool factory (or None to reset)."""
    with _POOL_LOCK:
        _POOL["factory"] = factory
        _POOL["pool"] = None
        _POOL["key"] = None


def get_worker_pool(channel_config_path, worker_max_tokens, worker_temperature):
    key = (channel_config_path, worker_max_tokens, worker_temperature)
    with _POOL_LOCK:
        if _POOL["pool"] is not None and _POOL["key"] == key:
            return _POOL["pool"]
        if _POOL["pool"] is not None:
            raise RuntimeError(f"worker pool already built with {_POOL['key']}, "
                               f"refusing to rebuild with {key}")
        if _POOL["factory"] is not None:
            pool = _POOL["factory"](*key)
        else:
            from llm_client import ChatLLM
            from route_dispatch import RouteDispatcher
            from worker_client import WorkerPool
            dispatcher = RouteDispatcher(config_path=channel_config_path)
            pool = WorkerPool(
                dispatcher, max_tokens=worker_max_tokens,
                make_client=lambda wid: ChatLLM(
                    dispatcher, worker_id=wid,
                    default_max_tokens=worker_max_tokens,
                    sampling={"temperature": worker_temperature}))
        _POOL["pool"] = pool
        _POOL["key"] = key
        return pool


def _as_bool(value, default=True):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("false", "0", "no", "off"):
            return False
        if text in ("true", "1", "yes", "on"):
            return True
        return default
    return default if value is None else bool(value)


def _semaphore(limit):
    loop = asyncio.get_running_loop()
    sem = _SEMAPHORES.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        _SEMAPHORES[loop] = sem
    return sem


async def _call_worker(pool, limit, **kwargs):
    """One frozen-worker call, off the event loop, bounded."""
    async with _semaphore(limit):
        try:
            return await asyncio.to_thread(pool.call, **kwargs)
        except Exception as exc:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            kind = type(exc).__name__
            return None, {"worker_id": kwargs.get("worker_id"),
                          "role": kwargs.get("role"),
                          "tag": kwargs.get("tag"), "success": False,
                          "error": f"{kind}: {exc}",
                          "wrapper_error": kind,
                          "input_tokens": 0, "output_tokens": 0}


_OBSERVATION_BLOCK = re.compile(
    r'<(observation|information)\b[^>]*>.*?</\1\s*>', re.DOTALL)


@dataclass
class _V2State:
    mode: Optional[str] = None
    selector_raw: str = ""
    selector_attempts: int = 0
    selector_valid: int = 0
    selector_violations: list = field(default_factory=list)
    worker_spans: list = field(default_factory=list)
    worker_menu_order: list = field(default_factory=list)
    rollout_n: int = -1
    forced_worker: str = ""
    explore_beta: float = 0.0
    injected_rules_chars: int = 0
    injected_rules_removed: int = 0
    turns: int = 0
    grammar_errors: list = field(default_factory=list)
    route_count: int = 0
    nodes: list = field(default_factory=list)        # V2 four-role node records
    layers: list = field(default_factory=list)       # per layer: node ids
    revision_loops: int = 0
    finished_with_summarizer: bool = False
    worker_records: list = field(default_factory=list)
    question: str = ""
    answer: Optional[str] = None
    answer_turn: str = ""
    code_meta: Optional[dict] = None
    code_scorer: str = ""
    calc_enabled: bool = False
    exec_runs: int = 0
    exec_passed_all: int = 0
    exec_sandbox_errors: int = 0
    stage2_alpha: float = 1.0
    stage2_forced: int = 1
    forced_mode: str = ""
    loop_errors: int = 0
    policy_invalid: str = ""
    policy_invalid_detail: str = ""
    cue_messages: int = 0
    state_headers: int = 0
    obs_blocks: int = 0            # observations injected
    obs_truncated: int = 0         # of those, how many lost body tokens
    obs_dropped_tokens: int = 0    # body tokens elided in total
    obs_budget_min: int = -1       # smallest per-observation budget granted
    policy_spans: list = field(default_factory=list)
    env_spans: list = field(default_factory=list)
    selector_span: tuple = ()
    mode_span_end: int = -1     # response index where the mode decision ENDS (last selector attempt)

    def record(self, code):
        self.grammar_errors.append(code)

    @property
    def is_agentic(self):
        return self.mode == "agentic"

    def action_state(self):
        """The dict actions.validate_layer expects."""
        return {"nodes": [{"id": n["id"], "role": n["role"]} for n in self.nodes],
                "layers": [list(ids) for ids in self.layers],
                "revision_loops": self.revision_loops,
                "finished": self.finished_with_summarizer}


@register("routeweaver_unified")
class UnifiedAgentLoop(RoutingAgentLoopBase):
    """V2 free-routing loop. Inherits token bookkeeping, replaces the grammar."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.get("router", {}) or {}
        self.selector_max_tokens = int(cfg.get("selector_max_tokens", 64))
        self.worker_ids = tuple(
            (cfg.get("worker_ids") or ",".join(DEFAULT_WORKER_IDS)).split(",")
            if isinstance(cfg.get("worker_ids", None), str)
            else (cfg.get("worker_ids") or DEFAULT_WORKER_IDS))
        self.worker_max_tokens = int(cfg.get("worker_max_tokens", 4096))
        self.worker_temperature = float(cfg.get("worker_temperature", 0.2))
        self.max_workers_v2 = int(cfg.get("max_workers_v2", MAX_WORKERS_V2))
        self.max_action_turns = int(cfg.get("max_action_turns",
                                            self.max_workers_v2 + 2))
        self.policy_violation_recovery = _as_bool(
            cfg.get("policy_violation_recovery", True), default=True)
        self.routing_state = _as_bool(cfg.get("routing_state", False),
                                      default=False)


    def _chat_delta_ids(self, text):
        """Environment text as a chat user turn (Qwen template tokens hardcoded)."""
        delta = ("<|im_end|>\n<|im_start|>user\n" + text
                 + "<|im_end|>\n<|im_start|>assistant\n")
        return self.tokenizer(delta, add_special_tokens=False)["input_ids"]

    def _sm_prompt_builder(self):
        """routing/mode_prompts.build_sm_prompt, imported lazily."""
        import mode_prompts
        return mode_prompts.build_sm_prompt

    def _inject_env(self, state, ids, context, response_ids, response_mask,
                    response_logprobs, logprobs_seen):
        """Environment tokens, mask 0, with the span recorded."""
        start = len(response_ids)
        self._inject_observation_ids(ids, context, response_ids, response_mask,
                                    response_logprobs, logprobs_seen)
        state.env_spans.append((start, len(response_ids)))

    async def _policy_turn(self, state, *args, **kwargs):
        """One generated turn, mask 1, with the span recorded."""
        response_ids = args[4]
        start = len(response_ids)
        text, logprobs_seen = await self._generate_turn(*args, **kwargs)
        state.policy_spans.append((start, len(response_ids)))
        try:
            for a0, b0, wid in worker_spans(
                    self.tokenizer, response_ids[start:],
                    allowed=set(self.worker_ids)):
                state.worker_spans.append((start + a0, start + b0, wid))
        except Exception:                                   # never kill a rollout
            state.record("worker_span_extraction_failed")
        return text, logprobs_seen

    def _budgeted_observation_ids(self, obs_text, state, used):
        """Tokenize a turn's environment text under a PER-OBSERVATION budget."""
        parts = self._split_observation_parts(obs_text)
        n_obs = sum(1 for _, is_obs in parts if is_obs)
        turn_budget = self._turn_observation_budget(used)
        if turn_budget <= 0:
            return []
        per_obs = self._layer_observation_budget(max(1, n_obs), turn_budget)
        stats = {}
        ids = []
        for text, is_obs in parts:
            if is_obs:
                ids.extend(self._observation_ids(text, per_obs, stats))
            else:
                ids.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        state.obs_blocks += stats.get("blocks", 0)
        state.obs_truncated += stats.get("truncated", 0)
        state.obs_dropped_tokens += stats.get("dropped_tokens", 0)
        if n_obs:
            state.obs_budget_min = (per_obs if state.obs_budget_min < 0
                                    else min(state.obs_budget_min, per_obs))
        return ids

    @staticmethod
    def _split_observation_parts(text):
        """Split environment text into [(segment, is_observation), ...]."""
        parts, last = [], 0
        for m in _OBSERVATION_BLOCK.finditer(text):
            if m.start() > last:
                parts.append((text[last:m.start()], False))
            parts.append((m.group(0), True))
            last = m.end()
        if last < len(text):
            parts.append((text[last:], False))
        return parts or [(text, False)]

    @staticmethod
    def _mask_accounting(state, response_mask):
        """What actually happened to each recorded span."""
        def span_values(spans):
            return [m for start, end in spans
                    for m in response_mask[start:min(end, len(response_mask))]]

        policy = span_values(state.policy_spans)
        env = span_values(state.env_spans)
        sel = (response_mask[state.selector_span[0]:
                             min(state.selector_span[1], len(response_mask))]
               if state.selector_span else [])
        return {
            "policy_token_count": int(sum(response_mask)),
            "env_token_count": int(len(response_mask) - sum(response_mask)),
            "policy_span_mask_min": int(min(policy)) if policy else 1,
            "env_span_mask_max": int(max(env)) if env else 0,
            "mode_span_mask_min": int(min(sel)) if sel else 0,
            "mode_span_tokens": len(sel),
            "mode_span_start": int(state.selector_span[0]) if state.selector_span else -1,
            "mode_span_end": int(state.mode_span_end if state.mode_span_end >= 0
                                 else (state.selector_span[1] if state.selector_span else -1)),
        }

    def _violation(self, state, codes, cue):
        """The ONE place a policy violation is turned into environment speech."""
        for code in codes:
            state.record(code)
        if self.policy_violation_recovery:
            state.cue_messages += 1
            return cue
        state.policy_invalid = str(codes[0]) if codes else "policy_invalid"
        state.policy_invalid_detail = str(cue)
        return None

    def _payload(self, kwargs):
        extra = dict(kwargs.get("extra_info") or {})
        payload = json.loads(extra.get("payload_json") or "{}")
        return extra, payload


    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        self._routeweaver_rollout_n = kwargs.get(ORDINAL_KEY)
        extra, payload = self._payload(kwargs)
        request_id = uuid4().hex
        prompt_ids = await self.apply_chat_template(messages)

        state = _V2State()
        state.question = str(payload.get("raw_question") or "")
        if str(extra.get("scorer_dataset") or "") in CODE_EXEC_SCORERS:
            meta = (payload.get("meta") or {})
            if meta.get("tests"):
                state.code_meta = meta
                state.code_scorer = str(extra.get("scorer_dataset"))
        elif str(extra.get("scorer_dataset") or "") in CALC_EXEC_SCORERS:
            state.calc_enabled = True

        response_ids, response_mask, response_logprobs = [], [], []
        logprobs_seen = False
        context = list(prompt_ids)

        mode, logprobs_seen = await self._declare_mode(
            state, sampling_params, request_id, context,
            response_ids, response_mask, response_logprobs, logprobs_seen, extra)

        if mode is None:
            return self._finish(prompt_ids, response_ids, response_mask,
                                response_logprobs, logprobs_seen, state, extra)

        state.mode = mode
        state.selector_valid = 1
        return await self._execute(
            mode, state, sampling_params, request_id, context, prompt_ids,
            response_ids, response_mask, response_logprobs, logprobs_seen, extra)

    async def _declare_mode(self, state, sampling_params, request_id, context,
                            response_ids, response_mask, response_logprobs,
                            logprobs_seen, extra):
        """FREE ROUTING: the policy declares the mode and keeps the gradient."""
        selector_params = dict(sampling_params)
        selector_params["max_tokens"] = self.selector_max_tokens
        mode = None
        for attempt in range(2):
            text, logprobs_seen = await self._policy_turn(
                state, request_id, context, selector_params, _MODE_STOPS,
                response_ids, response_mask, response_logprobs, logprobs_seen)
            if attempt == 0:
                state.selector_span = state.policy_spans[-1]
            state.mode_span_end = state.policy_spans[-1][1]
            state.selector_attempts += 1
            if attempt == 0:
                state.selector_raw = text
            if _OPEN_OBSERVATION.search(text) is not None:
                state.record("fabricated_observation_blocked")
            mode, codes = mode_selector.parse_selection(text)
            state.selector_violations = list(codes)
            if mode is not None:
                break
            for code in codes:
                state.record(code)
            if not self.policy_violation_recovery:
                state.policy_invalid = str(codes[0]) if codes else "mode_invalid"
                state.policy_invalid_detail = "selector declaration rejected"
                break
            if attempt == 0:
                state.cue_messages += 1
                self._inject_env(
                    state, self._chat_delta_ids(mode_selector.retry_cue()),
                    context, response_ids, response_mask, response_logprobs,
                    logprobs_seen)
        return mode, logprobs_seen

    async def _execute(self, mode, state, sampling_params, request_id, context,
                       prompt_ids, response_ids, response_mask,
                       response_logprobs, logprobs_seen, extra):
        """Everything after the declaration. Identical for free and forced."""
        _sid = (extra or {}).get("sample_id") or ""
        _bidx = (extra or {}).get("batch_index") or 0
        _order = we_display_order(_sid, list(self.worker_ids))
        _rn = getattr(self, "_routeweaver_rollout_n", None)
        _forced = (we_worker_for_rollout(_sid, _bidx, list(self.worker_ids), _rn)
                   if we_enabled() else None)
        state.rollout_n = -1 if _rn is None else int(_rn)
        state.worker_menu_order = list(_order)
        state.forced_worker = _forced or ""
        state.explore_beta = float(we_beta_for(_bidx))
        rules, removed = mode_selector.mode_rules(
            mode, state.question, worker_ids=_order,
            sm_prompt_builder=self._sm_prompt_builder(),
            predeclared=True, return_removed=True,
            routing_state=(self.routing_state and mode == "agentic"),
            max_workers=self.max_workers_v2, forced_worker=_forced)
        state.injected_rules_chars = len(rules)
        state.injected_rules_removed = len(removed)
        rule_ids = self._chat_delta_ids(rules)
        if len(response_ids) + len(rule_ids) <= self.response_length:
            self._inject_env(state, rule_ids, context, response_ids,
                             response_mask, response_logprobs, logprobs_seen)
        else:
            state.record("rules_budget_exhausted")

        stops = _AGENTIC_V2_STOPS if state.is_agentic else _TURN_STOPS
        while state.answer is None and state.turns < self.max_action_turns:
            if len(response_mask) >= self.response_length:
                break
            if self.routing_state and state.is_agentic:
                header_ids = self._chat_delta_ids(
                    route_syntax.routing_state(state.action_state(),
                                               max_workers=self.max_workers_v2))
                if len(response_ids) + len(header_ids) <= self.response_length:
                    self._inject_env(state, header_ids, context, response_ids,
                                     response_mask, response_logprobs,
                                     logprobs_seen)
                    state.state_headers += 1
            text, logprobs_seen = await self._policy_turn(
                state, request_id, context, sampling_params, stops,
                response_ids, response_mask, response_logprobs, logprobs_seen)
            state.turns += 1
            if _OPEN_OBSERVATION.search(text) is not None:
                state.record("fabricated_observation_blocked")

            obs_text = None
            try:
                if state.is_agentic:
                    obs_text = await self._agentic_turn(text, state)
                else:
                    obs_text = await self._single_multi_turn(text, state)
            except Exception as exc:                              # noqa: BLE001
                import traceback
                traceback.print_exc()
                state.record(f"loop_error:{type(exc).__name__}")
                state.loop_errors += 1
                break

            if state.policy_invalid:
                break
            if obs_text is None:            # answered, or nothing to inject
                continue
            obs_ids = self._budgeted_observation_ids(
                obs_text, state, used=len(response_ids))
            if not obs_ids:
                state.record("obs_budget_exhausted")
                break
            self._inject_env(
                state, obs_ids, context,
                response_ids, response_mask, response_logprobs, logprobs_seen)

        return self._finish(prompt_ids, response_ids, response_mask,
                            response_logprobs, logprobs_seen, state, extra)


    async def _single_multi_turn(self, text, state):
        from route_grammar import find_open
        from base_loop import _ANSWER_BLOCK

        if find_open("mode", text) != -1:
            return self._violation(state, ["mode_duplicate"],
                                   mode_selector.PREDECLARED_SENTENCE)
        answers = _ANSWER_BLOCK.findall(text)
        route, route_err = parse_route(text)
        if answers and route is None:
            need = 1 if state.mode == "single" else 2
            if state.route_count >= need:
                state.answer = answers[-1].strip()
                state.answer_turn = text
                return None
            return self._violation(
                state, ["answer_premature"],
                "Answer only after the required worker calls have run.")
        if route is None:
            return self._violation(
                state, [route_err.code.value],
                f"That turn was not executed: {route_err.detail}")
        limit = 1 if state.mode == "single" else 4
        if state.route_count >= limit:
            return self._violation(
                state, ["route_limit_exceeded"],
                f"You have used the {limit} worker call(s) this mode allows.")
        spec, verr = validate_route(route)
        if verr is not None:
            return self._violation(
                state, [verr.code.value],
                f"That turn was not executed: {verr.detail}")
        if route.model_id not in self.worker_ids:
            return self._violation(
                state, ["worker_not_offered"],
                f"That turn was not executed: {route.model_id} is not one of "
                f"the workers you were given.")

        pool = get_worker_pool(self.channel_config_path, self.worker_max_tokens,
                              self.worker_temperature)
        reply, record = await _call_worker(
            pool, self.dispatch_concurrency, worker_id=route.model_id,
            raw_question=state.question, task=route.query, role=None, refs=None,
            include_question=True, tag=f"{state.mode}_turn{state.turns}")
        record["mode"] = state.mode
        record.setdefault("output", reply or "")
        state.worker_records.append(record)
        state.route_count += 1
        body = (reply or "").strip() if record.get("success") else ""
        return f"\n\n<observation>{body}</observation>\n\n"


    def _run_executor_code(self, state, reply):
        """Run an executor node's program against the row's hidden tests."""
        import bench_scoring
        try:
            verdict = bench_scoring.score(state.code_scorer, reply or "",
                                          ["(hidden tests)"], state.code_meta)
        except Exception as exc:                                  # noqa: BLE001
            state.exec_sandbox_errors += 1
            logger.warning("executor sandbox raised: %r", exc)
            return ""
        state.exec_runs += 1
        passed = verdict.get("tests_passed")
        total = verdict.get("tests_total")
        if not verdict.get("code_chars"):
            return ("\n\n[execution] no runnable code block found in this "
                    "output; nothing was executed.")
        if verdict.get("correct"):
            state.exec_passed_all += 1
            return f"\n\n[execution] all {total} hidden tests PASSED."
        detail = verdict.get("detail") or {}
        error = str(detail.get("error") or "").strip()
        failed_at = detail.get("failed_at")
        bits = [f"\n\n[execution] {passed}/{total} hidden tests passed"]
        if failed_at is not None:
            bits.append(f"; first failure at test index {failed_at}")
        if error:
            bits.append(f"; error: {error[:300]}")
        bits.append(".")
        return "".join(bits)

    def _run_executor_calc(self, state, reply):
        """Run an executor node's program for its VALUE and hand back stdout."""
        import re as _re
        import subprocess
        import sys
        import tempfile

        blocks = _re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply or "", _re.S)
        if not blocks:
            return ""
        code = blocks[-1].strip()
        if not code:
            return ""
        try:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "calc.py")
                with open(path, "w") as fh:
                    fh.write(code)
                proc = subprocess.run(
                    [sys.executable, path], cwd=td, capture_output=True,
                    text=True, timeout=CALC_TIMEOUT_S,
                    env={"PATH": os.environ.get("PATH", ""),
                         "HOME": td, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            state.exec_runs += 1
            return (f"\n\n[calc] the program did not finish within "
                    f"{CALC_TIMEOUT_S}s; nothing was computed.")
        except Exception as exc:                                  # noqa: BLE001
            state.exec_sandbox_errors += 1
            logger.warning("calc sandbox raised: %r", exc)
            return ""

        state.exec_runs += 1
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else "non-zero exit, no stderr"
            return f"\n\n[calc] the program errored: {tail[:300]}"
        if not out:
            return "\n\n[calc] the program ran but printed nothing."
        state.exec_passed_all += 1
        if len(out) > CALC_STDOUT_CHARS:
            out = out[:CALC_STDOUT_CHARS] + " ...[truncated]"
        return f"\n\n[calc] the program printed:\n{out}"

    async def _agentic_turn(self, text, state):
        from base_loop import _ANSWER_BLOCK

        layer, errors = route_syntax.parse_routes(text, expect_mode=False)
        if layer:
            codes, _ = v2_actions.validate_layer(
                layer, state.action_state(), list(self.worker_ids),
                max_workers=self.max_workers_v2)
            errors = sorted(set(errors) | set(codes))
        if errors:
            answers = _ANSWER_BLOCK.findall(text)
            if answers and state.finished_with_summarizer:
                state.answer = answers[-1].strip()
                state.answer_turn = text
                return None
            return self._violation(
                state, errors,
                route_syntax.violation_cue(errors, state.action_state(),
                                           repeated=False,
                                           max_workers=self.max_workers_v2))

        layer_index = len(state.layers) + 1
        prepared = []
        for offset, action in enumerate(layer):
            node_id = f"n{len(state.nodes) + offset + 1}"
            refs = [(r, self._role_of(state, r), self._text_of(state, r))
                    for r in action["refs"]]
            prepared.append((node_id, action, v2_roles.build_worker_payload(
                action["role"], state.question, action["instruction"], refs)))
        revision = any(a["role"] == "executor" and
                       v2_actions.is_revision(a, state.action_state())
                       for a in layer)

        pool = get_worker_pool(self.channel_config_path, self.worker_max_tokens,
                              self.worker_temperature)
        results = await asyncio.gather(*[
            _call_worker(pool, self.dispatch_concurrency,
                         worker_id=action["model"], raw_question=state.question,
                         task=action["instruction"], role=action["role"],
                         refs=None, include_question=True,
                         tag=f"{node_id}_{action['role']}",
                         override_payload=payload,
                         exclude_texts=[self._text_of(state, r)
                                        for r in action["refs"]],
                         max_tokens=(4096 if action["role"] == "summarizer"
                                     else None))
            for node_id, action, payload in prepared])

        obs_parts, ids = [], []
        for (node_id, action, payload), (reply, record) in zip(prepared, results):
            from worker_client import extract_answer
            candidate = ""
            if action["role"] in ("executor", "summarizer") and reply:
                candidate, _src = extract_answer(reply)
            verdict_text = ""
            if (reply and record.get("success")
                    and action["role"] == "executor"
                    and (state.code_meta is not None or state.calc_enabled)):
                verdict_text = (self._run_executor_code(state, reply)
                                if state.code_meta is not None
                                else self._run_executor_calc(state, reply))
            node = {"id": node_id, "layer": layer_index, "role": action["role"],
                    "model": action["model"], "refs": list(action["refs"]),
                    "instruction": action["instruction"],
                    "output": verdict_text.lstrip("\n") + ("\n\n" if verdict_text else "") + (reply or ""),
                    "execution_verdict": verdict_text.strip(),
                    "candidate_answer": candidate,
                    "success": bool(record.get("success"))}
            state.nodes.append(node)
            record["role"] = action["role"]
            record["node_id"] = node_id
            state.worker_records.append(record)
            ids.append(node_id)
            body = ""
            if record.get("success"):
                head = verdict_text.lstrip("\n")
                body = (head + "\n\n" if head else "") + (reply or "").strip()
            obs_parts.append(f'<observation id="{node_id}" '
                             f'role="{action["role"]}">{body}</observation>')
        state.layers.append(ids)
        if revision:
            state.revision_loops += 1
        if any(a["role"] == v2_roles.TERMINAL_ROLE for a in layer):
            state.finished_with_summarizer = True
        return "\n" + "\n\n".join(obs_parts) + "\n"

    @staticmethod
    def _role_of(state, node_id):
        for n in state.nodes:
            if n["id"] == node_id:
                return n["role"]
        return "unknown"

    @staticmethod
    def _text_of(state, node_id):
        for n in state.nodes:
            if n["id"] == node_id:
                return n["output"]
        return ""


    def _finish(self, prompt_ids, response_ids, response_mask,
                response_logprobs, logprobs_seen, state, extra):
        assert len(response_ids) == len(response_mask), \
            f"ids/mask misaligned: {len(response_ids)} vs {len(response_mask)}"
        if logprobs_seen:
            assert len(response_logprobs) == len(response_ids), \
                f"ids/logprobs misaligned: {len(response_logprobs)} vs {len(response_ids)}"
        summarizer = [n for n in state.nodes
                      if n["role"] == v2_roles.TERMINAL_ROLE]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(response_logprobs[: self.response_length]
                               if logprobs_seen else None),
            num_turns=state.turns + 1,
            metrics={},
            extra_fields={
                "grammar_version": "v2_route_four_role",
                "selector_mode": state.mode or "",
                "selector_valid": int(state.selector_valid),
                "selector_raw": state.selector_raw,
                "selector_attempts": int(state.selector_attempts),
                "selector_violations": list(state.selector_violations),
                "worker_menu_order": "|".join(state.worker_menu_order),
                "forced_worker": str(state.forced_worker),
                "rollout_n": int(state.rollout_n),
                "worker_spans": ";".join("%d,%d,%s" % t for t in state.worker_spans),
                "explore_beta": float(state.explore_beta),
                "injected_rules_chars": int(state.injected_rules_chars),
                "injected_rules_removed": int(state.injected_rules_removed),
                "mode": state.mode or "",
                "executed_mode": state.mode or "",
                "route_count": int(state.route_count),
                "exec_runs": int(state.exec_runs),
                "exec_passed_all": int(state.exec_passed_all),
                "exec_sandbox_errors": int(state.exec_sandbox_errors),
                "worker_calls": len(state.worker_records),
                "failed_worker_calls": sum(
                    1 for r in state.worker_records if not r.get("success")),
                "obs_blocks": int(state.obs_blocks),
                "obs_truncated": int(state.obs_truncated),
                "obs_truncated_any": int(state.obs_truncated > 0),
                "obs_dropped_tokens": int(state.obs_dropped_tokens),
                "obs_budget_min": int(state.obs_budget_min),
                "grammar_errors": list(state.grammar_errors),
                "turns": int(state.turns),
                "role_sequence": [n["role"] for n in state.nodes],
                "layers": [list(ids) for ids in state.layers],
                "layer_shape": ["+".join(n["role"] for n in state.nodes
                                         if n["id"] in ids)
                                for ids in state.layers],
                "max_layer_width": max((len(i) for i in state.layers), default=0),
                "revision_loops": int(state.revision_loops),
                "finished_with_summarizer": bool(state.finished_with_summarizer),
                "summarizer_answer": (summarizer[-1]["candidate_answer"]
                                      if summarizer else ""),
                "summarizer_output": (summarizer[-1]["output"][:24000]
                                      if summarizer else ""),
                "router_answer": state.answer or "",
                "router_final_turn": (state.answer_turn or "")[:24000],
                "worker_records": state.worker_records,
                "loop_errors": int(state.loop_errors),
                "policy_violation_recovery": int(self.policy_violation_recovery),
                "policy_invalid": str(state.policy_invalid),
                "policy_invalid_detail": str(state.policy_invalid_detail)[:500],
                "cue_messages": int(state.cue_messages),
                "routing_state_enabled": int(self.routing_state),
                "state_headers": int(state.state_headers),
                "wrapper_errors": sum(1 for r in state.worker_records
                                      if r.get("wrapper_error")),
                "sample_id": str(extra.get("sample_id") or ""),
                "domain": str(extra.get("domain") or ""),
                "pool": str(extra.get("pool") or ""),
                "rfm_dataset": str(extra.get("dataset") or ""),
                "scorer_dataset": str(extra.get("scorer_dataset") or ""),
                **self._mask_accounting(state, response_mask),
            },
        )
