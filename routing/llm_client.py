"""Raw chat-completions client over the same channel stack."""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import requests

from route_dispatch import (  # noqa: E402
    _EXHAUSTED,
    MAX_ATTEMPTS_PER_CHANNEL,
    NETWORK_COOLDOWN_SECONDS,
    REQUEST_SEED,
    REQUEST_TEMPERATURE,
    RouteDispatcher,
    _classify_status,
)


class _SyntheticResponse:
    """A collected SSE stream wearing a non-streaming response's interface,
    so _parse_response and everything downstream stay byte-identical.
    """

    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    @property
    def text(self):
        import json as _json
        return _json.dumps(self._body)[:2000]


def _stream_post(url, headers, payload, timeout):
    """POST with stream=True and collect the SSE into one response."""
    import json as _json
    resp = requests.post(url, headers=headers,
                         json={**payload, "stream": True,
                               "stream_options": {"include_usage": True}},
                         timeout=timeout, stream=True)
    if resp.status_code != 200:
        return resp
    content, reasoning, usage, finish = [], [], {}, None
    try:
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                chunk = _json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    finally:
        resp.close()
    body = {"choices": [{"message": {"content": "".join(content),
                                     "reasoning_content": "".join(reasoning)},
                         "finish_reason": finish}],
            "usage": usage}
    return _SyntheticResponse(200, body)


class LLMCallError(RuntimeError):
    """No channel could serve a router turn."""


class ChatLLM:
    """Raw-chat client reusing a RouteDispatcher's channels and auth."""

    def __init__(self, dispatcher: RouteDispatcher, worker_id: str = "worker_3",
                 default_max_tokens: int = None, max_rounds: int = 6,
                 retry_wait: float = 15.0, total_budget: float = 900.0,
                 sampling: dict = None):
        self.sampling = dict(sampling or {})
        self.dispatcher = dispatcher
        self.worker_id = worker_id
        self.max_rounds = max_rounds
        self.retry_wait = retry_wait
        self.total_budget = total_budget
        self.default_max_tokens = default_max_tokens
        self.entries = dispatcher._enabled_entries(worker_id)
        if not self.entries:
            raise LLMCallError(f"no enabled channel for router model {worker_id!r}")

    def chat(self, messages, tag: str = "", max_tokens: int = None,
             record_role: str = "router_turn", sampling: dict = None):
        """One router turn, with bounded wait-and-retry."""
        started = time.monotonic()
        deadline = started + self.total_budget
        failed = {"failed_input_tokens": 0, "failed_output_tokens": 0,
                  "failed_cached_input_tokens": 0}
        retry_errors = []
        attempt_count = 0
        effective_sampling = {**self.sampling, **(sampling or {})}
        for round_index in range(self.max_rounds):
            result, attempts, round_errors = self._attempt_round(
                messages, tag, max_tokens, started, failed, record_role,
                effective_sampling)
            attempt_count += attempts
            retry_errors.extend(round_errors)
            if result is not None:
                result["attempt_count"] = attempt_count
                result["retry_error_codes"] = list(retry_errors)
                result["router_rounds"] = round_index + 1
                return result.pop("_text"), result
            left = deadline - time.monotonic()
            if left <= 0 or round_index == self.max_rounds - 1:
                break
            wait = min(self.retry_wait * (1 if attempts == 0 else 2), left)
            time.sleep(wait)
        raise LLMCallError(
            f"router call failed after {attempt_count} attempts in "
            f"{time.monotonic() - started:.0f}s: {retry_errors}")

    def _attempt_round(self, messages, tag, max_tokens, started, failed,
                       record_role="router_turn", sampling=None):
        """One walk over the channel list. Returns (record|None, attempts,
        errors); the record carries the reply under '_text'.
        """
        d = self.dispatcher
        retry_errors = []
        attempt_count = 0
        for entry_index, entry in enumerate(self.entries):
            last_resort = entry_index == len(self.entries) - 1
            channel = d.channels[entry["channel"]]
            auth = d._auth[entry["channel"]]
            timeout = entry.get("timeout", channel.get("timeout", 120))
            payload = {
                "model": entry["model"],
                "messages": messages,
                "temperature": REQUEST_TEMPERATURE,
                "seed": REQUEST_SEED,
                "max_tokens": (max_tokens or self.default_max_tokens
                               or entry.get("max_tokens", 1024)),
            }
            payload.update(entry.get("extra_body") or {})
            payload.update(sampling or {})
            load_failures = 0
            stop_file = channel.get("spend_stop_file")
            if stop_file and os.path.exists(os.path.expanduser(str(stop_file))):
                retry_errors.append(f"{entry['channel']}: spend cap reached "
                                    f"({stop_file} exists)")
                continue
            while load_failures < MAX_ATTEMPTS_PER_CHANNEL:
                key = auth.acquire(self.worker_id, last_resort=last_resort,
                                   retry_budget_left=MAX_ATTEMPTS_PER_CHANNEL - load_failures)
                if key is _EXHAUSTED:
                    break
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                attempt_count += 1
                try:
                    if channel.get("stream"):
                        resp = _stream_post(channel["base_url"], headers,
                                            payload, (timeout, timeout))
                    else:
                        resp = requests.post(channel["base_url"], headers=headers,
                                             json=payload, timeout=timeout)
                except requests.RequestException as exc:
                    retry_errors.append(f"{entry['channel']}:network:{type(exc).__name__}")
                    load_failures += 1
                    auth.set_cooldown(key, NETWORK_COOLDOWN_SECONDS, kind="network")
                    continue
                if resp.status_code == 200:
                    parsed = RouteDispatcher._parse_response(resp)
                    if parsed is None:
                        retry_errors.append(f"{entry['channel']}:unparseable")
                        break
                    text, input_tokens, output_tokens, diag = parsed
                    auth.report_success(key)
                    record = {
                        "_text": text,
                        "worker_id": self.worker_id,
                        "channel_id": entry["channel"],
                        "api_model_name": entry["model"],
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": diag.get("cached_input_tokens", 0),
                        "success": True,
                        "error_code": None,
                        "record_role": record_role,
                        "tag": tag,
                        "final_content_missing": diag.get("final_content_missing", False),
                        "latency_seconds": round(time.monotonic() - started, 3),
                        **failed,
                    }
                    return record, attempt_count, retry_errors
                status_class, cooldown = _classify_status(resp.status_code)
                retry_errors.append(f"{entry['channel']}:{resp.status_code}")
                usage = RouteDispatcher._usage_only(resp)
                if usage:
                    failed["failed_input_tokens"] += usage[0]
                    failed["failed_output_tokens"] += usage[1]
                    failed["failed_cached_input_tokens"] += usage[2]
                if status_class == "fatal":
                    break  # e.g. 402/404: a fact about the channel, not the key
                if status_class != "auth":
                    load_failures += 1
                auth.set_cooldown(key, cooldown, kind=status_class)
        return None, attempt_count, retry_errors

    def probe(self):
        """Send one cheap request; raise LLMCallError if no channel answers."""
        text, record = self.chat(
            [{"role": "user", "content": "Reply with exactly: pong"}],
            tag="probe", max_tokens=8)
        return {"text": text, "channel": record["channel_id"],
                "latency_seconds": record["latency_seconds"]}
