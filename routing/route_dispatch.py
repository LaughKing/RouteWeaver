"""Multi-channel provider dispatch for the router's worker calls."""

import json
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from route_service import AGENT_PROMPT
from worker_registry import list_worker_ids

_QUERY_ANCHOR = "Here is the sub-question for you to assist with:"
assert _QUERY_ANCHOR in AGENT_PROMPT
WORKER_PROMPT = AGENT_PROMPT.replace(
    _QUERY_ANCHOR,
    "Answer the following subtask directly and concisely. Do NOT show your reasoning process. "
    + _QUERY_ANCHOR)

DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'route_channels.yaml'))

MAX_ATTEMPTS_PER_ROUTE = 4
MAX_ATTEMPTS_PER_JOB = MAX_ATTEMPTS_PER_ROUTE
JOB_TIME_BUDGET_SECONDS = 300.0

MAX_ATTEMPTS_PER_CHANNEL = 2

DEADLINE_SAFETY_MARGIN_SECONDS = 5.0
MIN_ATTEMPT_SECONDS = 10.0

AUTH_COOLDOWN_SECONDS = 1800.0
RATE_LIMIT_COOLDOWN_SECONDS = 30.0
CAPACITY_COOLDOWN_SECONDS = 20.0
SERVER_ERROR_COOLDOWN_SECONDS = 20.0
NETWORK_COOLDOWN_SECONDS = 15.0

EXHAUSTED_WAIT_CAP_SECONDS = 30.0
EXHAUSTED_WAIT_MIN_SECONDS = 0.5
EXHAUSTED_WAIT_BACKOFF = 1.8
EXHAUSTED_MAX_WAITS = 6

REQUEST_MAX_TOKENS = 512
REQUEST_TEMPERATURE = 0.0
REQUEST_SEED = 42

_EXHAUSTED = object()


def _key_tag(key: Optional[str]) -> str:
    """A stable, non-reversible label for a credential, for logs and metrics."""
    if not key:
        return "none"
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class DispatchErrorCode(Enum):
    NO_ENABLED_CHANNEL = 'no_enabled_channel'
    ALL_CHANNELS_FAILED = 'all_channels_failed'
    JOB_BUDGET_EXCEEDED = 'job_budget_exceeded'


@dataclass(frozen=True)
class DispatchResult:
    text: str
    worker_id: str
    channel_id: Optional[str]       # channel that served the call; None on failure
    api_model_name: Optional[str]   # channel-specific model name actually used
    input_tokens: int
    output_tokens: int
    error_code: Optional[DispatchErrorCode] = None
    final_content_missing: bool = False
    reasoning_present: bool = False
    reasoning_chars: int = 0
    cached_input_tokens: int = 0
    initial_provider: Optional[str] = None
    final_provider: Optional[str] = None
    attempt_count: int = 0
    keys_tried: int = 0
    retry_error_codes: Tuple[str, ...] = ()
    fallback_used: bool = False
    final_success: bool = False
    latency_seconds: float = 0.0
    failed_input_tokens: int = 0
    failed_output_tokens: int = 0
    failed_cached_input_tokens: int = 0


def _classify_status(status_code: int) -> Tuple[str, Optional[float]]:
    """Map an HTTP status to (class, cooldown_seconds). cooldown None means the
    key is fine but this channel cannot serve the request (e.g. 404/410).
    """
    if status_code in (401, 403):
        return 'auth', AUTH_COOLDOWN_SECONDS
    if status_code == 429:
        return 'rate_limit', RATE_LIMIT_COOLDOWN_SECONDS
    if status_code == 503:
        return 'capacity', CAPACITY_COOLDOWN_SECONDS
    if status_code >= 500:
        return 'server_error', SERVER_ERROR_COOLDOWN_SECONDS
    return 'fatal', None


class _NoAuth:
    """Channels without authentication (local vLLM). Still carries a cooldown
    slot so a down server is skipped instead of hammered.
    """

    def __init__(self, now=time.monotonic):
        self._now = now
        self._cooldown_until = 0.0
        self._rejects_us = False
        self._lock = threading.Lock()

    def acquire(self, worker_id, last_resort=False, retry_budget_left=0):
        with self._lock:
            if self._rejects_us or self._cooldown_until > self._now():
                return _EXHAUSTED
            return None

    def wait_hint(self, last_resort=False):
        """Seconds until acquire() could succeed, or None if it never will."""
        with self._lock:
            if self._rejects_us:
                return None       # misconfigured endpoint; waiting cannot fix it
            return max(0.0, self._cooldown_until - self._now())

    def report_success(self, key):
        pass

    def set_cooldown(self, key, seconds, kind='transient'):
        with self._lock:
            self._cooldown_until = self._now() + seconds
            if kind == 'auth':
                self._rejects_us = True


class _SingleKeyFile:
    """One key, read from one line of a plain-text file."""

    def __init__(self, path, now=time.monotonic):
        self._path = os.path.expanduser(path)
        self._now = now
        self._key = None
        self._cooldown_until = 0.0
        self._dead = set()
        self._lock = threading.Lock()

    def acquire(self, worker_id, last_resort=False, retry_budget_left=0):
        with self._lock:
            if (self._cooldown_until > self._now()
                    and not last_resort and retry_budget_left <= 0):
                return _EXHAUSTED
            if self._key is None:
                self._key = self._read_key()
            if self._key in self._dead:
                self._key = None
                return _EXHAUSTED
            return self._key

    def _read_key(self):
        with open(self._path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
        raise ValueError(f"No key found in {self._path}")

    def wait_hint(self, last_resort=False):
        """Seconds until acquire() could succeed, or None if it never will."""
        with self._lock:
            if self._key is not None and self._key in self._dead:
                return None
            return max(0.0, self._cooldown_until - self._now())

    def report_success(self, key):
        pass

    def set_cooldown(self, key, seconds, kind='transient'):
        with self._lock:
            self._cooldown_until = self._now() + seconds
            if kind == 'auth' and key is not None:
                self._dead.add(key)
            self._key = None


_AUTH_CACHE: Dict[Tuple[str, str], Any] = {}
_AUTH_CACHE_LOCK = threading.Lock()


def _make_auth(auth_cfg: Dict[str, Any], now=time.monotonic):
    auth_type = auth_cfg.get('type', 'none')
    if auth_type == 'none':
        return _NoAuth(now=now)
    if auth_type != 'single_key_file':
        raise ValueError(f"Unknown auth type: {auth_type!r}")
    cache_key = (auth_type, os.path.expanduser(auth_cfg['path']))
    with _AUTH_CACHE_LOCK:
        if cache_key not in _AUTH_CACHE:
            _AUTH_CACHE[cache_key] = _SingleKeyFile(auth_cfg['path'], now=now)
        return _AUTH_CACHE[cache_key]


class RouteDispatcher:
    """Resolve (RouteCall, WorkerSpec) jobs to DispatchResults over the channels."""

    def __init__(self, config_path=None, post=requests.post, now=time.monotonic,
                 pool_size=2, sleep=time.sleep):
        path = os.path.expanduser(
            config_path or os.environ.get('ROUTE_CHANNEL_CONFIG', DEFAULT_CONFIG_PATH))
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.channels: Dict[str, Dict] = cfg['channels']
        self.workers: Dict[str, List[Dict]] = cfg['workers']
        self._post, self._now = post, now
        self._sleep = sleep
        self._pool_size = pool_size
        self._auth = {name: _make_auth(ch.get('auth', {'type': 'none'}), now=now)
                      for name, ch in self.channels.items()}

        unreachable = [w for w in list_worker_ids() if not self._enabled_entries(w)]
        if unreachable:
            print(f"[route_dispatch WARNING] workers with no enabled channel "
                  f"(routes to them will get empty observations): {unreachable}")


    def _enabled_entries(self, worker_id: str) -> List[Dict]:
        return [entry for entry in self.workers.get(worker_id, [])
                if self.channels.get(entry['channel'], {}).get('enabled', True)]

    def dispatch(self, jobs: List[Tuple[Any, Any]]) -> List[DispatchResult]:
        """Order-aligned with jobs; every job yields a DispatchResult, never an
        exception -- rollout must not crash on dispatch problems.
        """
        if not jobs:
            return []
        with ThreadPool(min(self._pool_size, len(jobs))) as pool:
            return pool.map(self._safe_run_job, jobs)

    @staticmethod
    def _failure(spec, code: DispatchErrorCode, trace=None) -> DispatchResult:
        trace = trace or {}
        return DispatchResult(text='', worker_id=spec.model_id, channel_id=None,
                              api_model_name=None, input_tokens=0, output_tokens=0,
                              error_code=code, final_success=False, **trace)

    def _safe_run_job(self, job) -> DispatchResult:
        try:
            return self._run_job(job)
        except Exception as exc:  # noqa: broad by design -- never crash rollout
            spec = job[1]
            print(f"[route_dispatch ERROR] unexpected failure for {spec.model_id!r}: {exc!r}")
            return self._failure(spec, DispatchErrorCode.ALL_CHANNELS_FAILED)

    def _run_job(self, job) -> DispatchResult:
        route, spec = job[0], job[1]
        entries = self._enabled_entries(spec.model_id)
        started = self._now()
        if not entries:
            return self._failure(spec, DispatchErrorCode.NO_ENABLED_CHANNEL)

        content = WORKER_PROMPT.format_map({'query': route.query})

        initial_provider = entries[0]['channel']
        attempt_count = 0
        keys_seen = set()
        retry_errors: List[str] = []
        failed_usage = {'failed_input_tokens': 0, 'failed_output_tokens': 0,
                        'failed_cached_input_tokens': 0}

        def trace(final_channel=None):
            return dict(initial_provider=initial_provider,
                        final_provider=final_channel,
                        attempt_count=attempt_count,
                        keys_tried=len(keys_seen),
                        retry_error_codes=tuple(retry_errors),
                        fallback_used=bool(final_channel and final_channel != initial_provider),
                        latency_seconds=round(self._now() - started, 3),
                        **failed_usage)

        def record_failed_usage(resp):
            """Charge-worthy usage from an attempt that produced no observation."""
            usage = self._usage_only(resp)
            if not usage:
                return
            failed_usage['failed_input_tokens'] += usage[0]
            failed_usage['failed_output_tokens'] += usage[1]
            failed_usage['failed_cached_input_tokens'] += usage[2]

        deadline = started + JOB_TIME_BUDGET_SECONDS

        def remaining():
            return deadline - self._now()

        attempts_left = MAX_ATTEMPTS_PER_ROUTE
        for entry_index, entry in enumerate(entries):
            last_resort = entry_index == len(entries) - 1
            channel = self.channels[entry['channel']]
            auth = self._auth[entry['channel']]
            payload = {
                'model': entry['model'],
                'messages': [{'role': 'user', 'content': content}],
                'temperature': REQUEST_TEMPERATURE,
                'seed': REQUEST_SEED,
                'max_tokens': entry.get('max_tokens', REQUEST_MAX_TOKENS),
            }
            payload.update(entry.get('extra_body') or {})
            request_timeout = entry.get('timeout', channel.get('timeout', 120))
            load_failures = 0
            waits_used = 0
            wait_backoff = EXHAUSTED_WAIT_MIN_SECONDS
            while True:
                left = remaining()
                if attempts_left <= 0 or left <= MIN_ATTEMPT_SECONDS:
                    print(f"[route_dispatch] route budget exhausted for "
                          f"{spec.model_id!r} ({left:.1f}s left, "
                          f"{attempt_count} attempts made)")
                    return self._failure(spec, DispatchErrorCode.JOB_BUDGET_EXCEEDED,
                                         trace())
                if load_failures >= MAX_ATTEMPTS_PER_CHANNEL:
                    print(f"[route_dispatch] {entry['channel']} gave "
                          f"{load_failures} overloaded attempts for "
                          f"{spec.model_id!r}; moving on")
                    break
                key = auth.acquire(route.model_id, last_resort=last_resort,
                                   retry_budget_left=MAX_ATTEMPTS_PER_CHANNEL - load_failures)
                if key is _EXHAUSTED:
                    if not last_resort:
                        break  # another channel could serve this; go there now
                    hint = (auth.wait_hint(last_resort=last_resort)
                            if hasattr(auth, 'wait_hint') else None)
                    if hint is None:
                        print(f"[route_dispatch] {entry['channel']} has no "
                              f"recoverable key for {spec.model_id!r}; not waiting")
                        break
                    if waits_used >= EXHAUSTED_MAX_WAITS:
                        print(f"[route_dispatch] {entry['channel']} still cooling "
                              f"after {waits_used} waits for {spec.model_id!r}; "
                              f"giving up")
                        break
                    nap = min(max(hint, wait_backoff), EXHAUSTED_WAIT_CAP_SECONDS,
                              left - MIN_ATTEMPT_SECONDS)
                    if nap <= 0:
                        break  # no budget left to wait AND still attempt
                    waits_used += 1
                    wait_backoff *= EXHAUSTED_WAIT_BACKOFF
                    retry_errors.append(f"{entry['channel']}:wait")
                    self._sleep(nap)
                    continue
                waits_used = 0          # a key was handed out; the wait paid off
                wait_backoff = EXHAUSTED_WAIT_MIN_SECONDS
                attempt_timeout = min(request_timeout, left - DEADLINE_SAFETY_MARGIN_SECONDS)
                attempts_left -= 1
                attempt_count += 1
                keys_seen.add(_key_tag(key))
                headers = {'Authorization': f'Bearer {key}'} if key else {}
                try:
                    resp = self._post(channel['base_url'], headers=headers, json=payload,
                                      timeout=attempt_timeout)
                except requests.RequestException as exc:
                    print(f"[route_dispatch] {entry['channel']} network error "
                          f"(key {_key_tag(key)}): {type(exc).__name__}")
                    retry_errors.append(f"{entry['channel']}:network")
                    load_failures += 1
                    auth.set_cooldown(key, NETWORK_COOLDOWN_SECONDS, kind='network')
                    continue  # next healthy key immediately, or next channel
                if resp.status_code == 200:
                    parsed = self._parse_response(resp)
                    if parsed is None:
                        retry_errors.append(f"{entry['channel']}:unparseable")
                        record_failed_usage(resp)
                        break  # unusable body from this channel; try the next one
                    text, input_tokens, output_tokens, diagnostics = parsed
                    auth.report_success(key)
                    return DispatchResult(text=text, worker_id=spec.model_id,
                                          channel_id=entry['channel'],
                                          api_model_name=entry['model'],
                                          input_tokens=input_tokens,
                                          output_tokens=output_tokens,
                                          final_success=True,
                                          **diagnostics,
                                          **trace(entry['channel']))
                status_class, cooldown = _classify_status(resp.status_code)
                retry_errors.append(f"{entry['channel']}:{resp.status_code}")
                record_failed_usage(resp)
                print(f"[route_dispatch] {entry['channel']} {status_class} "
                      f"({resp.status_code}) key {_key_tag(key)}")
                if status_class == 'fatal':
                    break  # e.g. 404/410: key is fine, channel can't serve this model
                if status_class == 'auth':
                    attempts_left += 1
                else:
                    load_failures += 1
                auth.set_cooldown(key, cooldown, kind=status_class)

        return self._failure(spec, DispatchErrorCode.ALL_CHANNELS_FAILED, trace())

    @staticmethod
    def _usage_only(resp) -> Optional[Tuple[int, int, int]]:
        """(input, output, cached) from a body that produced no observation."""
        try:
            usage = (resp.json() or {}).get('usage') or {}
        except Exception:
            return None
        if not usage:
            return None
        input_tokens = int(usage.get('prompt_tokens', 0) or 0)
        output_tokens = int(usage.get('completion_tokens', 0) or 0)
        details = usage.get('prompt_tokens_details') or {}
        cached = usage.get('prompt_cache_hit_tokens')
        if cached is None:
            cached = details.get('cached_tokens')
        cached = max(0, min(int(cached or 0), input_tokens))
        if not (input_tokens or output_tokens):
            return None
        return input_tokens, output_tokens, cached

    @staticmethod
    def _parse_response(resp) -> Optional[Tuple[str, int, int, dict]]:
        """The observation is message.content, and ONLY message.content."""
        try:
            data = resp.json()
            message = data['choices'][0]['message']
            content = (message.get('content') or '').strip()
            reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
            usage = data.get('usage') or {}
            input_tokens = int(usage.get('prompt_tokens', 0) or 0)
            output_tokens = int(usage.get('completion_tokens', 0) or 0)
            details = usage.get('prompt_tokens_details') or {}
            cached = usage.get('prompt_cache_hit_tokens')
            if cached is None:
                cached = details.get('cached_tokens')
            cached_input_tokens = int(cached or 0)
            cached_input_tokens = max(0, min(cached_input_tokens, input_tokens))
            diagnostics = {
                'final_content_missing': not content,
                'reasoning_present': bool(reasoning),
                'reasoning_chars': len(reasoning),
                'cached_input_tokens': cached_input_tokens,
            }
            if diagnostics['final_content_missing'] and diagnostics['reasoning_present']:
                print(f"[route_dispatch] worker returned reasoning but no final "
                      f"content ({diagnostics['reasoning_chars']} reasoning chars "
                      f"dropped); observation will be empty")
            return content, input_tokens, output_tokens, diagnostics
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"[route_dispatch] unparseable response: {exc}")
            return None
