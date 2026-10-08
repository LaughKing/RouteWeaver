"""Multi-channel provider dispatch for the router's worker calls.

Channel topology (base URLs, key sources, per-worker candidate channels and
priority) lives in a runtime YAML config (configs/route_channels.yaml, override
via constructor arg or ROUTE_CHANNEL_CONFIG env var) -- never hardcoded here.
API keys are referenced by path in that config, never inlined. Workers are not
bound to a provider: a worker may list several candidate channels, tried in
priority order.

Failure handling never sleeps in-line: a failing key/channel receives a
status-dependent cooldown and the job immediately moves on to the next
channel. Per-job attempt and time budgets bound the worst case so one bad
worker cannot stall a rollout batch. Cooldowns are runtime state only, never
persisted, and they expire on their own; the one permanent verdict is a 401,
which retires that credential for the rest of the process.

Pricing is deliberately absent here: results carry raw token usage and the
(worker_id, channel_id, api_model_name) provenance. Pricing is fixed per
worker_id (rewards/cost_model.py) regardless of which channel served the call.

Concurrency note (Ray): run_llm_loop -- and therefore dispatch -- executes in
the trainer driver process, not inside Ray remote workers; only
generate_sequences is remoted. Auth/cooldown state is shared between
dispatcher instances in the same process (train + validation managers) via a
module-level cache keyed by (auth type, key path). If dispatch is ever moved
into per-actor Ray processes, each process maintains its own independent
cooldown state (each actor re-discovers a dead credential once); true cross-process
sharing would need an external store and is out of scope.

A job is (RouteCall, WorkerSpec) and the request body is always plain text:
the router has no way to express an image route and no channel is configured
to carry one.
"""

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

# Grammar-chain worker prompt: legacy AGENT_PROMPT plus a directness directive.
# A reasoning-series worker otherwise dumps its chain of thought into content
# and burns the whole token budget before the answer
# (observation truncation keeps the head, so a late answer never survives).
# Anchored insertion so legacy route_service stays untouched.
_QUERY_ANCHOR = "Here is the sub-question for you to assist with:"
assert _QUERY_ANCHOR in AGENT_PROMPT
WORKER_PROMPT = AGENT_PROMPT.replace(
    _QUERY_ANCHOR,
    "Answer the following subtask directly and concisely. Do NOT show your reasoning process. "
    + _QUERY_ANCHOR)

DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'route_channels.yaml'))

# Route-level budgets. Two independent bounds, because either one alone has
# been observed to fail:
#
#   * a static attempt cap says nothing about wall clock when a provider hangs;
#   * a wall-clock budget checked only BETWEEN attempts lets the last attempt
#     start just under the line and then run its own full timeout past it,
#     which overruns the budget by a whole channel timeout.
#
# So the deadline is enforced INSIDE each attempt as well: every request gets
# timeout = min(channel timeout, remaining - safety margin), which makes the
# budget the real ceiling no matter how the channel timeouts are configured.
MAX_ATTEMPTS_PER_ROUTE = 4
MAX_ATTEMPTS_PER_JOB = MAX_ATTEMPTS_PER_ROUTE      # legacy name, same bound
# Must stay comfortably ABOVE any single request timeout, or one hung request
# eats the whole budget and the fallback is never asked. Shortening the channel
# timeout instead is worse: a worker that genuinely takes a minute then fails
# where it would have answered. The budget is the right knob, and it holds the
# slowest channel timeout twice over -- room for the bounded retries plus a
# fallback.
JOB_TIME_BUDGET_SECONDS = 300.0

# Per-CHANNEL attempt cap. The job-level cap alone is not enough: a worker
# whose only channel is cooling down can spend every attempt inside that one
# channel and never reach its fallback. A couple of attempts is the point where
# "this moment is unlucky" stops being a better explanation than "this provider
# is saturated right now".
MAX_ATTEMPTS_PER_CHANNEL = 2

# Time reserved for finishing the job after the last request returns, and the
# shortest attempt worth starting at all. A request given 2s is a request that
# will time out and cost an attempt for nothing.
DEADLINE_SAFETY_MARGIN_SECONDS = 5.0
MIN_ATTEMPT_SECONDS = 10.0

# status-dependent cooldowns (seconds). A cooled key is skipped rather than
# slept on WHILE ANOTHER CHANNEL COULD SERVE THE JOB -- see the wait loop in
# _run_job for what happens when none can.
AUTH_COOLDOWN_SECONDS = 1800.0
RATE_LIMIT_COOLDOWN_SECONDS = 30.0
CAPACITY_COOLDOWN_SECONDS = 20.0
SERVER_ERROR_COOLDOWN_SECONDS = 20.0
NETWORK_COOLDOWN_SECONDS = 15.0

# Waiting for a credential to come back, on a channel that is the ONLY path to
# the worker. Skipping instead of waiting is correct when there is somewhere
# else to go; on a single-key channel it is not, and the difference is not
# small: one 503 cools the key, and every route arriving inside that window is
# failed WITHOUT ISSUING A REQUEST. The cooldown is a statement about the
# provider's near future, so on a single-path worker the right response is to
# wait it out, not to hand the policy an empty observation and let it learn
# that the worker is bad.
#
# Bounds, in order of which one usually bites: the route deadline
# (JOB_TIME_BUDGET_SECONDS) caps total wall clock, EXHAUSTED_MAX_WAITS caps how
# many times one route may wait, and EXHAUSTED_WAIT_CAP_SECONDS keeps a single
# sleep from swallowing the whole budget in one go. Backoff is multiplicative
# so a provider that is genuinely down is asked progressively less often.
EXHAUSTED_WAIT_CAP_SECONDS = 30.0
EXHAUSTED_WAIT_MIN_SECONDS = 0.5
EXHAUSTED_WAIT_BACKOFF = 1.8
EXHAUSTED_MAX_WAITS = 6

REQUEST_MAX_TOKENS = 512
REQUEST_TEMPERATURE = 0.0
REQUEST_SEED = 42

# auth.acquire() sentinel: every key of this channel is cooling down right now
_EXHAUSTED = object()


def _key_tag(key: Optional[str]) -> str:
    """A stable, non-reversible label for a credential, for logs and metrics.

    Keys must never reach a log file, a rollout dump or wandb. Eight hex
    characters of a digest are enough to follow one credential across a run
    and useless for authenticating as it.
    """
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
    # Structured facts about a reasoning-series reply. Deliberately NOT the
    # reasoning text: whether it existed and how long it was is enough to
    # diagnose a truncated worker, and the text itself must never travel any
    # further than this process (see _parse_response).
    final_content_missing: bool = False
    reasoning_present: bool = False
    reasoning_chars: int = 0
    # Cached prompt tokens, as reported by the provider. A SUBSET of
    # input_tokens, never additional to it -- billing both at full price would
    # charge the same tokens twice.
    cached_input_tokens: int = 0
    # Provenance of the attempt chain. Which provider was asked first, which
    # one answered, how many requests it took and why each earlier one failed.
    # Without this a fallback is invisible: the record would say "deepseek
    # answered" whether that took one request or six, over either provider.
    initial_provider: Optional[str] = None
    final_provider: Optional[str] = None
    attempt_count: int = 0
    # distinct credentials this job used: whether a retry moved to another
    # key or hit the same one again
    keys_tried: int = 0
    retry_error_codes: Tuple[str, ...] = ()
    fallback_used: bool = False
    final_success: bool = False
    latency_seconds: float = 0.0
    # Usage reported by attempts that did NOT produce the observation. Billed
    # in addition to the successful attempt (the provider charged for them),
    # never instead of it.
    failed_input_tokens: int = 0
    failed_output_tokens: int = 0
    failed_cached_input_tokens: int = 0


def _classify_status(status_code: int) -> Tuple[str, Optional[float]]:
    """Map an HTTP status to (class, cooldown_seconds). cooldown None means the
    key is fine but this channel cannot serve the request (e.g. 404/410)."""
    if status_code in (401, 403):
        return 'auth', AUTH_COOLDOWN_SECONDS
    if status_code == 429:
        return 'rate_limit', RATE_LIMIT_COOLDOWN_SECONDS
    if status_code == 503:
        return 'capacity', CAPACITY_COOLDOWN_SECONDS
    if status_code >= 500:
        return 'server_error', SERVER_ERROR_COOLDOWN_SECONDS
    # 404 stays fatal for the channel: a provider that does not serve this
    # model answers 404 however often it is asked. It is a fact about the
    # channel, not about the credential.
    return 'fatal', None


class _NoAuth:
    """Channels without authentication (local vLLM). Still carries a cooldown
    slot so a down server is skipped instead of hammered."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self._cooldown_until = 0.0
        # An unauthenticated endpoint that answers 401 is misconfigured, not
        # busy; waiting cannot fix it and the caller refunds the route budget
        # for every auth failure, so without this the walk is unbounded.
        self._rejects_us = False
        self._lock = threading.Lock()

    def acquire(self, worker_id, last_resort=False, retry_budget_left=0):
        # last_resort and retry_budget_left are accepted for protocol
        # uniformity and ignored: a single unauthenticated endpoint has nothing
        # else to try, so it falls over on the first failure.
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
    """One key, read from one line of a plain-text file.

    A cooldown on a ONE-key channel cannot mean "try another key": there is
    only one credential, so the signal only ever means "wait", and waiting is
    not something dispatch does. Skipping the attempt instead would abandon the
    channel after a single transient failure, so the key is handed out while
    the channel's own bounded retry budget remains
    (MAX_ATTEMPTS_PER_CHANNEL), or on the final candidate channel where giving
    up produces an empty observation rather than a slow answer. The budget is
    what bounds the hammering; the cooldown never could.

    The cached key is dropped on failure, so a credential rotated on disk is
    picked up on the next attempt without restarting training.

    A credential the provider REJECTED (401/403) is refused for good. It is a
    verdict on the key, not on the moment, and the caller refunds the route
    budget on an auth failure -- so re-reading the same dead key would loop
    until the job deadline stopped it.

    Blank lines and '#' comments are skipped, so the file can carry a "do not
    commit this" header without that header becoming the credential.
    """

    def __init__(self, path, now=time.monotonic):
        self._path = os.path.expanduser(path)
        self._now = now
        self._key = None
        self._cooldown_until = 0.0
        # Credentials this channel has seen rejected outright. Held by VALUE,
        # not by a flag on the channel, so rotating the file still works: a new
        # key reads differently and is tried, the dead one never is again.
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
                # Drop the cache as well, or a key rotated on disk is never
                # seen: the refusal would keep answering from the dead value.
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
        """Seconds until acquire() could succeed, or None if it never will.

        A key the provider REJECTED is refused by acquire() for good, so
        waiting on it is waiting for nothing. Any other cooldown expires.
        """
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
                # 401/403 is a verdict on the credential, not on the moment.
                self._dead.add(key)
            self._key = None




# Shared across dispatcher instances in this process (train + validation), so
# cooldown knowledge learned by one is visible to the other. Keyed by
# (auth type, path); 'none' auth has no shareable state and is not cached.
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
    """Resolves (RouteCall, WorkerSpec) jobs to DispatchResults over the
    configured channels. post/now are injectable for tests."""

    # pool_size is the ONLY throttle between the rollout and the providers,
    # and it is deliberately small. Too many concurrent calls turn into
    # provider-side failures, which reach the policy as empty observations --
    # i.e. as "this worker is unreliable" rather than as a channel that was
    # asked too fast. Rollout-level concurrency is set separately, per run.
    def __init__(self, config_path=None, post=requests.post, now=time.monotonic,
                 pool_size=2, sleep=time.sleep):
        path = os.path.expanduser(
            config_path or os.environ.get('ROUTE_CHANNEL_CONFIG', DEFAULT_CONFIG_PATH))
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.channels: Dict[str, Dict] = cfg['channels']
        self.workers: Dict[str, List[Dict]] = cfg['workers']
        self._post, self._now = post, now
        # injectable so the wait loop is testable without real wall clock
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
        exception -- rollout must not crash on dispatch problems."""
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

        # attempt-chain provenance, carried into the route record whatever the
        # outcome. Every candidate channel serves the SAME canonical worker at
        # the same settings (configs/route_channels.yaml), so a fallback changes
        # who answered, never what was asked.
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
            """Charge-worthy usage from an attempt that produced no observation.

            Most non-200 bodies carry none; when one does, the provider still
            billed it, and leaving it out understates cost by exactly the
            amount that retries add.
            """
            usage = self._usage_only(resp)
            if not usage:
                return
            failed_usage['failed_input_tokens'] += usage[0]
            failed_usage['failed_output_tokens'] += usage[1]
            failed_usage['failed_cached_input_tokens'] += usage[2]

        # ROUTE-level hard deadline. Every attempt on every channel shares it,
        # and it bounds time INSIDE a request as well as between requests.
        deadline = started + JOB_TIME_BUDGET_SECONDS

        def remaining():
            return deadline - self._now()

        attempts_left = MAX_ATTEMPTS_PER_ROUTE
        for entry_index, entry in enumerate(entries):
            # on the final candidate channel there is nowhere left to fall
            # back to, so provider-level cooldowns are bypassed and any healthy
            # key is used
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
            # per-entry provider params from runtime config, e.g. thinking
            # suppression ({"thinking": {"type": "disabled"}}); this is the
            # default-"direct" execution mode -- the future skill attribute
            # will expose this switch to the policy
            payload.update(entry.get('extra_body') or {})
            # A worker entry may cap its own request timeout below the channel
            # default. The channel value is set for the workers that use it as
            # a PRIMARY; a fallback to an endpoint measured hanging needs a
            # tighter bound, because a hung fallback blocks the whole rollout
            # batch behind a dispatch pool of 2.
            request_timeout = entry.get('timeout', channel.get('timeout', 120))
            # Only LOAD failures (429/5xx/timeout) count against the channel
            # budget. A 401 is a verdict on the credential, not on the
            # provider: the credential is refused from then on, so walking past
            # it costs one request and settles the question, where charging it
            # to the retry budget would spend the budget on a channel that
            # cannot answer at all.
            load_failures = 0
            waits_used = 0
            wait_backoff = EXHAUSTED_WAIT_MIN_SECONDS
            while True:
                left = remaining()
                if attempts_left <= 0 or left <= MIN_ATTEMPT_SECONDS:
                    # Not enough budget to be worth starting: an attempt given
                    # a few seconds only buys a timeout and spends a retry.
                    print(f"[route_dispatch] route budget exhausted for "
                          f"{spec.model_id!r} ({left:.1f}s left, "
                          f"{attempt_count} attempts made)")
                    return self._failure(spec, DispatchErrorCode.JOB_BUDGET_EXCEEDED,
                                         trace())
                if load_failures >= MAX_ATTEMPTS_PER_CHANNEL:
                    # This channel has had its turn. Falling through now is the
                    # whole point: retrying a saturated provider a fourth time
                    # costs a request and yields the same 529, while the next
                    # channel is idle.
                    print(f"[route_dispatch] {entry['channel']} gave "
                          f"{load_failures} overloaded attempts for "
                          f"{spec.model_id!r}; moving on")
                    break
                key = auth.acquire(route.model_id, last_resort=last_resort,
                                   retry_budget_left=MAX_ATTEMPTS_PER_CHANNEL - load_failures)
                if key is _EXHAUSTED:
                    if not last_resort:
                        break  # another channel could serve this; go there now
                    # Nowhere left to fall back to. Giving up here is what turns
                    # a 20s provider cooldown into an empty observation, and an
                    # empty observation is indistinguishable to the policy from
                    # "this worker answered badly" -- so it teaches the wrong
                    # thing about a worker that was never actually asked. Wait
                    # for a key instead, bounded three ways (see the constants).
                    hint = (auth.wait_hint(last_resort=last_resort)
                            if hasattr(auth, 'wait_hint') else None)
                    if hint is None:
                        # no key can ever come back (all rejected outright)
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
                    # The deadline wins over the channel timeout. Without this
                    # the last attempt starts just under the budget and then
                    # runs its own full timeout past it -- the
                    # failure mode exactly.
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
                    # Refund the route budget: a 401 retires the credential,
                    # so walking past it is progress rather than a retry, and
                    # the attempts stay available for a channel that can
                    # actually answer. This cannot loop -- the retired
                    # credential makes the next acquire() return _EXHAUSTED --
                    # and the route deadline bounds the wall clock anyway.
                    attempts_left += 1
                else:
                    load_failures += 1
                auth.set_cooldown(key, cooldown, kind=status_class)
                # no sleeping: move straight to the next healthy key / channel

        return self._failure(spec, DispatchErrorCode.ALL_CHANNELS_FAILED, trace())

    @staticmethod
    def _usage_only(resp) -> Optional[Tuple[int, int, int]]:
        """(input, output, cached) from a body that produced no observation.

        Never raises and never touches message content -- a failed attempt's
        body is not an answer, and the only thing worth taking from it is what
        the provider says it charged.
        """
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
        """The observation is message.content, and ONLY message.content.

        Reasoning-series workers return their chain of thought in a separate
        `reasoning_content` field. This used to fall back to it when content
        was empty, which meant a worker that spent its whole budget thinking
        handed the router its raw reasoning as if it were an answer -- text
        that then entered the trajectory, the next turn's prompt and the
        rollout dump. Under the frozen direct-worker setting reasoning is
        suppressed and the field does not appear, but the guarantee must not
        depend on a provider honouring a flag.

        An empty content with reasoning present is a worker that never produced
        a final answer. That is reported as such (final_content_missing) and
        handled by the caller's existing empty-observation path; the reasoning
        text is dropped here and never leaves this function. Its presence and
        length are kept as structured diagnostics.
        """
        try:
            data = resp.json()
            message = data['choices'][0]['message']
            content = (message.get('content') or '').strip()
            reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
            usage = data.get('usage') or {}
            input_tokens = int(usage.get('prompt_tokens', 0) or 0)
            # completion_tokens already INCLUDES reasoning_tokens, so
            # reasoning must not be added again here.
            output_tokens = int(usage.get('completion_tokens', 0) or 0)
            # Cached input, reported under two different names depending on
            # the provider -- and some report both for the same call, so the
            # two are the SAME number and must not be summed. Absent on
            # providers with no prompt caching -> 0.
            details = usage.get('prompt_tokens_details') or {}
            cached = usage.get('prompt_cache_hit_tokens')
            if cached is None:
                cached = details.get('cached_tokens')
            cached_input_tokens = int(cached or 0)
            # never let a provider report more cache than prompt
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
