"""Worker-level early exploration: a prompt-level FORCED WORKER curriculum.

Same shape as exploration_schedule.py, and for the same reasons. Everything is a
deterministic function of (sample_id, batch_index) taken from the parquet, so:

  * every rollout of a query gets the SAME verdict and the SAME worker, which
    keeps a GRPO group inside one prompt (group_key is 1:1 with sample_id, and
    grpo_gated takes std within the group);
  * a resume reproduces the schedule bit-for-bit -- nothing is sampled at
    runtime, and batch_index comes from the row, not from a counter.

The forced worker is expressed ONLY as a sentence in the turn-2 injected rules.
The full candidate list is still shown, the router still writes the whole
<route model="..."> itself, and the reward is defined over the tokens it
actually emitted. Nothing rewrites an action after the fact: substituting the
dispatched model would make the reward a statement about a trajectory the
policy never produced.

Compliance is therefore NOT guaranteed, only measured: the loop records
forced_worker alongside the ids actually written.
"""
import hashlib
import os

def _f(name, d): return float(os.environ.get(name, d))
def _i(name, d): return int(os.environ.get(name, d))
BETA0 = _f("WORKER_EXPLORE_BETA0", "0.75")
STEPS = int(os.environ.get("WORKER_EXPLORE_STEPS", "15"))
START = int(os.environ.get("WORKER_EXPLORE_START", "0"))
ENABLED = os.environ.get("WORKER_EXPLORE", "0") == "1"
SHUFFLE = os.environ.get("WORKER_ORDER_SHUFFLE", "0") == "1"


def _u(salt, key):
    """Uniform in [0,1) from a stable hash. No RNG state, no run-to-run drift."""
    h = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def beta_for(batch_index, beta0=None, steps=None, start=None):
    """Fraction of QUERIES that carry a forced worker at this step."""
    b0 = _f("WORKER_EXPLORE_BETA0", "0.75") if beta0 is None else beta0
    n = _i("WORKER_EXPLORE_STEPS", "15") if steps is None else steps
    s = _i("WORKER_EXPLORE_START", "0") if start is None else start
    t = int(batch_index) - int(s)
    return max(0.0, min(1.0, b0 * (1.0 - t / float(n))))


def forced_worker(sample_id, batch_index, pool, beta0=None, steps=None, start=None):
    """The worker this query is told to use, or None for free generation.

    Target is a uniform round-robin over the pool keyed on sample_id alone, so
    coverage is even and does not follow the policy's own preferences.
    """
    if not sample_id or not pool:
        return None
    if _u("worker-explore-draw", sample_id) >= beta_for(batch_index, beta0, steps, start):
        return None
    return pool[int(_u("worker-explore-target", sample_id) * len(pool)) % len(pool)]


def display_order(sample_id, pool):
    """Per-query permutation of the candidate list. Identity when disabled.

    Order is a rendering choice only: validate_route resolves by registry
    lookup, so no parser, dispatcher or reward path reads a position.
    """
    ids = list(pool)
    if os.environ.get("WORKER_ORDER_SHUFFLE", "0") != "1" or not sample_id:
        return ids
    return [w for _, w in sorted((_u("worker-order", f"{sample_id}|{w}"), w) for w in ids)]


# ---------------------------------------------------------------- design ---
# 4 workers x 2 rollouts inside one K=8 group. WHICH 4 rotates across queries
# on the COMPLEMENT OF THE FANO PLANE: 7 blocks of size 4 over 7 points where
# every worker sits in exactly 4 blocks and every PAIR of workers co-occurs in
# exactly 2. That is the only 7-point design with both balanced exposure and
# balanced pair coverage, so no worker and no head-to-head comparison is
# systematically under-sampled. (Fano is 2-(7,3,1); its complement is
# 2-(7,4,2) with r = b - r_fano = 4.)
_FANO = ((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2))
BLOCKS = tuple(tuple(i for i in range(7) if i not in b) for b in _FANO)
PAIRS_PER = 2
BLOCKS_PER_WORKER = 4


def block_for(sample_id, pool):
    """The 4 workers this query rotates over, as pool ids."""
    if len(pool) != 7:
        raise ValueError(f"the 4x2 design is defined for a 7-worker pool, got {len(pool)}")
    b = BLOCKS[int(_u("worker-explore-block", sample_id) * len(BLOCKS)) % len(BLOCKS)]
    return [pool[i] for i in b]


def worker_for_rollout(sample_id, batch_index, pool, rollout_n,
                       beta0=None, steps=None, start=None):
    """The worker rollout `rollout_n` of this query is told to use, or None.

    Whether the QUERY explores is the beta draw on sample_id alone, so all 8
    rollouts of a group are exploration rollouts together or none are -- the
    group never mixes an exploration prompt with a free one, which is what
    keeps mu_G a single well-defined baseline. WITHIN the group the worker is a
    function of rollout_n, so the 8 rollouts carry 4 different worker names,
    2 each. That is the whole point: identical names across a group make
    sum_i A_i * dlogpi(w|ctx) collapse to dlogpi * sum_i A_i == 0.
    """
    if not sample_id or not pool or rollout_n is None:
        return None
    if _u("worker-explore-draw", sample_id) >= beta_for(batch_index, beta0, steps, start):
        return None
    four = block_for(sample_id, pool)
    # rollout_n is the GROUP-LOCAL ordinal (trainers/rollout_ordinal_hook), so with
    # K=8 the pairs (0,1),(2,3),(4,5),(6,7) map onto the block's 4 workers,
    # 2 rollouts each. The modulo is a guard, not the mechanism: it only fires
    # if K ever exceeds 2*len(block), and pairing no longer depends on batch
    # layout the way a positional counter did.
    per = max(1, int(os.environ.get("ROLLOUT_N", "8")) // len(four))
    return four[(int(rollout_n) // per) % len(four)]


FORCED_LINE = (
    "For this episode you must route to {worker}. Name that exact id in every "
    "worker call you make in this trajectory. The id list and the grammar above "
    "are unchanged; this episode only fixes which id you name."
)


def enabled():
    """Re-read at call time so a launcher's export is honoured after import."""
    return os.environ.get("WORKER_EXPLORE", "0") == "1"
