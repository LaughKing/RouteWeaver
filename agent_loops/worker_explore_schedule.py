"""Worker-level early exploration: a prompt-level FORCED WORKER curriculum."""
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
    """The worker this query is told to use, or None for free generation."""
    if not sample_id or not pool:
        return None
    if _u("worker-explore-draw", sample_id) >= beta_for(batch_index, beta0, steps, start):
        return None
    return pool[int(_u("worker-explore-target", sample_id) * len(pool)) % len(pool)]


def display_order(sample_id, pool):
    """Per-query permutation of the candidate list. Identity when disabled."""
    ids = list(pool)
    if os.environ.get("WORKER_ORDER_SHUFFLE", "0") != "1" or not sample_id:
        return ids
    return [w for _, w in sorted((_u("worker-order", f"{sample_id}|{w}"), w) for w in ids)]


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
    """The worker rollout `rollout_n` of this query is told to use, or None."""
    if not sample_id or not pool or rollout_n is None:
        return None
    if _u("worker-explore-draw", sample_id) >= beta_for(batch_index, beta0, steps, start):
        return None
    four = block_for(sample_id, pool)
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
