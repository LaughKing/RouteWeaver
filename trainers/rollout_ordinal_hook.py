"""Group-local rollout ordinal, assigned before the batch is chunked.

WHY NOT verl's `rollout_n`. It is a POSITIONAL counter: get_trajectory_info
increments it while `index[i-1] == index[i]`, so it is only the within-group
ordinal when the grouping column happens to change between groups. In practice
it ran 0..255 across a 256-row batch -- a batch-global counter -- and
the 4x2 rotation only landed correctly because a group happened to occupy an
8-aligned contiguous block. Any dataloader shuffle, batch reorder or K change
would have silently mis-paired the workers.

WHAT THIS DOES INSTEAD. Rows are numbered by their position WITHIN THEIR OWN
GROUP, where the group is read from the data (`uid`, else
`(sample_id, forced_mode)` out of extra_info) -- the same key GRPO reduces over.
The ordinal is attached to non_tensor_batch in AgentLoopManager.generate_sequences,
BEFORE `prompts.chunk(...)`, exactly like verl's own `priority` column two lines
above, so it travels with the row through chunking, reordering and concat, and
every agent loop receives it as an ordinary kwarg.

ZERO verl edits -- installed by a post-import hook in sitecustomize.py.
"""
import functools

KEY = "routeweaver_group_ordinal"


def _group_keys(non_tensor, n):
    uid = non_tensor.get("uid")
    if uid is not None:
        return [str(v) for v in uid]
    extra = non_tensor.get("extra_info")
    if extra is None:
        return None
    out = []
    for i in range(n):
        e = extra[i] or {}
        out.append("%s|%s" % (e.get("sample_id"), e.get("forced_mode")))
    return out


def assign_ordinals(prompts) -> bool:
    """-> True if the column was written."""
    import numpy as np
    n = len(prompts)
    keys = _group_keys(prompts.non_tensor_batch, n)
    if keys is None:
        return False
    seen = {}
    ordinal = np.zeros(n, dtype=np.int64)
    for i, k in enumerate(keys):
        ordinal[i] = seen.get(k, 0)
        seen[k] = ordinal[i] + 1
    prompts.non_tensor_batch[KEY] = ordinal
    return True


def install(module) -> bool:
    """Wrap AgentLoopManager.generate_sequences, matching its own sync/async
    nature. verl has both shapes across versions -- wrapping a SYNC method with
    an `async def` hands every caller a coroutine instead of a DataProto, which
    is the `'coroutine' object has no attribute 'meta_info'` crash. So the shape
    is detected rather than assumed."""
    import inspect
    manager = getattr(module, "AgentLoopManager", None)
    if manager is None:
        return False
    original = getattr(manager, "generate_sequences", None)
    if original is None or getattr(original, "_routeweaver_group_ordinal", False):
        return False

    def _prep(prompts):
        try:
            if KEY not in prompts.non_tensor_batch:
                assign_ordinals(prompts)
        except Exception:                       # never kill a step over a column
            import traceback
            traceback.print_exc()

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def generate_sequences(self, prompts, *a, **kw):
            _prep(prompts)
            return await original(self, prompts, *a, **kw)
    else:
        @functools.wraps(original)
        def generate_sequences(self, prompts, *a, **kw):
            _prep(prompts)
            return original(self, prompts, *a, **kw)

    generate_sequences._routeweaver_group_ordinal = True
    manager.generate_sequences = generate_sequences
    print("[rollout_ordinal] wrapped AgentLoopManager.generate_sequences (%s)"
          % ("async" if inspect.iscoroutinefunction(original) else "sync"), flush=True)
    return True
