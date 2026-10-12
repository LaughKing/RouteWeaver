"""Group-local rollout ordinal, assigned before the batch is chunked."""
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
    """Wrap AgentLoopManager.generate_sequences, preserving its sync/async shape."""
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
