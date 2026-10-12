# Copyright 2026 RouteWeaver.
"""Per-step routing metrics, into whatever verl logs to."""
import math
import os
from collections import Counter, defaultdict

MODES = ("single", "multi", "agentic")
SYSTEM_KINDS = ("execution_timeout", "provider_error", "sandbox_error",
                "wrapper_error")


def _column(batch, name):
    columns = getattr(batch, "non_tensor_batch", {}) or {}
    value = columns.get(name)
    return None if value is None else list(value)


def train_metrics(batch) -> dict:
    """routeweaver/* metrics for one step. Never raises: a metric must not stop a run.
    """
    out = {}
    modes = _column(batch, "mode")
    if modes is None:
        return out
    n = len(modes)
    uids = _column(batch, "uid") or list(range(n))
    rewards = _column(batch, "reward")
    if rewards is None:
        scores = batch.batch.get("token_level_scores")
        rewards = scores.sum(-1).tolist() if scores is not None else [0.0] * n
    infra = _column(batch, "infra_failed") or [0] * n
    kinds = _column(batch, "failure_kind") or [""] * n
    domains = _column(batch, "rfm_domain") or [""] * n
    valid = _column(batch, "selector_valid") or [0] * n

    counts = Counter(m if m in MODES else "invalid" for m in modes)
    for mode in MODES + ("invalid",):
        out[f"routeweaver/mode_share/{mode}"] = counts.get(mode, 0) / max(1, n)
    out["routeweaver/selector_valid_rate"] = sum(int(v) for v in valid) / max(1, n)
    attempts = _column(batch, "selector_attempts") or []
    if attempts:
        out["routeweaver/selector_retry_rate"] = sum(
            1 for a in attempts if int(a) > 1) / max(1, len(attempts))

    import math
    total = sum(counts.values())
    out["routeweaver/mode_entropy"] = -sum(
        (c / total) * math.log(c / total) for c in counts.values() if c)

    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    unique_modes, zero_var, group_std = [], 0, []
    for rows in groups.values():
        unique_modes.append(len({modes[i] for i in rows}))
        vals = [float(rewards[i]) for i in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        group_std.append(var ** 0.5)
        if var ** 0.5 < 1e-4:
            zero_var += 1
    out["routeweaver/group_size_mean"] = n / max(1, len(groups))
    out["routeweaver/group_unique_modes_mean"] = sum(unique_modes) / max(1, len(unique_modes))
    out["routeweaver/group_all_same_mode_ratio"] = sum(
        1 for u in unique_modes if u == 1) / max(1, len(unique_modes))
    out["routeweaver/zero_variance_group_ratio"] = zero_var / max(1, len(groups))
    out["routeweaver/group_reward_std_mean"] = sum(group_std) / max(1, len(group_std))

    out["routeweaver/reward_mean_all_rows"] = sum(float(r) for r in rewards) / max(1, n)
    clean = [float(r) for r, f in zip(rewards, infra) if int(f) == 0]
    out["routeweaver/reward_mean_excluding_system"] = (
        sum(clean) / len(clean) if clean else 0.0)
    out["routeweaver/infra_failed_rate"] = sum(int(f) for f in infra) / max(1, n)
    kind_counts = Counter(kinds)
    for kind in ("correct", "genuine_incorrect", "parser_error") + SYSTEM_KINDS:
        out[f"routeweaver/failure_kind/{kind}"] = kind_counts.get(kind, 0) / max(1, n)

    for mode in MODES:
        rows = [float(rewards[i]) for i in range(n) if modes[i] == mode]
        if rows:
            out[f"routeweaver/reward_by_mode/{mode}"] = sum(rows) / len(rows)
    for domain in sorted(set(d for d in domains if d)):
        rows = [float(rewards[i]) for i in range(n) if domains[i] == domain]
        if rows:
            out[f"routeweaver/reward_by_domain/{domain}"] = sum(rows) / len(rows)
            out[f"routeweaver/rows_by_domain/{domain}"] = len(rows)

    pol_min = _column(batch, "policy_span_mask_min")
    env_max = _column(batch, "env_span_mask_max")
    mode_min = _column(batch, "mode_span_mask_min")
    mode_tok = _column(batch, "mode_span_tokens")
    conditional = bool(_column(batch, "forced_mode"))
    if pol_min and env_max and (mode_min or conditional):
        if conditional:
            violations = sum(1 for a, b in zip(pol_min, env_max)
                             if int(a) != 1 or int(b) != 0)
        else:
            violations = sum(1 for a, b, c in zip(pol_min, env_max, mode_min)
                             if int(a) != 1 or int(b) != 0 or int(c) != 1)
        out["routeweaver/mask_violations"] = violations
        out["routeweaver/mask_policy_span_min"] = min(int(v) for v in pol_min)
        out["routeweaver/mask_env_span_max"] = max(int(v) for v in env_max)
        if mode_min and not conditional:
            out["routeweaver/mask_mode_span_min"] = min(int(v) for v in mode_min)
    if mode_tok:
        out["routeweaver/mode_span_tokens_mean"] = sum(
            float(v) for v in mode_tok) / len(mode_tok)
    for name in ("policy_token_count", "env_token_count"):
        values = _column(batch, name)
        if values:
            out[f"routeweaver/{name}_mean"] = sum(float(v) for v in values) / len(values)

    for name, key in (("worker_calls", "worker_calls"),
                      ("failed_worker_calls", "failed_worker_calls"),
                      ("route_count", "route_count"),
                      ("turns", "turns"),
                      ("max_layer_width", "max_layer_width"),
                      ("worker_output_tokens", "worker_output_tokens"),
                      ("grammar_invalid", "grammar_invalid"),
                      ("loop_errors", "loop_errors"),
                      ("wrapper_errors", "wrapper_errors")):
        values = _column(batch, key)
        if values:
            out[f"routeweaver/{name}_mean"] = sum(float(v) for v in values) / len(values)
    return out


def m1_metrics(batch) -> dict:
    """Metrics sliced by forced mode. Inert without a forced_mode column."""
    out = {}
    modes = _column(batch, "forced_mode")
    if not modes:
        return out
    n = len(modes)
    uids = _column(batch, "uid") or list(range(n))
    domains = _column(batch, "rfm_domain") or [""] * n
    rewards = _column(batch, "reward_value")
    if rewards is None:
        scores = batch.batch.get("token_level_scores")
        rewards = scores.sum(-1).tolist() if scores is not None else [0.0] * n

    def col(name, default=0):
        return _column(batch, name) or [default] * n

    correct = col("correct"); infra = col("infra_failed")
    term = col("terminal_invalid"); fpv = col("first_pass_valid")
    evv = col("eventual_valid"); cues = col("cue_count")
    prem = col("answer_premature"); refs = col("refs_violation")
    cue_msgs = col("cue_messages"); heads = col("state_headers")
    codes = [str(c or "") for c in
             (_column(batch, "policy_invalid") or [""] * n)]
    pol_inv = [1 if str(v) else 0
               for v in (_column(batch, "policy_invalid") or [""] * n)]
    calls = col("worker_calls"); turns = col("turns")
    sigs = _column(batch, "route_signature") or [""] * n

    for mode in MODES:
        idx = [i for i in range(n) if modes[i] == mode]
        if not idx:
            continue
        k = f"m1/{mode}"
        m = lambda seq: sum(float(seq[i]) for i in idx) / len(idx)
        out[f"{k}/rows"] = len(idx)
        out[f"{k}/reward_mean"] = m(rewards)
        out[f"{k}/task_accuracy"] = m(correct)
        out[f"{k}/first_pass_valid"] = m(fpv)
        out[f"{k}/eventual_valid"] = m(evv)
        out[f"{k}/terminal_invalid_rate"] = m(term)
        out[f"{k}/infra_rate"] = m(infra)
        out[f"{k}/cue_count_mean"] = m(cues)
        out[f"{k}/cue_messages_mean"] = m(cue_msgs)
        out[f"{k}/state_headers_mean"] = m(heads)
        out[f"{k}/policy_invalid_rate"] = m(pol_inv)
        out[f"{k}/answer_premature_rate"] = m(prem)
        out[f"{k}/refs_violation_rate"] = m(refs)
        seen = Counter(codes[i] for i in idx if codes[i])
        for code in ("ref_unknown", "ref_same_layer", "verifier_needs_refs",
                     "must_summarize_now", "action_missing"):
            out[f"{k}/violation/{code}"] = seen.get(code, 0) / len(idx)
        for code, hits in seen.items():
            if code not in ("ref_unknown", "ref_same_layer",
                            "verifier_needs_refs", "must_summarize_now",
                            "action_missing"):
                out[f"{k}/violation/{code}"] = hits / len(idx)
        out[f"{k}/worker_calls_mean"] = m(calls)
        out[f"{k}/turns_mean"] = m(turns)
        uniq = len({sigs[i] for i in idx})
        out[f"{k}/unique_route_ratio"] = uniq / max(1, len(idx))
        groups = defaultdict(list)
        for i in idx:
            groups[uids[i]].append(i)
        zero = usable = 0
        for rows in groups.values():
            vals = [float(rewards[i]) for i in rows]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            spread = var ** 0.5
            zero += int(spread < 1e-4)
            kept = [i for i in rows if not int(infra[i] or 0)]
            usable += int(len(kept) >= 2 and spread >= 1e-4)
        out[f"{k}/groups"] = len(groups)
        out[f"{k}/zero_variance_group_ratio"] = zero / max(1, len(groups))
        out[f"{k}/usable_group_ratio"] = usable / max(1, len(groups))
        out[f"{k}/effective_groups"] = len(groups) - zero
        for domain in sorted({d for d in domains if d}):
            sub = [i for i in idx if domains[i] == domain]
            if sub:
                out[f"m1/by_domain/{domain}/{mode}/reward_mean"] = (
                    sum(float(rewards[i]) for i in sub) / len(sub))
                out[f"m1/by_domain/{domain}/{mode}/task_accuracy"] = (
                    sum(float(correct[i]) for i in sub) / len(sub))
                out[f"m1/by_domain/{domain}/{mode}/rows"] = len(sub)

    per_uid_modes, per_uid_ids = defaultdict(set), defaultdict(set)
    sids = _column(batch, "rfm_sample_id") or [""] * n
    for i in range(n):
        per_uid_modes[uids[i]].add(modes[i])
        per_uid_ids[uids[i]].add(sids[i])
    out["m1/groups_spanning_modes"] = sum(
        1 for v in per_uid_modes.values() if len(v) > 1)
    out["m1/groups_spanning_queries"] = sum(
        1 for v in per_uid_ids.values() if len(v) > 1)
    sizes = Counter(len(v) for v in
                    [[i for i in range(n) if uids[i] == u] for u in set(uids)])
    out["m1/group_size_min"] = min(sizes) if sizes else 0
    fmask = _column(batch, "forced_mode_span_mask_max")
    if fmask:
        out["m1/forced_mode_span_mask_max"] = max(int(v) for v in fmask)
    executed = _column(batch, "mode") or []
    if executed:
        s2 = _column(batch, "stage2_forced")
        pairs = [(a, b) for i, (a, b) in enumerate(zip(modes, executed))
                 if not s2 or int(s2[i]) == 1]
        if pairs:
            out["m1/forced_mode_honoured"] = sum(
                1 for a, b in pairs if a == b) / len(pairs)
        out["m1/forced_mode_honoured_rows"] = len(pairs)
    return out


def stage2_metrics(batch) -> dict:
    """Annealing view. Inert unless the batch carries stage2_alpha."""
    alphas = _column(batch, "stage2_alpha")
    if not alphas:
        return {}
    out = {}
    n = len(alphas)
    forced = [int(v) for v in (_column(batch, "stage2_forced") or [1] * n)]
    executed = _column(batch, "mode") or [""] * n
    rewards = _column(batch, "reward")
    if rewards is None:
        scores = batch.batch.get("token_level_scores")
        rewards = scores.sum(-1).tolist() if scores is not None else [0.0] * n
    uids = _column(batch, "uid") or list(range(n))

    out["stage2/alpha"] = sum(float(a) for a in alphas) / max(1, n)
    out["stage2/forced_ratio"] = sum(forced) / max(1, n)
    out["stage2/free_ratio"] = 1.0 - out["stage2/forced_ratio"]
    out["stage2/rows"] = n

    for arm, want in (("forced", 1), ("free", 0)):
        idx = [i for i in range(n) if forced[i] == want]
        out[f"stage2/{arm}/rows"] = len(idx)
        if not idx:
            continue
        out[f"stage2/{arm}/reward_mean"] = sum(
            float(rewards[i]) for i in idx) / len(idx)
        counts = Counter(str(executed[i] or "invalid") for i in idx)
        for mode in ("single", "multi", "agentic"):
            out[f"stage2/{arm}/mode_share/{mode}"] = counts.get(mode, 0) / len(idx)
        out[f"stage2/{arm}/mode_share/invalid"] = (
            len(idx) - sum(counts.get(m, 0) for m in
                           ("single", "multi", "agentic"))) / len(idx)

    per_uid = {}
    for i in range(n):
        per_uid.setdefault(uids[i], set()).add(forced[i])
    out["stage2/groups_mixing_forced_free"] = sum(
        1 for v in per_uid.values() if len(v) > 1)
    out["stage2/groups"] = len(per_uid)

    fmin = [int(v) for v in (_column(batch, "free_mode_span_mask_min") or [])
            if int(v) >= 0]
    if fmin:
        out["stage2/free_mode_span_mask_min"] = min(fmin)
        out["stage2/free_mode_span_rows"] = len(fmin)
    return out


def routing_metrics(batch) -> dict:
    """What the FREE arm actually decides, and how sure it is."""
    mode_min = _column(batch, "mode_span_mask_min")
    if not mode_min:
        return {}
    n = len(mode_min)
    modes = _column(batch, "mode") or [""] * n
    domains = _column(batch, "rfm_domain") or [""] * n
    sids = _column(batch, "rfm_sample_id") or [""] * n
    free = [i for i in range(n) if int(mode_min[i] or 0) == 1]
    out = {"routing/free_rows": len(free)}
    if not free:
        return out

    MODES = ("single", "multi", "agentic")

    for m in MODES:
        out[f"routing/share/{m}"] = sum(
            1 for i in free if modes[i] == m) / len(free)
    for domain in sorted({domains[i] for i in free if domains[i]}):
        sub = [i for i in free if domains[i] == domain]
        for m in MODES:
            out[f"routing/share_by_domain/{domain}/{m}"] = sum(
                1 for i in sub if modes[i] == m) / len(sub)

    from collections import defaultdict
    per_q = defaultdict(list)
    for i in free:
        per_q[sids[i]].append(modes[i])
    ents, unanimous = [], 0
    for picks in per_q.values():
        k = len(picks)
        if k < 2:
            continue
        counts = [picks.count(m) for m in MODES]
        h = 0.0
        for c in counts:
            if c:
                pr = c / k
                h -= pr * math.log(pr)
        ents.append(h)
        unanimous += int(max(counts) == k)
    if ents:
        out["routing/per_question_entropy"] = sum(ents) / len(ents)
        out["routing/unanimous_group_ratio"] = unanimous / len(ents)
        out["routing/groups_scored"] = len(ents)

    rewards = _column(batch, "reward")
    if rewards is None:
        scores = getattr(batch, "batch", {}).get("token_level_scores")
        rewards = (scores.sum(-1).tolist() if scores is not None
                   else [0.0] * n)
    per_g = defaultdict(list)
    for i in free:
        per_g[sids[i]].append(i)
    kinds = {"one_mode_flat": 0, "one_mode_split": 0,
             "multi_mode_flat": 0, "multi_mode_split": 0}
    for rows in per_g.values():
        if len(rows) < 2:
            continue
        ms = {modes[i] for i in rows}
        rs = {round(float(rewards[i]), 6) for i in rows}
        key = ("multi_mode" if len(ms) > 1 else "one_mode") + (
            "_split" if len(rs) > 1 else "_flat")
        kinds[key] += 1
    tot = sum(kinds.values())
    if tot:
        for k, v in kinds.items():
            out[f"routing/group_kind/{k}"] = v / tot
        out["routing/mode_selection_signal"] = kinds["multi_mode_split"] / tot
    return out


def exec_metrics(batch) -> dict:
    """In-loop APPS execution. Inert unless the loop ran any."""
    runs = _column(batch, "exec_runs")
    if not runs:
        return {}
    total = sum(int(v) for v in runs)
    passed = sum(int(v) for v in (_column(batch, "exec_passed_all") or []))
    errs = sum(int(v) for v in (_column(batch, "exec_sandbox_errors") or []))
    out = {"routeweaver/exec/runs": total,
           "routeweaver/exec/sandbox_errors": errs,
           "routeweaver/exec/rows_with_exec": sum(1 for v in runs if int(v))}
    if total:
        out["routeweaver/exec/pass_all_rate"] = passed / total
    return out


def push_run_provenance():
    """Stamp the W&B run's config with what verl does not put there itself."""
    keys = ("ROUTEWEAVER_GIT_COMMIT", "ROUTEWEAVER_GIT_BRANCH", "ROUTEWEAVER_INIT_CKPT",
            "ROUTEWEAVER_INIT_GLOBAL_STEP", "ROUTEWEAVER_TRAIN_PARQUET",
            "ROUTEWEAVER_TRAIN_PARQUET_SHA", "ROUTEWEAVER_PROBE_SET",
            "ROUTEWEAVER_PROBE_SHA", "ROUTEWEAVER_SEED", "ROUTEWEAVER_REWARD_DEF",
            "ROUTEWEAVER_ROUTING_STATE", "ROUTEWEAVER_POLICY_VIOLATION_RECOVERY",
            "ROUTEWEAVER_SAVE_STEPS")
    payload = {k.lower().replace("routeweaver_", "routeweaver/"): os.environ[k]
               for k in keys if os.environ.get(k)}
    if not payload:
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update(payload, allow_val_change=True)
            print(f"[train_metrics] pushed {len(payload)} provenance keys "
                  f"to W&B config", flush=True)
    except Exception:                          # pragma: no cover
        import traceback
        traceback.print_exc()


def hier_metrics(batch) -> dict:
    """Hierarchical-advantage metrics. Inert unless ROUTEWEAVER_COMET=1."""
    if os.environ.get("ROUTEWEAVER_COMET", "0") != "1":
        return {}
    import json
    import comet_grpo
    out = dict(comet_grpo.LAST_STATS)
    d = os.environ.get("ROUTEWEAVER_HOOK_METRICS_DIR", "")
    ex = _column(batch, "extra_info") or []
    bi = int(dict(ex[0]).get("batch_index", -1)) if ex else -1
    path = os.path.join(d, f"hook_{bi}.json") if d and bi >= 0 else ""
    if path and os.path.exists(path):
        with open(path) as f:
            out.update(json.load(f))
    else:
        out["hier/hook_metrics_missing"] = 1
    return out


def obs_budget_metrics(batch) -> dict:
    """Observation truncation, overall / by mode / by domain."""
    out = {}
    blocks = _column(batch, "obs_blocks")
    if blocks is None:
        return out
    n = len(blocks)
    modes = _column(batch, "mode") or [""] * n
    domains = _column(batch, "rfm_domain") or [""] * n
    trunc = _column(batch, "obs_truncated") or [0] * n
    any_t = _column(batch, "obs_truncated_any") or [0] * n
    dropped = _column(batch, "obs_dropped_tokens") or [0] * n
    budget = _column(batch, "obs_budget_min") or [-1] * n

    def emit(prefix, idx):
        if not idx:
            return
        nb = sum(float(blocks[i]) for i in idx)
        out[f"{prefix}/rows"] = len(idx)
        out[f"{prefix}/row_truncated_rate"] = (
            sum(float(any_t[i]) for i in idx) / len(idx))
        out[f"{prefix}/obs_truncated_rate"] = (
            sum(float(trunc[i]) for i in idx) / nb if nb else 0.0)
        out[f"{prefix}/blocks_mean"] = nb / len(idx)
        out[f"{prefix}/dropped_tokens_mean"] = (
            sum(float(dropped[i]) for i in idx) / len(idx))
        granted = [float(budget[i]) for i in idx if float(budget[i]) >= 0]
        if granted:
            out[f"{prefix}/budget_min"] = min(granted)
            out[f"{prefix}/budget_mean"] = sum(granted) / len(granted)

    emit("obs", list(range(n)))
    for mode in sorted({str(m) for m in modes if m}):
        emit(f"obs/by_mode/{mode}", [i for i in range(n) if modes[i] == mode])
    for domain in sorted({str(d) for d in domains if d}):
        emit(f"obs/by_domain/{domain}",
             [i for i in range(n) if domains[i] == domain])
    return out


def install_metrics_hook(ray_trainer_module) -> bool:
    if os.environ.get("ROUTEWEAVER_METRICS", "") != "1":
        return False
    import functools

    original = getattr(ray_trainer_module, "compute_data_metrics", None)
    if original is None or getattr(original, "_routeweaver_metrics", False):
        return False

    pushed = {"done": False}

    @functools.wraps(original)
    def compute_data_metrics(batch, *args, **kwargs):
        metrics = original(batch, *args, **kwargs)
        try:
            metrics.update(train_metrics(batch))
            metrics.update(m1_metrics(batch))
            metrics.update(stage2_metrics(batch))
            metrics.update(routing_metrics(batch))
            metrics.update(exec_metrics(batch))
            metrics.update(hier_metrics(batch))
            metrics.update(obs_budget_metrics(batch))
        except Exception:                      # pragma: no cover
            import traceback
            traceback.print_exc()
        if not pushed["done"]:
            pushed["done"] = True
            push_run_provenance()
        return metrics

    compute_data_metrics._routeweaver_metrics = True
    ray_trainer_module.compute_data_metrics = compute_data_metrics
    return True
