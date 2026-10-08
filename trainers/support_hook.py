# Copyright 2026 RouteWeaver.
"""Wire the minimum-support auxiliary loss into the actor's optimizer step.

WHERE IT ATTACHES. verl's BaseEngine.train_batch is

    optimizer_zero_grad()  ->  forward_backward_batch()  ->  optimizer_step()

with backward only ACCUMULATING into .grad and no step inside. Wrapping
`optimizer_step` therefore lets the auxiliary forward/backward land in the SAME
optimizer step as GRPO's, which is the whole requirement -- and the auxiliary
sequences never enter forward_backward_batch, so they cannot touch the rollout,
the advantage, the PPO ratio or any loss normalisation.

WHY NO BATCH PLUMBING. The auxiliary batch is a pure function of (dataset, step):
the parquet is written in DomainQuotaBatchSampler order and read by a
SequentialSampler with shuffle=False, so training step g consumes parquet rows
[32*(g-1), 32*g). ppo_mini_batch_size * rollout.n equals the full batch and
ppo_epochs is 1, so there is exactly ONE optimizer step per training step -- the
hook counts its own calls from the resume step and indexes the parquet directly.
Nothing has to be threaded through DataProto, micro-batching or the loss fn.

MEMORY. The loss depends on theta only through the 3B candidate scores, so
    grad_theta L = sum_i (dL/ds_i) * grad_theta s_i
Pass 1 scores every chunk under no_grad and computes dL/ds exactly; pass 2
re-forwards each chunk with grad and backwards it with those upstream weights.
That is mathematically identical to one big graph and bounds activation memory.
"""
import json
import os
import threading

import torch

import support_loss as S

_LOCK = threading.Lock()
_STATE = {"installed": False, "hook": None}
LAST_METRICS = {}


class SupportHook:
    def __init__(self, parquet, model_path, start_batch, steps, lambda0,
                 floor, batch_size=32, chunk=12, max_len=2048):
        self.parquet = parquet
        self.start_batch = int(start_batch)
        self.steps = int(steps)
        self.lambda0 = float(lambda0)
        self.floor = float(floor)
        self.batch_size = int(batch_size)
        self.chunk = int(chunk)
        self.max_len = int(max_len)
        self.calls = 0
        self._prompts = None
        self._tok = None
        self._model_path = model_path
        self._by_batch = None          # batch_index -> [(sample_id, arm, prompt)]

    # ---------------------------------------------------------------- data --
    def _load(self):
        if self._prompts is not None:
            return
        import pandas as pd
        from transformers import AutoTokenizer
        frame = pd.read_parquet(self.parquet, columns=["prompt", "extra_info"])
        self._prompts = [p[0]["content"] for p in frame["prompt"]]
        self._tok = AutoTokenizer.from_pretrained(self._model_path)
        # UNIQUE queries per batch_index, parquet order. In the 32x4 layout every
        # row is its own query so this is the old [32*g, 32*(g+1)) slice; in the
        # MD layout a controlled query has 3 rows and must be scored once.
        by_batch, seen = {}, set()
        for text, extra in zip(self._prompts, frame["extra_info"]):
            extra = dict(extra or {})
            bi = int(extra.get("batch_index", -1))
            sid = str(extra.get("sample_id") or "")
            if (bi, sid) in seen:
                continue
            seen.add((bi, sid))
            by_batch.setdefault(bi, []).append((sid, text))
        self._by_batch = by_batch

    def alpha(self, batch_index):
        t = batch_index - self.start_batch
        return min(1.0, max(0.0, 1.0 - t / float(self.steps)))

    def _build(self, batch_index, device):
        """96 sequences [chat(selector prompt) | <mode>m</mode>], left padded."""
        self._load()
        entries = self._by_batch.get(int(batch_index)) or []
        rows = [text for _, text in entries]
        if not rows:
            return None
        self._entries = entries
        tok = self._tok
        seqs, lens = [], []
        for text in rows:
            chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False)
            p_ids = tok(chat, add_special_tokens=False).input_ids
            if len(p_ids) > self.max_len:          # keep the tail: the question
                p_ids = p_ids[-self.max_len:]      # and the response header
            for m in S.MODES:
                full = p_ids + tok(f"<mode>{m}</mode>",
                                   add_special_tokens=False).input_ids
                seqs.append(full)
                lens.append(len(full) - len(p_ids))
        width = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        return ids.to(device), mask.to(device), torch.tensor(lens), max(lens) + 1

    # --------------------------------------------------------------- score --
    @staticmethod
    def _scores(module, ids, mask, lens, keep, lo, hi):
        """Candidate logprobs for rows [lo,hi). Left pad => explicit position_ids:
        the default arange would hand a real token a pad slot's index and rotate
        every RoPE angle."""
        sub_ids, sub_mask = ids[lo:hi], mask[lo:hi]
        pos = (sub_mask.cumsum(-1) - 1).clamp(min=0)
        # Call the ROOT module, never a submodule. Under FSDP2 the parameters
        # are DTensors and only the wrapped root's pre-forward hooks unshard
        # them; going straight to .model/.lm_head raises "got mixed torch.Tensor
        # and DTensor".
        out_obj = module(input_ids=sub_ids, attention_mask=sub_mask,
                         position_ids=pos, return_dict=True, use_cache=False)
        width = sub_ids.shape[1]
        lp = getattr(out_obj, "log_probs", None)
        if lp is None:
            # no fused kernels: standard head, take logprobs of the taken tokens
            logp = torch.log_softmax(out_obj.logits.float(), dim=-1)
            lp = logp[:, :-1].gather(-1, sub_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            base = width              # same indexing as the fused path:
            #   lp[:, j] = log P(token j+1); only the length differs (W-1 vs W)
            #   and index W-2 is the last one either path ever reads.
        else:
            # use_fused_kernels: forward_with_torch_backend returns per-token
            # log P(token j+1) at index j over the FULL width, with no vocab
            # tensor ever materialised. The final index wraps (torch.roll) and
            # is never read: a declaration's last token sits at width-1, whose
            # logprob is at width-2.
            base = width
        out, first = [], []
        # MODE_SCORE_SPAN=content (recommended): score only the mode CONTENT
        # tokens (declaration indices 3 .. n-4: "single" | "multi" | "ag"+"entic").
        # "full" (the legacy span) also sums "<mode>" and
        # "</mode>". The legacy form is WRONG for this policy: it emits the
        # closing ">" merged with the newline (">\n", p=1.0), so the canonical
        # bare ">" is off-distribution (logprob -11..-19) and its value differs
        # by candidate, swamping the real content-token gap: the full-span
        # distribution looks nearly uniform where the next-token one is a point
        # mass.
        span = os.environ.get("MODE_SCORE_SPAN", "full")
        for i in range(sub_ids.shape[0]):
            n = int(lens[lo + i])
            # tokens [width-n, width-1] are the declaration; the logprob of the
            # token at k lives at index k-1.
            if span == "content":
                out.append(lp[i, base - 1 - n + 3:base - 4].sum())
            else:
                out.append(lp[i, base - 1 - n:base - 1].sum())
            # the FIRST content token after "<mode>" (declaration index 3):
            # "single" / "multi" / "ag" -- the single-token categorical view.
            first.append(lp[i, base - 1 - n + 3])
        SupportHook._last_first = torch.stack(first)
        return torch.stack(out)

    # ---------------------------------------------------------------- step --
    def run(self, engine):
        """One auxiliary forward/backward, accumulating into .grad."""
        # The step is (first batch_index in the parquet) + (trainer step
        # already completed at resume) + (calls so far). The old form
        # `start_batch + calls` silently assumed the parquet starts at batch 0
        # AND the run starts at step 0 -- wrong by the resume offset on any
        # resumed run. Resuming at step 25 on a parquet that starts at batch 25
        # would have replayed batch 25 as batch 50.
        self._load()
        first = min(self._by_batch) if self._by_batch else 0
        resume_step = int(os.environ.get("ROUTEWEAVER_RESUME_STEP", "0"))
        batch_index = first + resume_step + self.calls
        self.calls += 1
        alpha = self.alpha(batch_index)
        lam = S.lambda_for(alpha, self.lambda0)
        metrics = {"support/alpha": alpha, "support/lambda": lam,
                   "support/batch_index": batch_index}
        if lam <= 0.0:
            LAST_METRICS.clear(); LAST_METRICS.update(metrics)
            self._publish(batch_index, metrics)
            return metrics

        module = engine.module
        device = next(module.parameters()).device
        built = self._build(batch_index, device)
        if built is None:
            LAST_METRICS.clear(); LAST_METRICS.update(metrics)
            return metrics
        ids, mask, lens, keep = built
        n = ids.shape[0]

        # pass 1: scores without grad -> the exact upstream dL/ds
        with torch.no_grad():
            parts, firsts = [], []
            for lo in range(0, n, self.chunk):
                parts.append(self._scores(module, ids, mask, lens, keep, lo,
                                          min(lo + self.chunk, n)))
                firsts.append(SupportHook._last_first)
        scores = torch.cat(parts).view(-1, 3)
        first_tok = torch.cat(firsts).view(-1, 3)      # diagnostic only
        scores_g = scores.detach().clone().requires_grad_(True)
        loss, pibar, penalty = S.support_loss(scores_g, floor=self.floor)
        total = lam * loss
        total.backward()
        dLds = scores_g.grad.detach().view(-1)          # (3B,)
        metrics["aux/dLds_norm"] = float(dLds.norm())

        # Gradient bookkeeping: .grad already holds GRPO's gradient. Snapshot
        # its local shards (CPU) so the auxiliary contribution can be measured
        # as ||g_after - g_before|| on this rank. Global norms = sqrt of the
        # sum over ranks of the printed squares. Off with MODE_GRAD_NORM=0.
        measure = os.environ.get("MODE_GRAD_NORM", "1") == "1"
        g0, grpo_sq = [], 0.0
        if measure:
            with torch.no_grad():
                for prm in module.parameters():
                    g = prm.grad
                    if g is None:
                        g0.append(None); continue
                    g = g.to_local() if hasattr(g, "to_local") else g
                    grpo_sq += float((g.float() ** 2).sum())
                    g0.append(g.detach().to("cpu", copy=True))

        # pass 2: re-forward with grad, backward with those weights.
        # sum_i (dL/ds_i) grad_theta s_i is exactly grad_theta (lam*L).
        for lo in range(0, n, self.chunk):
            hi = min(lo + self.chunk, n)
            s = self._scores(module, ids, mask, lens, keep, lo, hi)
            torch.autograd.backward(s, grad_tensors=dLds[lo:hi])

        if measure:
            aux_sq, tot_sq = 0.0, 0.0
            with torch.no_grad():
                for prm, old in zip(module.parameters(), g0):
                    g = prm.grad
                    if g is None:
                        continue
                    g = g.to_local() if hasattr(g, "to_local") else g
                    tot_sq += float((g.float() ** 2).sum())
                    d = g.float() - (old.to(g.device).float() if old is not None else 0.0)
                    aux_sq += float((d ** 2).sum())
            del g0
            metrics.update({"grad/grpo_sq_local": grpo_sq,
                            "grad/aux_sq_local": aux_sq,
                            "grad/total_sq_local": tot_sq,
                            "grad/grpo_norm_local": grpo_sq ** 0.5,
                            "grad/aux_norm_local": aux_sq ** 0.5,
                            "grad/aux_over_grpo_local": (aux_sq / grpo_sq) ** 0.5 if grpo_sq > 0 else -1.0})

        metrics.update({
            "support/loss": float(loss.detach()),
            "support/weighted_loss": float(lam * loss.detach()),
            "support/pibar_single": float(pibar[0]),
            "support/pibar_multi": float(pibar[1]),
            "support/pibar_agentic": float(pibar[2]),
            "support/penalty_single": float(penalty[0]),
            "support/penalty_multi": float(penalty[1]),
            "support/penalty_agentic": float(penalty[2]),
            "support/n_queries": int(scores.shape[0]),
        })
        # pi split by forced/free: free/pi_* is the routing prior on the
        # prompts whose mode token actually gets a gradient this step.
        try:
            import exploration_schedule as S2
            with torch.no_grad():
                pq_all = torch.softmax(scores, dim=-1)
            fr = [not S2.is_forced(sid, alpha) for sid, _ in self._entries]
            fr = torch.tensor(fr, device=pq_all.device)
            for name, sel in (("free", fr), ("forced", ~fr)):
                if int(sel.sum()) > 0:
                    m_ = pq_all[sel].mean(0)
                    metrics.update({f"{name}/pi_single": float(m_[0]), f"{name}/pi_multi": float(m_[1]),
                                    f"{name}/pi_agentic": float(m_[2]), f"{name}/n_queries": int(sel.sum())})
        except Exception as exc:                                   # noqa: BLE001
            metrics["hook/arm_split_error"] = 1
            print(f"[support] arm split failed: {exc!r}", flush=True)
        if os.environ.get("ROUTEWEAVER_COMET", "0") == "1":
            # per-mode mean pi over the whole batch (controlled + natural)
            with torch.no_grad():
                pq = torch.softmax(scores, dim=-1).mean(0)
                p1 = torch.softmax(first_tok, dim=-1).mean(0)
                H = -(torch.softmax(scores, dim=-1) * torch.log_softmax(scores, dim=-1)).sum(-1).mean()
            metrics.update({"mode/pibar_all_single": float(pq[0]),
                            "mode/pibar_all_multi": float(pq[1]),
                            "mode/pibar_all_agentic": float(pq[2]),
                            "mode/pi1_single": float(p1[0]),     # first-token view
                            "mode/pi1_multi": float(p1[1]),
                            "mode/pi1_agentic": float(p1[2]),
                            "mode/entropy_mean": float(H)})
        LAST_METRICS.clear(); LAST_METRICS.update(metrics)
        self._publish(batch_index, metrics)
        return metrics

    def _publish(self, batch_index, metrics):
        """Hand the driver's metrics hook this step's numbers via a file (the
        hook runs in the actor worker, wandb in the driver)."""
        d = os.environ.get("ROUTEWEAVER_HOOK_METRICS_DIR", "")
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"hook_{int(batch_index)}.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({k: v for k, v in metrics.items()
                       if isinstance(v, (int, float))}, f)
        os.replace(tmp, path)


def install(engine_module, class_name="FSDPEngine") -> bool:
    """Wrap the CONCRETE engine's optimizer_step. Enabled only when
    ROUTEWEAVER_SUPPORT=1.

    Must be the concrete class, not BaseEngine: FSDPEngine(BaseEngine) defines
    its own optimizer_step (fsdp/transformer_impl.py:665), so a patch on the
    base is shadowed and the hook runs never -- which is exactly what happened
    on the first two attempts, silently, while training looked healthy.

    A missing target raises when support is requested. Returning False there
    would reproduce the same silent no-op this comment exists to prevent.
    """
    with _LOCK:
        if _STATE["installed"]:
            return False
        if os.environ.get("ROUTEWEAVER_SUPPORT", "0") != "1":
            return False
        cls = getattr(engine_module, class_name, None)
        if cls is None:
            raise RuntimeError(
                f"ROUTEWEAVER_SUPPORT=1 but {class_name} is absent from "
                f"{engine_module.__name__}; refusing to train without the "
                f"auxiliary loss silently installed nowhere")
        if "optimizer_step" not in vars(cls):
            raise RuntimeError(
                f"ROUTEWEAVER_SUPPORT=1 but {class_name} does not define "
                f"optimizer_step itself; patching it would be shadowed")
        original = getattr(cls, "optimizer_step", None)
        if original is None or getattr(original, "_routeweaver_support", False):
            return False

        parquet = os.environ.get("ROUTEWEAVER_TRAIN_PARQUET")
        if not parquet:
            # The term scores the three candidate declarations of every query in
            # THIS batch, which it reads from the parquet. Without it there is
            # nothing to score, and silently skipping would drop a loss term the
            # run was configured to use.
            raise RuntimeError(
                "ROUTEWEAVER_SUPPORT=1 but ROUTEWEAVER_TRAIN_PARQUET is unset; "
                "launch through scripts/train_routeweaver.sh, which exports it")
        hook = SupportHook(
            parquet=parquet,
            model_path=os.environ.get("ROUTEWEAVER_TOKENIZER",
                                      os.environ.get("ROUTEWEAVER_INIT_CKPT")),
            start_batch=os.environ.get("FORCED_START_BATCH", 25),
            steps=os.environ.get("FORCED_ANNEAL_STEPS", 50),
            lambda0=os.environ.get("SUPPORT_LAMBDA0", S.SUPPORT_LAMBDA0),
            floor=os.environ.get("SUPPORT_FLOOR", S.SUPPORT_FLOOR),
            chunk=int(os.environ.get("SUPPORT_CHUNK", 12)),
            max_len=int(os.environ.get("SUPPORT_MAX_LEN", 2048)),
        )
        _STATE["hook"] = hook

        def optimizer_step(self, *args, **kwargs):
            # No actor/critic guard: GRPO has no value model, so every engine
            # that reaches optimizer_step here IS the actor. The first version
            # gated on hasattr(module, "get_output_embeddings"), which is False
            # on the FSDP-wrapped module -- the hook then skipped every step in
            # SILENCE and the run was plain annealing wearing Stage-2-v2's name.
            # An engine without an lm_head raises inside run() and is caught
            # below, which is the loud failure the guard only pretended to give.
            try:
                m = hook.run(self)
                print(f"[support] {json.dumps(m)}", flush=True)
            except Exception as exc:                              # noqa: BLE001
                import traceback
                print(f"[support] FAILED, skipping this step: {exc!r}", flush=True)
                traceback.print_exc()
            return original(self, *args, **kwargs)

        optimizer_step._routeweaver_support = True
        cls.optimizer_step = optimizer_step
        _STATE["installed"] = True
        print("[support] optimizer_step hook installed", flush=True)
        return True
