#!/usr/bin/env python
"""Merge an FSDP2 sharded actor checkpoint into plain HF safetensors.

verl writes model_world_size_N_rank_i.pt holding DTensors: each rank's file has
every parameter KEY, but each value is that rank's shard of the global tensor.
A world_size=2 checkpoint therefore cannot be loaded on one GPU -- verl looks
for model_world_size_1_rank_0.pt and there is none. Merged HF weights carry the
same parameters with no world size baked in, so they load on any card count.

The merge concatenates each rank's LOCAL shard along the sharded dimension.
`full_tensor()` looks like the obvious call and is wrong here: it gathers over
the live process group, and in a 1-process group it returns only that process's
shard -- silently half the rows. The first attempt did exactly that and wrote an
embed_tokens of [75968, 2560] against a config asking for [151936, 2560].

Every parameter in this checkpoint is Shard(dim=0) over a 1-D mesh of size 2, so
torch.cat(locals, dim=0) reconstructs it, and DTensor still carries the global
shape to assert the result against.
"""
import argparse, os, shutil, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".../global_step_N/actor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.ckpt)
                    if f.startswith("model_world_size_") and f.endswith(".pt"))
    if not shards:
        sys.exit(f"no model_world_size_*.pt in {a.ckpt}")
    print(f"shards: {shards}")

    from torch.distributed.tensor import DTensor
    dt = getattr(torch, a.dtype)
    parts, globals_, plain = {}, {}, {}
    for name in shards:
        sd = torch.load(os.path.join(a.ckpt, name), map_location="cpu",
                        weights_only=False)
        for k, v in sd.items():
            if isinstance(v, DTensor):
                parts.setdefault(k, []).append(v.to_local())
                globals_[k] = tuple(v.shape)
            else:
                plain[k] = v
        del sd
        print(f"  read {name}")

    merged = {}
    for k, chunks in parts.items():
        t = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        want = globals_[k]
        if tuple(t.shape) != want:
            sys.exit(f"FATAL {k}: merged {tuple(t.shape)} != global {want}")
        merged[k] = t.to(dt)
    for k, v in plain.items():
        merged.setdefault(k, v.to(dt) if hasattr(v, "to") else v)
    print(f"  merged {len(merged)} params, all shapes match their global shape")

    os.makedirs(a.out, exist_ok=True)
    hf = os.path.join(a.ckpt, "huggingface")
    for f in os.listdir(hf):                       # config + tokenizer
        shutil.copy2(os.path.join(hf, f), os.path.join(a.out, f))

    from safetensors.torch import save_file
    merged = {k: v.contiguous() for k, v in merged.items()}
    save_file(merged, os.path.join(a.out, "model.safetensors"),
              metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in merged.values())
    print(f"wrote {a.out}  {len(merged)} tensors  {total/2**30:.2f} GB")


if __name__ == "__main__":
    main()
