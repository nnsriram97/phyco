#!/usr/bin/env python3
"""
Bake the multi-branch ControlNet weights into the merged base DiT so PhyCo can be
loaded from a SINGLE checkpoint (no --controlnet_branch_ckpts at inference).

The merged base DiT (`--base`) contains only the frozen backbone (net.* / net_ema.*,
no `controlnet_branches`). Each ControlNet branch was trained in its own run; at
inference `controlnet_{i+1}` is loaded from that run's `controlnet_branches.{i}.*`
keys. This script copies, for each branch index i, the `controlnet_branches.{i}.*`
tensors from the i-th branch checkpoint into the base, producing one self-contained
checkpoint identical to what the 4-file load would assemble at runtime.

Usage:
  python scripts/misc/consolidate_phyco_checkpoint.py \
    --base   <merged base DiT>.pt \
    --branch 0=<controlnet_1 run>.pt \
    --branch 1=<controlnet_2 run>.pt \
    --branch 2=<controlnet_3 run>.pt \
    --output <consolidated>.pt
"""
import argparse
from pathlib import Path

import torch


def _state(ck):
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Merged base DiT checkpoint (backbone, no controlnet_branches).")
    p.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="IDX=PATH",
        help="Branch index and the checkpoint to pull controlnet_branches.IDX.* from. Repeat per branch.",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"[info] loading base DiT: {args.base}")
    out = dict(_state(torch.load(args.base, map_location="cpu")))
    existing_cb = [k for k in out if "controlnet_branches" in k]
    if existing_cb:
        print(f"[warn] base already has {len(existing_cb)} controlnet_branches keys; they will be overwritten per branch")

    prefixes = ("net.", "net_ema.")
    for spec in args.branch:
        idx_str, path = spec.split("=", 1)
        idx = int(idx_str)
        print(f"[info] branch {idx}: pulling controlnet_branches.{idx}.* from {path}")
        bstate = _state(torch.load(path, map_location="cpu"))
        copied = 0
        for pref in prefixes:
            needle = f"{pref}controlnet_branches.{idx}."
            for k, v in bstate.items():
                if k.startswith(needle):
                    out[k] = v
                    copied += 1
        if copied == 0:
            raise RuntimeError(f"No keys matching *controlnet_branches.{idx}.* found in {path}")
        print(f"[info]   copied {copied} tensors for branch {idx}")

    n_net = len([k for k in out if k.startswith("net.") and not k.startswith("net_ema.")])
    n_cb = len([k for k in out if k.startswith("net.") and "controlnet_branches" in k])
    branch_idxs = sorted({k.split("controlnet_branches.")[1].split(".")[0] for k in out if "controlnet_branches" in k})
    print(f"[info] consolidated net.* keys: {n_net} (of which {n_cb} are controlnet_branches); branch indices: {branch_idxs}")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, outp)
    print(f"[success] wrote {outp}")


if __name__ == "__main__":
    main()
