#!/usr/bin/env python3
"""Merge a Kev v1.0 PEFT LoRA adapter into a Qwen3.5 base checkpoint, in place.

The adapter targets the text stack as `base_model.model.layers.*` while the
multimodal base stores those weights as `model.language_model.layers.*`, so
`peft.merge_and_unload` cannot resolve the names. The math is the same as
peft's fp32 `get_delta_weight`: W += (lora_alpha / r) * B @ A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ADAPTER_PREFIX = "base_model.model."
BASE_PREFIX = "model.language_model."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("base_dir", type=Path)
    args = parser.parse_args()

    cfg = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]

    with safe_open(args.adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        adapter = {k: f.get_tensor(k) for k in f.keys()}

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter.items():
        if not key.startswith(ADAPTER_PREFIX) or not key.endswith(".weight"):
            sys.exit(f"unexpected adapter tensor name: {key}")
        stem, lora_key = key.rsplit(".lora_", 1)
        which, suffix = lora_key.split(".", 1)
        if suffix != "weight" or which not in ("A", "B"):
            sys.exit(f"unexpected adapter tensor name: {key}")
        pairs.setdefault(stem, {})[which] = tensor

    index_path = args.base_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        weight_map = None

    shard_deltas: dict[str, list[tuple[str, dict[str, torch.Tensor]]]] = {}
    missing = []
    for stem, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            sys.exit(f"incomplete lora pair for {stem}")
        target = BASE_PREFIX + stem[len(ADAPTER_PREFIX):] + ".weight"
        shard = weight_map.get(target) if weight_map else "model.safetensors"
        if shard is None:
            missing.append(target)
            continue
        shard_deltas.setdefault(shard, []).append((target, ab))

    applied = 0
    for shard, deltas in sorted(shard_deltas.items()):
        path = args.base_dir / shard
        tensors = load_file(path)
        for target, ab in deltas:
            if target not in tensors:
                missing.append(target)
                continue
            base = tensors[target].to(torch.float32)
            delta = (ab["B"].to(torch.float32) @ ab["A"].to(torch.float32)) * scale
            if base.shape != delta.shape:
                sys.exit(f"shape mismatch for {target}: base {tuple(base.shape)} delta {tuple(delta.shape)}")
            tensors[target] = base + delta
            applied += 1
        save_file(tensors, path, metadata={"format": "pt"})
        print(f"{shard}: merged {len(deltas)} tensors")

    print(f"applied {applied}/{len(pairs)} deltas, missing {len(missing)}")
    for name in missing:
        print(f"  MISSING {name}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
