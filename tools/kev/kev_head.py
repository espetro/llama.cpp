#!/usr/bin/env python3
"""Convert a Kev v1.0 head.pt checkpoint to the head.json format kev_pack expects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("head_pt", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    sd = torch.load(args.head_pt, map_location="cpu", weights_only=True)
    head = sd["head"]

    q_weight = head["q.weight"].float()
    k_weight = head["k.weight"].float()
    head_dim, hidden = q_weight.shape

    out = {
        "hidden_size": hidden,
        "head_dim": head_dim,
        "temperature": float(sd["temperature"]),
        "q_weight": q_weight.tolist(),
        "k_weight": k_weight.tolist(),
        "q_bias": head["q.bias"].float().tolist(),
        "k_bias": head["k.bias"].float().tolist(),
    }
    for extra in ("base", "base_revision", "lora", "temperature_fit", "weights_dtype"):
        if extra in sd:
            out[extra] = sd[extra]

    args.out.write_text(json.dumps(out))
    print(f"{args.out}: head_dim={head_dim} hidden={hidden} temperature={out['temperature']}")


if __name__ == "__main__":
    main()
