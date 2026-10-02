#!/usr/bin/env python3
"""Pack a Kev pointer head and metadata into a model GGUF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "gguf-py"))

import gguf


TOKENS = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "opt_start": "<|box_start|>",
    "opt_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}


def find_token(tokens: list[str], value: str) -> int:
    try:
        return tokens.index(value)
    except ValueError as exc:
        raise ValueError(f"token not found in source vocabulary: {value}") from exc


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)


def source_name(manifest_path: Path | None, head: dict) -> str:
    if manifest_path is None:
        return "jaredpalmer/kev"
    manifest = json.loads(manifest_path.read_text())
    source = manifest.get("run", "jaredpalmer/kev")
    revision = manifest.get("revision", manifest.get("rev"))
    return f"{source}@{revision}" if revision else source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.head.suffix != ".json":
        parser.error("--head must point to head.json")

    reader = gguf.GGUFReader(args.gguf, "r")
    head = json.loads(args.head.read_text())
    hidden = int(head["hidden_size"])
    head_dim = int(head["head_dim"])

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise ValueError("source GGUF has no architecture metadata")
    writer = gguf.GGUFWriter(args.out, arch=arch_field.contents(), endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = int(alignment.contents())
    copy_metadata(reader, writer)

    source = source_name(args.manifest, head)
    writer.add_key_value(gguf.Keys.General.NAME, f"Kev {source.rsplit('/', 1)[-1]}", gguf.GGUFValueType.STRING)
    writer.add_key_value(gguf.Keys.General.TYPE, "model", gguf.GGUFValueType.STRING)
    tags_key = gguf.Keys.General.TAGS
    existing_tags = reader.get_field(tags_key)
    tags = existing_tags.contents() if existing_tags is not None else []
    if "kev" not in tags:
        tags = [*tags, "kev"]
    writer.add_key_value(tags_key, tags, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    tokens_field = reader.get_field(gguf.Keys.Tokenizer.LIST)
    if tokens_field is None:
        raise ValueError("source GGUF has no tokenizer vocabulary")
    tokens = tokens_field.contents()

    writer.add_uint32("kev.version", 1)
    writer.add_float32("kev.temperature", float(head["temperature"]))
    writer.add_uint32("kev.head_dim", head_dim)
    for name, token in TOKENS.items():
        writer.add_int32(f"kev.tokens.{name}", find_token(tokens, token))
    writer.add_uint32("kev.limits.max_state", 8192)
    writer.add_uint32("kev.limits.max_row", 8192)
    writer.add_string("kev.source", source)

    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.data.shape, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)

    for name in ("q_weight", "k_weight"):
        values = np.asarray(head[name], dtype=np.float32)
        if values.shape != (head_dim, hidden):
            raise ValueError(f"{name} has shape {values.shape}, expected {(head_dim, hidden)}")
        # head.json rows are [head_dim, hidden]; GGUF ne is [hidden, head_dim].
        writer.add_tensor(f"dec.head_{name[0]}.weight", values)

    for name in ("q_bias", "k_bias"):
        values = np.asarray(head[name], dtype=np.float32)
        if values.shape != (head_dim,):
            raise ValueError(f"{name} has shape {values.shape}, expected {(head_dim,)}")
        writer.add_tensor(f"dec.head_{name[0]}.bias", values)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
