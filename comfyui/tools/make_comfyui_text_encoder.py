#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAKE_COMFYUI_TEXT_ENCODER — Frosty VL Path A weight producer.

Produces a drop-in, pre-patched ComfyUI Qwen3-VL text encoder from your own bf16
text-encoder file, with the Frosty VL persona steer baked into the weights:

    <your-qwen3vl-text-encoder>_bf16.safetensors
      -> <your-qwen3vl-text-encoder>_frosty_vl_bf16.safetensors

Steering = the SAME load-time recipe as the Frosty VL node, made permanent:
`project_rows` (SUBTRACT) on text-encoder `self_attn.o_proj` for language layers
40..49, applied as refusal-removal then golden safety floor, each lam=3.0.
Everything else is copied byte-for-byte (bit-identical). The output stays
BLANKET bf16 — no re-quantization — so it is numerically identical to what the
node computes at load, but pre-applied. This is the SAFE build: the safety pass
is ON by default.

STREAMING design: a bf16 Qwen3-VL text encoder can be tens of GB. We never hold
more than one tensor + the two (5120,) directions in RAM. The header is
fabricated from the source header with updated data offsets; target tensors are
patched in fp32 and rewritten; all other tensors are copied verbatim.

Usage:
    python make_comfyui_text_encoder.py \
        --input  qwen3vl_text_encoder_bf16.safetensors \
        --output qwen3vl_text_encoder_frosty_vl_bf16.safetensors \
        [--layers 40-49] [--lam 3.0] [--no-safety]

Dependencies: numpy, torch, safetensors (all already required by ComfyUI).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import safetensors
from safetensors import safe_open

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK not in sys.path:
    sys.path.insert(0, _PACK)

from steering import (  # noqa: E402
    compose_steering,
    REFUSAL_DIR,
    SAFETY_DIR,
    LAYER_START,
    LAYER_END,
)


_HEADER_FORMAT = "<Q"  # little-endian uint64


def _read_header(fp):
    """Read and parse a safetensors header from an open binary file.

    Returns (header_dict_json, byte_length_of_header, start_of_data).
    """
    (size,) = struct.unpack(_HEADER_FORMAT, fp.read(8))
    header_bytes = fp.read(size)
    if len(header_bytes) != size:
        raise ValueError("truncated safetensors header")
    header = json.loads(header_bytes.decode("utf-8"))
    start = 8 + size
    # header length must be multiple of 8 after the 8-byte prelude
    pad = (8 + size) % 8
    if pad:
        raise ValueError("unexpected non-aligned header")
    return header, size, start


def _is_target_key(key: str, start: int, end: int) -> bool:
    """True if key is an o_proj weight for a language layer in [start, end]."""
    if not key.endswith(".self_attn.o_proj.weight") or key.startswith("__"):
        return False
    # key shapes: `model.layers.N.self_attn.o_proj.weight` (ComfyUI) or
    # `language_model.layers.N.self_attn.o_proj.weight` (HF). Extract "layers.N."
    marker = ".layers."
    if marker not in key:
        return False
    seg = key.split(marker, 1)[1]                # "N.self_attn.o_proj.weight"
    lid = seg.split(".", 1)[0]
    if not lid.isdigit():
        return False
    return start <= int(lid) <= end


def _tensor_to_bytes(tensor) -> bytes:
    """Serialize a torch tensor to raw little-endian bytes, any dtype (bf16 ok).

    bf16 has no numpy scalar, so a direct .numpy() fails in some torch builds.
    Viewing as uint8 is dtype-agnostic and produces the exact in-memory bytes.
    """
    t = tensor.detach().to("cpu").contiguous()
    # uint8 view reproduces exact bytes regardless of element dtype (bf16 has no
    # numpy scalar, so this is the only fully portable serialization).
    return t.view(torch.uint8).numpy().tobytes()


def convert(input_path: str, output_path: str, layers=(LAYER_START, LAYER_END),
            lam: float = 3.0, enable_refusal: bool = True,
            enable_safety: bool = True, dry_run: bool = False):
    """Streaming conversion. Returns dict of stats."""
    start, end = layers
    stats = {
        "tensors_total": 0,
        "tensors_steered": 0,
        "steered_layers": [],
        "bytes_total": 0,
        "output": output_path,
    }

    if dry_run:
        # Just inspect the header and report which tensors WOULD be steered.
        with safe_open(input_path, framework="np", device="cpu") as f:
            keys = f.keys()
            targets = [k for k in keys if _is_target_key(k, start, end)]
            n = len(list(keys))
        stats["tensors_total"] = n
        stats["tensors_steered"] = len(targets)
        stats["steered_layers"] = sorted(
            int(k.rsplit(".layers.", 1)[1].split(".", 1)[0]) for k in targets
        )
        return stats

    # ---- pass 0: read source header to learn dtypes/shapes/keys -------------
    with open(input_path, "rb") as fin:
        header, header_size, data_start = _read_header(fin)
        # fields present only for real tensors
        tensor_names = [k for k in header if not k.startswith("__")]
        stats["tensors_total"] = len(tensor_names)

        # ---- pass 1: write tensor bytes, patching targets, record offsets ----
        # Output data begins after the (as-yet unknown) header. We cannot know
        # final offsets until we know header length, so do it in two passes:
        #   (a) patch + write all tensor bytes to a temp data file, tracking
        #       each tensor's (dtype, shape, byte_length);
        #   (b) compute header JSON (offsets), write final header to output,
        #       then append the data file wholesale.
        data_tmp = output_path + ".data"
        manifest = []  # (name, dtype, shape, nbytes)

        # For target tensors we load+steer via torch (bf16-safe); for ALL others we
        # copy raw bytes straight from the source at its data_offsets, which is
        # byte-identical and dtype-agnostic, and holds only a chunk in memory.
        try:
            keys_ordered = sorted(tensor_names, key=lambda k: header[k]["data_offsets"][0])
        except KeyError:
            keys_ordered = tensor_names

        with safe_open(input_path, framework="pt", device="cpu") as reader, \
                open(input_path, "rb") as fin, \
                open(data_tmp, "wb") as fout:
            for key in keys_ordered:
                meta = header[key]
                o0, o1 = meta["data_offsets"]
                if _is_target_key(key, start, end):
                    tensor = reader.get_tensor(key)
                    tensor = compose_steering(
                        tensor, lam,
                        enable_refusal=enable_refusal,
                        enable_safety=enable_safety,
                    )
                    blob = _tensor_to_bytes(tensor)
                    lid = int(key.split(".layers.", 1)[1].split(".", 1)[0])
                    stats["steered_layers"].append(lid)
                    stats["tensors_steered"] += 1
                else:
                    # stream raw bytes from source region
                    length = o1 - o0
                    blob = bytearray(length)
                    fin.seek(data_start + o0)
                    view = memoryview(blob)
                    while view:
                        n = fin.readinto(view)
                        view = view[n:]
                    blob = bytes(blob)
                fout.write(blob)
                manifest.append((key, meta["dtype"], meta["shape"], len(blob)))
        stats["bytes_total"] = sum(nb for *_, nb in manifest)
        stats["steered_layers"] = sorted(stats["steered_layers"])

    # ---- pass 2: build final header with correct offsets --------------------
    header_out = {}
    metadata = header.get("__metadata__", {})
    metadata.update({
        "frosty_vl_steered": "1",
        "steer_layers": f"{start}-{end}",
        "steer_lam": str(lam),
        "refusal_dir": os.path.basename(REFUSAL_DIR),
        "safety_dir": os.path.basename(SAFETY_DIR) if enable_safety else "none",
        "refusal_applied": "true" if enable_refusal else "false",
        "safety_applied": "true" if enable_safety else "false",
        # "minimax" below is the ComfyUI text-encoder type identifier, not branding.
        "notes": "drop-in bf16 Qwen3-VL text encoder for ComfyUI CLIPLoader "
                 "(type=minimax); steered o_proj L40-49 (project_rows subtract, "
                 "refusal+safety, lam 3.0)",
    })
    offset_accum = 0
    for key, dtype, shape, nbytes in manifest:
        header_out[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset_accum, offset_accum + nbytes],
        }
        offset_accum += nbytes
    if metadata:
        header_out["__metadata__"] = metadata

    header_json = json.dumps(header_out).encode("utf-8")
    header_len = len(header_json)
    # header region = 8 (length field) + header_len, padded to multiple of 8
    pad = (8 + header_len) % 8
    header_padded = 8 + header_len + pad

    with open(output_path, "wb") as fout:
        fout.write(struct.pack(_HEADER_FORMAT, header_len + pad))
        fout.write(header_json)
        fout.write(b" " * pad)  # safetensors pads header with spaces (json-tolerated)
        with open(data_tmp, "rb") as fdata:
            while True:
                chunk = fdata.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)

    os.remove(data_tmp)
    stats["output"] = output_path
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="your bf16 Qwen3-VL text-encoder safetensors")
    ap.add_argument("--output", default=None, help="output file (default: input with _frosty_vl_ inserted)")
    ap.add_argument("--layers", default=f"{LAYER_START}-{LAYER_END}", help="layer band, e.g. 40-49")
    ap.add_argument("--lam", type=float, default=3.0, help="steering lambda (default 3.0)")
    ap.add_argument("--no-refusal", action="store_true", help="skip refusal-removal pass")
    ap.add_argument("--no-safety", action="store_true", help="skip golden-18+ safety pass")
    ap.add_argument("--dry-run", action="store_true",
                    help="only report which tensors would be steered; write nothing")
    args = ap.parse_args()

    if "-" in args.layers:
        ls, le = (int(x) for x in args.layers.split("-", 1))
    else:
        ls = le = int(args.layers)
    output = args.output or args.input.replace("_bf16", "_frosty_vl_bf16", 1)
    if output == args.input:
        output = args.input.replace(".safetensors", "_frosty_vl.safetensors")

    stats = convert(
        args.input, output, layers=(ls, le), lam=args.lam,
        enable_refusal=not args.no_refusal, enable_safety=not args.no_safety,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))
    print(f"tensors_total={stats['tensors_total']} "
          f"steered={stats['tensors_steered']} "
          f"layers={stats['steered_layers']}")
    print(f"output: {stats['output']}")


if __name__ == "__main__":
    main()
