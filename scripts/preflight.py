#!/usr/bin/env python3
"""Validate the Frosty VL (Qwen3-VL persona plugin) deployment without loading model weights."""
import json
import os
import sys
from pathlib import Path

import numpy as np


def fail(message):
    print("FAIL:", message)
    return False


model = Path(os.environ.get("FVL_MODEL", "/models/base-pipeline"))
refusal = Path(os.environ.get("FVL_REFUSAL_DIR", "/app/server/data/refusal_dir_L50.npy"))
safety = Path(os.environ.get("FVL_SAFETY_DIR", "/app/server/data/safety_dir_L50.npy"))
refusal_lam = float(os.environ.get("FVL_ABL_LAM", "3.0"))
safety_lam = float(os.environ.get("FVL_SAFETY_LAM", "3.0"))
ok = True

components = [
    "modular_model_index.json",
    "text_encoder",
    "tokenizer",
    "processor",
    "transformer",
    "transformer_ref",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
]
for name in components:
    path = model / name
    if not path.exists():
        ok = fail("missing model component %s" % path)

enabled_directions = []
if refusal_lam != 0:
    enabled_directions.append(("refusal", refusal))
if safety_lam != 0:
    enabled_directions.append(("safety", safety))

for label, path in enabled_directions:
    if not path.is_file():
        ok = fail("missing %s direction %s" % (label, path))
        continue
    vector = np.load(path, mmap_mode="r")
    if vector.ndim != 1 or vector.shape[0] != 5120:
        ok = fail("%s direction has shape %s; expected (5120,)" % (label, vector.shape))
    else:
        print("OK: %s direction %s" % (label, path))

try:
    # diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
    # integration - not a MiniMax product dependency.
    from diffusers import MiniMaxH3Ref2VABlocks, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    print("OK: Diffusers Qwen3-VL modular integration imports")
except Exception as exc:
    ok = fail("Diffusers Qwen3-VL modular integration: %s" % exc)

try:
    import torch
    count = torch.cuda.device_count()
    print("CUDA devices:", count)
    if count < 2:
        ok = fail("two CUDA devices are required for the as-built topology")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        print("  cuda:%d %s %.1f GiB" % (index, props.name, props.total_memory / 2**30))
    if count >= 2 and any(torch.cuda.get_device_properties(i).total_memory < 175 * 2**30 for i in (0, 1)):
        ok = fail("cuda:0 and cuda:1 should each expose at least 175 GiB")
except Exception as exc:
    ok = fail("CUDA/PyTorch check: %s" % exc)

index_path = model / "modular_model_index.json"
if index_path.is_file():
    try:
        index = json.loads(index_path.read_text())
        print("Model index class:", index.get("_class_name"))
    except Exception as exc:
        ok = fail("could not parse modular_model_index.json: %s" % exc)

if not ok:
    sys.exit(1)
print("PASS: deployment artifacts and runtime prerequisites look ready")
