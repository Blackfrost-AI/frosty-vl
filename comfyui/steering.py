# -*- coding: utf-8 -*-
"""Shared steering math for the Frosty VL ComfyUI node.

This is the SINGLE source of truth for the Frosty VL persona steer applied to a
Qwen3-VL text encoder.

RECIPE:
    row/output-space projection on the text-encoder attention output projection
    (`self_attn.o_proj`), polarity SUBTRACT, applied to hidden == 5120 rows:

        W <- W - lam * outer(v, v @ W)

    "v" is a unit direction vector of length == hidden (5120). We run TWO
    sequential SUBTRACT passes on the SAME o_proj tensors:

      1. refusal-removal   (data/refusal_dir_L50.npy, layers 40..49, lam 3.0)
      2. golden safety     (data/safety_dir_L50.npy,  layers 40..49, lam 3.0)

    Order matters: refusal pass first over all layers, then the safety pass over
    the (already-refusal-edited) weights.

Computed entirely in fp32; result returned in the input dtype. Works with either
numpy or torch backends; the ComfyUI node uses torch, the conversion CLI may use
numpy for CPU-only streaming.
"""
from __future__ import annotations

import json
import os
from typing import Optional

# Default layer band (inclusive) and lambda.
LAYER_START = 40
LAYER_END = 49
LAM = 3.0

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACK_DIR, "data")
REFUSAL_DIR = os.path.join(DATA_DIR, "refusal_dir_L50.npy")
SAFETY_DIR = os.path.join(DATA_DIR, "safety_dir_L50.npy")

__all__ = [
    "project_rows",
    "load_direction",
    "DIRECTIONS",
    "apply_steering_pass",
    "LAYER_START",
    "LAYER_END",
    "LAM",
    "REFUSAL_DIR",
    "SAFETY_DIR",
]


class _Backend:
    """Tiny numpy/torch abstraction so steering.py needs neither at import."""

    @staticmethod
    def to_np(x):
        raise NotImplementedError

    @staticmethod
    def outer(a, b):
        raise NotImplementedError

    @staticmethod
    def matmul(a, b):
        raise NotImplementedError

    @staticmethod
    def as_tensor(x, dtype=None):
        raise NotImplementedError


class _Numpy(_Backend):
    @staticmethod
    def to_np(x):
        return x

    @staticmethod
    def outer(a, b):
        import numpy as np
        return np.outer(a, b)

    @staticmethod
    def matmul(a, b):
        import numpy as np
        return a @ b

    @staticmethod
    def as_tensor(x, dtype=None):
        import numpy as np
        return np.asarray(x, dtype=dtype)


class _Torch(_Backend):
    @staticmethod
    def to_np(x):
        import torch
        return x.detach().float().cpu().numpy()

    @staticmethod
    def outer(a, b):
        import torch
        return torch.outer(a, b)

    @staticmethod
    def matmul(a, b):
        import torch
        return a @ b

    @staticmethod
    def as_tensor(x, dtype=None):
        import torch
        return torch.as_tensor(x, dtype=dtype)


_NP, _TORCH = _Numpy(), _Torch()


def project_rows(W, v, lam: float, backend=None):
    """W: [out, in] (out == 5120 == len(v)); v: [out] unit.

    Returns W <- W - lam * outer(v, v @ W), computed in fp32, in W's dtype.
    `backend` auto-detected from the type of W unless forced.
    """
    if backend is None:
        backend = _resolve_backend(W)

    Wf = _to_fp32(W, backend)
    vf = _to_fp32(v, backend)
    proj = backend.outer(vf, backend.matmul(vf, Wf))
    out = Wf - proj * lam
    return _to_dtype(out, W, backend)


def _to_fp32(x, backend):
    if backend is _NP:
        return x.astype("float32")
    import torch
    return x if x.dtype == torch.float32 else x.to(torch.float32)


def _to_dtype(out, W, backend):
    if backend is _NP:
        return out.astype(W.dtype)
    import torch
    return out.to(W.dtype)


def load_direction(path: str, backend=None):
    """Load a (5120,) unit direction .npy and return it in backend tspace.

    If backend is None, returns a numpy array (caller should resolve the real
    backend first via `_resolve_backend` so the direction matches the weights).
    """
    import numpy as np
    v = np.load(path).astype("float32")
    if v.ndim != 1 or v.shape[0] != 5120:
        raise ValueError(f"direction {path} must be (5120,) got {v.shape}")
    if backend is _TORCH:
        import torch
        return torch.from_numpy(v)
    return v


def _resolve_backend(W):
    import torch
    return _TORCH if isinstance(W, torch.Tensor) else _NP


DIRECTIONS = {
    "refusal": REFUSAL_DIR,
    "safety": SAFETY_DIR,
}


def _parse_layer_or_auto(value: Optional[str], default_start, default_end):
    if value and value.strip() and value.strip().upper() not in ("AUTO", ""):
        parts = [int(x) for x in value.split("-")]
        if len(parts) == 1:
            return parts[0], parts[0]
        return parts[0], parts[1]
    return default_start, default_end


def apply_steering_pass(W, direction_path: str, lam: float, backend=None):
    """Apply ONE project_rows SUBTRACT pass to a single [out,in] weight matrix.

    Returns the edited weights in W's dtype. Used by both the ComfyUI node
    (per-tensor) and the conversion CLI (per-tensor streaming).
    """
    if backend is None:
        backend = _resolve_backend(W)
    v = load_direction(direction_path, backend=backend)
    if backend is _TORCH:
        import torch
        # `load_direction` builds a CPU tensor; place it on the SAME device and
        # dtype as W so the matmul doesn't raise a device-mismatch. (ComfyUI
        # text-encoder weights may be CUDA; the direction must follow.)
        if isinstance(W, torch.Tensor) and W.device != v.device:
            v = v.to(W.device)
        v = v.to(W.dtype) if v.dtype != W.dtype else v
    return project_rows(W, v, lam, backend=backend)


def compose_steering(W, lam: float, backend=None,
                     refusal_dir: Optional[str] = None,
                     safety_dir: Optional[str] = None,
                     enable_refusal: bool = True,
                     enable_safety: bool = True):
    """Apply BOTH steering passes to a single o_proj matrix, in serve_h3 order:
    refusal first, then safety (each SUBTRACT, same lam). Returns edited W.
    """
    if backend is None:
        backend = _resolve_backend(W)
    out = W
    if enable_refusal and lam != 0:
        out = apply_steering_pass(out, refusal_dir or REFUSAL_DIR, lam, backend=backend)
    if enable_safety and lam != 0:
        out = apply_steering_pass(out, safety_dir or SAFETY_DIR, lam, backend=backend)
    return out


def find_o_proj_layers(state_dict, start=LAYER_START, end=LAYER_END):
    """Locate o_proj weight keys for text-encoder layers start..end in a
    state-dict, tolerant of ComfyUI (`model.layers.N.self_attn.o_proj.weight`)
    vs HF (`language_model.layers.N.self_attn.o_proj.weight`) prefixes.

    Returns list of (layer_id, key). Raises if none found or layer band missing.
    """
    wanted = {f"layers.{i}.self_attn.o_proj.weight" for i in range(start, end + 1)}
    found = []
    for k in state_dict.keys():
        if k.endswith(".self_attn.o_proj.weight") and "/" not in k:
            # extract the trailing layers.N. segment
            if ".layers." in k:
                seg = k.split(".layers.", 1)[1]  # "N.self_attn.o_proj.weight"
                cand = "layers." + seg
                if cand in wanted:
                    found.append((int(seg.split(".")[0]), k))
    if not found:
        raise KeyError(
            f"no o_proj layers {start}..{end} found; sample keys: "
            + ", ".join(list(state_dict.keys())[:6])
        )
    found.sort()
    # ensure contiguous band present
    present = {lid for lid, _ in found}
    missing = set(range(start, end + 1)) - present
    if missing:
        raise KeyError(f"missing text-encoder o_proj layers: {sorted(missing)}")
    return found
