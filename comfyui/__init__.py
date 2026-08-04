# -*- coding: utf-8 -*-
"""Frosty VL — ComfyUI persona-steering node for Qwen3-VL text encoders.

Bring your own Qwen3-VL model; the steering is applied in memory at load time,
so the on-disk weights are never modified. The node applies a `project_rows`
patch on the text encoder's `self_attn.o_proj` layers 40..49 with
refusal-removal + golden safety floor (lam 3.0). This is the SAFE release: the
safety floor is ON by default.

Two ways to use:
  Path A (pre-patched weights): run tools/make_comfyui_text_encoder.py to bake
      the steer into a drop-in text-encoder safetensors file, place it in
      ComfyUI/models/text_encoders/ and select it in CLIPLoader
      (type `minimax` — the ComfyUI text-encoder type identifier, not product
      branding). No custom node needed.
  Path B (load-time steering node): use the `FrostyVLPersonaSteering` node on the
      CLIP coming out of CLIPLoader to steer the loaded weights in memory.

ComfyUI custom-node contract. Importing this file must not crash ComfyUI, so
nothing here hard-requires torch / comfy at import time — the node module only
imports them lazily when a node is actually instantiated.
"""
from __future__ import annotations

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_m = {}
_d = {}

try:
    from .nodes.frosty_vl_steering import (
        NODE_CLASS_MAPPINGS as _m,
        NODE_DISPLAY_NAME_MAPPINGS as _d,
    )
except (ImportError, ModuleNotFoundError):
    # Fallback for hyphenated install-folder names that some ComfyUI loaders
    # mis-identify as package roots: import via absolute path instead.
    import os as _os
    import sys as _sys
    _PACK = _os.path.dirname(_os.path.abspath(__file__))
    if _PACK not in _sys.path:
        _sys.path.insert(0, _PACK)
    from nodes.frosty_vl_steering import (  # noqa: E402
        NODE_CLASS_MAPPINGS as _m,
        NODE_DISPLAY_NAME_MAPPINGS as _d,
    )
except Exception as _e:  # pragma: no cover - defensive for hostile envs
    import traceback
    traceback.print_exc()

NODE_CLASS_MAPPINGS.update(_m)
NODE_DISPLAY_NAME_MAPPINGS.update(_d)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
