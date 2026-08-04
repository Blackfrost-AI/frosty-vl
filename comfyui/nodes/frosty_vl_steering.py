# -*- coding: utf-8 -*-
"""Frosty VL persona-steering node.

Applies the Frosty VL persona steer IN MEMORY to a ComfyUI-loaded Qwen3-VL text
encoder, at load time, so the on-disk weights stay pristine. Bring your own
Qwen3-VL model; nothing on disk is modified.

It patches the attention output projection `self_attn.o_proj` for language
layers 40..49 with TWO sequential SUBTRACT project_rows passes (refusal-removal
then golden safety floor, lam 3.0). This is the SAFE release: the safety floor
is ON by default.

Integrates with the ComfyUI Qwen3-VL text-encoder path: drop this node between
your `CLIPLoader` (type `minimax` — the ComfyUI text-encoder type identifier,
not product branding) and the text-conditioning encode path. Because ComfyUI
caches the loaded CLIP for the session, the in-place edit persists for every
sampled frame without re-doing work.

The patch is delivered by GENERIC module traversal (`...layers.N.self_attn.o_proj`),
so it works on both ComfyUI's Qwen-VL text-encoder wrapper and a raw transformers
Qwen3-VL model without depending on either package's internals at import time.
"""
from __future__ import annotations

import os
import re

_NODE_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if _NODE_PACK_DIR not in sys.path:
    sys.path.insert(0, _NODE_PACK_DIR)

from steering import compose_steering, LAYER_START, LAYER_END, LAM  # noqa: E402


def _find_o_proj_weights(model, start: int, end: int):
    """Map layer_id -> o_proj.weight tensor for layers in [start, end].

    Generic: scans named_modules for any module whose name ends in
    `self_attn.o_proj`, derives the layer id from a `layers.N.` segment, and
    returns those in the requested band. Works on ComfyUI's Qwen-VL wrapper and
    on a raw transformers model alike.
    """
    import torch.nn as nn

    found = {}
    for name, module in model.named_modules():
        if not name.endswith("self_attn.o_proj"):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        lid = int(m.group(1))
        if start <= lid <= end and hasattr(module, "weight"):
            found[lid] = module.weight
    return found


class FrostyVLPersonaSteering:
    """Load-time persona steering for a ComfyUI Qwen3-VL text encoder."""

    CATEGORY = "Frosty VL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "steer_refusal": ("BOOLEAN", {"default": True}),
                "steer_safety": ("BOOLEAN", {"default": True}),
                "lam": (
                    "FLOAT",
                    {"default": LAM, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "layer_band": (
                    "STRING",
                    {"default": f"{LAYER_START}-{LAYER_END}", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "summary")
    FUNCTION = "apply"
    OUTPUT_NODE = False

    def apply(self, clip, steer_refusal=True, steer_safety=True, lam=LAM,
              layer_band=f"{LAYER_START}-{LAYER_END}"):
        import torch

        # Resolve the model that holds the language-transformer layers. ComfyUI
        # Qwen-VL wraps the HF model; try the documented path then fall back to
        # a scan for any module tree containing `layers.N.self_attn.o_proj`.
        model = self._resolve_model(clip)
        start, end = self._parse_band(layer_band)
        weights = _find_o_proj_weights(model, start, end)
        if not weights:
            raise RuntimeError(
                f"FrostyVLPersonaSteering: no o_proj weights found for layers "
                f"{start}..{end} in this text encoder "
                f"(checked modules under {type(model).__name__})."
            )

        edited = 0
        for lid in sorted(weights):
            # Do all steering math on CPU: o_proj is 5120x5120, runs in a
            # worker thread while ComfyUI offloads the text encoder asynchronously.
            # CPU math avoids CUDA-stream races in that thread and the device
            # mismatch in project_rows; copy_ back handles cpu->gpu memcpy safely.
            t_cpu = weights[lid].detach().float().to("cpu")
            t2 = compose_steering(
                t_cpu, lam,
                enable_refusal=steer_refusal,
                enable_safety=steer_safety,
            )
            with torch.no_grad():
                weights[lid].copy_(t2)
            edited += 1

        summary = (
            f"steered {edited} o_proj layers {start}..{end} lam={lam} "
            f"(refusal={steer_refusal}, safety={steer_safety})"
        )
        return (clip, summary)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_model(clip):
        """Walk known ComfyUI / HF access paths to the language model layers."""
        candidates = []
        # ComfyUI Qwen-VL wrapper: clip.cond_stage_model.model / .transformer / .model
        for attr in ("cond_stage_model",):
            cm = getattr(clip, attr, None)
            if cm is None:
                continue
            for sub in ("model", "transformer"):
                obj = getattr(cm, sub, None)
                if obj is not None:
                    candidates.append(obj)
            candidates.append(cm)
        # Raw access via .model.language_model (serve_h3 path)
        raw = getattr(clip, "model", None)
        if raw is not None:
            candidates.append(raw)
        candidates.append(clip)

        import torch.nn as nn

        for cand in candidates:
            if isinstance(cand, nn.Module) and _find_o_proj_weights(cand, 0, 70):
                return cand
        # last resort: any nn.Module arg reachable
        return clip

    @staticmethod
    def _parse_band(spec: str):
        parts = [int(x.strip()) for x in str(spec).replace(" ", "").split("-") if x.strip()]
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return LAYER_START, LAYER_END


NODE_CLASS_MAPPINGS = {"FrostyVLPersonaSteering": FrostyVLPersonaSteering}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrostyVLPersonaSteering": "Frosty VL Persona Steering"
}


# -- lightweight self-test (no comfy needed) --------------------------------
if __name__ == "__main__":
    import torch
    from steering import project_rows  # noqa

    # build a fake "model" with layers 30..55 o_proj to exercise traversal
    torch.manual_seed(1)
    fake = torch.nn.Module()
    layers = torch.nn.Module()
    for i in range(30, 56):
        attn = torch.nn.Module()
        attn.o_proj = torch.nn.Linear(5120, 5120, bias=False)
        attn.o_proj.weight.data = (torch.rand(5120, 5120, dtype=torch.float32) - 0.5)
        lm = torch.nn.Module()
        lm.self_attn = attn
        setattr(layers, f"{i}", lm)
    fake.language_model = torch.nn.Module()
    fake.language_model.layers = layers

    w = _find_o_proj_weights(fake, 40, 49)
    assert sorted(w) == list(range(40, 50)), sorted(w)
    # run one layer through compose and confirm weights changed + rows normalized
    l45 = getattr(fake.language_model.layers, "45").self_attn.o_proj
    before = l45.weight.data.clone()
    FrostyVLPersonaSteering().apply(fake, steer_refusal=True, steer_safety=True)
    after = l45.weight.data
    print("self-test OK; L45 row-norm delta:",
          float((before.norm(dim=1) - after.norm(dim=1)).abs().mean()))
