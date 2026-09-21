"""Reversible DWM-equivalent activation projection for Qwen Image 2.1.

The installed Qwen Image engine uses NF4 writers.  Dequantizing, changing, and
requantizing those weights would add codec error.  For a bias-free writer the
activation projection below is algebraically identical to Frosty VL's row
weight projection, while leaving both disk and in-memory weights untouched.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from safetensors import safe_open

EXPECTED_LAYERS = 36
EXPECTED_HIDDEN = 4096
_request_scale: ContextVar[float] = ContextVar("qwen_image_dwm_scale", default=1.0)


def _parse_layers(value: str) -> tuple[int, ...]:
    chosen = set()
    for segment in value.split(","):
        segment = segment.strip()
        if not segment:
            continue
        pieces = segment.split("-", 1)
        start = int(pieces[0])
        end = int(pieces[1]) if len(pieces) == 2 else start
        if end < start:
            raise ValueError(f"descending DWM layer range: {segment}")
        chosen.update(range(start, end + 1))
    layers = tuple(sorted(chosen))
    if not layers or layers[0] < 0 or layers[-1] >= EXPECTED_LAYERS:
        raise ValueError(f"FVL_IMAGE_DWM_LAYERS must select layers within 0..{EXPECTED_LAYERS - 1}")
    return layers


def _project(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    if hidden.shape[-1] != EXPECTED_HIDDEN:
        raise ValueError(f"Qwen Image DWM expected hidden width {EXPECTED_HIDDEN}, got {hidden.shape[-1]}")
    strength = float(alpha) * float(_request_scale.get())
    if strength == 0:
        return hidden
    values = hidden.float()
    vector = direction.to(device=hidden.device, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1.0e-12)
    coefficient = torch.matmul(values, vector)
    return (values - strength * coefficient.unsqueeze(-1) * vector).to(hidden.dtype)


def _hidden(output) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value):
        raise TypeError(f"DWM hook output is not a tensor: {type(output).__name__}")
    return value


def _replace(output, hidden: torch.Tensor):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported DWM hook output: {type(output).__name__}")


@contextmanager
def request_scale(value: float):
    token = _request_scale.set(float(value))
    try:
        yield
    finally:
        _request_scale.reset(token)


def configure_from_environment(text_encoder) -> dict:
    bank_value = os.environ.get("FVL_IMAGE_DWM_BANK", "").strip()
    if not bank_value:
        return {"enabled": False, "mode": "clean", "weights_modified": False}

    bank = Path(bank_value).expanduser().resolve()
    if not bank.is_file():
        raise RuntimeError(f"Qwen Image DWM bank not found: {bank}")
    layers = _parse_layers(os.environ.get("FVL_IMAGE_DWM_LAYERS", ""))
    alpha_attention = float(os.environ.get("FVL_IMAGE_DWM_ATTN_ALPHA", "1.0"))
    alpha_mlp = float(os.environ.get("FVL_IMAGE_DWM_MLP_ALPHA", "1.0"))
    if alpha_attention < 0 or alpha_mlp < 0:
        raise ValueError("Qwen Image DWM alpha values must be non-negative")

    with safe_open(str(bank), framework="pt", device="cpu") as handle:
        attention = handle.get_tensor("direction.attention_output_last").float()
        mlp = handle.get_tensor("direction.mlp_output_last").float()
        metadata = handle.metadata() or {}
    expected = (EXPECTED_LAYERS, EXPECTED_HIDDEN)
    if tuple(attention.shape) != expected or tuple(mlp.shape) != expected:
        raise RuntimeError(
            f"Qwen Image DWM bank must be {expected}; got {tuple(attention.shape)} and {tuple(mlp.shape)}"
        )
    if metadata.get("revision") != "b3179ad355be050328e483a9dfdd9e60cd62adfa":
        raise RuntimeError("Qwen Image DWM bank source revision mismatch")

    model_layers = text_encoder.model.language_model.layers
    if len(model_layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"Qwen Image encoder has {len(model_layers)} layers, expected {EXPECTED_LAYERS}")
    handles = []
    for layer_id in layers:
        layer = model_layers[layer_id]

        def attention_hook(_module, _inputs, output, index=layer_id):
            return _replace(output, _project(_hidden(output), attention[index], alpha_attention))

        def mlp_hook(_module, _inputs, output, index=layer_id):
            return _replace(output, _project(_hidden(output), mlp[index], alpha_mlp))

        if alpha_attention:
            handles.append(layer.self_attn.register_forward_hook(attention_hook))
        if alpha_mlp:
            handles.append(layer.mlp.register_forward_hook(mlp_hook))
    # Keep handles with the encoder for explicit lifetime and future clean-mode removal.
    text_encoder._blackfrost_image_dwm_handles = handles
    return {
        "enabled": True,
        "mode": "runtime_activation_projection",
        "bank": str(bank),
        "layers": list(layers),
        "alpha_attention": alpha_attention,
        "alpha_mlp": alpha_mlp,
        "weights_modified": False,
        "source_revision": metadata["revision"],
        "pairs": int(metadata.get("pairs", "0")),
    }

