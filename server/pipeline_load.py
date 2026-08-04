"""Local-load fix for Frosty VL (Qwen3-VL persona plugin) using the experimental Diffusers modular pipeline."""
import os

import torch
# diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
# integration - not a MiniMax product dependency.
from diffusers import MiniMaxH3Ref2VABlocks, ModularPipeline

# diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
# integration - not a MiniMax product dependency. diffusers uses this id to
# resolve component *classes* offline; it is NOT a weights source.
HUB_REPO = "MiniMaxAI/MiniMax-H3"
DEFAULT_MODEL = "/models/base-pipeline"


def _remap_local(pipe, model_dir: str) -> None:
    """Point Hub-backed ComponentSpecs at the complete local checkpoint root."""
    specs = getattr(pipe, "_component_specs", None) or {}
    for name, spec in specs.items():
        original = getattr(spec, "pretrained_model_name_or_path", None)
        if isinstance(original, str) and original.startswith(HUB_REPO):
            spec.pretrained_model_name_or_path = model_dir
            print("  remap %s -> local (%s)" % (name, getattr(spec, "subfolder", "?")), flush=True)


def load_pipeline(
    model_dir: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    offline: bool = True,
    workflow: str = "fl2va",
):
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("loading ModularPipeline from", model_dir, flush=True)
    if workflow == "ref2va":
        # diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL
        # text-encoder integration - not a MiniMax product dependency.
        pipe = MiniMaxH3Ref2VABlocks().init_pipeline(model_dir)
    elif workflow in ("t2va", "fl2va"):
        pipe = ModularPipeline.from_pretrained(model_dir)
    else:
        raise ValueError("unknown Frosty VL workflow: %s" % workflow)

    _remap_local(pipe, model_dir)
    pipe.load_components(dtype=dtype)
    pipe.to(device)
    print("Frosty VL %s loaded on %s (dtype=%s)" % (workflow, device, dtype), flush=True)
    return pipe
