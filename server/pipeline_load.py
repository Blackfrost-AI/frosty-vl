# -*- coding: utf-8 -*-
"""Local-load fix for the Frosty VL (Qwen3-VL persona plugin) modular pipeline.

WHY THIS EXISTS
  The base pipeline's `modular_model_index.json` hardcodes an upstream hub repo
  id as `pretrained_model_name_or_path` for every component. diffusers'
  ModularPipeline does NOT substitute the local root, so any non-hub / offline
  load fails: components (notably the Qwen3-VL `text_encoder`) resolve against
  the empty hub cache and register as `None`, crashing generation with
  `'NoneType' object has no attribute 'dtype'`.

  Two conditions unblock a fully-local, hermetic load:
    1. Remap each ComponentSpec's `pretrained_model_name_or_path` to the local
       model dir (subfolder is preserved, so nothing touches the hub).
    2. `accelerate` must be installed — the transformer sets
       `keep_in_fp32_modules` (proj_in/audio_proj_in/proj_out/audio_proj_out),
       which requires `low_cpu_mem_usage=True`, which requires accelerate.

RESULT
  All core components load from the local dir: text_encoder (Qwen3-VL, bf16),
  transformer (bf16), vae (fp32), audio_vae (fp32), tokenizer, processor,
  scheduler, audio_scheduler. `transformer_ref` loads only for the Ref2VA
  reference branch and is intentionally absent on the FL2VA/t2va path.

USAGE
    from pipeline_load import load_pipeline
    pipe = load_pipeline()             # loads + .to("cuda")
    state = pipe(prompt="...")

    # Optional: compile the repeated transformer blocks. This is lazy (the
    # first generation compiles) and may retain substantial CUDA cache memory.
    pipe = load_pipeline(compile_repeated=True)
"""
import os
import torch
# diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
# integration - external library API, not branding.
from diffusers import MiniMaxH3Ref2VABlocks, ModularPipeline

# diffusers uses this id to resolve component *classes* offline; it is the
# external library API, not branding, and is NOT a weights source.
HUB_REPO = "MiniMaxAI/MiniMax-H3"
DEFAULT_MODEL = "/models/base-pipeline"


def _remap_local(pipe, model_dir: str) -> None:
    """Rewrite any ComponentSpec pointing at the hub repo id to the local dir."""
    specs = getattr(pipe, "_component_specs", None) or {}
    for name, spec in specs.items():
        orig = getattr(spec, "pretrained_model_name_or_path", None)
        if isinstance(orig, str) and orig.startswith(HUB_REPO):
            spec.pretrained_model_name_or_path = model_dir
            print("  remap %s -> local (%s)" % (name, getattr(spec, "subfolder", "?")), flush=True)


def load_pipeline(
    model_dir: str = DEFAULT_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    offline: bool = True,
    workflow: str = "fl2va",
    compile_repeated: bool = False,
):
    """Load the Frosty VL ModularPipeline fully from the LOCAL model dir.

    Hermetic by default: sets HF_HUB_OFFLINE so the (empty) hub cache can't
    silently intercept a component. Remaps all component paths to model_dir.
    Returns a pipeline already moved to `device` with components loaded.

    ``compile_repeated`` opts into Diffusers regional ``torch.compile`` for the
    declared transformer/refiner blocks. Compilation is lazy, so the first
    request for each input shape pays the compile cost and may reserve much more
    CUDA memory. It is deliberately disabled by default.
    """
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("loading ModularPipeline from", model_dir, flush=True)
    if workflow == "ref2va":
        # Ref2VA is a second block graph over the same repository. Constructing
        # it through its blocks selects transformer_ref instead of transformer.
        pipe = MiniMaxH3Ref2VABlocks().init_pipeline(model_dir)
    elif workflow in ("t2va", "fl2va"):
        pipe = ModularPipeline.from_pretrained(model_dir)
    else:
        raise ValueError("unknown Frosty VL workflow: %s" % workflow)
    _remap_local(pipe, model_dir)
    pipe.load_components(dtype=dtype)
    pipe.to(device)
    if compile_repeated:
        component_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        transformer = pipe.components.get(component_name)
        if transformer is None:
            raise RuntimeError("cannot compile missing component: %s" % component_name)
        repeated = getattr(transformer, "_repeated_blocks", ())
        region_count = sum(
            module.__class__.__name__ in repeated for module in transformer.modules()
        )
        transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)
        print(
            "Frosty VL %s regional compile enabled for %d repeated blocks "
            "(fullgraph=True, dynamic=True; compilation is lazy)"
            % (workflow, region_count),
            flush=True,
        )
    print("Frosty VL %s loaded on %s (dtype=%s)" % (workflow, device, dtype), flush=True)
    return pipe
