"""Qwen Image 2.1 engine. Native Windows/CUDA; local weights; one GPU worker.

The optional NF4 loader quantizes linear layers in memory, leaving the source
checkpoint intact. An optional image-specific DWM profile projects text-encoder
writer activations at runtime and likewise never changes weights.
"""
from __future__ import annotations

import base64
import gc
import io
import json
import math
import os
import queue
import secrets
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator
from . import qwen_prompt_enhance
from .image_dwm import configure_from_environment, request_scale
from .image_library import ImageLibrary, TYPES

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_DIR = Path(os.environ.get("FVL_IMAGE_MODEL_DIR", "models/Qwen-Image-2.1")).resolve()
OUT_DIR = Path(os.environ.get("FVL_IMAGE_OUTPUT_DIR", "outputs/images")).resolve()
QUANTIZATION = os.environ.get("FVL_IMAGE_QUANTIZATION", "nf4")
DWM_DEFAULT_SCALE = float(os.environ.get("FVL_IMAGE_DWM_DEFAULT_SCALE", "0.0"))
if not 0.0 <= DWM_DEFAULT_SCALE <= 2.0:
    raise ValueError("FVL_IMAGE_DWM_DEFAULT_SCALE must be between 0 and 2")
DWM_STATUS = {"enabled": False, "mode": "clean", "weights_modified": False}
MAX_BODY = 36_000_000
CAPABILITIES = ["text_to_image", "image_edit", "multi_reference", "transparent_png",
                "transparent_edit", "subject_extraction", "mask_edit", "annotation_edit"]


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=12000)
    mode: Literal["auto", "generate", "edit", "transparent", "extract", "masked", "annotate"] = "auto"
    images_b64: list[str] = Field(default_factory=list, max_length=10)
    mask_b64: str | None = None
    preserve_unmasked: bool = True
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1024, ge=256, le=4096)
    num_inference_steps: int = Field(default=40, ge=1, le=80)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    negative_prompt: str = Field(default="", max_length=4000)
    true_cfg_scale: float = Field(default=1.0, ge=1.0, le=10.0)
    use_kv_cache: bool = True
    reference_resolution: Literal[256, 512, 1024] = 512
    n: int = Field(default=1, ge=1, le=4)
    enhance_prompt: bool = False
    auto_aspect_ratio: bool = False
    dwm_scale: float = Field(default=DWM_DEFAULT_SCALE, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_options(self):
        if self.mode == "auto":
            self.mode = "edit" if self.images_b64 else "generate"
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("Enter an image description or editing instruction")
        if self.width % 32 or self.height % 32:
            raise ValueError("Width and height must be multiples of 32")
        if self.width * self.height > 4_500_000:
            raise ValueError("Maximum output area is 4.5 megapixels")
        if self.auto_aspect_ratio and not self.enhance_prompt:
            raise ValueError("Automatic aspect ratio requires automatic prompt enhancement")
        if self.mode in {"edit", "extract", "masked", "annotate"} and not self.images_b64:
            raise ValueError("This mode needs a reference image")
        if self.mode == "generate" and self.images_b64:
            raise ValueError("Choose Edit or another reference mode to use uploaded images")
        if self.mode == "masked" and not self.mask_b64:
            raise ValueError("Paint or upload a mask for a masked edit")
        if self.mask_b64 and self.mode != "masked":
            raise ValueError("Masks are accepted only in masked-edit mode")
        if self.mask_b64 and len(self.images_b64) > 9:
            raise ValueError("Use up to nine references plus one mask")
        if self.negative_prompt.strip() and self.true_cfg_scale <= 1:
            raise ValueError("A negative prompt requires guidance above 1")
        if self.true_cfg_scale > 1 and not self.negative_prompt.strip():
            raise ValueError("Guidance above 1 requires a negative prompt")
        if sum(map(len, self.images_b64)) + len(self.mask_b64 or "") > 32_000_000:
            raise ValueError("Combined reference images exceed 32 MB; resize them first")
        return self


def decode_image(encoded: str) -> Image.Image:
    if len(encoded) > 12_000_000:
        raise ValueError("An uploaded image exceeds 12 MB")
    try:
        raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Use PNG, JPEG, or WebP")
            if source.width * source.height > 16_000_000:
                raise ValueError("Reference images must be at most 16 megapixels")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid reference image: " + str(exc)) from exc


def prepare_inputs(spec: ImageRequest):
    images = [decode_image(value) for value in spec.images_b64]
    if sum(image.width * image.height for image in images) > 40_000_000:
        raise ValueError("Combined references exceed 40 megapixels")
    prompt = spec.prompt
    mask = None
    if spec.mode == "transparent":
        prompt = "This is an RGBA image with transparency. " + prompt + ". The image has alpha channel and the background is transparent."
    elif spec.mode == "extract":
        prompt = "Extract the requested subject from the first image as an RGBA image with a transparent background. Preserve the subject's appearance. " + prompt
    elif spec.mode == "masked":
        mask = decode_image(spec.mask_b64).convert("L")
        if mask.size != images[0].size:
            raise ValueError("The mask must have the same dimensions as the first reference")
        if mask.getextrema()[1] == 0:
            raise ValueError("The mask is empty; paint the area to edit in white")
        images.append(mask.convert("RGBA"))
        prompt = ("Edit the first image in the region marked white in the last image (the black-and-white edit mask). "
                  "Preserve the other regions and remove any mask markings from the result. " + prompt)
    elif spec.mode == "annotate":
        prompt = "Follow the editing annotations on the first reference image. Remove the annotation markings in the finished image. " + prompt
    return prompt, images, mask


def load_pipeline():
    global DWM_STATUS
    import torch
    from diffusers import BitsAndBytesConfig as DiffusionQuantization
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import BitsAndBytesConfig as EncoderQuantization
    from transformers import Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This engine requires an available CUDA GPU")
    if not (MODEL_DIR / "model_index.json").is_file():
        raise RuntimeError("Local Qwen Image 2.1 checkpoint is incomplete")
    common = dict(local_files_only=True, low_cpu_mem_usage=True)
    if QUANTIZATION == "nf4":
        quant = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        print("[qwen-image] Loading NF4 text encoder", flush=True)
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR / "text_encoder"), quantization_config=EncoderQuantization(**quant),
            dtype=torch.bfloat16, device_map={"": 0}, **common)
        DWM_STATUS = configure_from_environment(encoder)
        print(f"[qwen-image] DWM profile: {DWM_STATUS}", flush=True)
        print("[qwen-image] Loading NF4 image transformer", flush=True)
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            str(MODEL_DIR / "transformer"), quantization_config=DiffusionQuantization(**quant),
            torch_dtype=torch.bfloat16, device_map={"": 0}, **common)
        pipe = QwenImage21Pipeline.from_pretrained(
            str(MODEL_DIR), text_encoder=encoder, transformer=transformer,
            torch_dtype=torch.bfloat16, **common)
        pipe.to("cuda")
    elif QUANTIZATION == "bf16-offload":
        pipe = QwenImage21Pipeline.from_pretrained(str(MODEL_DIR), torch_dtype=torch.bfloat16, **common)
        DWM_STATUS = configure_from_environment(pipe.text_encoder)
        print(f"[qwen-image] DWM profile: {DWM_STATUS}", flush=True)
        pipe.enable_model_cpu_offload()
    else:
        raise ValueError("FVL_IMAGE_QUANTIZATION must be nf4 or bf16-offload")
    # Small default tiles leave visible seams in RGBA output. Decode standard
    # canvases in one pass, with overlapping 1K tiles for larger canvases.
    pipe.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=768, tile_sample_stride_width=768)
    pipe.set_progress_bar_config(disable=True)
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


class ImageEngine:
    def __init__(self, loader=load_pipeline, output_dir=OUT_DIR):
        self.loader = loader
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.library = ImageLibrary(self.output)
        self.pipe = None
        self.loading = False
        self.error = None
        self.jobs = {}
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=4)
        self.active = None
        self.initialized = False

    @property
    def ready(self):
        return self.initialized or self.pipe is not None

    def start(self):
        self.loading = True
        threading.Thread(target=self._worker, name="qwen-image-worker", daemon=True).start()

    def _worker(self):
        try:
            self.pipe = self.loader()
            self.initialized = True
            print("[qwen-image] Ready", flush=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            self.loading = False
        if self.pipe is None:
            return
        while True:
            job_id, spec, prepared = self.pending.get()
            try:
                if self.get(job_id).get("kind") == "enhance":
                    self._enhance(job_id, spec, prepared)
                else:
                    self._render(job_id, spec, prepared)
            except Exception as exc:
                traceback.print_exc()
                self.update(job_id, status="error", stage="Stopped", error=f"{type(exc).__name__}: {exc}")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                self.active = None
                self.pending.task_done()

    def update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(changes, updated_at=time.time())
            return dict(self.jobs[job_id])

    def submit(self, spec, kind="image"):
        if not self.ready:
            raise HTTPException(503, self.error or "The model is still loading")
        prepared = prepare_inputs(spec)
        if kind == "enhance" or spec.enhance_prompt:
            task, _ = qwen_prompt_enhance.checkpoint_for(MODEL_DIR, prepared[1])
            if not qwen_prompt_enhance.available(MODEL_DIR)[task]:
                raise HTTPException(503, "The official prompt enhancer is still downloading")
        job_id = "img_" + secrets.token_hex(12)
        seed = spec.seed if spec.seed is not None else secrets.randbelow(2**31)
        spec = spec.model_copy(update={"seed": seed})
        with self.lock:
            if self.pending.full():
                raise HTTPException(429, "The image queue is full; wait for a render to finish")
            while len(self.jobs) >= 100:
                old = next((k for k, v in self.jobs.items() if v["status"] in {"done", "error", "cancelled"}), None)
                if old is None:
                    break
                self.jobs.pop(old)
            self.jobs[job_id] = dict(id=job_id, kind=kind, status="queued", stage="Queued", seed=seed,
                                     created_at=time.time(), outputs=[], cancel=False,
                                     completed_steps=0, total_steps=spec.num_inference_steps * spec.n)
            self.pending.put_nowait((job_id, spec, prepared))
        return self.get(job_id)

    def _enhance(self, job_id, spec, prepared, finish=True):
        import torch
        if self.get(job_id)["cancel"]:
            self.update(job_id, status="cancelled", stage="Cancelled")
            return
        self.active = job_id
        started = time.monotonic()
        self.update(job_id, status="running", stage="Making room for the prompt enhancer")
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        result = qwen_prompt_enhance.enhance(MODEL_DIR, prepared[0], prepared[1], spec.seed,
            lambda **changes: self.update(job_id, **changes), lambda: self.get(job_id)["cancel"])
        if result is None:
            self.update(job_id, status="cancelled", stage="Cancelled")
            return
        self.update(job_id, status="done" if finish else "running", stage="Prompt ready to review",
                    enhancement=result, original_prompt=spec.prompt,
                    enhancement_seconds=round(time.monotonic() - started, 2))
        if finish:
            self.update(job_id, seconds=round(time.monotonic() - started, 2))
        return result

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "Unknown image job")
            return dict(self.jobs[job_id])

    def _render(self, job_id, spec, prepared):
        import torch
        if self.get(job_id)["cancel"]:
            self.update(job_id, status="cancelled", stage="Cancelled")
            return
        self.active = job_id
        started = time.monotonic()
        self.update(job_id, status="running", stage="Encoding prompt and references")
        original_prompt = spec.prompt
        if spec.enhance_prompt:
            result = self._enhance(job_id, spec, prepared, finish=False)
            if result is None:
                return
            changes = dict(prompt=result["prompt"], enhance_prompt=False, auto_aspect_ratio=False)
            if spec.auto_aspect_ratio:
                ratios = {"1:1": 1, "4:3": 4/3, "3:4": 3/4, "3:2": 3/2,
                          "2:3": 2/3, "16:9": 16/9, "9:16": 9/16}
                ratio = ratios.get(result.get("wh_ratio"))
                if ratio is None and result.get("ratio_follow", "").startswith("<image"):
                    try:
                        ref = prepared[1][int(result["ratio_follow"][6:-1]) - 1]
                        ratio = min(ratios.values(), key=lambda r: abs(r - ref.width / ref.height))
                    except (ValueError, IndexError):
                        pass
                if ratio:
                    area = spec.width * spec.height
                    changes.update(width=max(256, min(4096, int(math.sqrt(area*ratio)/32)*32)),
                                   height=max(256, min(4096, int(math.sqrt(area/ratio)/32)*32)))
            spec = ImageRequest.model_validate(spec.model_dump() | changes)
            prepared = prepare_inputs(spec)
        if self.pipe is None:
            self.update(job_id, stage="Loading image model after prompt enhancement")
            self.pipe = self.loader()
            if self.get(job_id)["cancel"]:
                self.update(job_id, status="cancelled", stage="Cancelled")
                return
        prompt, images, mask = prepared
        outputs = []
        for index in range(spec.n):
            if self.get(job_id)["cancel"]:
                self.update(job_id, status="cancelled", stage="Cancelled", outputs=outputs)
                return
            def on_step(pipe, step, timestep, callback_kwargs):
                if self.get(job_id)["cancel"]:
                    pipe._interrupt = True
                self.update(job_id, stage=f"Image {index + 1} of {spec.n} · step {step + 1} of {spec.num_inference_steps}",
                            completed_steps=index * spec.num_inference_steps + step + 1)
                return callback_kwargs

            seed = (spec.seed + index) % (2**63)
            with request_scale(spec.dwm_scale):
                result = self.pipe(prompt=prompt, image=images or None, width=spec.width, height=spec.height,
                                   output_resolution=spec.reference_resolution,
                                   num_inference_steps=spec.num_inference_steps,
                                   true_cfg_scale=spec.true_cfg_scale, negative_prompt=spec.negative_prompt or None,
                                   use_kv_cache=spec.use_kv_cache, generator=torch.Generator("cuda").manual_seed(seed),
                                   callback_on_step_end=on_step).images[0].convert("RGBA")
            if self.get(job_id)["cancel"]:
                self.update(job_id, status="cancelled", stage="Cancelled", outputs=outputs)
                return
            if mask is not None and spec.preserve_unmasked:
                original = images[0].resize(result.size, Image.Resampling.LANCZOS)
                result = Image.composite(result, original, mask.resize(result.size, Image.Resampling.NEAREST))
            name = f"qwen21_{time.strftime('%Y%m%d_%H%M%S')}_{seed}_{secrets.token_hex(4)}.png"
            metadata = dict(prompt=original_prompt, effective_prompt=prompt, seed=seed, mode=spec.mode,
                            prompt_enhancement=self.get(job_id).get("enhancement"),
                            engine_id="qwen-image-2.1", engine_label="Qwen Image 2.1", created_at=time.time(),
                            width=result.width, height=result.height, format="RGBA PNG", reference_count=len(spec.images_b64),
                            num_inference_steps=spec.num_inference_steps, quantization=QUANTIZATION,
                            true_cfg_scale=spec.true_cfg_scale, use_kv_cache=spec.use_kv_cache,
                            dwm_scale=spec.dwm_scale, dwm_profile=DWM_STATUS,
                            reference_resolution=spec.reference_resolution,
                            preserve_unmasked=spec.preserve_unmasked if mask is not None else None)
            saved = self.library.publish(result, name, metadata)
            outputs.append(dict(id=saved["id"], name=name, file_url="/files/" + name,
                                seed=seed, width=result.width, height=result.height))
            self.update(job_id, outputs=list(outputs))
        self.update(job_id, status="done", stage="Complete", seconds=round(time.monotonic() - started, 2), outputs=outputs)


engine = ImageEngine()


@asynccontextmanager
async def lifespan(app):
    engine.start()
    yield


app = FastAPI(title="Frosty Studio — Qwen Image 2.1", lifespan=lifespan)


@app.middleware("http")
async def body_limit(request: Request, call_next):
    if request.method == "POST":
        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY:
                return JSONResponse({"detail": "Request exceeds 36 MB"}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
    return await call_next(request)


@app.get("/health")
def health():
    return dict(status="error" if engine.error else "loading" if engine.loading else "ok",
                ready=engine.ready, loading=engine.loading, error=engine.error,
                model=MODEL_ID, quantization=QUANTIZATION, capabilities=CAPABILITIES,
                dwm=DWM_STATUS, dwm_default_scale=DWM_DEFAULT_SCALE,
                prompt_enhancement=qwen_prompt_enhance.available(MODEL_DIR),
                active_job=engine.active, queued=engine.pending.qsize())


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "Qwen"}]}


@app.post("/jobs", status_code=202)
def submit(spec: ImageRequest):
    try:
        return engine.submit(spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/enhance", status_code=202)
def enhance_prompt(spec: ImageRequest):
    try:
        return engine.submit(spec, kind="enhance")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/jobs/{job_id}")
def job(job_id: str):
    return engine.get(job_id)


@app.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    current = engine.get(job_id)
    if current["status"] not in {"done", "error", "cancelled"}:
        return engine.update(job_id, cancel=True, stage="Cancelling after the current operation")
    return current


class LibrarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[str] = Field(min_length=1, max_length=100)


@app.get("/gallery")
def gallery():
    return engine.library.gallery()


@app.get("/gallery/trash")
def gallery_trash():
    return engine.library.trash_items()


def library_action(selection, action):
    results = []
    for identifier in dict.fromkeys(selection.ids):
        try:
            entry = action(identifier)
            results.append(dict(id=identifier, ok=True, trash_id=entry["id"],
                                asset_id=entry["asset_id"], name=entry["name"], state=entry["state"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append(dict(id=identifier, ok=False, error=str(exc)))
    return dict(ok=all(item["ok"] for item in results), results=results)


@app.post("/gallery/trash")
def trash_images(selection: LibrarySelection):
    return library_action(selection, engine.library.move_to_trash)


@app.post("/gallery/restore")
def restore_images(selection: LibrarySelection):
    return library_action(selection, engine.library.restore)


@app.get("/files/{name}")
def output_file(name: str):
    try:
        candidate = engine.library.file(name)
    except (OSError, ValueError):
        raise HTTPException(404)
    return FileResponse(candidate, media_type=TYPES[candidate.suffix.lower()])
