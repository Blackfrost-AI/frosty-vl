"""Serve Wan 2.2 TI2V through the small, engine-neutral Studio API.

This process is intentionally separate from the currently loaded renderer. It
owns one GPU, writes into the shared gallery directory, and implements the same
``POST /generate`` and ``GET /files/{name}`` contract used by the Studio proxy.
"""
import base64
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image as PILImage, ImageOps
from pydantic import BaseModel


MODEL = os.environ.get("FROST_MODEL", "/models/Wan2.2-TI2V-5B-Diffusers")
MODEL_ID = os.environ.get("FROST_MODEL_ID", "wan22-ti2v-5b")
DEVICE = os.environ.get("FROST_DEVICE", "cuda:2")
OUT_DIR = Path(os.environ.get("FROST_OUTPUT_DIR", "/outputs")).expanduser().resolve()
OFFLINE = os.environ.get("FROST_OFFLINE", "1").lower() not in {"0", "false", "no"}
FPS = 24

RESOLUTIONS = {
    "480p": {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640),
    },
    "720p": {
        "16:9": (1280, 704),
        "9:16": (704, 1280),
        "1:1": (960, 960),
    },
}
DEFAULT_NEGATIVE = (
    "overexposed, static, blurry details, subtitles, watermark, worst quality, low quality, "
    "JPEG artifacts, deformed, disfigured, malformed limbs, fused fingers, cluttered background"
)

app = FastAPI(title="Frosty engine: Wan 2.2 TI2V 5B")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipe_t2v = None
pipe_i2v = None
load_error = None
loading = False
loaded_at = None
started_at = time.time()
_LOAD_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


class GenRequest(BaseModel):
    prompt: str
    seed: int | None = None
    image_b64: str | None = None
    identity_lock: bool = False
    duration_seconds: float | None = 3.0
    negative_prompt: str | None = None
    resolution: str | None = "480p"
    aspect_ratio: str | None = "16:9"
    num_inference_steps: int | None = 30
    guidance_scale: float | None = 5.0


def _load():
    global pipe_t2v, pipe_i2v, load_error, loading, loaded_at
    with _LOAD_LOCK:
        if pipe_t2v is not None or loading:
            return
        loading = True
        try:
            model_path = Path(MODEL)
            if OFFLINE and not (model_path / "model_index.json").is_file():
                raise FileNotFoundError("Wan model is not downloaded at %s" % MODEL)
            if OFFLINE:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print("[wan] loading %s on %s" % (MODEL, DEVICE), flush=True)
            vae = AutoencoderKLWan.from_pretrained(
                MODEL,
                subfolder="vae",
                dtype=torch.float32,
                local_files_only=OFFLINE,
            )
            loaded_t2v = WanPipeline.from_pretrained(
                MODEL,
                vae=vae,
                dtype=torch.bfloat16,
                local_files_only=OFFLINE,
                low_cpu_mem_usage=True,
            )
            i2v_scheduler = FlowMatchEulerDiscreteScheduler.from_config(loaded_t2v.scheduler.config)
            loaded_i2v = WanImageToVideoPipeline(
                tokenizer=loaded_t2v.tokenizer,
                text_encoder=loaded_t2v.text_encoder,
                vae=loaded_t2v.vae,
                scheduler=i2v_scheduler,
                image_processor=None,
                image_encoder=None,
                transformer=loaded_t2v.transformer,
                transformer_2=loaded_t2v.transformer_2,
                boundary_ratio=loaded_t2v.config.boundary_ratio,
                expand_timesteps=loaded_t2v.config.expand_timesteps,
            )
            loaded_t2v.to(DEVICE)
            loaded_i2v.to(DEVICE)
            loaded_t2v.set_progress_bar_config(desc="Wan 2.2")
            loaded_i2v.set_progress_bar_config(desc="Wan 2.2 I2V")
            pipe_t2v = loaded_t2v
            pipe_i2v = loaded_i2v
            loaded_at = time.time()
            print("[wan] ready in %.1fs" % (loaded_at - started_at), flush=True)
        except Exception as exc:
            load_error = "%s: %s" % (type(exc).__name__, exc)
            print("[wan] load failed: %s" % load_error, flush=True)
        finally:
            loading = False


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, name="wan-loader", daemon=True).start()


def _decode_image(image_b64):
    if not image_b64:
        return None
    encoded = image_b64.split(",", 1)[-1].strip()
    if len(encoded) > 28_000_000:
        raise ValueError("image is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = PILImage.open(io.BytesIO(raw))
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid uploaded image: %s" % exc) from exc


def _dimensions(resolution, aspect_ratio):
    resolution = resolution if resolution in RESOLUTIONS else "480p"
    aspect_ratio = aspect_ratio if aspect_ratio in RESOLUTIONS[resolution] else "16:9"
    return RESOLUTIONS[resolution][aspect_ratio], resolution, aspect_ratio


def _num_frames(seconds):
    seconds = 3.0 if seconds is None else float(seconds)
    if not 1.0 <= seconds <= 5.1:
        raise ValueError("Wan duration must be between 1 and 5 seconds")
    requested = max(25, int(round(seconds * FPS)))
    # Wan requires 4*k+1 frames.
    return max(25, 4 * round((requested - 1) / 4) + 1)


def _fit_image(image, width, height):
    return ImageOps.fit(image, (width, height), method=PILImage.Resampling.LANCZOS)


def _write_metadata(path, metadata):
    sidecar = path.with_name(path.name + ".json")
    temporary = sidecar.with_name(sidecar.name + ".tmp-" + uuid.uuid4().hex[:8])
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def _generate(request):
    if pipe_t2v is None:
        if load_error:
            raise RuntimeError(load_error)
        raise RuntimeError("Wan 2.2 is still loading")
    prompt = request.prompt.strip()
    if not prompt:
        raise ValueError("prompt is required")
    if len(prompt) > 6000:
        raise ValueError("prompt is too long")
    if request.identity_lock:
        raise ValueError("Wan 2.2 supports image guidance, not Same Face lock")
    (width, height), resolution, aspect_ratio = _dimensions(request.resolution, request.aspect_ratio)
    frames = _num_frames(request.duration_seconds)
    steps = max(4, min(60, int(request.num_inference_steps or 30)))
    guidance = max(1.0, min(10.0, float(request.guidance_scale or 5.0)))
    seed = int(request.seed if request.seed is not None else int.from_bytes(os.urandom(4), "big") % (2**31))
    image = _decode_image(request.image_b64)
    output = OUT_DIR / ("wan22_%s.mp4" % uuid.uuid4().hex[:12])
    kwargs = {
        "prompt": prompt,
        "negative_prompt": (request.negative_prompt or DEFAULT_NEGATIVE).strip(),
        "height": height,
        "width": width,
        "num_frames": frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": torch.Generator(device=DEVICE).manual_seed(seed),
    }
    mode = "text-to-video"
    selected_pipe = pipe_t2v
    if image is not None:
        kwargs["image"] = _fit_image(image, width, height)
        selected_pipe = pipe_i2v
        mode = "image-to-video"
    began = time.time()
    with _GEN_LOCK, torch.inference_mode():
        result = selected_pipe(**kwargs).frames[0]
        export_to_video(result, str(output), fps=FPS, quality=8, macro_block_size=16)
    elapsed = round(time.time() - began, 1)
    metadata = {
        "created_at": time.time(),
        "prompt": prompt,
        "seed": seed,
        "mode": mode,
        "engine_id": MODEL_ID,
        "engine_label": "Wan 2.2 TI2V 5B",
        "duration_seconds": frames / FPS,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    _write_metadata(output, metadata)
    print("[wan] %s %dx%d %df/%d steps in %.1fs -> %s" % (
        mode, width, height, frames, steps, elapsed, output.name
    ), flush=True)
    return output, metadata, elapsed


@app.get("/")
def info():
    return {
        "service": "Frosty universal video engine",
        "engine_id": MODEL_ID,
        "model": MODEL,
        "ready": pipe_t2v is not None,
        "loading": loading,
        "error": load_error,
        "device": DEVICE,
        "capabilities": ["text_to_video", "image_to_video"],
    }


@app.get("/health")
def health():
    return {
        "status": "error" if load_error else ("ok" if pipe_t2v is not None else "loading"),
        "ready": pipe_t2v is not None,
        "loading": loading,
        "error": load_error,
        "engine_id": MODEL_ID,
        "uptime_s": round(time.time() - started_at),
    }


@app.post("/generate")
def generate(request: GenRequest):
    if pipe_t2v is None:
        raise HTTPException(status_code=503, detail=load_error or "Wan 2.2 is still loading")
    try:
        output, metadata, elapsed = _generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "file": str(output),
        "size": output.stat().st_size,
        "mode": metadata["mode"],
        "engine_id": MODEL_ID,
        "seed": metadata["seed"],
        "duration_seconds": metadata["duration_seconds"],
        "seconds": elapsed,
        "settings": {
            "resolution": metadata["resolution"],
            "aspect_ratio": metadata["aspect_ratio"],
            "num_inference_steps": metadata["num_inference_steps"],
            "guidance_scale": metadata["guidance_scale"],
        },
    }


@app.get("/files/{name}")
def files(name: str):
    clean = Path(name).name
    path = (OUT_DIR / clean).resolve()
    if path.parent != OUT_DIR or not path.is_file() or path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
