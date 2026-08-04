"""Serve the Frosty VL (Qwen3-VL persona plugin) FL2VA and Ref2VA pipelines over FastAPI.

The customer's base pipeline stays pristine on disk. Persona-steering and
golden-safety directions are projected onto the Qwen3-VL text encoder in
memory at startup.
"""
import base64
import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_refusal import project_rows
from pipeline_load import load_pipeline

# diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
# integration - not a MiniMax product dependency.
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.utils.export_utils import encode_video
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Frosty VL server")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("FVL_MODEL", "/models/base-pipeline")
OUT_DIR = Path(os.environ.get("FVL_SERVE_OUT", os.path.expanduser("~/frosty-vl/outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_FILE = os.environ.get("FVL_REFUSAL_DIR", os.path.join(HERE, "data", "refusal_dir_L50.npy"))
ABL_START = int(os.environ.get("FVL_ABL_START", "40"))
ABL_END = int(os.environ.get("FVL_ABL_END", "49"))
ABL_LAM = float(os.environ.get("FVL_ABL_LAM", "3.0"))
SAFETY_DIR = os.environ.get("FVL_SAFETY_DIR", os.path.join(HERE, "data", "safety_dir_L50.npy"))
SAFETY_START = int(os.environ.get("FVL_SAFETY_START", "40"))
SAFETY_END = int(os.environ.get("FVL_SAFETY_END", "49"))
SAFETY_LAM = float(os.environ.get("FVL_SAFETY_LAM", "3.0"))
SEED = int(os.environ.get("FVL_SEED", "42"))

FL_DEVICE = os.environ.get("FVL_DEVICE", "cuda:0")
REF_DEVICE = os.environ.get("FVL_REF_DEVICE", "cuda:1")
MODEL_ID = os.environ.get("FVL_MODEL_ID", "frosty-vl")
PUBLIC_BASE = os.environ.get("FVL_PUBLIC_BASE", "http://127.0.0.1:8899").rstrip("/")

pipe = None
pipe_ref = None
ref_loading = False
ref_load_error = None
started = time.time()
_REF_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


def _validate_artifacts():
    required = [Path(MODEL) / "modular_model_index.json"]
    if ABL_LAM != 0:
        required.append(Path(DIR_FILE))
    if SAFETY_LAM != 0:
        required.append(Path(SAFETY_DIR))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Frosty VL deployment artifacts: %s" % ", ".join(missing))


def _apply_edits(target_pipe, label: str):
    layers = target_pipe.components["text_encoder"].model.language_model.layers

    if ABL_LAM != 0:
        direction = torch.from_numpy(np.load(DIR_FILE))
        for layer_id in range(ABL_START, ABL_END + 1):
            output = layers[layer_id].self_attn.o_proj
            with torch.no_grad():
                output.weight.copy_(project_rows(output.weight.detach(), direction, ABL_LAM))
        print(
            "[server] %s refusal-removal applied to text_encoder layers %d..%d lam=%.2f"
            % (label, ABL_START, ABL_END, ABL_LAM),
            flush=True,
        )

    if SAFETY_LAM != 0:
        safety_direction = torch.from_numpy(np.load(SAFETY_DIR))
        for layer_id in range(SAFETY_START, SAFETY_END + 1):
            output = layers[layer_id].self_attn.o_proj
            with torch.no_grad():
                output.weight.copy_(project_rows(output.weight.detach(), safety_direction, SAFETY_LAM))
        print(
            "[server] %s golden-safety floor applied to text_encoder layers %d..%d lam=%.2f"
            % (label, SAFETY_START, SAFETY_END, SAFETY_LAM),
            flush=True,
        )

    if ABL_LAM == 0 and SAFETY_LAM == 0:
        print("[server] %s serving clean checkpoint (no load-time projection)" % label, flush=True)


def _load_fl2va():
    global pipe
    print("[server] loading FL2VA from %s on %s" % (MODEL, FL_DEVICE), flush=True)
    pipe = load_pipeline(model_dir=MODEL, device=FL_DEVICE, workflow="fl2va")
    _apply_edits(pipe, "FL2VA")
    print("[server] FL2VA ready in %.1fs" % (time.time() - started), flush=True)


def _load_ref2va():
    global pipe_ref, ref_loading, ref_load_error
    with _REF_LOCK:
        if pipe_ref is not None:
            return pipe_ref
        if ref_load_error is not None:
            raise RuntimeError(ref_load_error)
        ref_loading = True
        try:
            print("[server] loading Ref2VA from %s on %s" % (MODEL, REF_DEVICE), flush=True)
            loaded = load_pipeline(model_dir=MODEL, device=REF_DEVICE, workflow="ref2va")
            _apply_edits(loaded, "Ref2VA")
            pipe_ref = loaded
            print("[server] Ref2VA identity pipeline ready", flush=True)
        except Exception as exc:
            ref_load_error = "%s: %s" % (type(exc).__name__, exc)
            print("[server] Ref2VA load failed: %s" % ref_load_error, flush=True)
            raise
        finally:
            ref_loading = False
    return pipe_ref


_validate_artifacts()
_load_fl2va()
threading.Thread(target=_load_ref2va, name="frosty-vl-ref2va-loader", daemon=True).start()


class GenRequest(BaseModel):
    prompt: str
    seed: int | None = None
    image_b64: str | None = None
    identity_lock: bool = False
    duration_seconds: float | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    seed: int | None = None


def _decode_image(image_b64: str | None):
    if not image_b64:
        return None
    encoded = image_b64.split(",", 1)[-1].strip()
    if len(encoded) > 28_000_000:
        raise ValueError("image is too large (20 MB maximum after encoding)")
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = PILImage.open(io.BytesIO(raw))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid uploaded image: %s" % exc) from exc
    if image.width < 64 or image.height < 64:
        raise ValueError("image must be at least 64x64 pixels")
    if image.width * image.height > 40_000_000:
        raise ValueError("image dimensions are too large")
    return image


def _num_frames(duration_seconds: float | None) -> int:
    """Map seconds to Frosty VL's required 17*n+5 frame count."""
    seconds = 5.0 if duration_seconds is None else float(duration_seconds)
    if not 5.0 <= seconds <= 15.0:
        raise ValueError("duration must be between 5 and 15 seconds")
    requested = int(round(seconds * 24))
    n = max(7, (requested - 5 + 16) // 17)
    return min(345, 17 * n + 5)


def _identity_prompt(prompt: str) -> str:
    return (
        "subject_definitions:\n"
        "<Subject 1> is the same person shown in <Picture 1>. Preserve their facial identity, face shape, "
        "skin details, hair, and distinguishing features.\n\n"
        "summary:\n[identity reference] Create the requested scene with <Subject 1> as the main person, keeping "
        "the same recognizable identity throughout.\n\n"
        "retention_analysis:\n<Subject 1>: fully_preserved - identity and facial characteristics remain "
        "consistent while pose, action, wardrobe, setting, and camera may follow the scene direction.\n\n"
        "detailed_description:\n" + prompt
    )


def _state_value(state, key):
    getters = (
        lambda: getattr(state, key),
        lambda: state.get(key),
        lambda: state[key],
        lambda: state.get_intermediate(key),
        lambda: state.values.get(key),
    )
    for getter in getters:
        try:
            value = getter()
            if value is not None:
                return value
        except Exception:
            pass
    return None


def _normalise_video(video):
    while isinstance(video, (list, tuple)) and len(video) == 1 and not isinstance(video[0], PILImage.Image):
        video = video[0]
    if not isinstance(video, (list, tuple, np.ndarray)) and not torch.is_tensor(video):
        return [video]
    return video


def _generate(
    prompt: str,
    seed: int,
    image=None,
    identity_lock: bool = False,
    duration_seconds: float | None = None,
) -> Path:
    output = OUT_DIR / ("frostyvl_%s.mp4" % uuid.uuid4().hex[:12])
    generated_at = time.time()
    frames = _num_frames(duration_seconds)

    # One lock protects both experimental modular pipelines from overlapping
    # requests. It also gives deterministic queue ordering to the UI.
    with _GEN_LOCK:
        if identity_lock:
            if image is None:
                raise ValueError("same-person lock requires an uploaded image")
            target_pipe = _load_ref2va()
            state = target_pipe(
                prompt=_identity_prompt(prompt),
                # diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL
            # text-encoder integration - not a MiniMax product dependency.
            references=[MiniMaxH3Reference(image=image)],
                num_frames=frames,
                generator=torch.Generator().manual_seed(seed),
            )
            mode = "identity-reference"
        else:
            kwargs = {
                "prompt": prompt,
                "num_frames": frames,
                "generator": torch.Generator().manual_seed(seed),
            }
            if image is not None:
                kwargs["image"] = image
            state = pipe(**kwargs)
            mode = "image-to-video" if image is not None else "text-to-video"

    print("[server] %s '%s' -> gen %.0fs" % (mode, prompt[:40], time.time() - generated_at), flush=True)
    video = _state_value(state, "videos")
    if video is None:
        raise RuntimeError("no videos in pipeline state")
    video = _normalise_video(video)

    audio = _state_value(state, "audio")
    while isinstance(audio, (list, tuple)) and len(audio) == 1:
        audio = audio[0]
    sampling_rate = _state_value(state, "sampling_rate")

    print(
        "[server] video=%s audio=%s sr=%s" % (type(video).__name__, type(audio).__name__, sampling_rate),
        flush=True,
    )
    encode_video(
        video=video,
        fps=24,
        output_path=str(output),
        audio=audio,
        audio_sample_rate=sampling_rate,
    )
    print("[server] wrote %s (%d bytes)" % (output, output.stat().st_size), flush=True)
    return output


def _extract_prompt(messages) -> str:
    parts = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
    return "\n".join(parts).strip() or "A cinematic establishing shot"


def _file_url(name: str) -> str:
    return "%s/files/%s" % (PUBLIC_BASE, name)


@app.get("/")
def info():
    safety_on = SAFETY_LAM != 0
    return {
        "service": "Frosty VL",
        "model": MODEL,
        "model_id": MODEL_ID,
        "refusal_removal": ABL_LAM != 0,
        "ablate_layers": [ABL_START, ABL_END],
        "ablate_lam": ABL_LAM,
        "golden_safety_floor": safety_on,
        "safety_layers": [SAFETY_START, SAFETY_END],
        "safety_lam": SAFETY_LAM,
        "ready": pipe is not None,
        "reference_ready": pipe_ref is not None,
        "reference_loading": ref_loading,
        "reference_error": ref_load_error,
        "uptime_s": round(time.time() - started),
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if pipe is not None else "loading",
        "ready": pipe is not None,
        "reference_ready": pipe_ref is not None,
        "reference_loading": ref_loading,
        "reference_error": ref_load_error,
    }


@app.get("/files/{name}")
def files(name: str):
    path = OUT_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/generate")
def generate_get(prompt: str, json: int = 0, seed: int | None = None):
    output = _generate(prompt, seed if seed is not None else SEED)
    if json:
        return {"prompt": prompt, "file": str(output), "size": output.stat().st_size}
    return FileResponse(output, media_type="video/mp4", filename=output.name)


@app.post("/generate")
def generate_post(request: GenRequest):
    image = _decode_image(request.image_b64)
    output = _generate(
        request.prompt,
        request.seed if request.seed is not None else SEED,
        image=image,
        identity_lock=request.identity_lock,
        duration_seconds=request.duration_seconds,
    )
    mode = "identity-reference" if request.identity_lock else ("image-to-video" if image else "text-to-video")
    return {
        "prompt": request.prompt,
        "file": str(output),
        "size": output.stat().st_size,
        "mode": mode,
        "duration_seconds": _num_frames(request.duration_seconds) / 24.0,
    }


@app.get("/v1/models")
def models_list():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(started),
                "owned_by": "blackfrost",
                "permission": [],
                "root": MODEL_ID,
                "parent": None,
            }
        ],
    }


@app.get("/v1/models/{model_id}")
def models_get(model_id: str):
    if model_id != MODEL_ID:
        return {"object": "model", "id": model_id, "error": "unknown model"}
    return {"id": MODEL_ID, "object": "model", "created": int(started), "owned_by": "blackfrost"}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    prompt = _extract_prompt(request.messages)
    output = _generate(prompt, request.seed if request.seed is not None else SEED)
    url = _file_url(output.name)
    return {
        "id": "chatcmpl-%s" % uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Generated video+audio for prompt: %s (file: %s)" % (prompt, url),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "video_url": url,
        "video_file": str(output),
    }


@app.post("/v1/responses")
def responses(request: ChatRequest):
    prompt = _extract_prompt(request.messages)
    output = _generate(prompt, request.seed if request.seed is not None else SEED)
    url = _file_url(output.name)
    return {
        "id": "resp_%s" % uuid.uuid4().hex[:12],
        "object": "response",
        "status": "completed",
        "model": request.model or MODEL_ID,
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Video: %s" % url}]}
        ],
        "video_url": url,
        "video_file": str(output),
    }
