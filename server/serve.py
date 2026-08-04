# -*- coding: utf-8 -*-
"""Host the Frosty VL (Qwen3-VL persona plugin) pipeline as an HTTP service.

Purpose: let a human A/B-test the restricted build live. The pipeline is loaded
once at startup and the established row-projection weight change (project_rows,
on text-encoder self_attn.o_proj) is applied IN MEMORY (never written to disk):

  refusal : W <- W - lam_r * outer(v_r, v_r @ W)   [v_r = refusal dir, L40-49, lam 3.0]
  safety  : W <- W - lam_s * outer(v_s, v_s @ W)   [v_s = golden-safety dir (under-18 floor)]

  refusal only (FVL_SAFETY_LAM=0)         -> model id  frosty-vl
  restricted, DEFAULT (FVL_SAFETY_LAM!=0) -> model id  frosty-vl-restricted

The restricted build bakes in the ONLY floor we keep: no exploitation of minors
/ no sexual images of children / anything under 18. It is refusal+safety applied
to the clean weights (the on-disk base pipeline stays byte-identical; both edits
are load-time only).

Endpoints
  GET /                        -> service info + current config (refusal + safety)
  GET /health                  -> 200 once model loaded
  GET /generate?prompt=...     -> video+audio mp4 (Content-Type video/mp4)
  GET /generate?prompt=...&json=1 -> JSON metadata
  POST /generate  {"prompt": ...} -> mp4 metadata

Generation mirrors the proven baseline recipe (49-step, seed 42, the diffusers
call + encode_video mux with audio).
"""
import base64
import io
import os
import sys
import threading
import time
import uuid
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_load import load_pipeline
from apply_refusal import project_rows
# diffusers Apache-2.0 reference pipeline classes for the Qwen3-VL text-encoder
# integration - external library API, not branding.
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.utils.export_utils import encode_video
from fastapi import FastAPI
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
# Golden safety floor (under-18 / no exploitation). ON by default.
SAFETY_DIR = os.environ.get("FVL_SAFETY_DIR", os.path.join(HERE, "data", "safety_dir_L50.npy"))
SAFETY_START = int(os.environ.get("FVL_SAFETY_START", "40"))
SAFETY_END = int(os.environ.get("FVL_SAFETY_END", "49"))
SAFETY_LAM = float(os.environ.get("FVL_SAFETY_LAM", "3.0"))
SEED = int(os.environ.get("FVL_SEED", "42"))
FL_DEVICE = os.environ.get("FVL_DEVICE", "cuda:0")
REF_DEVICE = os.environ.get("FVL_REF_DEVICE", "cuda:1")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be a boolean value" % name)


# The per-workflow overrides allow staged activation because FL2VA and Ref2VA
# run on GPUs with different free-memory margins. The global flag is a
# convenience only when both devices have already been qualified.
COMPILE_REPEATED = _env_bool("FVL_COMPILE_REPEATED")
COMPILE_FL2VA = _env_bool("FVL_COMPILE_FL2VA", COMPILE_REPEATED)
COMPILE_REF2VA = _env_bool("FVL_COMPILE_REF2VA", COMPILE_REPEATED)

pipe = None
pipe_ref = None
ref_loading = False
ref_load_error = None
_REF_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()
started = time.time()


def _apply_edits(target_pipe, label: str):
    te = target_pipe.components["text_encoder"]
    layers = te.model.language_model.layers
    if ABL_LAM != 0:
        v = torch.from_numpy(np.load(DIR_FILE))
        for lid in range(ABL_START, ABL_END + 1):
            op = layers[lid].self_attn.o_proj
            with torch.no_grad():
                op.weight.copy_(project_rows(op.weight.detach(), v, ABL_LAM))
        print("[server] %s refusal-removal applied to text_encoder layers %d..%d lam=%.2f"
              % (label, ABL_START, ABL_END, ABL_LAM), flush=True)
    if SAFETY_LAM != 0:
        vs = torch.from_numpy(np.load(SAFETY_DIR))
        for lid in range(SAFETY_START, SAFETY_END + 1):
            op = layers[lid].self_attn.o_proj
            with torch.no_grad():
                op.weight.copy_(project_rows(op.weight.detach(), vs, SAFETY_LAM))
        print("[server] %s golden-safety floor applied to text_encoder layers %d..%d lam=%.2f"
              % (label, SAFETY_START, SAFETY_END, SAFETY_LAM), flush=True)
    if ABL_LAM == 0 and SAFETY_LAM == 0:
        print("[server] %s serving clean baseline (no edit)" % label, flush=True)


def _load():
    global pipe
    print("[server] loading FL2VA pipeline from %s on %s" % (MODEL, FL_DEVICE), flush=True)
    pipe = load_pipeline(
        model_dir=MODEL,
        device=FL_DEVICE,
        workflow="fl2va",
        compile_repeated=COMPILE_FL2VA,
    )
    _apply_edits(pipe, "FL2VA")
    print("[server] FL2VA ready in %.1fs" % (time.time() - started), flush=True)


def _load_ref():
    """Load the identity/reference pipeline once, on the otherwise idle GPU 1."""
    global pipe_ref, ref_loading, ref_load_error
    with _REF_LOCK:
        if pipe_ref is not None:
            return pipe_ref
        if ref_load_error is not None:
            raise RuntimeError(ref_load_error)
        ref_loading = True
        try:
            print("[server] loading Ref2VA pipeline from %s on %s" % (MODEL, REF_DEVICE), flush=True)
            loaded = load_pipeline(
                model_dir=MODEL,
                device=REF_DEVICE,
                workflow="ref2va",
                compile_repeated=COMPILE_REF2VA,
            )
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


_load()
# Loading a second full stack would not fit beside FL2VA on GPU 0. GPU 1 is idle
# and large enough, so warm Ref2VA there without delaying basic service.
threading.Thread(target=_load_ref, name="frosty-vl-ref2va-loader", daemon=True).start()


class GenRequest(BaseModel):
    prompt: str
    seed: int | None = None
    image_b64: str | None = None
    identity_lock: bool = False
    duration_seconds: float | None = None


def _decode_image(image_b64: str | None):
    if not image_b64:
        return None
    encoded = image_b64.split(",", 1)[-1].strip()
    if len(encoded) > 28_000_000:
        raise ValueError("image is too large (20 MB maximum after encoding)")
    try:
        data = base64.b64decode(encoded, validate=True)
        image = PILImage.open(io.BytesIO(data))
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
    """Return a valid 17*n+5 frame count between 5s and its real ceiling."""
    seconds = 5.0 if duration_seconds is None else float(duration_seconds)
    if not 5.0 <= seconds <= 15.0:
        raise ValueError("duration must be between 5 and 15 seconds")
    requested = int(round(seconds * 24))
    n = max(7, (requested - 5 + 16) // 17)
    return min(345, 17 * n + 5)  # 362 frames would exceed the 15s validation ceiling


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


def _generate(prompt: str, seed: int, image=None, identity_lock: bool = False,
              duration_seconds: float | None = None) -> Path:
    out = OUT_DIR / ("frostyvl_%s.mp4" % uuid.uuid4().hex[:12])
    tg = time.time()
    frames = _num_frames(duration_seconds)
    with _GEN_LOCK:
        if identity_lock:
            if image is None:
                raise ValueError("same-person lock requires an uploaded image")
            target_pipe = _load_ref()
            state = target_pipe(
                prompt=_identity_prompt(prompt),
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
    print("[server] %s '%s' -> gen %.0fs" % (mode, prompt[:40], time.time() - tg), flush=True)

    def fetch(st, key):
        for getter in (lambda: getattr(st, key), lambda: st.get(key), lambda: st[key],
                       lambda: st.get_intermediate(key), lambda: st.values.get(key)):
            try:
                val = getter()
                if val is not None:
                    return val
            except Exception:
                pass
        return None

    # Robust output extraction. The pipeline "videos" may be a list of PIL
    # Image frames (pass the WHOLE list to encode_video — it stacks them), a
    # single PIL Image (wrap), or an ndarray/tensor of all frames (pass through).
    # Only unwrap a single-element container if it wraps exactly ONE non-PIL
    # object; never collapse a genuine PIL frame list into a bare Image.
    def norm_video(v):
        while (isinstance(v, (list, tuple)) and len(v) == 1
               and not isinstance(v[0], PILImage.Image)):
            v = v[0]
        if not isinstance(v, (list, tuple)) and not isinstance(v, np.ndarray) \
           and not torch.is_tensor(v):
            return [v]  # bare PIL Image (or scalar) -> single-frame list
        return v

    videos = fetch(state, "videos")
    if videos is None:
        raise RuntimeError("no videos in pipeline state")
    video = norm_video(videos)
    audio = fetch(state, "audio")
    sr = fetch(state, "sampling_rate")
    a = audio
    while isinstance(a, (list, tuple)) and len(a) == 1:
        a = a[0]
    aud = a
    print("[server] video=%s audio=%s sr=%s" % (type(video).__name__, type(aud).__name__, sr), flush=True)
    encode_video(video=video, fps=24, output_path=str(out), audio=aud, audio_sample_rate=sr)
    print("[server] wrote %s (%d bytes)" % (out, os.path.getsize(out)), flush=True)
    return out


MODEL_ID = os.environ.get("FVL_MODEL_ID",
                          "frosty-vl-restricted" if SAFETY_LAM != 0 else "frosty-vl")
BASE = os.environ.get("FVL_PUBLIC_BASE", "http://127.0.0.1:8899")


def _file_url(name: str) -> str:
    return "%s/files/%s" % (BASE.rstrip("/"), name)


def _extract_prompt(messages) -> str:
    """Concatenate role=user contents (accept str or list-of-content-parts)."""
    parts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if m.get("role") == "system":
            continue
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
    return "\n".join(parts).strip() or "A cinematic establishing shot"


@app.get("/v1/models")
def models_list():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": int(started),
         "owned_by": "blackfrost", "permission": [], "root": MODEL_ID, "parent": None}
    ]}


@app.get("/v1/models/{mid}")
def models_get(mid: str):
    if mid != MODEL_ID:
        return {"object": "model", "id": mid, "error": "unknown model"}
    return {"id": MODEL_ID, "object": "model", "created": int(started), "owned_by": "blackfrost"}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    seed: int | None = None


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = _extract_prompt(req.messages)
    seed = req.seed if req.seed is not None else SEED
    out = _generate(prompt, seed)
    url = _file_url(out.name)
    # OpenAI-compatible message: short text + attachable video URL in metadata
    content = "Generated video+audio for prompt: %s (file: %s)" % (prompt, url)
    return {
        "id": "chatcmpl-%s" % uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "video_url": url,
        "video_file": str(out),
    }


@app.post("/v1/responses")
def responses(req: ChatRequest):
    prompt = _extract_prompt(req.messages)
    seed = req.seed if req.seed is not None else SEED
    out = _generate(prompt, seed)
    url = _file_url(out.name)
    return {"id": "resp_%s" % uuid.uuid4().hex[:12], "object": "response",
            "status": "completed", "model": req.model or MODEL_ID,
            "output": [{"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": \
                              "Video: %s" % url}]}],
            "video_url": url, "video_file": str(out)}


@app.get("/files/{name}")
def files(name: str):
    p = OUT_DIR / Path(name).name
    if not p.exists():
        return {"error": "not found"}, 404
    return FileResponse(p, media_type="video/mp4", filename=name)


@app.get("/")
def info():
    safety_on = SAFETY_LAM != 0
    return {"service": "Frosty VL", "model": MODEL, "model_id": MODEL_ID,
            "refusal_removal": ABL_LAM != 0,
            "ablate_layers": [ABL_START, ABL_END], "ablate_lam": ABL_LAM,
            "golden_safety_floor": safety_on,
            "safety_layers": [SAFETY_START, SAFETY_END], "safety_lam": SAFETY_LAM,
            "ready": pipe is not None, "reference_ready": pipe_ref is not None,
            "reference_loading": ref_loading, "reference_error": ref_load_error,
            "regional_compile": {"fl2va": COMPILE_FL2VA, "ref2va": COMPILE_REF2VA},
            "uptime_s": round(time.time() - started)}


@app.get("/health")
def health():
    return {"status": "ok" if pipe is not None else "loading", "ready": pipe is not None,
            "reference_ready": pipe_ref is not None, "reference_loading": ref_loading,
            "reference_error": ref_load_error,
            "regional_compile": {"fl2va": COMPILE_FL2VA, "ref2va": COMPILE_REF2VA}}


@app.get("/generate")
def generate_get(prompt: str, json: int = 0, seed: int | None = None):
    out = _generate(prompt, seed if seed is not None else SEED)
    if json:
        return {"prompt": prompt, "file": str(out), "size": os.path.getsize(out),
                "config": {"ablate_layers": [ABL_START, ABL_END], "ablate_lam": ABL_LAM}}
    return FileResponse(out, media_type="video/mp4", filename=out.name)


@app.post("/generate")
def generate_post(req: GenRequest):
    image = _decode_image(req.image_b64)
    out = _generate(req.prompt, req.seed if req.seed is not None else SEED,
                    image=image, identity_lock=req.identity_lock,
                    duration_seconds=req.duration_seconds)
    return {"prompt": req.prompt, "file": str(out), "size": os.path.getsize(out),
            "mode": "identity-reference" if req.identity_lock else
                    ("image-to-video" if image is not None else "text-to-video"),
            "duration_seconds": _num_frames(req.duration_seconds) / 24.0}
