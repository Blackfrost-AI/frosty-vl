"""Loopback-only Studio adapter for an already-loaded, pristine base pipeline.

This module never loads or edits model weights.  ``start_server`` receives the
native Diffusers pipeline object from the clean baseline session and only adds
the small HTTP/file/muxing contract expected by Frosty Studio.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import uvicorn
from diffusers.utils.export_utils import encode_video
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image as PILImage, ImageOps
from pydantic import BaseModel


FPS = 24


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
        raise ValueError("image is too large")
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
    seconds = 5.0 if duration_seconds is None else float(duration_seconds)
    if not 5.0 <= seconds <= 15.0:
        raise ValueError("duration must be between 5 and 15 seconds")
    requested = int(round(seconds * FPS))
    n = max(7, (requested - 5 + 16) // 17)
    return min(345, 17 * n + 5)


def _fetch(state, key):
    for getter in (
        lambda: getattr(state, key),
        lambda: state.get(key),
        lambda: state[key],
        lambda: state.get_intermediate(key),
        lambda: state.values.get(key),
    ):
        try:
            value = getter()
            if value is not None:
                return value
        except Exception:
            pass
    return None


def _normalize_video(video):
    while (
        isinstance(video, (list, tuple))
        and len(video) == 1
        and not isinstance(video[0], PILImage.Image)
    ):
        video = video[0]
    if not isinstance(video, (list, tuple, np.ndarray)) and not torch.is_tensor(video):
        return [video]
    return video


def start_server(
    pipe,
    host: str = "127.0.0.1",
    port: int = 8902,
    output_dir: str = os.path.expanduser("~/frosty-vl/outputs"),
):
    """Start a daemon HTTP adapter around ``pipe`` and return server/thread."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generation_lock = threading.Lock()
    started_at = time.time()
    app = FastAPI(title="Frosty VL pristine baseline")

    @app.get("/")
    def info():
        return {
            "service": "Frosty VL pristine baseline",
            "engine_id": "pristine-base",
            "ready": True,
            "weight_edits": False,
            "audio_sampling_rate": pipe.audio_sampling_rate,
            "device": str(next(pipe.transformer.parameters()).device),
            "capabilities": ["text_to_video", "image_to_video", "native_audio"],
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "ready": True,
            "engine_id": "pristine-base",
            "weight_edits": False,
            "audio_sampling_rate": pipe.audio_sampling_rate,
            "uptime_s": round(time.time() - started_at),
        }

    @app.post("/generate")
    def generate(request: GenRequest):
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        if len(prompt) > 12_000:
            raise HTTPException(status_code=400, detail="prompt is too long")
        if request.identity_lock:
            raise HTTPException(status_code=400, detail="pristine FL2VA has no Same Face lock")
        try:
            image = _decode_image(request.image_b64)
            frames = _num_frames(request.duration_seconds)
            seed = int(request.seed if request.seed is not None else 42)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        output = out_dir / ("frostyvl_clean_%s.mp4" % uuid.uuid4().hex[:12])
        kwargs = {
            "prompt": prompt,
            "num_frames": frames,
            "generator": torch.Generator().manual_seed(seed),
        }
        if image is not None:
            kwargs["image"] = image

        began = time.time()
        with generation_lock, torch.inference_mode():
            state = pipe(**kwargs)
            video = _fetch(state, "videos")
            audio = _fetch(state, "audio")
            sample_rate = _fetch(state, "sampling_rate") or pipe.audio_sampling_rate
            if video is None:
                raise HTTPException(status_code=500, detail="pipeline returned no video")
            while isinstance(audio, (list, tuple)) and len(audio) == 1:
                audio = audio[0]
            encode_video(
                video=_normalize_video(video),
                fps=FPS,
                output_path=str(output),
                audio=audio,
                audio_sample_rate=sample_rate,
            )

        return {
            "file": str(output),
            "size": output.stat().st_size,
            "mode": "image-to-video" if image is not None else "text-to-video",
            "engine_id": "pristine-base",
            "seed": seed,
            "duration_seconds": frames / FPS,
            "audio_sampling_rate": sample_rate,
            "seconds": round(time.time() - began, 1),
        }

    @app.get("/files/{name}")
    def files(name: str):
        clean = Path(name).name
        path = (out_dir / clean).resolve()
        if path.parent != out_dir or not path.is_file() or path.suffix.lower() != ".mp4":
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="frosty-vl-clean-http", daemon=True)
    thread.start()
    return server, thread
