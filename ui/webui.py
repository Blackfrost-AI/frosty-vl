#!/usr/bin/env python3
"""Frosty Studio — browser UI and proxy for a local video render engine.

Runs beside the render server and supports text/image-to-video, optional
same-person references, multi-scene movie jobs, and a persistent output gallery.

    python3 ui/webui.py
    # http://localhost:8890

Env: FVL_BASE, FVL_UI_HOST, FVL_UI_PORT, FVL_GEN_TIMEOUT, FVL_FFMPEG,
FVL_GALLERY_DIR, FVL_ENGINES_FILE, FVL_GALLERY_LIMIT.
"""
import base64
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.video_library import VideoLibrary
from server.image_library import ImageLibrary
from ui.video_jobs import VideoJobs, QueueFull, Cancelled

FVL_BASE = (os.environ.get("FVL_BASE") or "http://127.0.0.1:8899").rstrip("/")
UI_HOST = os.environ.get("FVL_UI_HOST") or "127.0.0.1"
UI_PORT = int(os.environ.get("FVL_UI_PORT") or "8890")
GEN_TIMEOUT = int(os.environ.get("FVL_GEN_TIMEOUT") or "2400")
FFMPEG = os.environ.get("FVL_FFMPEG") or "/usr/bin/ffmpeg"
_GALLERY_VALUE = (
    os.environ.get("FVL_GALLERY_DIR")
    or os.environ.get("FVL_SERVE_OUT")
)
GALLERY_DIR = Path(_GALLERY_VALUE).expanduser().resolve() if _GALLERY_VALUE else None
GALLERY_LIMIT = max(1, min(500, int(os.environ.get("FVL_GALLERY_LIMIT", "200"))))
if GALLERY_DIR:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def _load_engines():
    config_path = os.environ.get("FVL_ENGINES_FILE")
    data = None
    if config_path:
        try:
            data = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
        except Exception as exc:
            print("engine config could not be read: %s" % exc, flush=True)
    if not isinstance(data, dict):
        data = {
            "default": "current",
            "engines": [{
                "id": "current",
                "label": "Current engine",
                "base": FVL_BASE,
                "description": "The currently configured video engine.",
                "capabilities": ["text_to_video", "image_to_video", "same_face", "native_audio", "scene_lab"],
                "controls": {"durations": [5, 8, 10, 12, 14]},
            }],
        }
    engines = []
    seen = set()
    for raw in data.get("engines") or []:
        if not isinstance(raw, dict):
            continue
        engine_id = str(raw.get("id") or "").strip()
        base = str(raw.get("base") or "").rstrip("/")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", engine_id) or not base.startswith(("http://", "https://")):
            continue
        if engine_id in seen:
            continue
        seen.add(engine_id)
        engines.append({
            "id": engine_id,
            "label": str(raw.get("label") or engine_id),
            "base": base,
            "description": str(raw.get("description") or "Video generation engine"),
            "capabilities": [str(value) for value in (raw.get("capabilities") or [])],
            "controls": raw.get("controls") if isinstance(raw.get("controls"), dict) else {},
        })
    if not engines:
        raise RuntimeError("engine configuration contains no usable engines")
    default_id = str(data.get("default") or "")
    if default_id not in seen:
        default_id = engines[0]["id"]
    return default_id, engines


DEFAULT_ENGINE_ID, ENGINES = _load_engines()
ENGINES_BY_ID = {engine["id"]: engine for engine in ENGINES}

_CACHE = {}  # token -> (content_type, bytes)
_CACHE_LOCK = threading.Lock()
_VIDEO_LIBRARY = None
_VIDEO_LIBRARY_LOCK = threading.RLock()


def _video_library():
    global _VIDEO_LIBRARY
    if not GALLERY_DIR:
        raise ValueError("Video library storage is not configured; set FVL_GALLERY_DIR")
    with _VIDEO_LIBRARY_LOCK:
        if _VIDEO_LIBRARY is None or _VIDEO_LIBRARY.root != GALLERY_DIR.resolve():
            _VIDEO_LIBRARY = VideoLibrary(GALLERY_DIR)
        return _VIDEO_LIBRARY

MEDIA_EXT = (".mp4", ".webm", ".gif", ".mov", ".m4v", ".png", ".jpg", ".jpeg", ".webp")
CT_BY_EXT = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif",
    ".mov": "video/quicktime", ".m4v": "video/mp4",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
}


def _gallery_path(name):
    """Resolve a gallery basename without allowing path traversal or symlinks."""
    if not GALLERY_DIR:
        return None
    clean = str(name)
    if not clean or any(c in clean for c in '/\\:\x00') or clean.rstrip(' .') != clean:
        return None
    if not clean.lower().endswith(MEDIA_EXT):
        return None
    candidate = GALLERY_DIR / clean
    if candidate.is_symlink() or candidate.resolve().parent != GALLERY_DIR.resolve():
        return None
    return candidate


def _metadata_path(media_path):
    return media_path.with_name(media_path.name + ".json")


def _record_gallery_metadata(name, metadata):
    path = _gallery_path(name)
    if not path or not path.is_file():
        return
    if path.suffix.lower() in VideoLibrary.media_types:
        _video_library().metadata(name, metadata or {})
        return
    allowed = {
        key: value for key, value in (metadata or {}).items()
        if key in {
            "prompt", "seed", "mode", "duration_seconds", "scenes", "engine_id", "engine_label",
            "resolution", "aspect_ratio", "num_inference_steps", "guidance_scale",
            "prompt_format",
        }
        and isinstance(value, (str, int, float, bool, list, type(None)))
    }
    allowed.setdefault("created_at", path.stat().st_mtime)
    sidecar = _metadata_path(path)
    temporary = sidecar.with_name(sidecar.name + ".tmp-" + secrets.token_hex(4))
    temporary.write_text(json.dumps(allowed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def _store_gallery_file(source, prefix, metadata):
    """Copy an assembled/local clip into persistent storage, if configured."""
    if not GALLERY_DIR:
        return None
    name = "%s_%s_%s.mp4" % (prefix, time.strftime("%Y%m%d_%H%M%S"), secrets.token_hex(4))
    return _video_library().publish_file(source, name, metadata)["file_url"]


def _gallery_items():
    if not GALLERY_DIR:
        return None
    # Keep the original mixed-media endpoint compatible. New video endpoints use
    # VideoLibrary directly; image-default requests proxy to the image engine.
    items = _video_library().gallery()["items"]
    for item in ImageLibrary(GALLERY_DIR).gallery()["items"]:
        item["file_url"] = "/api/gallery/file?name=" + quote(item["name"])
        items.append(item)
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[:GALLERY_LIMIT]


def _content_type_for(name):
    for ext, content_type in CT_BY_EXT.items():
        if name.lower().endswith(ext):
            return content_type
    return "application/octet-stream"


def _normalize_prompt(request):
    """Return canonical prompt text and its declared format.

    The engine's native API accepts prompt text. Structured mode therefore
    validates a JSON object and serializes it as readable JSON text rather than
    pretending its fields are separate generation parameters.
    """
    prompt_format = str(request.get("prompt_format") or "text").strip().lower()
    if prompt_format not in ("text", "json"):
        raise ValueError("prompt format must be text or json")
    raw_prompt = request.get("prompt")
    if prompt_format == "json":
        if isinstance(raw_prompt, dict):
            structured = raw_prompt
        elif isinstance(raw_prompt, str):
            if not raw_prompt.strip():
                raise ValueError("empty JSON prompt")
            try:
                structured = json.loads(raw_prompt)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON prompt at line %d, column %d: %s"
                    % (exc.lineno, exc.colno, exc.msg)
                ) from exc
        else:
            raise ValueError("JSON prompt must be an object")
        if not isinstance(structured, dict):
            raise ValueError("JSON prompt must be an object")
        if not structured:
            raise ValueError("JSON prompt cannot be empty")
        prompt = json.dumps(structured, ensure_ascii=False, indent=2)
    else:
        if not isinstance(raw_prompt, str):
            raise ValueError("text prompt must be a string")
        prompt = raw_prompt.strip()
        if not prompt:
            raise ValueError("empty prompt")
    if len(prompt) > 12_000:
        raise ValueError("prompt is too long (12,000 characters maximum)")
    return prompt, prompt_format


def _cache_media(data, content_type="video/mp4", prefix="media"):
    token = "%s_%s" % (prefix, secrets.token_urlsafe(9))
    with _CACHE_LOCK:
        _CACHE[token] = (content_type, data)
        while len(_CACHE) > 32:
            _CACHE.pop(next(iter(_CACHE)))
    return "/api/file?cache=" + token


def find_file_ref(obj):
    """Return (kind, value), with kind in name/url/b64, from a render response."""
    found = []

    def walk(value, key=None):
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, child_key)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str):
            found.append((key, value))

    walk(obj)
    for key, value in found:
        low = value.lower()
        if "/files/" in low or any(low.endswith(ext) for ext in MEDIA_EXT):
            return ("url" if "/files/" in low or low.startswith("http") else "name", value)
    b64_re = re.compile(r"^[A-Za-z0-9+/=\s]+$")
    for key, value in found:
        if len(value) > 2000 and b64_re.match(value):
            return "b64", value
    for key, value in found:
        if (key or "").lower() in ("name", "file", "filename", "output", "path", "url", "video"):
            return "name", value
    return None, None


def _engine(engine_id):
    return ENGINES_BY_ID.get(str(engine_id or "")) or ENGINES_BY_ID[DEFAULT_ENGINE_ID]


def _image_engine():
    default = _engine(None)
    if "text_to_image" in default["capabilities"]:
        return default
    return next((item for item in ENGINES if "text_to_image" in item["capabilities"]), None)


def _workspaces():
    return {"image": any("text_to_image" in item["capabilities"] for item in ENGINES),
            "video": any("text_to_video" in item["capabilities"] for item in ENGINES)}


def _public_engine(engine, health=None):
    result = {key: engine[key] for key in ("id", "label", "description", "capabilities", "controls")}
    result["health"] = health or {"ready": False, "status": "unreachable"}
    return result


def http_json(method, path, payload=None, timeout=30, base=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        (base or FVL_BASE) + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def http_bytes(path, timeout=120, base=None):
    with urllib.request.urlopen((base or FVL_BASE) + path, timeout=timeout) as response:
        return response.status, response.headers.get("Content-Type", "application/octet-stream"), response.read()


def _remote_generate(payload, engine=None):
    engine = engine or _engine(None)
    try:
        _, raw = http_json("POST", "/generate", payload, timeout=GEN_TIMEOUT, base=engine["base"])
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:1200].decode(errors="replace")
        raise RuntimeError("%s returned HTTP %s: %s" % (engine["label"], exc.code, detail)) from exc
    except Exception as exc:
        raise RuntimeError("%s call failed: %s" % (engine["label"], exc)) from exc
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"_raw_text": raw[:1000].decode(errors="replace")}
    kind, value = find_file_ref(parsed)
    if kind is None:
        raise RuntimeError("render finished but no media reference was returned: %s" % _prune(parsed))
    return parsed, kind, value


def _remote_name(value):
    if "/files/" in value:
        return value.split("/files/")[-1].split("?")[0].split("/")[-1]
    return value.split("?")[0].split("/")[-1]


def _result_url(kind, value, prefix="gen", metadata=None, engine=None):
    if kind == "b64":
        if GALLERY_DIR:
            with tempfile.NamedTemporaryFile(prefix=prefix + "_", suffix=".mp4", delete=False) as temporary:
                temporary.write(base64.b64decode(value, validate=False))
                temporary_path = Path(temporary.name)
            try:
                saved_url = _store_gallery_file(temporary_path, prefix, metadata or {})
            finally:
                temporary_path.unlink(missing_ok=True)
            if saved_url:
                return saved_url
        return _cache_media(base64.b64decode(value, validate=False), "video/mp4", prefix)
    name = _remote_name(value)
    local_path = _gallery_path(name)
    if local_path and local_path.is_file():
        _record_gallery_metadata(name, metadata or {})
        return "/api/gallery/file?name=" + quote(name)
    fallback = "/api/file?name=" + quote(name)
    if engine:
        fallback += "&engine=" + quote(engine["id"])
    return fallback


def _result_bytes(kind, value, engine=None):
    if kind == "b64":
        return "video/mp4", base64.b64decode(value, validate=False)
    name = _remote_name(value)
    _, content_type, data = http_bytes(
        "/files/" + quote(name), timeout=180, base=(engine or _engine(None))["base"]
    )
    if not content_type or content_type == "application/octet-stream":
        content_type = _content_type_for(name)
    return content_type, data


def _prune(obj):
    if isinstance(obj, dict):
        return {key: _prune(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_prune(value) for value in obj]
    if isinstance(obj, str) and len(obj) > 400:
        return "<str len=%d>" % len(obj)
    return obj


def _extract_last_frame(clip_path):
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-sseof", "-0.12", "-i", str(clip_path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False,
    )
    if result.returncode or not result.stdout:
        raise RuntimeError("could not extract continuity frame: " + result.stderr.decode(errors="replace")[-500:])
    return "data:image/png;base64," + base64.b64encode(result.stdout).decode()


def _concat_clips(paths, output_path):
    list_path = output_path.with_suffix(".txt")
    list_path.write_text("".join("file '%s'\n" % str(path).replace("'", "'\\''") for path in paths))
    common = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", str(list_path)]
    copy_run = subprocess.run(
        common + ["-c", "copy", "-movflags", "+faststart", str(output_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False,
    )
    if copy_run.returncode:
        encode_run = subprocess.run(
            common + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
                      "-movflags", "+faststart", str(output_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900, check=False,
        )
        if encode_run.returncode:
            raise RuntimeError("scene assembly failed: " + encode_run.stderr.decode(errors="replace")[-1200:])
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError("scene assembly produced an empty movie")


def _run_video(request, report, cancelled):
    engine = _engine(request["engine_id"])
    base_seed = request["seed"] if request["seed"] is not None else secrets.randbelow(2**31)
    request["seed"] = base_seed
    report(engine_label=engine["label"], base_seed=base_seed, used_seed=base_seed)
    metadata = {key: request.get(key) for key in VideoLibrary.metadata_fields}
    metadata.update(engine_id=engine["id"], engine_label=engine["label"])
    if request["kind"] == "clip":
        if cancelled():
            raise Cancelled()
        report(stage="Rendering clip")
        payload = {k: v for k, v in request.items() if k in VIDEO_RENDER_FIELDS}
        parsed, kind, value = _remote_generate(payload, engine)
        metadata.update(mode=parsed.get("mode", "image to video" if request.get("image_b64") else "text to video"))
        url = _result_url(kind, value, metadata=metadata, engine=engine)
        return dict(file_url=url, mode=metadata["mode"], used_seed=base_seed,
                    duration_seconds=parsed.get("duration_seconds") or request["duration_seconds"])

    scenes = request["scenes"]
    original_image = request.get("image_b64")
    identity_lock = bool(request.get("identity_lock"))
    continuity = bool(request.get("continuity", True)) and not identity_lock
    scene_urls = []
    with tempfile.TemporaryDirectory(prefix="frosty-vl-scenes-") as temp_dir:
        clips = []
        current_image = original_image
        for index, scene in enumerate(scenes):
            if cancelled():
                raise Cancelled()
            report(stage="Rendering scene %d of %d" % (index+1, len(scenes)), current_scene=index+1)
            prompt = ("Overall story and continuity: " + request.get("story", "") + "\n\n"
                      "Scene %d of %d: %s\nPreserve established character, production design, lighting and visual language."
                      % (index+1, len(scenes), scene["prompt"]))
            payload = {k: v for k, v in request.items() if k in VIDEO_RENDER_FIELDS}
            payload.update(prompt=prompt, seed=(base_seed+index) % (2**31), duration_seconds=scene["duration"])
            if identity_lock:
                payload.update(image_b64=original_image, identity_lock=True)
            elif current_image:
                payload["image_b64"] = current_image
            _, kind, value = _remote_generate(payload, engine)
            scene_metadata = dict(metadata, prompt=scene["prompt"], seed=payload["seed"],
                                  mode="scene %d of %d" % (index+1, len(scenes)), duration_seconds=scene["duration"])
            scene_urls.append(_result_url(kind, value, "scene", scene_metadata, engine))
            report(completed_scenes=index+1, scene_urls=list(scene_urls))
            if cancelled():
                raise Cancelled()
            _, data = _result_bytes(kind, value, engine)
            clip = Path(temp_dir) / ("scene_%02d.mp4" % index)
            clip.write_bytes(data)
            clips.append(clip)
            if continuity and index < len(scenes)-1:
                report(stage="Linking scene %d to scene %d" % (index+1, index+2))
                current_image = _extract_last_frame(clip)
        if cancelled():
            raise Cancelled()
        report(stage="Assembling movie", current_scene=None)
        final = Path(temp_dir) / "movie.mp4"
        _concat_clips(clips, final)
        metadata.update(prompt=request.get("story") or "%d-scene movie" % len(scenes), mode="scene lab",
                        scenes=scenes, duration_seconds=sum(x["duration"] for x in scenes))
        url = _store_gallery_file(final, "scene_lab", metadata) or _cache_media(final.read_bytes(), "video/mp4", "movie")
        return dict(file_url=url, scene_urls=scene_urls, mode="scene lab", used_seed=base_seed)


VIDEO_RENDER_FIELDS = {"prompt", "seed", "image_b64", "identity_lock", "duration_seconds", "negative_prompt",
                       "resolution", "aspect_ratio", "num_inference_steps", "guidance_scale"}
VIDEO_JOBS = VideoJobs(_run_video)


def _validate_video_request(raw, kind=None):
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object")
    allowed = VIDEO_RENDER_FIELDS | {"engine_id", "kind", "prompt_format", "story", "scenes", "continuity", "request_id"}
    if set(raw)-allowed:
        raise ValueError("Unknown video settings: " + ", ".join(sorted(set(raw)-allowed)))
    request = dict(raw)
    videos = [e for e in ENGINES if "text_to_video" in e["capabilities"]]
    if not videos:
        raise ValueError("No video engine is configured")
    selected = request.get("engine_id") or next((e["id"] for e in videos if e["id"] == DEFAULT_ENGINE_ID), videos[0]["id"])
    engine = ENGINES_BY_ID.get(selected)
    if not engine or "text_to_video" not in engine["capabilities"]:
        raise ValueError("Select a configured video engine")
    request["engine_id"] = selected
    request["kind"] = kind or request.get("kind", "clip")
    if request["kind"] not in {"clip", "scenes"}:
        raise ValueError("Video kind must be clip or scenes")
    caps, controls = set(engine["capabilities"]), engine.get("controls", {})
    for flag in ("identity_lock", "continuity"):
        if flag in request and not isinstance(request[flag], bool):
            raise ValueError(flag + " must be true or false")
    image = request.get("image_b64")
    if image:
        if "image_to_video" not in caps:
            raise ValueError("This engine does not support image input")
        if not isinstance(image, str) or len(image)>12_000_000:
            raise ValueError("Image input must be base64 and at most 12 MB")
        try:
            image_bytes = base64.b64decode(image.split(',', 1)[-1], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid base64 image") from exc
        if not image_bytes.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'RIFF')):
            raise ValueError("Use a PNG, JPEG or WebP image")
    if request.get("identity_lock") and (not image or "same_face" not in caps):
        raise ValueError("Same Face needs a reference and an engine with same_face capability")
    if request.get("request_id") is not None and (not isinstance(request["request_id"], str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", request["request_id"])):
        raise ValueError("Invalid request_id")
    seed = request.get("seed")
    if seed in (None, "", -1, "-1"):
        request["seed"] = None  # Randomize after idempotency matching, not during validation.
    else:
        if isinstance(seed, bool) or not str(seed).isdigit() or not 0<=int(seed)<2**31:
            raise ValueError("Seed must be an integer from 0 to 2147483647")
        request["seed"] = int(seed)
    durations = [float(x) for x in controls.get("durations", [])]
    def duration(value):
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid duration") from exc
        if not math.isfinite(number) or not 1<=number<=15 or (durations and number not in durations):
            raise ValueError("Choose a duration supported by the selected engine")
        return number
    if request["kind"] == "scenes":
        if "scene_lab" not in caps:
            raise ValueError("This engine does not support Scene Lab")
        scenes = request.get("scenes")
        if not isinstance(scenes, list) or not 2<=len(scenes)<=8:
            raise ValueError("Add between 2 and 8 scenes")
        clean = []
        for scene in scenes:
            if not isinstance(scene, dict):
                raise ValueError("Each scene must be an object")
            prompt, _ = _normalize_prompt({"prompt": scene.get("prompt")})
            clean.append({"prompt": prompt, "duration": duration(scene.get("duration", durations[0] if durations else 5))})
        request["scenes"] = clean
        story = request.get("story", "")
        if not isinstance(story, str) or len(story)>12000:
            raise ValueError("Story must be text, at most 12000 characters")
        request["story"] = story.strip()
        if request.get("continuity", True) and not request.get("identity_lock") and "image_to_video" not in caps:
            raise ValueError("Match-cut continuity needs an image-to-video engine")
    else:
        if request.get("scenes"):
            raise ValueError("Use kind=scenes for Scene Lab")
        request["prompt"], request["prompt_format"] = _normalize_prompt(request)
        request["duration_seconds"] = duration(request.get("duration_seconds", durations[0] if durations else 5))
    for key, config_key in (("resolution", "resolutions"), ("aspect_ratio", "aspect_ratios")):
        if key in request and request[key] not in controls.get(config_key, []):
            raise ValueError(key + " is not supported by the selected engine")
    for key, config_key in (("num_inference_steps", "steps"), ("guidance_scale", "guidance")):
        if key in request:
            bounds = controls.get(config_key)
            value = request[key]
            if not bounds or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + " is not supported or invalid")
            if not bounds["min"]<=value<=bounds["max"] or (key=="num_inference_steps" and int(value)!=value):
                raise ValueError(key + " is outside engine limits")
    if request.get("negative_prompt"):
        if not controls.get("negative_prompt") or not isinstance(request["negative_prompt"], str) or len(request["negative_prompt"])>4000:
            raise ValueError("Negative prompt is unsupported or too long")
    return request


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, body, content_type):
        """Serve a cached/proxied clip with byte ranges so seeking works."""
        total = len(body)
        start, end, code = 0, max(0, total - 1), 200
        header = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)$", header)
        if match and total:
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), total - 1) if right else total - 1
            elif right:
                start = max(0, total - int(right))
            if start >= total or end < start:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % total)
                self.end_headers()
                return
            code = 206
        chunk = body[start:end + 1]
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, total))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(chunk)

    def _send_media_file(self, path, content_type):
        """Stream a persistent clip with byte ranges without loading it into RAM."""
        total = path.stat().st_size
        start, end, code = 0, max(0, total - 1), 200
        header = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)$", header)
        if match and total:
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), total - 1) if right else total - 1
            elif right:
                start = max(0, total - int(right))
            if start >= total or end < start:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % total)
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Disposition", 'inline; filename="%s"' % path.name.replace('"', ""))
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, total))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        remaining = end - start + 1
        with path.open("rb") as media:
            media.seek(start)
            while remaining:
                chunk = media.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _read_json(self, limit=36_000_000):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > limit:
            raise ValueError("request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def _image_proxy(self, method, path, payload=None):
        engine = _image_engine()
        if engine is None:
            return self._send(404, json.dumps({"detail": "No image engine is configured"}))
        try:
            status, raw = http_json(method, path, payload, timeout=15, base=engine["base"])
            return self._send(status, raw)
        except urllib.error.HTTPError as exc:
            return self._send(exc.code, exc.read(12000))
        except Exception as exc:
            return self._send(502, json.dumps({"detail": "Image engine unavailable: " + str(exc)}))

    def _image_file(self, name):
        engine = _image_engine()
        if not engine:
            return self._send(404, "No image engine", "text/plain")
        if not name or any(char in name for char in "/\\:\x00"):
            return self._send(404, "Image not found", "text/plain")
        try:
            status, content_type, data = http_bytes("/files/" + quote(name, safe=""), timeout=30, base=engine["base"])
            return self._send(status, data, content_type)
        except urllib.error.HTTPError as exc:
            return self._send(exc.code, exc.read(12000))
        except Exception:
            return self._send(502, "Image engine unavailable", "text/plain")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/videos/jobs":
            return self._send(200, json.dumps(dict(ok=True, items=VIDEO_JOBS.list())))
        job_match = re.fullmatch(r"/api/videos/jobs/(vid_[a-f0-9]{24})", parsed_path.path)
        if job_match:
            try:
                return self._send(200, json.dumps(VIDEO_JOBS.get(job_match[1])))
            except KeyError as exc:
                return self._send(404, json.dumps(dict(ok=False, error=str(exc))))
        if parsed_path.path in {"/api/videos/gallery", "/api/videos/gallery/trash"}:
            try:
                library = _video_library()
                data = library.trash_items() if parsed_path.path.endswith("trash") else library.gallery()
                return self._send(200, json.dumps(data))
            except (OSError, ValueError) as exc:
                return self._send(503, json.dumps(dict(ok=False, error=str(exc))))
        if parsed_path.path.startswith("/api/videos/files/"):
            try:
                name = unquote(parsed_path.path.removeprefix("/api/videos/files/"))
                file = _video_library().file(name)
                return self._send_media_file(file, _content_type_for(name))
            except (OSError, ValueError):
                return self._send(404, "Video not found", "text/plain")
        if parsed_path.path == "/api/workspaces":
            return self._send(200, json.dumps(_workspaces()))
        if parsed_path.path in ("/", "/image", "/video"):
            image_view = parsed_path.path == "/image" or (parsed_path.path == "/" and "text_to_image" in _engine(None)["capabilities"])
            workspace = "image" if image_view else "video"
            if not _workspaces()[workspace]:
                return self._send(404, "This workspace has no configured engine.", "text/plain")
            if image_view:
                return self._send(200, Path(__file__).with_name("image_studio.html").read_bytes(), "text/html; charset=utf-8")
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if parsed_path.path in ("/api/images/health", "/api/images/gallery", "/api/images/gallery/trash"):
            return self._image_proxy("GET", parsed_path.path.removeprefix("/api/images"))
        if parsed_path.path.startswith("/api/images/files/"):
            return self._image_file(unquote(parsed_path.path.removeprefix("/api/images/files/")))
        if parsed_path.path in ("/image_studio.js", "/image_studio.css", "/video_studio.js", "/video_studio.css", "/studio_theme.css"):
            kind = "text/javascript" if parsed_path.path.endswith(".js") else "text/css"
            return self._send(200, Path(__file__).with_name(parsed_path.path[1:]).read_bytes(), kind + "; charset=utf-8")
        if re.fullmatch(r"/api/images/jobs/img_[a-f0-9]{24}", parsed_path.path):
            return self._image_proxy("GET", parsed_path.path.removeprefix("/api/images"))
        if parsed_path.path == "/api/health":
            engine = _engine(parse_qs(parsed_path.query).get("engine", [DEFAULT_ENGINE_ID])[0])
            try:
                status, raw = http_json("GET", "/health", timeout=8, base=engine["base"])
                info = json.loads(raw)
                info["ok"] = status == 200 and bool(info.get("ready"))
                info["engine_id"] = engine["id"]
                return self._send(200, json.dumps(info))
            except Exception as exc:
                return self._send(200, json.dumps({"ok": False, "engine_id": engine["id"], "error": str(exc)}))
        if parsed_path.path == "/api/engines":
            engines = []
            for engine in ENGINES:
                try:
                    status, raw = http_json("GET", "/health", timeout=2, base=engine["base"])
                    health = json.loads(raw)
                    health["reachable"] = status == 200
                except Exception as exc:
                    health = {"ready": False, "reachable": False, "status": "unreachable", "error": str(exc)}
                engines.append(_public_engine(engine, health))
            return self._send(200, json.dumps({
                "ok": True, "default": DEFAULT_ENGINE_ID, "engines": engines,
            }))
        if parsed_path.path == "/api/gallery":
            if parse_qs(parsed_path.query).get("workspace") != ["video"] and "text_to_image" in _engine(None)["capabilities"]:
                return self._image_proxy("GET", "/gallery")
            try:
                items = _gallery_items()
                if items is None:
                    return self._send(200, json.dumps({
                        "ok": False, "items": [],
                        "error": "persistent gallery storage is not configured",
                    }))
                return self._send(200, json.dumps({"ok": True, "count": len(items), "items": items}))
            except Exception as exc:
                return self._send(200, json.dumps({"ok": False, "items": [], "error": str(exc)}))
        if parsed_path.path == "/api/gallery/file":
            name = parse_qs(parsed_path.query).get("name", [""])[0]
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and "text_to_image" in _engine(None)["capabilities"]:
                return self._image_file(name)
            if Path(name).suffix.lower() in VideoLibrary.media_types and GALLERY_DIR:
                try:
                    path = _video_library().file(name)
                except (OSError, ValueError):
                    return self._send(404, "Video not found", "text/plain")
            else:
                path = _gallery_path(name)
            if not path or not path.is_file():
                return self._send(404, "not found", "text/plain")
            try:
                return self._send_media_file(path, _content_type_for(path.name))
            except (BrokenPipeError, ConnectionResetError):
                return
        if parsed_path.path == "/api/file":
            query = parse_qs(parsed_path.query)
            engine = _engine(query.get("engine", [DEFAULT_ENGINE_ID])[0])
            if "cache" in query:
                token = query["cache"][0]
                with _CACHE_LOCK:
                    hit = _CACHE.get(token)
                if not hit:
                    return self._send(404, "not in cache", "text/plain")
                return self._send_media(hit[1], hit[0])
            name = query.get("name", [""])[0].split("/")[-1]
            if not name:
                return self._send(400, "missing name", "text/plain")
            if "text_to_image" in engine["capabilities"]:
                return self._image_file(name)
            if GALLERY_DIR and name in _video_library()._hidden():
                return self._send(404, "Video is in Trash", "text/plain")
            local_path = _gallery_path(name)
            if local_path and local_path.is_file():
                try:
                    return self._send_media_file(local_path, _content_type_for(local_path.name))
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                status, content_type, data = http_bytes(
                    "/files/" + quote(name), timeout=180, base=engine["base"]
                )
                if not content_type or content_type == "application/octet-stream":
                    content_type = _content_type_for(name)
                return self._send_media(data, content_type)
            except Exception as exc:
                return self._send(502, "file fetch failed: " + str(exc), "text/plain")
        if parsed_path.path == "/api/lab/status":
            job_id = parse_qs(parsed_path.query).get("job", [""])[0]
            try:
                return self._send(200, json.dumps(VIDEO_JOBS.get(job_id)))
            except KeyError as exc:
                return self._send(404, json.dumps(dict(ok=False, error=str(exc))))
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._send(403, json.dumps({"detail": "Cross-origin writes are not accepted"}))
        path = urlparse(self.path).path
        if path in {"/api/images/jobs", "/api/images/enhance", "/api/images/gallery/trash", "/api/images/gallery/restore"} or re.fullmatch(r"/api/images/jobs/img_[a-f0-9]{24}/cancel", path):
            try:
                payload = self._read_json()
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
            except Exception as exc:
                return self._send(400, json.dumps({"detail": str(exc)}))
            return self._image_proxy("POST", path.removeprefix("/api/images"), payload)
        if path in {"/api/videos/gallery/trash", "/api/videos/gallery/restore"}:
            try:
                request = self._read_json(limit=32000)
                ids = request.get("ids") if isinstance(request, dict) else None
                if not isinstance(ids, list) or not 1<=len(ids)<=100 or not all(isinstance(x, str) for x in ids):
                    raise ValueError("Provide between 1 and 100 asset or Trash IDs")
                library = _video_library()
                method = library.restore if path.endswith("restore") else library.move_to_trash
                results = []
                for identifier in dict.fromkeys(ids):
                    try:
                        entry = method(identifier)
                        results.append(dict(ok=True, id=identifier, name=entry["name"], trash_id=entry["id"]))
                    except (OSError, ValueError, KeyError) as exc:
                        results.append(dict(ok=False, id=identifier, error=str(exc)))
                return self._send(200, json.dumps(dict(ok=all(x["ok"] for x in results), results=results)))
            except (OSError, ValueError) as exc:
                return self._send(400, json.dumps(dict(ok=False, error=str(exc))))
        cancel_match = re.fullmatch(r"/api/videos/jobs/(vid_[a-f0-9]{24})/cancel", path)
        if cancel_match:
            try:
                self._read_json(limit=32000)
                return self._send(200, json.dumps(VIDEO_JOBS.cancel(cancel_match[1])))
            except KeyError as exc:
                return self._send(404, json.dumps(dict(ok=False, error=str(exc))))
            except ValueError as exc:
                return self._send(400, json.dumps(dict(ok=False, error=str(exc))))
        if path not in {"/api/videos/jobs", "/api/generate", "/api/lab"}:
            return self._send(404, "not found", "text/plain")
        try:
            request = _validate_video_request(self._read_json(), "scenes" if path=="/api/lab" else None)
            job = VIDEO_JOBS.submit(request)
            if path == "/api/generate":
                # Compatibility route; the UI and agents use the nonblocking jobs API.
                deadline = time.monotonic() + GEN_TIMEOUT
                while job["status"] not in {"done", "error", "cancelled"} and time.monotonic()<deadline:
                    time.sleep(.1)
                    job = VIDEO_JOBS.get(job["id"])
            return self._send(200 if job["status"] in {"done", "error", "cancelled"} else 202, json.dumps(job))
        except QueueFull as exc:
            return self._send(429, json.dumps(dict(ok=False, error=str(exc))))
        except (ValueError, TypeError) as exc:
            return self._send(400, json.dumps(dict(ok=False, error=str(exc))))


PAGE = Path(__file__).with_name("video_studio.html").read_text(encoding="utf-8")


def main():
    server = ThreadingHTTPServer((UI_HOST, UI_PORT), Handler)
    print("Frosty Studio  ->  http://localhost:%d" % UI_PORT)
    print("proxying to %s   (Ctrl-C to stop)" % FVL_BASE)
    print("gallery: %s" % (GALLERY_DIR or "disabled (set FVL_GALLERY_DIR)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
