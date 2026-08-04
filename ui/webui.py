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
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

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
_JOBS = {}
_JOBS_LOCK = threading.Lock()

MEDIA_EXT = (".mp4", ".webm", ".gif", ".mov", ".m4v")
CT_BY_EXT = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif",
    ".mov": "video/quicktime", ".m4v": "video/mp4",
}


def _gallery_path(name):
    """Resolve a gallery basename without allowing path traversal or symlinks."""
    if not GALLERY_DIR:
        return None
    clean = Path(str(name)).name
    if not clean.lower().endswith(MEDIA_EXT):
        return None
    candidate = (GALLERY_DIR / clean).resolve()
    if candidate.parent != GALLERY_DIR or candidate.is_symlink():
        return None
    return candidate


def _metadata_path(media_path):
    return media_path.with_name(media_path.name + ".json")


def _record_gallery_metadata(name, metadata):
    path = _gallery_path(name)
    if not path or not path.is_file():
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
    destination = _gallery_path(name)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    _record_gallery_metadata(name, metadata)
    return "/api/gallery/file?name=" + quote(name)


def _gallery_items():
    if not GALLERY_DIR:
        return None
    items = []
    for path in GALLERY_DIR.iterdir():
        if path.is_symlink() or not path.is_file() or not path.name.lower().endswith(MEDIA_EXT):
            continue
        try:
            stat = path.stat()
            metadata = {}
            sidecar = _metadata_path(path)
            if sidecar.is_file() and not sidecar.is_symlink():
                parsed = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    metadata = parsed
            created_at = float(metadata.get("created_at") or stat.st_mtime)
            items.append({
                "name": path.name,
                "file_url": "/api/gallery/file?name=" + quote(path.name),
                "size": stat.st_size,
                "created_at": created_at,
                "prompt": str(metadata.get("prompt") or ""),
                "prompt_format": str(metadata.get("prompt_format") or "text"),
                "seed": metadata.get("seed"),
                "mode": str(metadata.get("mode") or "saved video"),
                "duration_seconds": metadata.get("duration_seconds"),
                "scenes": metadata.get("scenes"),
                "engine_id": metadata.get("engine_id"),
                "engine_label": metadata.get("engine_label"),
                "resolution": metadata.get("resolution"),
                "aspect_ratio": metadata.get("aspect_ratio"),
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items[:GALLERY_LIMIT]


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


def _job_update(job_id, **changes):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(changes)
            _JOBS[job_id]["updated_at"] = time.time()


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


def _run_lab_job(job_id, request):
    started_at = time.time()
    engine = _engine(request.get("engine_id"))
    scenes = request["scenes"]
    story = (request.get("story") or "").strip()
    original_image = request.get("image_b64") or None
    identity_lock = bool(request.get("identity_lock") and original_image)
    continuity = bool(request.get("continuity", True)) and not identity_lock
    seed_value = request.get("seed")
    try:
        base_seed = int(seed_value) if str(seed_value).strip() not in ("", "-1") else secrets.randbelow(2**31)
    except Exception:
        base_seed = secrets.randbelow(2**31)

    try:
        _job_update(job_id, status="running", stage="Preparing scenes", base_seed=base_seed)
        current_image = original_image
        scene_urls = []
        with tempfile.TemporaryDirectory(prefix="frosty-vl-scenes-") as temp_dir:
            clip_paths = []
            for index, scene in enumerate(scenes):
                _job_update(
                    job_id, stage="Rendering scene %d of %d" % (index + 1, len(scenes)),
                    current_scene=index + 1, completed_scenes=index,
                )
                directions = []
                if story:
                    directions.append("Overall story and continuity: " + story)
                directions.append("Scene %d of %d: %s" % (index + 1, len(scenes), scene["prompt"].strip()))
                if len(scenes) > 1:
                    directions.append(
                        "This is one shot in the same film. Preserve established character, production design, "
                        "lighting logic, and visual language while following this scene's action."
                    )
                payload = {
                    "prompt": "\n\n".join(directions),
                    "seed": (base_seed + index) % (2**31),
                    "duration_seconds": float(scene.get("duration", 5)),
                }
                for key in (
                    "negative_prompt", "resolution", "aspect_ratio", "num_inference_steps", "guidance_scale"
                ):
                    if request.get(key) not in (None, ""):
                        payload[key] = request[key]
                if identity_lock:
                    payload["image_b64"] = original_image
                    payload["identity_lock"] = True
                elif current_image:
                    payload["image_b64"] = current_image

                _, kind, value = _remote_generate(payload, engine)
                scene_urls.append(_result_url(kind, value, "scene", {
                    "prompt": scene["prompt"].strip(),
                    "seed": payload["seed"],
                    "mode": "scene %d of %d" % (index + 1, len(scenes)),
                    "duration_seconds": float(scene.get("duration", 5)),
                    "engine_id": engine["id"],
                    "engine_label": engine["label"],
                }, engine=engine))
                _, clip_data = _result_bytes(kind, value, engine)
                clip_path = Path(temp_dir) / ("scene_%02d.mp4" % (index + 1))
                clip_path.write_bytes(clip_data)
                clip_paths.append(clip_path)

                if continuity and index < len(scenes) - 1:
                    _job_update(job_id, stage="Linking scene %d to scene %d" % (index + 1, index + 2))
                    current_image = _extract_last_frame(clip_path)
                _job_update(job_id, completed_scenes=index + 1, scene_urls=list(scene_urls))

            _job_update(job_id, stage="Assembling the final movie", current_scene=None)
            final_path = Path(temp_dir) / "frostyvl_scene_lab.mp4"
            _concat_clips(clip_paths, final_path)
            final_url = _store_gallery_file(final_path, "scene_lab", {
                "prompt": story or "%d-scene movie" % len(scenes),
                "seed": base_seed,
                "mode": "scene lab",
                "engine_id": engine["id"],
                "engine_label": engine["label"],
                "duration_seconds": sum(float(scene.get("duration", 5)) for scene in scenes),
                "scenes": [scene["prompt"].strip() for scene in scenes],
            }) or _cache_media(final_path.read_bytes(), "video/mp4", "movie")

        _job_update(
            job_id, status="done", stage="Complete", file_url=final_url, scene_urls=scene_urls,
            seconds=round(time.time() - started_at, 1), completed_scenes=len(scenes),
        )
    except Exception as exc:
        _job_update(
            job_id, status="error", stage="Stopped", error="%s: %s" % (type(exc).__name__, exc),
            seconds=round(time.time() - started_at, 1),
        )


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
        if length > limit:
            raise ValueError("request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
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
            with _JOBS_LOCK:
                job = dict(_JOBS.get(job_id) or {})
            if not job:
                return self._send(404, json.dumps({"ok": False, "error": "unknown scene job"}))
            job["ok"] = job.get("status") != "error"
            return self._send(200, json.dumps(job))
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/generate", "/api/lab"):
            return self._send(404, "not found", "text/plain")
        try:
            request = self._read_json()
        except Exception as exc:
            return self._send(400, json.dumps({"ok": False, "error": "bad request: %s" % exc}))
        requested_engine_id = str(request.get("engine_id") or DEFAULT_ENGINE_ID)
        if requested_engine_id not in ENGINES_BY_ID:
            return self._send(400, json.dumps({"ok": False, "error": "unknown model selection"}))
        engine = ENGINES_BY_ID[requested_engine_id]
        capabilities = set(engine["capabilities"])
        allowed_durations = [float(value) for value in (engine["controls"].get("durations") or [])]

        if path == "/api/lab":
            if "scene_lab" not in capabilities:
                return self._send(400, json.dumps({"ok": False, "error": "this model does not support Scene Lab"}))
            scenes = request.get("scenes")
            if not isinstance(scenes, list) or not 2 <= len(scenes) <= 8:
                return self._send(400, json.dumps({"ok": False, "error": "add between 2 and 8 scenes"}))
            for index, scene in enumerate(scenes):
                if not isinstance(scene, dict) or not (scene.get("prompt") or "").strip():
                    return self._send(400, json.dumps({"ok": False, "error": "scene %d needs a prompt" % (index + 1)}))
                try:
                    duration = float(scene.get("duration", 5))
                except Exception:
                    duration = 0
                if allowed_durations and duration not in allowed_durations:
                    choices = ", ".join("%g" % value for value in allowed_durations)
                    return self._send(400, json.dumps({
                        "ok": False, "error": "scene duration for %s must be one of: %s seconds" % (engine["label"], choices)
                    }))
                if not allowed_durations and not 1 <= duration <= 15:
                    return self._send(400, json.dumps({"ok": False, "error": "scene durations must be 1–15 seconds"}))
            if request.get("identity_lock") and not request.get("image_b64"):
                return self._send(400, json.dumps({"ok": False, "error": "same-person lock needs an image"}))
            if request.get("identity_lock") and "same_face" not in capabilities:
                return self._send(400, json.dumps({"ok": False, "error": "%s does not support Same Face lock" % engine["label"]}))
            job_id = "lab_" + secrets.token_urlsafe(8)
            with _JOBS_LOCK:
                while len(_JOBS) > 20:
                    _JOBS.pop(next(iter(_JOBS)))
                _JOBS[job_id] = {
                    "job": job_id, "status": "queued", "stage": "Queued", "total_scenes": len(scenes),
                    "completed_scenes": 0, "scene_urls": [], "created_at": time.time(),
                    "engine_id": engine["id"], "engine_label": engine["label"],
                }
            threading.Thread(target=_run_lab_job, args=(job_id, request), daemon=True).start()
            return self._send(202, json.dumps({"ok": True, "job": job_id}))

        try:
            prompt, prompt_format = _normalize_prompt(request)
        except ValueError as exc:
            return self._send(400, json.dumps({"ok": False, "error": str(exc)}))
        if request.get("identity_lock") and not request.get("image_b64"):
            return self._send(400, json.dumps({"ok": False, "error": "same-person lock needs an image"}))
        if request.get("identity_lock") and "same_face" not in capabilities:
            return self._send(400, json.dumps({"ok": False, "error": "%s does not support Same Face lock" % engine["label"]}))
        if request.get("image_b64") and "image_to_video" not in capabilities:
            return self._send(400, json.dumps({"ok": False, "error": "%s does not support image input" % engine["label"]}))
        seed_value = request.get("seed")
        try:
            seed = int(seed_value) if str(seed_value).strip() not in ("", "-1", "None") else secrets.randbelow(2**31)
        except Exception:
            seed = secrets.randbelow(2**31)
        payload = {"prompt": prompt, "seed": seed}
        for key in (
            "image_b64", "identity_lock", "duration_seconds", "negative_prompt", "resolution",
            "aspect_ratio", "num_inference_steps", "guidance_scale",
        ):
            if request.get(key) not in (None, "", False):
                payload[key] = request[key]
        started_at = time.time()
        try:
            parsed, kind, value = _remote_generate(payload, engine)
            file_url = _result_url(kind, value, metadata={
                "prompt": prompt,
                "prompt_format": prompt_format,
                "seed": seed,
                "mode": parsed.get("mode", "video"),
                "duration_seconds": parsed.get("duration_seconds") or request.get("duration_seconds"),
                "engine_id": engine["id"],
                "engine_label": engine["label"],
                "resolution": request.get("resolution"),
                "aspect_ratio": request.get("aspect_ratio"),
                "num_inference_steps": request.get("num_inference_steps"),
                "guidance_scale": request.get("guidance_scale"),
            }, engine=engine)
        except Exception as exc:
            return self._send(200, json.dumps({
                "ok": False, "error": str(exc), "seconds": round(time.time() - started_at, 1),
            }))
        return self._send(200, json.dumps({
            "ok": True, "file_url": file_url, "seconds": round(time.time() - started_at, 1),
            "used_seed": seed, "mode": parsed.get("mode", "video"),
            "prompt_format": prompt_format,
            "duration_seconds": parsed.get("duration_seconds"), "engine_id": engine["id"],
            "engine_label": engine["label"], "raw": _prune(parsed),
        }))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#ffffff">
<title>Frosty Studio</title>
<style>
  :root{--bg:#f3f6fb;--card:#fff;--ink:#172234;--muted:#617086;--line:#d7e0ec;
    --accent:#1565c0;--accent-soft:#e9f2fd;--warm:#ef6c00;--ok:#27823b;--bad:#b13c17;--shadow:0 12px 35px rgba(30,58,95,.07)}
  *{box-sizing:border-box} html{-webkit-text-size-adjust:100%} body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:radial-gradient(circle at 50% -160px,#dfeeff 0,transparent 440px),var(--bg);color:var(--ink)}
  header{padding:16px 24px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.9);backdrop-filter:blur(12px);
    display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:20} header h1{font-size:18px;margin:0;letter-spacing:.2px}
  .mark{width:31px;height:31px;border-radius:9px;background:linear-gradient(145deg,#1976d2,#0d47a1);color:#fff;
    display:grid;place-items:center;font-weight:850;box-shadow:0 4px 12px rgba(21,101,192,.25)}
  .dot{width:9px;height:9px;border-radius:50%;background:#aeb9c7;margin-left:auto}.dot.up{background:var(--ok)}.dot.down{background:var(--bad)}
  .sub{color:var(--muted);font-size:13px} main{max-width:1080px;margin:24px auto 50px;padding:0 18px;display:grid;gap:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:20px;box-shadow:var(--shadow)}
  .engine-bar{display:grid;grid-template-columns:minmax(220px,310px) minmax(0,1fr) auto;gap:16px;align-items:end;padding:17px 20px;background:#fff}
  .engine-copy{align-self:center;min-width:0}.engine-copy b{display:block;font-size:14px}.engine-copy .sub{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.engine-caps{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.cap{padding:3px 7px;border-radius:99px;background:#eef4fb;color:#45627f;font-size:11px;font-weight:750}.cap.on{background:#e8f5eb;color:#27733a}
  .settings-panel{padding:18px 20px;border-top:1px solid var(--line);background:#f8fbff}.settings-grid{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px}.settings-wide{grid-column:1/-1}.settings-panel textarea{min-height:72px}.settings-note{padding:12px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--muted)}
  .hero{padding:0;overflow:hidden}.tabs{display:flex;padding:7px;background:#edf2f8;border-bottom:1px solid var(--line);gap:6px;overflow-x:auto}
  .tab{border:0;background:transparent;color:var(--muted);padding:10px 16px;border-radius:9px;font-weight:750;white-space:nowrap}.tab.active{background:#fff;color:var(--accent);box-shadow:0 1px 5px rgba(30,55,90,.11)}
  .panel{display:none;padding:22px}.panel.active{display:block}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:17px}
  h2{font-size:18px;margin:0 0 3px} label{display:block;font-weight:700;font-size:13px;margin:0 0 7px}
  textarea,input,select{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:9px;font:inherit;color:var(--ink);background:#fbfdff;outline:none}
  textarea:focus,input:focus,select:focus{border-color:#7cadde;box-shadow:0 0 0 3px rgba(21,101,192,.09)} textarea{min-height:105px;resize:vertical}
  .composer{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(250px,.75fr);gap:18px}.field+ .field{margin-top:14px}
  .prompt-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:7px}.prompt-head>label{margin:0}
  .prompt-controls{display:flex;align-items:flex-end;gap:7px;flex-wrap:wrap;justify-content:flex-end}.prompt-format{display:flex;align-items:center;gap:7px;margin:0;color:var(--muted)}.prompt-format span{font-size:12px}.prompt-format select{width:auto;min-width:145px;padding:7px 9px;font-size:13px}
  .prompt-controls button{min-height:34px;padding:7px 10px;font-size:12px}.prompt-note{display:block;margin-top:7px;min-height:20px}.json-valid{color:var(--ok)}.json-invalid{color:var(--bad)}
  textarea.json-prompt{min-height:420px;font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;font-size:13px;line-height:1.5;tab-size:2;white-space:pre}
  .drop{min-height:151px;border:1.5px dashed #a9bdd5;border-radius:11px;background:#f8fbff;display:grid;place-items:center;text-align:center;padding:14px;cursor:pointer;color:var(--muted);transition:.18s}
  .drop:hover,.drop.drag{border-color:var(--accent);background:var(--accent-soft)}.drop .icon{font-size:25px;display:block;margin-bottom:4px}.drop b{color:var(--accent)}
  .drop.disabled{opacity:.55;cursor:not-allowed}.drop.disabled:hover{border-color:#a9bdd5;background:#f8fbff}
  .preview{position:relative;width:100%;height:150px}.preview img{width:100%;height:100%;object-fit:cover;border-radius:9px}.preview .shade{position:absolute;inset:auto 0 0;padding:22px 9px 8px;color:#fff;text-align:left;font-size:12px;background:linear-gradient(transparent,rgba(0,0,0,.72));border-radius:0 0 9px 9px}
  .remove-image{position:absolute;right:7px;top:7px;border:0;border-radius:50%;width:29px;height:29px;padding:0;background:rgba(15,25,38,.75);color:#fff;font-weight:800}
  .options{display:grid;gap:10px;margin-top:11px}.toggle-line{display:flex;align-items:flex-start;gap:10px;padding:10px 11px;border:1px solid var(--line);border-radius:10px;background:#fbfdff}
  .toggle-line input{width:auto;margin:4px 0 0;accent-color:var(--accent)}.toggle-line b{display:block;font-size:13px}.toggle-line small{display:block;color:var(--muted);line-height:1.35;margin-top:2px}
  .row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-top:15px}.row .seed{flex:0 0 155px}.row .duration{flex:0 0 150px}.grow{flex:1}
  button{cursor:pointer;border:0;border-radius:9px;padding:11px 18px;min-height:44px;font:inherit;font-weight:750;touch-action:manipulation}.go{background:var(--accent);color:#fff;box-shadow:0 5px 12px rgba(21,101,192,.18)}
  .go:hover{background:#0e59ad}.go:disabled{background:#9db8d6;box-shadow:none;cursor:not-allowed}.ghost{background:#edf3fa;color:var(--accent)}.danger{background:#fff0ec;color:var(--bad);padding:7px 10px}
  .status{margin-top:15px;font-size:14px;color:var(--muted);min-height:22px;display:flex;align-items:center;gap:9px}.spin{width:15px;height:15px;border:2px solid #cfe0f5;border-top-color:var(--accent);border-radius:50%;animation:s .8s linear infinite;display:inline-block}@keyframes s{to{transform:rotate(360deg)}}
  .err{color:var(--bad)}.result video{width:100%;border-radius:11px;background:#000;margin-top:7px;max-height:660px}.result-tools{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:9px}
  a.dl{color:var(--accent);font-weight:700;text-decoration:none}.badge{display:inline-flex;padding:4px 8px;border-radius:99px;background:var(--accent-soft);color:var(--accent);font-size:12px;font-weight:700}
  .lab-layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:18px}.scenes{display:grid;gap:10px}.scene{border:1px solid var(--line);border-radius:11px;padding:12px;background:#fbfdff}
  .scene-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.scene-head b{font-size:13px}.scene-grid{display:grid;grid-template-columns:minmax(0,1fr) 120px;gap:10px}.scene textarea{min-height:78px}
  .add-scene{width:100%;margin-top:10px;border:1px dashed #a9bdd5;background:#f8fbff;color:var(--accent)}.lab-side{align-self:start;position:sticky;top:82px}.lab-side .drop{min-height:125px}
  .progress{height:7px;background:#e4ebf4;border-radius:99px;overflow:hidden;margin-top:10px}.progress>span{height:100%;display:block;background:linear-gradient(90deg,var(--accent),#45a4ff);transition:width .4s}
  .scene-clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:9px;margin-top:12px}.scene-clips video{width:100%;border-radius:8px;background:#000}
  .gallery-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:14px}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}.tile{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#fff}.tile video{width:100%;aspect-ratio:16/9;object-fit:contain;display:block;background:#000}.tile .meta{padding:10px;font-size:12px;color:var(--muted);display:grid;gap:5px}.tile .meta b{color:var(--ink)}.tile-tools{display:flex;align-items:center;justify-content:space-between;gap:8px}.empty-gallery{grid-column:1/-1;padding:38px 18px;text-align:center;border:1px dashed #b8c8da;border-radius:11px;background:#f8fbff}.filename{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  @media(max-width:760px){body{background:var(--bg)}header{padding:10px 12px;gap:9px}header .tagline{display:none}header h1{font-size:16px}#ep{max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.mark{width:29px;height:29px}main{margin:10px auto 28px;padding:0 8px;gap:10px}.card{border-radius:12px}.engine-bar{grid-template-columns:1fr auto;padding:13px 14px;gap:10px}.engine-copy{grid-column:1/-1;grid-row:2}.engine-copy .sub{white-space:normal}.settings-panel{padding:14px}.settings-grid{grid-template-columns:1fr 1fr}.composer,.lab-layout{grid-template-columns:1fr;gap:14px}.prompt-head{align-items:flex-start;flex-direction:column}.prompt-controls{justify-content:flex-start}.prompt-format select{font-size:16px}.lab-side{position:static}.scene-grid{grid-template-columns:1fr}.panel{padding:14px}.panel-head,.gallery-head{flex-wrap:wrap;margin-bottom:14px}.tabs{padding:5px;gap:4px}.tab{flex:1 0 auto;padding:10px 12px}.drop{min-height:112px}.preview{height:142px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.row .seed,.row .duration{min-width:0;flex:initial}.row .grow{display:none}.row #rnd,.row .go{grid-column:1/-1;width:100%}.result-tools{align-items:flex-start;flex-wrap:wrap}.result-tools .dl{padding:8px 0}.scene{padding:10px}.scene textarea{min-height:92px}.status{align-items:flex-start;overflow-wrap:anywhere}textarea,input,select{font-size:16px}.gallery{grid-template-columns:1fr}.tile .meta{padding:11px}.scene-clips{grid-template-columns:1fr}}
  @media(min-width:521px) and (max-width:760px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><span class="mark">❄</span><h1>Frosty Studio</h1><span class="sub tagline">universal video</span><span id="dot" class="dot"></span><span class="sub" id="ep">connecting…</span></header>
<main>
  <section class="card hero">
    <div class="engine-bar">
      <div><label for="engine-select">Video model</label><select id="engine-select" aria-label="Video model"><option>Loading models…</option></select></div>
      <div class="engine-copy"><b id="engine-title">Connecting to engines…</b><span class="sub" id="engine-description">Checking the available render models.</span><div class="engine-caps" id="engine-caps"></div></div>
      <button class="ghost" id="settings-toggle" type="button" aria-expanded="false">Settings</button>
    </div>
    <div class="settings-panel" id="settings-panel" hidden>
      <div class="settings-grid">
        <div id="resolution-field"><label for="resolution">Resolution</label><select id="resolution"></select></div>
        <div id="aspect-field"><label for="aspect-ratio">Frame</label><select id="aspect-ratio"></select></div>
        <div id="steps-field"><label for="steps">Quality steps</label><input id="steps" type="number" inputmode="numeric"></div>
        <div id="guidance-field"><label for="guidance">Prompt strength</label><input id="guidance" type="number" inputmode="decimal"></div>
        <div class="settings-wide" id="negative-field"><label for="negative-prompt">Avoid <span class="sub">(negative prompt)</span></label><textarea id="negative-prompt" placeholder="Things you do not want in the video…"></textarea></div>
        <div class="settings-wide settings-note" id="settings-note">This model uses its tuned defaults. Seed and duration remain available in the Create panel.</div>
      </div>
    </div>
    <nav class="tabs"><button class="tab active" data-tab="create">Create</button><button class="tab" data-tab="lab">Scene lab</button><button class="tab" data-tab="gallery">Gallery</button></nav>
    <div class="panel active" id="create-panel">
      <div class="panel-head"><div><h2>Create a clip</h2><div class="sub">Prompt from scratch or animate an uploaded image.</div></div><span class="badge" id="duration-badge">5–14 sec</span></div>
      <div class="composer">
        <div>
          <div class="prompt-head">
            <label for="p">Direction</label>
            <div class="prompt-controls">
              <label class="prompt-format" for="prompt-format"><span>Format</span><select id="prompt-format"><option value="text">Natural language</option><option value="json">Structured JSON</option></select></label>
              <button class="ghost" id="json-template" type="button" hidden>Load detailed template</button>
              <button class="ghost" id="json-format" type="button" hidden>Format JSON</button>
            </div>
          </div>
          <textarea id="p" placeholder="Describe the subject, action, camera, lighting, dialogue, and sound…">A red fox trotting through a snowy pine forest, cinematic tracking shot, soft daylight, natural forest ambience</textarea>
          <span class="sub prompt-note" id="prompt-note">Natural-language direction sent directly to the selected model.</span>
        </div>
        <div>
          <label>Source image <span class="sub">(optional)</span></label>
          <div class="drop image-drop"><div class="empty"><span class="icon">▧</span><b>Add an image</b><br><span class="sub">drop, paste, or choose a file</span></div><div class="preview" hidden></div></div>
          <input class="image-input" type="file" accept="image/*" hidden>
          <div class="options"><label class="toggle-line"><input id="lock" type="checkbox" disabled><span><b>Same Face</b><small id="lock-help">Use the image as an identity reference instead of the opening frame.</small></span></label></div>
        </div>
      </div>
      <div class="row">
        <div class="duration"><label for="duration">Length</label><select id="duration"><option value="5">5 seconds</option><option value="8">8 seconds</option><option value="10">10 seconds</option><option value="12">12 seconds</option><option value="14">14 seconds</option></select></div>
        <div class="seed"><label for="s">Seed <span class="sub">(blank = random)</span></label><input id="s" placeholder="random" inputmode="numeric"></div>
        <button class="ghost" id="rnd" type="button">Randomize</button><div class="grow"></div><button class="go" id="go" type="button">Generate clip</button>
      </div>
      <div class="status" id="status"></div><div class="result" id="result"></div>
    </div>

    <div class="panel" id="lab-panel">
      <div class="panel-head"><div><h2>Scene lab</h2><div class="sub">Build a longer movie shot by shot; the selected engine renders and joins every scene.</div></div><span class="badge" id="lab-engine-badge">selected model</span></div>
      <div class="field"><label for="story">Story + continuity notes <span class="sub">(optional)</span></label><textarea id="story" style="min-height:74px" placeholder="The same traveler crosses the desert at dusk. Warm 35mm film look, restrained camera movement, continuous wardrobe and weather."></textarea></div>
      <div class="lab-layout">
        <div><label>Scenes</label><div class="scenes" id="scenes"></div><button class="add-scene" id="add-scene" type="button">+ Add scene</button></div>
        <aside class="lab-side">
          <label>Character / opening image <span class="sub">(optional)</span></label>
          <div class="drop image-drop"><div class="empty"><span class="icon">▧</span><b>Add an image</b><br><span class="sub">shared with Create clip</span></div><div class="preview" hidden></div></div>
          <input class="image-input" type="file" accept="image/*" hidden>
          <div class="options">
            <label class="toggle-line"><input id="lab-lock" type="checkbox" disabled><span><b>Same Face</b><small id="lab-lock-help">Reference the same face in every scene.</small></span></label>
            <label class="toggle-line"><input id="continuity" type="checkbox" checked><span><b>Match-cut scenes</b><small>Start each new scene from the prior scene's last frame.</small></span></label>
          </div>
          <div class="field" style="margin-top:12px"><label for="lab-seed">Base seed</label><input id="lab-seed" placeholder="random" inputmode="numeric"></div>
          <button class="go" id="lab-go" type="button" style="width:100%;margin-top:12px">Generate movie</button>
        </aside>
      </div>
      <div class="status" id="lab-status"></div><div id="lab-progress" class="progress" hidden><span style="width:0"></span></div><div class="result" id="lab-result"></div>
    </div>

    <div class="panel" id="gallery-panel">
      <div class="gallery-head"><div><h2>Gallery</h2><div class="sub" id="gallery-summary">Loading saved videos…</div></div><button class="ghost" id="gallery-refresh" type="button">Refresh</button></div>
      <div class="gallery" id="gallery"><div class="empty-gallery sub">Loading your saved videos…</div></div>
    </div>
  </section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let imageData=null,imageName='',timer=null,t0=0,labTimer=null,engines=[],selectedEngineId='';
const capNames={text_to_video:'Text',image_to_video:'Image',same_face:'Same Face',native_audio:'Audio',scene_lab:'Scene Lab'};
const selectedEngine=()=>engines.find(x=>x.id===selectedEngineId)||engines[0]||null;
const hasCap=name=>!!selectedEngine()?.capabilities?.includes(name);
const jsonPromptTemplate={
  version:"1.0",
  summary:"A cinematic tracking shot of a red fox moving through a snowy pine forest at dawn.",
  intent:{mood:"quiet, alert, immersive",priority:"natural motion, coherent anatomy, and synchronized environmental sound"},
  subjects:[{
    id:"fox_1",
    type:"adult red fox",
    appearance:{fur:"rich red-orange winter coat",markings:"white chest and dark lower legs",condition:"healthy and dry"},
    action:"trot steadily along a narrow snow-covered trail, occasionally turning its ears toward distant sounds",
    performance:"natural animal behavior; calm but attentive",
    continuity:"preserve coat markings, scale, and body proportions for the entire shot"
  }],
  environment:{
    location:"dense alpine pine forest",
    time_of_day:"early dawn",
    weather:"cold, still air with very light drifting snow",
    ground:"fresh powder over a narrow forest trail",
    background:"layered pine trunks fading into pale blue atmospheric haze",
    physical_details:["small puffs of powder under each paw", "subtle breath vapor", "occasional snow falling from branches"]
  },
  timeline:[
    {time:"0.0-1.5s",visual_action:"begin on a wide profile view as the fox enters frame left",camera_action:"smooth lateral tracking begins",audio_event:"soft paw impacts and quiet winter wind"},
    {time:"1.5-3.8s",visual_action:"fox crosses the center of frame and briefly looks toward camera without stopping",camera_action:"ease into a medium profile framing with stable horizon",audio_event:"nearby branch creak and a distant bird call on the right"},
    {time:"3.8-5.2s",visual_action:"fox continues toward a brighter opening between the trees",camera_action:"gently lag behind and hold the final composition",audio_event:"paw sounds recede into natural forest ambience"}
  ],
  camera:{
    shot_design:"single continuous shot",
    framing:"wide-to-medium side profile",
    lens:"35mm spherical lens",
    movement:"low, stabilized lateral tracking at the fox's pace",
    height:"approximately 45 cm above the ground",
    focus:"continuous focus on the fox's eyes and head",
    depth_of_field:"moderate; subject sharp with softly layered background",
    composition:"fox travels left to right with open lead room",
    prohibited_moves:["no cuts", "no sudden zoom", "no orbit", "no handheld shake"]
  },
  lighting:{
    key:"soft cool dawn skylight",
    accent:"faint warm sunlight filtering between distant trees",
    contrast:"gentle and realistic",
    exposure:"retain detail in white snow and dark fur"
  },
  visual_style:{
    medium:"photoreal cinematic nature footage",
    color_palette:["snow white", "pine green", "cool blue shadows", "warm red fur"],
    texture:"fine fur, powder snow, and bark detail without oversharpening",
    motion_rendering:"natural shutter motion blur",
    grading:"restrained documentary color grade"
  },
  audio:{
    dialogue:[],
    ambience:"quiet stereo winter forest with light wind through pine needles",
    foreground_sounds:["soft rhythmic paw impacts in snow", "subtle fox breathing"],
    background_sounds:["one distant bird call from the right", "occasional branch creak"],
    music:"none",
    mix:"natural dynamic range; ambience remains below foreground movement",
    synchronization:"paw impacts and visible environmental motion must align precisely"
  },
  continuity:["one fox only", "consistent direction of travel", "stable weather and dawn lighting", "no unexplained objects entering the shot"],
  avoid:["anatomy distortion", "extra limbs or tails", "sliding feet", "flicker", "warped trees", "text", "logos", "subtitles", "music"]
};
let promptMode='text';
const promptDrafts={text:$('#p').value,json:JSON.stringify(jsonPromptTemplate,null,2)};
function parsedJsonPrompt(){let value;try{value=JSON.parse($('#p').value);}catch(error){throw new Error('Invalid JSON: '+error.message);}if(!value||Array.isArray(value)||typeof value!=='object')throw new Error('JSON prompt must be a non-empty object.');if(!Object.keys(value).length)throw new Error('JSON prompt cannot be empty.');return value;}
function promptForRequest(){if(promptMode==='json')return JSON.stringify(parsedJsonPrompt(),null,2);return $('#p').value.trim();}
function updatePromptNote(){const note=$('#prompt-note');if(promptMode!=='json'){note.className='sub prompt-note';note.textContent='Natural-language direction sent directly to the selected model.';return;}try{const value=parsedJsonPrompt(),chars=JSON.stringify(value,null,2).length;note.className='sub prompt-note json-valid';note.textContent='Valid JSON object · '+Object.keys(value).length+' top-level fields · '+chars.toLocaleString()+' characters. The model receives this structure as prompt text.';}catch(error){note.className='sub prompt-note json-invalid';note.textContent=error.message;}}
function setPromptMode(next){promptDrafts[promptMode]=$('#p').value;promptMode=next;$('#p').value=promptDrafts[promptMode];$('#p').classList.toggle('json-prompt',promptMode==='json');$('#p').placeholder=promptMode==='json'?'Enter one valid JSON object…':'Describe the subject, action, camera, lighting, dialogue, and sound…';$('#json-template').hidden=promptMode!=='json';$('#json-format').hidden=promptMode!=='json';updatePromptNote();}
$('#prompt-format').onchange=()=>setPromptMode($('#prompt-format').value);
$('#json-template').onclick=()=>{if($('#p').value.trim()&&!confirm('Replace the current JSON with the detailed template?'))return;$('#p').value=JSON.stringify(jsonPromptTemplate,null,2);promptDrafts.json=$('#p').value;updatePromptNote();};
$('#json-format').onclick=()=>{try{$('#p').value=JSON.stringify(parsedJsonPrompt(),null,2);promptDrafts.json=$('#p').value;updatePromptNote();}catch(error){updatePromptNote();$('#status').innerHTML='<span class="err">'+esc(error.message)+'</span>';}};
$('#p').addEventListener('input',()=>{promptDrafts[promptMode]=$('#p').value;updatePromptNote();});
$('#p').addEventListener('keydown',event=>{if(promptMode==='json'&&event.key==='Tab'){event.preventDefault();const start=event.target.selectionStart;event.target.setRangeText('  ',start,event.target.selectionEnd,'end');promptDrafts.json=event.target.value;updatePromptNote();}});
function setOptions(select,values,labels={}){const current=select.value;select.innerHTML='';(values||[]).forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=labels[value]||value;select.appendChild(option);});if([...select.options].some(x=>x.value===current))select.value=current;}
function settingsPayload(){const engine=selectedEngine(),controls=engine?.controls||{},payload={};
  if(controls.resolutions?.length)payload.resolution=$('#resolution').value;if(controls.aspect_ratios?.length)payload.aspect_ratio=$('#aspect-ratio').value;
  if(controls.steps)payload.num_inference_steps=Number($('#steps').value);if(controls.guidance)payload.guidance_scale=Number($('#guidance').value);
  if(controls.negative_prompt&&$('#negative-prompt').value.trim())payload.negative_prompt=$('#negative-prompt').value.trim();return payload;}
function applyEngine(){const engine=selectedEngine();if(!engine)return;const controls=engine.controls||{},health=engine.health||{};
  $('#engine-title').textContent=engine.label;$('#engine-description').textContent=engine.description;$('#lab-engine-badge').textContent=engine.label;
  $('#engine-caps').innerHTML='';engine.capabilities.forEach(name=>{const chip=document.createElement('span');chip.className='cap on';chip.textContent=capNames[name]||name.replaceAll('_',' ');$('#engine-caps').appendChild(chip);});
  $('#dot').className='dot '+(health.ready?'up':'down');$('#ep').textContent=health.ready?'ready':health.loading||health.status==='loading'?'loading model…':'model offline';
  const durations=(controls.durations||[5]).map(Number),durationLabels=Object.fromEntries(durations.map(n=>[n,n+' seconds']));setOptions($('#duration'),durations,durationLabels);
  $('#duration-badge').textContent=durations.length?(Math.min(...durations)+'–'+Math.max(...durations)+' sec'):'model default';
  if(typeof scenes!=='undefined'){scenes.forEach(scene=>{if(!durations.includes(Number(scene.duration)))scene.duration=durations[0];});renderScenes();}
  setOptions($('#resolution'),controls.resolutions||[]);setOptions($('#aspect-ratio'),controls.aspect_ratios||[]);
  $('#resolution-field').hidden=!controls.resolutions?.length;$('#aspect-field').hidden=!controls.aspect_ratios?.length;
  $('#steps-field').hidden=!controls.steps;$('#guidance-field').hidden=!controls.guidance;$('#negative-field').hidden=!controls.negative_prompt;
  if(controls.steps){$('#steps').min=controls.steps.min;$('#steps').max=controls.steps.max;if(!$('#steps').value)$('#steps').value=controls.steps.default;}
  if(controls.guidance){$('#guidance').min=controls.guidance.min;$('#guidance').max=controls.guidance.max;$('#guidance').step=controls.guidance.step||.5;if(!$('#guidance').value)$('#guidance').value=controls.guidance.default;}
  $('#settings-note').hidden=!!(controls.resolutions?.length||controls.aspect_ratios?.length||controls.steps||controls.guidance||controls.negative_prompt);
  if(!hasCap('same_face')){$('#lock').checked=false;$('#lab-lock').checked=false;$('#continuity').disabled=false;}$('#lock-help').textContent=hasCap('same_face')?'Use the image as an identity reference instead of the opening frame.':'This model uses the image as an opening-frame guide; identity lock is unavailable.';
  $('#lab-lock-help').textContent=hasCap('same_face')?'Reference the same face in every scene.':'Same Face is unavailable for this model.';renderImage();}
async function refreshEngines(){try{const previous=selectedEngineId||$('#engine-select').value,r=await fetch('/api/engines',{cache:'no-store'}),j=await r.json();if(!j.ok)throw new Error(j.error||'Could not load models');engines=j.engines||[];selectedEngineId=engines.some(x=>x.id===previous)?previous:(j.default||engines[0]?.id||'');
  const select=$('#engine-select');select.innerHTML='';engines.forEach(engine=>{const option=document.createElement('option');option.value=engine.id;const state=engine.health?.ready?'●':engine.health?.loading||engine.health?.status==='loading'?'◌':'○';option.textContent=state+' '+engine.label;select.appendChild(option);});select.value=selectedEngineId;applyEngine();}
  catch(e){$('#dot').className='dot down';$('#ep').textContent='model menu offline';$('#engine-description').textContent=e.message||e;}}

$('#engine-select').onchange=()=>{selectedEngineId=$('#engine-select').value;applyEngine();};
$('#settings-toggle').onclick=()=>{const panel=$('#settings-panel'),willOpen=panel.hidden;panel.hidden=!willOpen;$('#settings-toggle').setAttribute('aria-expanded',String(willOpen));$('#settings-toggle').textContent=willOpen?'Close settings':'Settings';};

$$('.tab').forEach(b=>b.onclick=()=>{$$('.tab').forEach(x=>x.classList.toggle('active',x===b));$$('.panel').forEach(x=>x.classList.remove('active'));$('#'+b.dataset.tab+'-panel').classList.add('active');if(b.dataset.tab==='gallery')loadGallery();});

function imageFile(file){return new Promise((resolve,reject)=>{if(!file||!file.type.startsWith('image/'))return reject(new Error('Choose an image file.'));
  const url=URL.createObjectURL(file),img=new Image();img.onload=()=>{URL.revokeObjectURL(url);let w=img.naturalWidth,h=img.naturalHeight,max=1600;
    if(Math.max(w,h)>max){const scale=max/Math.max(w,h);w=Math.round(w*scale);h=Math.round(h*scale);}const c=document.createElement('canvas');c.width=w;c.height=h;
    c.getContext('2d').drawImage(img,0,0,w,h);resolve(c.toDataURL('image/jpeg',.92));};img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('That image could not be read.'));};img.src=url;});}
async function setImage(file){try{imageData=await imageFile(file);imageName=file.name;renderImage();}catch(e){alert(e.message);}}
function renderImage(){const canImage=hasCap('image_to_video'),canLock=hasCap('same_face');$$('.image-drop').forEach(drop=>{const empty=drop.querySelector('.empty'),preview=drop.querySelector('.preview');drop.classList.toggle('disabled',!canImage);
  if(imageData){empty.hidden=true;preview.hidden=false;preview.innerHTML='<img src="'+imageData+'"><div class="shade">'+esc(imageName)+'</div><button class="remove-image" title="Remove image" type="button">×</button>';preview.querySelector('button').onclick=e=>{e.stopPropagation();imageData=null;imageName='';$('#lock').checked=false;$('#lab-lock').checked=false;renderImage();};}
  else{empty.hidden=false;preview.hidden=true;preview.innerHTML='';}});$$('#lock,#lab-lock').forEach(x=>x.disabled=!imageData||!canLock);}
$$('.image-drop').forEach((drop,i)=>{const input=$$('.image-input')[i];drop.onclick=()=>{if(hasCap('image_to_video'))input.click();};input.onchange=()=>input.files[0]&&setImage(input.files[0]);
  drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');e.dataTransfer.files[0]&&setImage(e.dataTransfer.files[0]);};});
document.addEventListener('paste',e=>{const f=[...(e.clipboardData?.files||[])].find(x=>x.type.startsWith('image/'));if(f)setImage(f);});
$('#lab-lock').onchange=()=>{$('#continuity').disabled=$('#lab-lock').checked;if($('#lab-lock').checked)$('#continuity').checked=false;};

$('#rnd').onclick=()=>{$('#s').value=Math.floor(Math.random()*2**31);};
function tick(){const elapsed=Math.round((Date.now()-t0)/1000);$('#status').innerHTML='<span class="spin"></span> Rendering with '+esc(selectedEngine()?.label||'the selected model')+'… '+elapsed+'s';}
function showResult(container,url,seed,mode,sceneUrls=[]){container.innerHTML='';const v=document.createElement('video');v.src=url;v.controls=true;v.autoplay=true;v.loop=sceneUrls.length===0;container.appendChild(v);
  const tools=document.createElement('div');tools.className='result-tools';tools.innerHTML='<span class="badge">'+esc(mode)+'</span><a class="dl" href="'+url+'" download="frostyvl_'+esc(seed)+'.mp4">Download video ↓</a>';container.appendChild(tools);
  if(sceneUrls.length){const clips=document.createElement('div');clips.className='scene-clips';sceneUrls.forEach((u,i)=>{const sv=document.createElement('video');sv.src=u;sv.controls=true;sv.muted=true;sv.title='Scene '+(i+1);clips.appendChild(sv);});container.appendChild(clips);}}
function formatBytes(bytes){const n=Number(bytes)||0;if(n<1024)return n+' B';if(n<1024**2)return (n/1024).toFixed(1)+' KB';if(n<1024**3)return (n/1024**2).toFixed(1)+' MB';return (n/1024**3).toFixed(1)+' GB';}
function formatDate(seconds){const d=new Date(Number(seconds)*1000);return Number.isNaN(d.valueOf())?'':d.toLocaleString([], {dateStyle:'medium',timeStyle:'short'});}
function renderGallery(items){const root=$('#gallery');root.innerHTML='';$('#gallery-summary').textContent=items.length?items.length+' saved video'+(items.length===1?'':'s'):'No saved videos yet';
  if(!items.length){root.innerHTML='<div class="empty-gallery"><b>Your videos will stay here</b><div class="sub">New clips and completed scene-lab movies are saved on this machine.</div></div>';return;}
  items.forEach(item=>{const tile=document.createElement('article');tile.className='tile';const video=document.createElement('video');video.src=item.file_url;video.controls=true;video.loop=true;video.muted=true;video.playsInline=true;video.preload='metadata';
    const meta=document.createElement('div');meta.className='meta';const title=document.createElement('b');title.textContent=(item.engine_label?item.engine_label+' · ':'')+(item.mode||'saved video')+(item.prompt_format==='json'?' · JSON':'')+(item.seed!==null&&item.seed!==undefined?' · seed '+item.seed:'');
    const prompt=document.createElement('div');prompt.textContent=item.prompt||item.name;prompt.title=item.prompt||item.name;
    const tools=document.createElement('div');tools.className='tile-tools';const details=document.createElement('span');details.textContent=[formatDate(item.created_at),formatBytes(item.size)].filter(Boolean).join(' · ');
    const link=document.createElement('a');link.className='dl';link.href=item.file_url;link.download=item.name;link.textContent='Download ↓';tools.append(details,link);meta.append(title,prompt,tools);tile.append(video,meta);root.appendChild(tile);});}
async function loadGallery(){try{$('#gallery-summary').textContent='Loading saved videos…';const r=await fetch('/api/gallery',{cache:'no-store'}),j=await r.json();if(!j.ok)throw new Error(j.error||'Gallery unavailable');renderGallery(j.items||[]);}
  catch(e){$('#gallery-summary').textContent='Gallery unavailable';$('#gallery').innerHTML='<div class="empty-gallery"><span class="err">'+esc(e.message||e)+'</span></div>';}}
$('#gallery-refresh').onclick=loadGallery;

async function generate(){const engine=selectedEngine();let prompt;try{prompt=promptForRequest();}catch(error){$('#status').innerHTML='<span class="err">'+esc(error.message)+'</span>';return;}if(!prompt){$('#status').innerHTML='<span class="err">Enter a direction first.</span>';return;}if(!engine?.health?.ready){$('#status').innerHTML='<span class="err">'+esc(engine?.label||'The selected model')+' is not ready yet.</span>';return;}if($('#lock').checked&&!imageData){$('#status').innerHTML='<span class="err">Add a reference image for Same Face.</span>';return;}
  $('#go').disabled=true;$('#result').innerHTML='';t0=Date.now();tick();timer=setInterval(tick,1000);
  try{const payload={prompt,prompt_format:promptMode,engine_id:engine.id,seed:$('#s').value.trim(),duration_seconds:Number($('#duration').value),identity_lock:$('#lock').checked,...settingsPayload()};if(imageData)payload.image_b64=imageData;
    const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),j=await r.json();clearInterval(timer);
    if(j.ok){$('#status').textContent=j.engine_label+' finished in '+j.seconds+'s · seed '+j.used_seed;showResult($('#result'),j.file_url,j.used_seed,j.engine_label+' · '+j.mode);loadGallery();}
    else $('#status').innerHTML='<span class="err">'+esc(j.error||'No media returned')+'</span>';
  }catch(e){clearInterval(timer);$('#status').innerHTML='<span class="err">Request failed: '+esc(e)+'</span>';}$('#go').disabled=false;}
$('#go').onclick=generate;$('#p').addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')generate();});

let scenes=[{prompt:'A wide establishing shot introduces the character and location, calm anticipation, cinematic natural sound',duration:5},{prompt:'A closer moving shot as the character begins the main action, building energy and visual detail',duration:5},{prompt:'A striking final shot resolves the moment and holds on an expressive ending, cinematic finish',duration:5}];
const sceneDurations=()=>selectedEngine()?.controls?.durations?.map(Number)||[5];
function renderScenes(){const root=$('#scenes'),durations=sceneDurations();root.innerHTML='';scenes.forEach((scene,i)=>{const el=document.createElement('div');el.className='scene';
  el.innerHTML='<div class="scene-head"><b>Scene '+(i+1)+'</b>'+(scenes.length>2?'<button class="danger" type="button">Remove</button>':'')+'</div><div class="scene-grid"><textarea placeholder="What happens in this shot?"></textarea><div><label>Length</label><select>'+durations.map(n=>'<option value="'+n+'" '+(Number(scene.duration)===n?'selected':'')+'>'+n+' sec</option>').join('')+'</select></div></div>';
  const ta=el.querySelector('textarea'),sel=el.querySelector('select');ta.value=scene.prompt;ta.oninput=()=>scene.prompt=ta.value;sel.onchange=()=>scene.duration=Number(sel.value);const rm=el.querySelector('.danger');if(rm)rm.onclick=()=>{scenes.splice(i,1);renderScenes();};root.appendChild(el);});}
$('#add-scene').onclick=()=>{if(scenes.length>=8)return;scenes.push({prompt:'',duration:sceneDurations()[0]});renderScenes();};renderScenes();

async function pollLab(job){try{const r=await fetch('/api/lab/status?job='+encodeURIComponent(job)),j=await r.json();const elapsed=Math.round((Date.now()-t0)/1000),pct=Math.max(3,Math.round(100*(j.completed_scenes||0)/(j.total_scenes||1)));
  $('#lab-progress').hidden=false;$('#lab-progress span').style.width=(j.status==='done'?100:pct)+'%';
  if(j.status==='done'){clearInterval(labTimer);$('#lab-go').disabled=false;$('#lab-status').textContent='Movie complete in '+j.seconds+'s · '+j.total_scenes+' scenes';showResult($('#lab-result'),j.file_url,j.base_seed,(j.engine_label||'Model')+' · scene-lab movie',j.scene_urls);loadGallery();return;}
  if(j.status==='error'){clearInterval(labTimer);$('#lab-go').disabled=false;$('#lab-status').innerHTML='<span class="err">'+esc(j.error)+'</span>';return;}
  $('#lab-status').innerHTML='<span class="spin"></span> '+esc(j.stage)+' · '+elapsed+'s';
  }catch(e){clearInterval(labTimer);$('#lab-go').disabled=false;$('#lab-status').innerHTML='<span class="err">Status check failed: '+esc(e)+'</span>';}}
async function generateLab(){const engine=selectedEngine();if(!engine?.health?.ready){$('#lab-status').innerHTML='<span class="err">'+esc(engine?.label||'The selected model')+' is not ready yet.</span>';return;}if(scenes.some(x=>!x.prompt.trim())){$('#lab-status').innerHTML='<span class="err">Every scene needs a direction.</span>';return;}if($('#lab-lock').checked&&!imageData){$('#lab-status').innerHTML='<span class="err">Add a character image for Same Face.</span>';return;}
  $('#lab-go').disabled=true;$('#lab-result').innerHTML='';$('#lab-progress').hidden=false;$('#lab-progress span').style.width='2%';t0=Date.now();$('#lab-status').innerHTML='<span class="spin"></span> Starting scene lab…';
  const payload={story:$('#story').value.trim(),scenes,engine_id:engine.id,seed:$('#lab-seed').value.trim(),identity_lock:$('#lab-lock').checked,continuity:$('#continuity').checked,...settingsPayload()};if(imageData)payload.image_b64=imageData;
  try{const r=await fetch('/api/lab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),j=await r.json();if(!j.ok)throw new Error(j.error||'Could not start lab');await pollLab(j.job);labTimer=setInterval(()=>pollLab(j.job),1800);}
  catch(e){$('#lab-go').disabled=false;$('#lab-status').innerHTML='<span class="err">'+esc(e.message||e)+'</span>';}}
$('#lab-go').onclick=generateLab;
refreshEngines();loadGallery();setInterval(refreshEngines,10000);
</script></body></html>"""


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
