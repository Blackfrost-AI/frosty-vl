import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def studio(monkeypatch):
    spec = importlib.util.spec_from_file_location("routing_webui", Path(__file__).resolve().parents[1] / "ui/webui.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    engines = [{"id": "video", "base": "http://127.0.0.1:19001", "capabilities": ["text_to_video"]},
               {"id": "image", "base": "http://127.0.0.1:19002", "capabilities": ["text_to_image"]}]
    monkeypatch.setattr(ui, "ENGINES", engines)
    monkeypatch.setattr(ui, "ENGINES_BY_ID", {item["id"]: item for item in engines})
    monkeypatch.setattr(ui, "DEFAULT_ENGINE_ID", "video")
    calls = []
    def upstream(method, path, payload=None, timeout=None, base=None):
        calls.append((method, path, payload, base))
        return 200, json.dumps({"ready": True, "ok": True, "items": []}).encode()
    monkeypatch.setattr(ui, "http_json", upstream)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield ui, "http://127.0.0.1:" + str(server.server_port), calls
    server.shutdown()
    server.server_close()
    thread.join()


def test_image_and_video_routes_coexist_with_video_default(studio):
    _, base, calls = studio
    assert b"engine-select" in urllib.request.urlopen(base + "/video").read()
    assert b"references-section" in urllib.request.urlopen(base + "/image").read()
    assert b"references-section" not in urllib.request.urlopen(base + "/").read()
    assert json.load(urllib.request.urlopen(base + "/api/workspaces")) == {"image": True, "video": True}
    urllib.request.urlopen(base + "/api/images/health").read()
    assert calls[-1] == ("GET", "/health", None, "http://127.0.0.1:19002")
    urllib.request.urlopen(base + "/api/images/gallery").read()
    assert calls[-1][1] == "/gallery"


def test_library_post_proxy_and_origin_check(studio):
    _, base, calls = studio
    body = json.dumps({"ids": ["image_" + "a" * 32]}).encode()
    request = urllib.request.Request(base + "/api/images/gallery/trash", data=body,
                                     headers={"Content-Type": "application/json", "Origin": base})
    assert urllib.request.urlopen(request).status == 200
    assert calls[-1][1] == "/gallery/trash" and calls[-1][3].endswith(":19002")
    request.add_header("Origin", "http://unrelated.invalid")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 403


def test_unconfigured_workspace_is_explicit(studio, monkeypatch):
    ui, base, _ = studio
    monkeypatch.setattr(ui, "ENGINES", [ui.ENGINES[0]])
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "/image")
    assert error.value.code == 404


def test_legacy_image_gallery_uses_library_with_image_default(studio, monkeypatch):
    ui, base, calls = studio
    monkeypatch.setattr(ui, "DEFAULT_ENGINE_ID", "image")
    urllib.request.urlopen(base + "/api/gallery").read()
    assert calls[-1] == ("GET", "/gallery", None, "http://127.0.0.1:19002")
    monkeypatch.setattr(ui, "_gallery_items", lambda: [])
    previous = len(calls)
    urllib.request.urlopen(base + "/api/gallery?workspace=video").read()
    assert len(calls) == previous
