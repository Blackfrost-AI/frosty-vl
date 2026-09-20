import base64
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import qwen_image_serve as q


def encoded(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


@pytest.fixture
def rgba():
    return encoded(Image.new("RGBA", (256, 256), (20, 40, 60, 80)))


@pytest.mark.parametrize("updates", [
    {"prompt": " "}, {"width": 257}, {"height": 255}, {"width": 4096, "height": 4096},
    {"n": 5}, {"seed": -1}, {"num_inference_steps": 0}, {"true_cfg_scale": 11},
    {"negative_prompt": "blur"}, {"true_cfg_scale": 2}, {"mode": "edit"}, {"mode": "extract"},
    {"mode": "masked"}, {"mode": "annotate"}, {"mask_b64": "x"}, {"invented_parameter": True},
])
def test_invalid_controls_rejected(updates):
    with pytest.raises(ValidationError):
        q.ImageRequest(**({"prompt": "a ceramic fox"} | updates))


def test_alpha_preserved(rgba):
    image = q.decode_image(rgba)
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (20, 40, 60, 80)


def test_invalid_bytes_rejected():
    with pytest.raises(ValueError):
        q.decode_image("not an image")


def test_reference_order_and_limits(rgba):
    spec = q.ImageRequest(prompt="combine these", mode="edit", images_b64=[rgba] * 10)
    _, images, _ = q.prepare_inputs(spec)
    assert len(images) == 10
    with pytest.raises(ValidationError):
        q.ImageRequest(prompt="combine", mode="edit", images_b64=[rgba] * 11)
    with pytest.raises(ValidationError):
        q.ImageRequest(prompt="combine", mode="masked", images_b64=[rgba] * 10, mask_b64=rgba)


def test_mask_controls(rgba):
    base = dict(prompt="change to red", mode="masked", images_b64=[rgba])
    for mask in [Image.new("L", (256, 256), 0), Image.new("L", (128, 128), 255)]:
        with pytest.raises(ValueError):
            q.prepare_inputs(q.ImageRequest(**base, mask_b64=encoded(mask)))
    spec = q.ImageRequest(**base, mask_b64=encoded(Image.new("L", (256, 256), 255)))
    prompt, images, mask = q.prepare_inputs(spec)
    assert len(images) == 2 and mask.size == (256, 256)
    assert "last image" in prompt


@pytest.mark.parametrize("mode,needle", [("transparent", "alpha channel"), ("extract", "Extract"), ("annotate", "annotations")])
def test_mode_instructions(mode, needle, rgba):
    prompt, images, _ = q.prepare_inputs(q.ImageRequest(prompt="a fox", mode=mode, images_b64=[rgba]))
    assert needle in prompt and len(images) == 1


@pytest.fixture
def engine(tmp_path, monkeypatch):
    e = q.ImageEngine(output_dir=tmp_path)
    e.pipe = object()
    monkeypatch.setattr(q, "engine", e)
    return e


def test_http_queue_cancel_controls(engine, rgba):
    client = TestClient(q.app)
    assert client.get("/health").json()["ready"] is True
    assert client.post("/jobs", json={"prompt": "x", "mode": "edit", "images_b64": ["invalid"]}).status_code == 422
    jobs = [client.post("/jobs", json={"prompt": "a fox", "seed": i}) for i in range(4)]
    assert all(r.status_code == 202 for r in jobs)
    assert client.post("/jobs", json={"prompt": "fifth"}).status_code == 429
    job_id = jobs[0].json()["id"]
    assert client.post(f"/jobs/{job_id}/cancel").json()["cancel"] is True
    assert client.get("/jobs/unknown").status_code == 404
    assert client.get("/files/nonexistent.png").status_code == 404


def test_not_ready_fails_explicitly(engine):
    engine.pipe = None
    client = TestClient(q.app)
    assert client.get("/health").json()["ready"] is False
    assert client.post("/jobs", json={"prompt": "fox"}).status_code == 503


def test_saved_rgba_mask_preservation_and_seed(engine, rgba, monkeypatch):
    class Generator:
        def __init__(self, device): pass
        def manual_seed(self, seed): return seed
    fake_torch = SimpleNamespace(Generator=Generator)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []
    class Pipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            kwargs["callback_on_step_end"](self, 0, 1, {})
            return SimpleNamespace(images=[Image.new("RGBA", (256, 256), (255, 0, 0, 180))])
    engine.pipe = Pipeline()
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (128, 0, 256, 256))
    spec = q.ImageRequest(prompt="paint right half", mode="masked", images_b64=[rgba], mask_b64=encoded(mask),
                          seed=42, n=2, width=256, height=256, num_inference_steps=1)
    job = engine.submit(spec)
    args = engine.pending.get_nowait()
    engine._render(*args)
    result = engine.get(job["id"])
    assert result["status"] == "done" and len(result["outputs"]) == 2
    assert [c["generator"] for c in calls] == [42, 43]
    image = Image.open(engine.output / result["outputs"][0]["name"])
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (20, 40, 60, 80)
    assert image.getpixel((255, 0)) == (255, 0, 0, 180)
    metadata = json.loads((engine.output / (result["outputs"][0]["name"] + ".json")).read_text())
    assert metadata["seed"] == 42 and metadata["preserve_unmasked"] is True
    assert not list(engine.output.glob("*.partial"))


def test_gallery_png_and_external_symlink_rejected(tmp_path):
    module = importlib.util.spec_from_file_location("image_webui_test", ROOT / "ui/webui.py")
    ui = importlib.util.module_from_spec(module)
    module.loader.exec_module(ui)
    ui.GALLERY_DIR = tmp_path
    image = tmp_path / "saved.png"
    Image.new("RGBA", (8, 8)).save(image)
    assert ui._gallery_items()[0]["content_type"] == "image/png"
    (tmp_path / "link.png").symlink_to(image)
    assert ui._gallery_path("link.png") is None
    assert ui._gallery_path("file.txt") is None


def test_enhancer_requires_verified_weights(engine, monkeypatch):
    monkeypatch.setattr(q.qwen_prompt_enhance, "available", lambda _: {"t2i": False, "edit": False})
    client = TestClient(q.app)
    assert client.post("/enhance", json={"prompt": "a fox"}).status_code == 503
    assert engine.pending.empty()


def test_enhancer_releases_image_model_and_keeps_queue_ready(engine, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setattr(q.qwen_prompt_enhance, "available", lambda _: {"t2i": True, "edit": True})
    engine.initialized = True
    def rewrite(model_dir, prompt, images, seed, update, cancelled):
        assert engine.pipe is None
        assert prompt == "a fox" and not images and seed == 42
        update(stage="Enhancing prompt", generated_tokens=16)
        return {"prompt": "A copper fox in warm studio light", "wh_ratio": "1:1", "task": "t2i"}
    monkeypatch.setattr(q.qwen_prompt_enhance, "enhance", rewrite)
    job = engine.submit(q.ImageRequest(prompt="a fox", seed=42), kind="enhance")
    engine._enhance(*engine.pending.get_nowait())
    result = engine.get(job["id"])
    assert result["status"] == "done" and result["original_prompt"] == "a fox"
    assert result["enhancement"]["prompt"].startswith("A copper")
    assert engine.ready and engine.pipe is None
    assert engine.submit(q.ImageRequest(prompt="next image"))["status"] == "queued"


def test_official_enhancer_parsing_controls():
    from server.vendor.qwen_pe import pe_core as core
    profile = core.get_profile("t2i")
    valid = core.parse_answer('{"rewritten_prompt":"A copper fox", "wh_ratio":"1:1"}', profile)
    assert valid["parse_ok"] and valid["positive_prompt"] == "A copper fox"
    assert not core.parse_answer("unfinished answer", profile)["parse_ok"]
    assert core.get_profile("edit").presence_penalty == 0
    assert profile.presence_penalty == 1.5


def test_automatic_enhance_then_render_preserves_provenance(engine, monkeypatch):
    class Generator:
        def __init__(self, device): pass
        def manual_seed(self, seed): return seed
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=Generator,
        cuda=SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setattr(q.qwen_prompt_enhance, "available", lambda _: {"t2i": True, "edit": True})
    monkeypatch.setattr(q.qwen_prompt_enhance, "enhance", lambda *args:
                        {"prompt": "A wide copper fox portrait", "wh_ratio": "16:9", "task": "t2i"})
    calls = []
    class Pipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGBA", (kwargs["width"], kwargs["height"]))])
    engine.loader = Pipeline
    engine.initialized = True
    spec = q.ImageRequest(prompt="a fox", seed=4, width=512, height=512, num_inference_steps=1,
                          enhance_prompt=True, auto_aspect_ratio=True)
    job = engine.submit(spec)
    engine._render(*engine.pending.get_nowait())
    result = engine.get(job["id"])
    assert result["status"] == "done" and result["kind"] == "image"
    assert calls[0]["prompt"] == "A wide copper fox portrait"
    assert (calls[0]["width"], calls[0]["height"]) == (672, 384)
    meta = json.loads((engine.output / (result["outputs"][0]["name"] + ".json")).read_text())
    assert meta["prompt"] == "a fox" and meta["effective_prompt"] == "A wide copper fox portrait"
    assert meta["prompt_enhancement"]["task"] == "t2i"
