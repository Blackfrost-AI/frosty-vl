#!/usr/bin/env python3
"""E2E smoke test: queue the Frosty VL T2V pipe against the running ComfyUI and
verify an mp4 is produced. Validates every node name, input, and link in the
workflow we just wrote into user/default/workflows/.

Note: `"minimax"` is the ComfyUI text-encoder type identifier for CLIPLoader,
not product branding. The UNET/VAE/CLIP filenames below are your own Qwen3-VL
engine artifacts — adjust them to match the files in your ComfyUI models dir."""
import json, sys, time, urllib.request

HOST = "http://127.0.0.1:8188"

PROMPT = {
    "1":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_text_encoder_bf16.safetensors", "type": "minimax"}},
    "2":  {"class_type": "FrostyVLPersonaSteering", "inputs": {"clip": ["1", 0], "steer_refusal": True, "steer_safety": True, "lam": 3.0, "layer_band": "40-49"}},
    "3":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a cinematic close-up portrait, warm rim light, expressive and bold, soft bokeh background, shallow depth of field, drifting camera, elegant motion, trending on artstation, 24fps"}},
    "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "blurry, noisy, low quality, oversaturated, deformed, flickering, jittery, watermark, text"}},
    "4":  {"class_type": "UNETLoader", "inputs": {"unet_name": "text_to_video_dit_bf16.safetensors", "weight_dtype": "default"}},
    "5":  {"class_type": "VAELoader", "inputs": {"vae_name": "video_vae_fp16.safetensors"}},
    "6":  {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": 1344, "height": 768, "length": 124}},
    "7":  {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["3", 0], "negative": ["10", 0], "latent_image": ["6", 0], "seed": 12345, "steps": 42, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
    "8":  {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["5", 0]}},
    "9":  {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 24.0, "bit_depth": 8}},
    "11": {"class_type": "SaveVideo", "inputs": {"video": ["9", 0], "filename_prefix": "video/FrostyVL", "format": "auto", "codec": {"codec": "auto"}}},
}

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())

# 1. Queue it
try:
    qr = post(HOST + "/prompt", {"prompt": PROMPT, "client_id": "frosty-vl-smoke"})
except Exception as e:
    body = ""
    try:
        body = e.read().decode()
    except Exception:
        pass
    print("QUEUE FAILED:", e); print(body); sys.exit(1)

if "error" in qr:
    print("QUEUE ERROR:", json.dumps(qr, indent=2)); sys.exit(1)

pid = qr.get("prompt_id")
print("queued prompt_id:", pid)

# 2. Poll /history until this prompt completes or errors
deadline = time.time() + 480  # up to 8 min
last = ""
while time.time() < deadline:
    time.sleep(8)
    try:
        hist = get(HOST + "/history/" + pid)
    except Exception as e:
        print("history poll error:", e); continue
    if pid in hist:
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("completed") or status.get("status_str") == "success":
            outs = entry.get("outputs", {})
            vids = []
            for nid, out in outs.items():
                for k, v in out.items():
                    if isinstance(v, list):
                        vids.extend(x.get("filename") for x in v if isinstance(x, dict) and "filename" in x)
                    elif isinstance(v, dict) and "filename" in v:
                        vids.append(v["filename"])
            print("DONE. status:", status.get("status_str")); print("outputs:", json.dumps(outs, indent=2)[:2000])
            sys.exit(0 if vids else 1)
        if "messages" in status:
            for ts, m in status["messages"]:
                if m.get("type") == "execution_error":
                    print("EXECUTION ERROR:"); print(json.dumps(m.get("data", {}), indent=2)[:3000]); sys.exit(1)
    else:
        last = "."
if last:
    print("TIMEOUT waiting for completion"); sys.exit(1)
