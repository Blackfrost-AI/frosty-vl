---
license: apache-2.0
language:
  - en
tags:
  - comfyui
  - text-to-video
  - image-to-video
  - video-generation
  - qwen3-vl
  - text-encoder
  - persona-steering
pipeline_tag: text-to-video
library_name: comfyui
inference: true
---

<div align="center">

# Frosty VL — ComfyUI node

### *Persona-steering for Qwen3-VL text encoders, applied in memory at load. Bring your own model; on-disk weights are never modified.*

</div>

---

## What this is

**Frosty VL** is a ComfyUI persona-steering node for **Qwen3-VL text encoders**.
It applies a load-time, in-memory `project_rows` patch to the text encoder's
`self_attn.o_proj` (language layers 40–49) — refusal-removal followed by a golden
safety floor, lam 3.0. **This is the SAFE release: the safety floor is ON by
default.** You bring your own Qwen3-VL-based model; the on-disk weights are never
touched.

It is **not** a fine-tune and it ships **no model weights**. Two ways to use it:

- **Path A — pre-patched weights:** bake the steer into a drop-in text encoder
  with `tools/make_comfyui_text_encoder.py`, drop it into
  `ComfyUI/models/text_encoders/`, and select it in `CLIPLoader`
  (type `minimax` — the ComfyUI text-encoder type identifier, not product
  branding). No custom node.
- **Path B — load-time steering node:** the `FrostyVLPersonaSteering` custom node
  applies the steer in memory to *any* Qwen3-VL text encoder at load, keeping
  your weights pristine.

Both paths change **only** the text encoder's `self_attn.o_proj` (language layers
40–49); everything else in your pipeline is untouched.

---

## 🛡️ The safety floor — what it will NOT generate

**18+.** If a child or a person under 18 is in the frame, they stay innocent:

- ✅ **No exploitation of minors.**
- ✅ **No sexual imagery of children.**
- ✅ **Anything involving children or persons under 18 stays wholesome.**

The sexualized-underage axis is driven *below zero* in hidden space and held
there (including on held-out boundary prompts); wholesome-minor content is
untouched.

---

## 🚀 How to use

### Install the node pack

```
ComfyUI/custom_nodes/
└── frosty-vl/
```

The steering vectors are **not** included in this public repo (open-core — they
are delivered to licensed customers). Place `refusal_dir_L50.npy` and
`safety_dir_L50.npy` in `data/` before use (see `data/README.md`).

### Path A — pre-patched text encoder (drop-in, no custom node)

Produce the steered file from your own bf16 Qwen3-VL text encoder (streaming
tool; low RAM even on large upstream files):

```bash
python frosty-vl/tools/make_comfyui_text_encoder.py \
  --input  qwen3vl_text_encoder_bf16.safetensors \
  --output ComfyUI/models/text_encoders/qwen3vl_text_encoder_frosty_vl_bf16.safetensors
```

Then, in your workflow, set `CLIPLoader` → the produced filename, type `minimax`.

### Path B — load-time steering node (weights stay pristine)

Use your own Qwen3-VL text encoder and insert the custom node:

```
CLIPLoader (type=minimax) ──▶ FrostyVLPersonaSteering ──▶ (conditioning encode)
```

Widgets: `steer_refusal`, `steer_safety` (on/off), `lam` (default `3.0`),
`layer_band` (default `40-49`). The steer is applied in memory at load and
persists for the session.

### Example workflows

- `workflows/frosty_vl_t2v.json` — a Qwen3-VL text-to-video graph with the
  steering node wired in.
- `workflows/frosty_vl_t2v_reference.json` — a raw ComfyUI export of the same
  reference graph.

Drop one into ComfyUI, adjust file names / prompt / seed to match your own
engine artifacts, and queue.

---

## 📦 Repository layout

| Path | Contents |
|---|---|
| `README.md` | This card |
| `LICENSE` | License terms |
| `steering.py` | Shared `project_rows` steering math (single source of truth) |
| `data/` | Steering direction vectors (`.npy` + `.json` provenance) — delivered to licensed customers, not in this public repo |
| `nodes/frosty_vl_steering.py` | `FrostyVLPersonaSteering` ComfyUI node |
| `tools/make_comfyui_text_encoder.py` | Path A streaming weight producer |
| `tools/smoke_t2v.py` | End-to-end smoke test against a running ComfyUI |
| `tools/watch_t2v.py` | History watcher for a queued prompt |
| `workflows/frosty_vl_t2v.json` | Example T2V workflow |
| `workflows/frosty_vl_t2v_reference.json` | Reference ComfyUI export |

---

## 📋 Quick facts

- **Applies to:** any Qwen3-VL text encoder (5120-wide hidden), loaded via
  ComfyUI `CLIPLoader` (type `minimax`) or a raw transformers model
- **Build:** `project_rows` SUBTRACT on o_proj L40–49, refusal then golden
  safety, lam 3.0, blanket bf16, no re-quantization
- **Removed:** the refusal bias that gates explicit generations
- **Kept (baked in):** the golden safety floor — the only rule that stays
- **Delivery:** pre-patched drop-in weights **and** a load-time steering node
- **On-disk weights are never modified** — steering is a runtime, in-memory
  projection onto a copy of the text encoder

### What you will never get out of it
- ❌ A sexualized toddler, baby, child, or preteen
- ❌ A silently-censored generation of otherwise-permitted adult content

---

## ✅ Verification

- Steering math on real-shape `[5120,5120]` bf16 tensors matches the shared
  `project_rows` recipe (single source of truth in `steering.py`).
- The conversion tool is a verified streaming editor: only the 10 target
  `o_proj` tensors are touched; every other tensor is copied byte-for-byte; the
  output header re-reads cleanly.
- The `FrostyVLPersonaSteering` node imports cleanly in both ComfyUI
  module-loading modes and its built-in self-test passes.
- End-to-end ComfyUI generation was not executed in this workspace; run
  `workflows/frosty_vl_t2v.json` once locally to confirm before widespread use.

---

## 📜 License

See `LICENSE`. The steering node code carried in this repo is MIT; the steering
vectors are delivered separately to licensed customers.

*Made for the creators. Free where it should be free. Firm where it must be firm.*
