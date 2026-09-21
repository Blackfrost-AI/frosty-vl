# Qwen Image 2.1 Studio

Qwen Image 2.1 is the primary image workspace in Frosty Studio.
It uses the official Qwen Image 2.1 Diffusers pipeline and local weights. The
existing video engines and their model loaders remain separate.

## Workspaces

- **Create / edit automatically:** no references creates an image; adding references
  selects editing. The uploader stays visible, with paste/drop, reorder, replace,
  remove, preview and additive gallery selection. Up to ten PNG/JPEG/WebP inputs
  with alpha preserved; seven aspect ratios and 512/1K/2K output sizes.
- **Transparent:** RGBA generation and editing of transparent source layers.
- **Extract subject:** prompt-guided subject extraction to a transparent PNG.
- **Mask edit:** paint or upload a mask. White selects the edit region. With
  “Keep pixels outside the mask unchanged,” the result is composited over the
  resized original after generation; untouched pixels and alpha are preserved.
  The mask occupies one image slot: at most nine references plus one mask.
- **Annotate:** paint or circle an area and describe the desired edit. The
  marked reference is passed to the model, which is asked to remove the marks.

All outputs are PNG. The gallery persists in the output directory. It supports
download, additive reference reuse, and individual/bulk **Delete**. Deleted
photos and their metadata move to app-managed **Trash**, with **Undo** and
**Restore**, including after a restart. There is no permanent-delete operation
or automatic purge. Interrupted moves are journaled for recovery; restore
conflicts retain both files. Images without sidecars are supported; the image
library does not delete videos.

Steps, seeds, sequential variations, negative
prompts with true CFG, reference resolution, and prefix caching are exposed.
Jobs have progress, bounded queuing, cancellation, and reconnect after a page
refresh. Restarting the engine interrupts queued/running jobs; saved images remain.

Model capabilities are distinct from hardware capacity. Ten high-resolution
references or 2K images need more memory than a small single-image request. Use
compact reference detail or turn off reference caching when memory is tight.

## Prompt enhancement

**Enhance prompt** uses the official separate 9B checkpoints: PE-T2I for creation
and PE-I2I when references are present. It returns an editable preview; the
original direction stays unchanged until **Use this prompt**. A suggested
aspect ratio is optional and never applied automatically.

The browser also offers **Enhance automatically when generating** (enabled by
default, matching Qwen's early-access demo). This runs enhancement and rendering
as one queued job, displays the prompt used, and retains the original direction
in image metadata. Applying a reviewed preview turns automatic enhancement off
to avoid rewriting twice. **Let the enhancer choose the aspect ratio** is a
separate opt-in; otherwise the selected preset or custom dimensions are kept.
Custom dimensions accept multiples of 32, from 256 to 4096, up to 4.5 MP.

Keep `Qwen-Image-2.1-PE-T2I` and `Qwen-Image-2.1-PE-I2I` next to the image model
directory. Each checkpoint must include its own `system_prompt.txt` and a
verified download receipt (`download-status.json` with `state: completed`).
The adapter uses the official sampling profiles and parsing code, with NF4
weight loading. Model reasoning is not displayed or saved.

One GPU worker handles both jobs. Enhancement unloads the image pipeline,
loads the appropriate enhancer, returns the preview, then releases it. The
next image job reloads the pipeline. This avoids keeping both model families
resident in the 16 GB GPU; switching adds load time. Cancellation is checked
between generated tokens and after checkpoint loading.

## Native Windows/CUDA profile

The integration was developed against PyTorch 2.14.0 + CUDA 13.0,
torchvision 0.29.0, Python 3.12, and the versions in `requirements-image.txt`.
Choose PyTorch wheels supported by the installed NVIDIA driver.

```powershell
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-image.txt
```

Set these in the engine terminal, then run one Uvicorn worker:

```powershell
$env:FVL_IMAGE_MODEL_DIR = 'D:\Models\Qwen-Image-2.1'
$env:FVL_IMAGE_OUTPUT_DIR = 'D:\AI\FrostyImage21\outputs'
$env:FVL_IMAGE_QUANTIZATION = 'nf4'
python -m uvicorn server.qwen_image_serve:app --host 127.0.0.1 --port 8899 --workers 1
```

`nf4` loads both the text encoder and image transformer in 4-bit with BF16
computation. Source files are unchanged. `bf16-offload` is also available for
hosts with enough system memory; it is not the small-memory Windows default.
Weights load offline, without `trust_remote_code`. VAE tiles are 1024 with a
768 stride. Standard square 1K images decode in one pass; larger canvases use
overlapping tiles. The upstream 256-pixel tile default produced visible seams
in the deployment's comparison probe.

In a second terminal, point the Studio at the image engine profile:

```powershell
$env:FVL_ENGINES_FILE = 'config/qwen-image-engines.json'
$env:FVL_GALLERY_DIR = 'D:\AI\FrostyImage21\outputs'
$env:FVL_UI_HOST = '127.0.0.1'
$env:FVL_UI_PORT = '8890'
python ui/webui.py
```

For WireGuard access, replace the Studio host with the machine's WireGuard IP
and add a firewall rule restricted to that interface and authorized overlay
sources. Keep the engine on loopback. No public listener or authentication is
added by these scripts; access control is provided by the private overlay.

For both workspaces, adapt `config/engines.combined.example.json` and point
`FVL_ENGINES_FILE` to it. Use separate backend ports. `/image` and `/video` select
the workspace; `/` follows the configured default. Image API and gallery routes
always use the image backend, even when video is the default. `FVL_GALLERY_DIR`
continues to configure the video gallery. Run one image-engine process per
output folder; it owns the image files and `.frosty-trash` journal.

Both workspaces can coexist without loading both models on the same GPU.

SGLang's September 20 Qwen Image 2.1 recipe targets CUDA on Linux and has no
verified published container for this integration. Its smallest tested recipe
is an RTX 4090 with 22.7 GiB request-phase VRAM. This profile uses Qwen's
recommended Diffusers pipeline with a native Windows NF4 loader instead.

## Image API

### Experimental Blackfrost image DWM

The image engine can load a model-specific 36x4096 direction bank and apply a
reversible activation projection to selected Qwen3-VL text-encoder attention
and MLP writer outputs. This is algebraically equivalent to the Frosty VL row
projection for bias-free writers, but is compatible with the NF4 runtime and
never edits or requantizes model weights.

Configure the profile at engine start with `FVL_IMAGE_DWM_BANK`,
`FVL_IMAGE_DWM_LAYERS`, `FVL_IMAGE_DWM_ATTN_ALPHA`, and
`FVL_IMAGE_DWM_MLP_ALPHA`. Requests may set `dwm_scale` from 0 (clean A/B
baseline) through 2. The default comes from `FVL_IMAGE_DWM_DEFAULT_SCALE` and
is 0 unless explicitly configured. Health and PNG sidecars record the active
profile and request scale.

The UI reads the enabled state and default from the image engine. DWM does not
alter the separate official prompt enhancers. Supply a matching direction bank;
no image DWM weights are bundled.

The browser-facing API accepts JSON at `POST /api/images/jobs`:

```json
{"prompt":"A ceramic blue fox on a warm white background","mode":"auto","width":512,"height":512,"num_inference_steps":40,"seed":42}
```

The HTTP 202 response contains a job `id`. Poll `GET /api/images/jobs/{id}`.
Cancel with `POST /api/images/jobs/{id}/cancel` and an empty JSON object.
Completed jobs include `outputs`, each with `name`, seed and dimensions.
Fetch images through `/api/images/files/{name}` and list them with
`GET /api/images/gallery`. `auto` is the default mode; it resolves to creation
or editing based on reference presence. Explicit legacy modes remain supported.

`POST /api/images/gallery/trash` accepts `{"ids":["image_<id>"]}` using IDs from
the gallery. Each result reports success/failure and its `trash_id`.
`GET /api/images/gallery/trash` lists recoverable items; restore them with
`POST /api/images/gallery/restore` and `{"ids":["<trash_id>"]}`. Repeated operations
are idempotent. The engine reports partial failures per item without discarding
successful results.

For editing, set `mode` to `edit`, `transparent`, `extract`, `masked`, or
`annotate`. Pass references as an ordered `images_b64` list of base64 PNG/JPEG/WebP
or data URLs. Masked editing additionally takes `mask_b64`, with the same size
as the first reference; the mask occupies one of the model's ten input slots.
`reference_resolution` accepts 256, 512 (default), or 1024 independently of output
dimensions. `n` runs one to four variations sequentially using consecutive seeds.

The image job API is asynchronous; it does not claim OpenAI Images API compatibility.
Raw engine routes are `/jobs`, `/jobs/{id}`, `/jobs/{id}/cancel`, `/files/{name}`,
`/health`, and `/v1/models`. Health distinguishes loading/error from readiness.

Prompt enhancement accepts the same request at `POST /api/images/enhance`
(engine `/enhance`), uses the same job polling/cancellation routes, and returns
an `enhancement` object containing `prompt`, `wh_ratio`, `ratio_follow`, `task`,
and model provenance. It does not automatically render an image or alter the
user's submitted direction. Health reports availability for each enhancer.
Image jobs accept `enhance_prompt: true` for automatic enhancement and optional
`auto_aspect_ratio: true` to adopt the enhancer's framing at approximately the
requested pixel area. Job results and PNG metadata retain the actual prompt used.

## Validation

```sh
python -m pytest -q tests/test_qwen_image.py
node --check ui/image_studio.js
```

Tests cover strict parameters, malformed inputs, transparency, masks, reference
limits, queue limits, cancellation, output persistence, and gallery confinement.
Real GPU generation still requires a live consumer-path check after installation.

Sources: [Qwen's release](https://github.com/QwenLM/Qwen-Image-2.1),
[Diffusers NF4 loading](https://huggingface.co/docs/diffusers/main/quantization/bitsandbytes),
[Windows bitsandbytes support](https://huggingface.co/docs/bitsandbytes/installation),
[SGLang recipe](https://docs.sglang.io/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1).
