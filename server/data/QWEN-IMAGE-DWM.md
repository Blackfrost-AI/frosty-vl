# Bundled Qwen Image 2.1 DWM runtime bank

Fresh Git clones and source ZIPs include the inference vectors needed by
Frosty's Image DWM control. No calibration dataset or capture run is needed.

| File | Purpose |
| --- | --- |
| [qwen-image21-dwm.safetensors](qwen-image21-dwm.safetensors) | The two runtime direction tensors; 1,180,160 bytes (about 1.2 MB) |
| [qwen-image21-dwm.json](qwen-image21-dwm.json) | Checksum, tensor layout, compatible revisions and example profile |

The bank contains only `direction.attention_output_last` and
`direction.mlp_output_last`, both FP32 with shape `[36, 4096]`. It was derived
from a 250-pair capture. Only the resulting vectors and small provenance metadata
are distributed; the pairs, per-pair activations and research dataset stay private.
Base-model weights must be obtained separately.

## Verify the bank

Run from the checkout root. PowerShell:

```powershell
$imageDwmManifest = Get-Content '.\server\data\qwen-image21-dwm.json' -Raw | ConvertFrom-Json
$imageDwmHash = (Get-FileHash '.\server\data\qwen-image21-dwm.safetensors' -Algorithm SHA256).Hash
if ($imageDwmHash -ne $imageDwmManifest.sha256) { throw 'Image DWM bank checksum mismatch' }
```

Linux/macOS, or any installed Python interpreter:

```sh
python -c 'import hashlib,json,pathlib; p=pathlib.Path("server/data/qwen-image21-dwm.safetensors"); m=json.loads(p.with_suffix(".json").read_text()); assert hashlib.sha256(p.read_bytes()).hexdigest()==m["sha256"], "Image DWM bank checksum mismatch"; print("Image DWM bank verified")'
```

Expected SHA-256:
`0f124108244a2ed289eb0f6db9fe9ece78125a1974f241f086933350457c8fef`.

## Enable the bundled profile

The [Windows setup launch block](../../docs/WINDOWS-IMAGE-SETUP.md#4-start-the-engine--terminal-a)
sets the following values. Preserve an existing operator profile when upgrading.
From the checkout root in the engine's terminal:

```powershell
$env:FVL_IMAGE_DWM_BANK = (Resolve-Path '.\server\data\qwen-image21-dwm.safetensors').Path
$env:FVL_IMAGE_DWM_LAYERS = '19-24'
$env:FVL_IMAGE_DWM_ATTN_ALPHA = '1.0'
$env:FVL_IMAGE_DWM_MLP_ALPHA = '1.0'
$env:FVL_IMAGE_DWM_DEFAULT_SCALE = '0.5'
```

For an existing native Linux Image installation:

```sh
export FVL_IMAGE_DWM_BANK="$PWD/server/data/qwen-image21-dwm.safetensors"
export FVL_IMAGE_DWM_LAYERS=19-24
export FVL_IMAGE_DWM_ATTN_ALPHA=1.0
export FVL_IMAGE_DWM_MLP_ALPHA=1.0
export FVL_IMAGE_DWM_DEFAULT_SCALE=0.5
```

Then start the engine with its normal model/output/quantization settings and one
worker. Environment changes apply at process startup. Drain active jobs before
restarting an existing engine; do not interrupt a running render.

Studio's `/api/images/health` should report `dwm.enabled: true`, layers 19–24,
and `dwm_default_scale: 0.5`. Reload Studio to read the profile. Each image's
sidecar records the profile and request strength.

Choose **0** in Studio for a clean baseline. To disable DWM hooks entirely,
clear `FVL_IMAGE_DWM_BANK` before starting the engine. DWM applies reversible
activation projections to the image model's text encoder; it never rewrites
model weights and does not alter the separate prompt-enhancer models.

## Compatibility and limits

The vectors were captured against Qwen Image 2.1 revision
`b3179ad355be050328e483a9dfdd9e60cd62adfa`. Revision
`790c92633540aa0cb11d9abf19eb46d861714758` has matching encoder configuration,
index and recorded shard hashes and was also verified with this bank.
The loader checks the bank's source revision and 36-layer, 4096-wide geometry.
Do not substitute directions from another model or bypass those checks.

The supplied profile is experimental. Live rendering verified that strength 0.5
changes output pixels and returning to strength 0 reproduces the clean result
with the same prompt and seed. This establishes operation, not an optimal
strength or a general improvement in image quality.

See the repository [distribution notice](../../LICENSE-NOTICE.md),
[LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) for existing terms.
