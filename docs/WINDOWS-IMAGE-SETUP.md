# Qwen Image 2.1 on Windows

[Start here](../GETTING-STARTED.md) · [Image feature guide](../IMAGE-STUDIO.md) · [Agent runbook](../AGENT-INSTRUCTIONS.md)

This is the native NVIDIA/CUDA Image profile. It starts **two processes**: the
image engine on loopback `8899`, and the browser Studio on loopback `8890`.
Optional MCP is a third, lightweight process. Docker and the video engine are not
needed. The documented NF4 profile was exercised on an RTX 4080 with 16 GB VRAM.

## 1. Prerequisites and folders

Install Git, Python 3.12 and an NVIDIA driver compatible with your chosen CUDA
PyTorch wheels. Use PowerShell. Check:

```powershell
git --version
py -3.12 --version
nvidia-smi
Get-PSDrive -PSProvider FileSystem
```

The example layout is `C:\AI\frosty-vl` for source, `C:\AI\frosty-image-venv` for
the runtime, `C:\Models` for weights, and `C:\AI\FrostyOutputs\images` for results.
Substitute your own persistent drive consistently. Keep output libraries and
weights outside the Git checkout.

The tested base snapshot is about 33.1 GB; both optional PE checkpoints add about
37.7 GB. Leave extra disk space for dependencies, cache, outputs and temporary
files. This storage estimate does not predict system RAM or peak VRAM use.

```powershell
New-Item -ItemType Directory -Force C:\AI, C:\Models, C:\AI\FrostyOutputs\images | Out-Null
Set-Location C:\AI
git clone https://github.com/Blackfrost-AI/frosty-vl.git
Set-Location C:\AI\frosty-vl
py -3.12 -m venv C:\AI\frosty-image-venv
$ImagePython = 'C:\AI\frosty-image-venv\Scripts\python.exe'
& $ImagePython -m pip install --upgrade pip
```

For an existing installation, inspect its files and environment first. Do not
clone over it, overwrite its configuration or replace its output folder.

## 2. Install the Image runtime

The tested profile uses PyTorch 2.14.0, torchvision 0.29.0 and CUDA 13.0 wheels.
Install them only with a compatible driver; verify current availability and driver
support in [PyTorch's installation guide](https://pytorch.org/get-started/locally/).
The pinned Image requirements include the Diffusers revision providing
`QwenImage21Pipeline`; a random older release may not contain it.

```powershell
& $ImagePython -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $ImagePython -m pip install -r requirements-image.txt
if ($LASTEXITCODE -ne 0) { throw 'Image dependencies failed' }
& $ImagePython -c "import torch; from diffusers import QwenImage21Pipeline; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

Do not install `requirements.txt` or `requirements-mcp.txt` into this environment.
If the checks fail, resolve the driver/wheel/import mismatch before loading models.

## 3. Download and verify the selected checkpoints

You may use an existing verified local copy. Otherwise, the official sources are:

| Component | Official source | Required? |
| --- | --- | --- |
| Base image pipeline | [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Yes |
| Creation prompt enhancer | [PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) | For creation enhancement |
| Reference-edit prompt enhancer | [PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) | For enhancement with references |

Review each model's terms at its source. These model files are not in the Frosty
Git repository. The following uses the exact revisions previously verified with
this integration, not an assertion that they will always be the latest versions.

Use a separate download environment so upgrading the Hub CLI cannot alter your
CUDA packages. Log in with `hf auth login` only if access requires it; never put a
token into the repository or the receipt.

```powershell
py -3.12 -m venv C:\AI\frosty-download-venv
C:\AI\frosty-download-venv\Scripts\python.exe -m pip install --upgrade huggingface_hub
$HubCli = 'C:\AI\frosty-download-venv\Scripts\hf.exe'
& $HubCli download --help
& $HubCli cache verify --help
```

This block downloads the base and **both optional enhancers**. Remove the two PE
entries for a base-only installation. It may transfer roughly 71 GB. Run it once
per destination; do not run simultaneous downloads into the same directory.

```powershell
$ModelRoot = 'C:\Models'
$Models = @(
    @{ Name = 'Qwen-Image-2.1'; Revision = 'b3179ad355be050328e483a9dfdd9e60cd62adfa' },
    @{ Name = 'Qwen-Image-2.1-PE-T2I'; Revision = 'f3ed7985c788ad75b3ab7223e0c4c51e2a43545b' },
    @{ Name = 'Qwen-Image-2.1-PE-I2I'; Revision = '72927bc08afc99b7888ceb7d7d51a12db3700bbd' }
)
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($Model in $Models) {
    $Repo = 'Qwen/' + $Model.Name
    $Destination = Join-Path $ModelRoot $Model.Name
    New-Item -ItemType Directory -Force $Destination | Out-Null
    $Receipt = Join-Path $Destination 'download-status.json'
    [IO.File]::WriteAllText($Receipt, '{"state":"verifying"}', $Utf8NoBom)
    & $HubCli download $Repo --revision $Model.Revision --local-dir $Destination
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Repo" }
    & $HubCli cache verify $Repo --revision $Model.Revision --local-dir $Destination --fail-on-missing-files
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $Repo" }
    if ($Model.Name -like '*-PE-*' -and -not (Test-Path (Join-Path $Destination 'system_prompt.txt'))) {
        throw "Missing enhancer system prompt: $Repo"
    }
    $Record = @{
        state = 'completed'; repo = $Repo; revision = $Model.Revision
        verification = 'hf cache verify --fail-on-missing-files'
        verified_at = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json
    [IO.File]::WriteAllText($Receipt, $Record, $Utf8NoBom)
}
```

[The Hub verification command](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-cache-verify)
checks file checksums. `--fail-on-missing-files` makes an incomplete snapshot fail.
The extra local `download-status.json` receipt is expected, so this recipe does
not reject extra files. A failed download or verification leaves a non-completed
receipt. Re-running uses the existing download state; it does not deliberately
purge the folder.

The final layout must be:

```text
C:\Models\
  Qwen-Image-2.1\
    model_index.json, text_encoder\, transformer\, vae\, ...
  Qwen-Image-2.1-PE-T2I\
    config.json, system_prompt.txt, download-status.json, ...
  Qwen-Image-2.1-PE-I2I\
    config.json, system_prompt.txt, download-status.json, ...
```

Do not manufacture `state: completed` to make a missing enhancer appear ready.
The app uses the marker as an availability gate; file verification happens during
setup, not every time the engine answers health.

## 4. Start the engine — terminal A

First verify the bundled bank against its [checksum manifest](../server/data/qwen-image21-dwm.json)
using the [PowerShell check](../server/data/QWEN-IMAGE-DWM.md#verify-the-bank).
The DWM values below are the experimental profile for a new install; preserve
your existing profile when upgrading.

```powershell
Set-Location C:\AI\frosty-vl
$env:FVL_IMAGE_MODEL_DIR = 'C:\Models\Qwen-Image-2.1'
$env:FVL_IMAGE_OUTPUT_DIR = 'C:\AI\FrostyOutputs\images'
$env:FVL_IMAGE_QUANTIZATION = 'nf4'
$env:FVL_IMAGE_DWM_BANK = (Resolve-Path '.\server\data\qwen-image21-dwm.safetensors').Path
$env:FVL_IMAGE_DWM_LAYERS = '19-24'
$env:FVL_IMAGE_DWM_ATTN_ALPHA = '1.0'
$env:FVL_IMAGE_DWM_MLP_ALPHA = '1.0'
$env:FVL_IMAGE_DWM_DEFAULT_SCALE = '0.5'
C:\AI\frosty-image-venv\Scripts\python.exe -m uvicorn server.qwen_image_serve:app --host 127.0.0.1 --port 8899 --workers 1
```

Leave this terminal open. Initial loading takes time. One worker owns the queue,
GPU pipeline, library and Trash journal. Do not use multiple Uvicorn workers or
point two engine processes at the same output directory.

Choose strength **0** in Studio for a clean baseline. No calibration pairs or
capture files are required to use the bank.

`nf4` quantizes the text encoder and transformer during loading with BF16 compute;
it does not rewrite source model files. `bf16-offload` is an alternative for
adequately provisioned hosts, not the verified small-memory Windows default.

## 5. Start Studio — terminal B

```powershell
Set-Location C:\AI\frosty-vl
$env:FVL_ENGINES_FILE = 'C:\AI\frosty-vl\config\qwen-image-engines.json'
$env:FVL_UI_HOST = '127.0.0.1'
$env:FVL_UI_PORT = '8890'
C:\AI\frosty-image-venv\Scripts\python.exe ui\webui.py
```

Open **http://127.0.0.1:8890/image**. Environment variables set in terminal A do
not automatically appear in terminal B; the examples set each process's needs.
`FVL_GALLERY_DIR` is for the separate video gallery. Image gallery requests go
through Studio to the engine's `FVL_IMAGE_OUTPUT_DIR`.

## 6. Verify a first image

In a third PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8899/health | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8890/api/engines | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8890/api/images/health | ConvertTo-Json -Depth 8
```

Wait for `ready: true` and check that the reported model is Qwen Image 2.1.
Then in the browser:

1. Choose Create/edit automatically with no references, 512 size and one variation.
2. Turn **Enhance automatically when generating** off for the first base-only test.
3. Use a short direction such as “A blue ceramic mug on a plain white table.”
4. Generate, wait for completion, and open/download the PNG. Check the gallery.
5. If enhancers were installed, test Enhance prompt, review its result, and apply
   it. Test reference editing separately with a small disposable image.

An HTTP/API alternative, still through Studio:

```powershell
$Body = @{ prompt = 'A blue ceramic mug on a plain white table'; mode = 'auto'; width = 512; height = 512; num_inference_steps = 20; seed = 42; enhance_prompt = $false } | ConvertTo-Json
$Job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8890/api/images/jobs -ContentType 'application/json' -Body $Body
$Job.id
Invoke-RestMethod ("http://127.0.0.1:8890/api/images/jobs/" + $Job.id) | ConvertTo-Json -Depth 8
```

Poll the last command until `done`, `error` or `cancelled`; inspect the result, not
just the initial job ID. See [Image API](../IMAGE-STUDIO.md#image-api) for outputs.

## 7. Optional additions

- **MCP:** follow [MCP.md](../MCP.md) in a separate environment. Point it at Studio
  `http://127.0.0.1:8890`, not engine `8899`.
- **Video:** first set up its backend independently, then follow
  [combined workspaces](../GETTING-STARTED.md#combined-workspaces).
- **WireGuard/SSH:** use [remote access](../GETTING-STARTED.md#remote-use). Native
  Studio binding is `FVL_UI_HOST`; the raw engine remains loopback. No network
  access rule or tunnel is created automatically.
- **Image DWM:** the runtime bank is included and the launch block above enables
  its profile. See the [bank guide](../server/data/QWEN-IMAGE-DWM.md) for verification,
  Linux settings and clean-mode operation. Preserve an existing operator profile.

## Stop, restart and update

Use Ctrl+C in each owned foreground terminal. Environment settings apply to the
process started from that terminal; closing PowerShell does not create a service
or persistent configuration. Re-run the relevant blocks to restart. Saved images
and Trash remain on disk; queued/running jobs do not resume after restart.

After installation is verified, you can save the same commands in private launcher
scripts outside the checkout. Automatic startup is optional. For upgrades, follow
[the update procedure](DEVELOPMENT.md#updating-an-installation), preserve your paths
and DWM settings, and rerun a small consumer test. The video `scripts/update.sh`
is not an Image installer.
