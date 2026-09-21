# Install Frosty Studio with an agent

This runbook is for an assistant setting up Frosty on behalf of its user. Read it
before acting. The same rules apply whether the assistant runs locally or over SSH.
[Project overview](README.md) · [Setup chooser](GETTING-STARTED.md) · [Repository rules](AGENTS.md)

## 1. Establish the requested installation

Reuse facts the user already supplied. Ask only for missing answers that affect
correctness; bundle them instead of repeating permission questions at each step.

| Decision | Information you need |
| --- | --- |
| Components | Image, video, MCP companion only, or a combination |
| Target | Actual host, OS, architecture and connection; verify hostname before changing it |
| Existing installation | Repository path, environment, configured engine URLs and current service ownership |
| Models | Existing local paths or the specific approved source/revision to download |
| Storage | Persistent model/output paths and enough free space |
| Access | Local browser, existing VPN/private network, or SSH tunnel |
| Agent client | MCP client, its configuration mechanism and the Studio URL it can reach |

An MCP-only request does not require choosing a model. An Image request does not
require video steering artifacts or two B200 GPUs. Do not guess a target from a
remembered IP address or stop an unrelated GPU workload to make room.

## 2. Inspect and report a short plan

Use read-only checks first:

```sh
git status --short
git rev-parse HEAD
python --version
nvidia-smi
```

On Windows, `py -3.12 --version` selects the documented Image interpreter. GPU
checks are unnecessary on an MCP-only computer. Inspect available disk space,
listening ports and the exact existing service/task/container when relevant.
For Docker video, also inspect `docker version` and `docker compose version`.

Report the selected guide, destination folders, engines/ports and outstanding
prerequisites. Continue with steps already authorized by the request. Ask before
unrequested large downloads, paid resources, service replacement or wider network
access. Do not accept a model license on the user's behalf.

## 3. Install only the selected path

### A. Qwen Image 2.1

Follow [docs/WINDOWS-IMAGE-SETUP.md](docs/WINDOWS-IMAGE-SETUP.md) in order.

- Create a dedicated Python 3.12 environment and install the documented CUDA
  wheels plus `requirements-image.txt`. Confirm CUDA before loading weights.
- Use the official base snapshot or the user's existing verified copy. The
  optional PE-T2I/I2I checkpoints live next to it with their exact folder names.
- Verify downloads at a pinned revision before writing `download-status.json`.
  Merely creating a completion marker does not verify any files.
- Set `FVL_IMAGE_MODEL_DIR`, `FVL_IMAGE_OUTPUT_DIR` and quantization explicitly.
  Preserve an existing DWM bank/layer/strength configuration; a new setup can run
  without an image DWM bank.
- Run one Uvicorn worker for the image engine. A second terminal runs Studio
  using `config/qwen-image-engines.json`. Never reuse the video Docker recipe.
- Disable automatic enhancement for the first base-only image test. Enable it
  only when the selected official enhancer is actually available.

For another OS or GPU, treat this as a port requiring validation, not a tested
copy of the Windows profile. Check current upstream driver/runtime compatibility
before changing the recipe.

### B. Video

Follow [docs/VIDEO-SETUP.md](docs/VIDEO-SETUP.md).

- Confirm the complete modular diffusion pipeline, component layout, compatible
  text encoder and loader imports. Do not pass a standalone Qwen3-VL chat model.
- Check separately provided video direction artifacts against their recorded
  shapes/checksums. They are absent from a public clone. Report missing files;
  do not fake them or disable existing steering settings to pass preflight.
- Configure persistent paths, explicit GPU IDs and loopback host-published ports.
- Build and run the no-load preflight before starting the selected services.
  Preflight does not prove a render will fit or finish.
- Wait for the required readiness fields. Identity mode requires its reference
  pipeline ready; Wan has different capabilities and its own checkpoint.
- Keep the shipped safety settings unless the user explicitly requests a change.

The existing Compose profile restarts services unless stopped and may start after
Docker/host restart. Explain that behavior before handing off. Creating additional
scheduled tasks or host services is a separate choice; do not silently add them.

### C. MCP companion only

Follow [MCP.md](MCP.md).

- Create `.venv-mcp`, install only `requirements-mcp.txt` and run launcher `--help`.
- Configure the client's actual Python and launcher **absolute** paths; do not
  leave `/absolute/path` or `C:\path` placeholders in a live configuration.
- Set `FROSTY_STUDIO_URL` to Studio, normally port `8890`. From another machine,
  loopback refers to that other machine; use the existing tunnel/VPN as appropriate.
- Prefer stdio for a local client. A stdio process can appear to wait silently;
  it is waiting for protocol input, not displaying an interactive shell.
- Optional HTTP uses `/mcp`, a specific private address, and a bearer token for
  non-loopback access. Store tokens in the client's secret mechanism.
- Keep local-file reads disabled unless needed. `FROSTY_MCP_INPUT_DIRS` should
  contain only approved input folders. These paths belong to the companion host.

## 4. Connect workspaces and route access

Use [GETTING-STARTED.md](GETTING-STARTED.md#combined-workspaces) for a combined
Studio. Engine configuration is a description of already running backends, not a
model scheduler. Reserve distinct backend ports and sufficient GPU resources.

Native defaults: Studio `8890`, Image `8899`, example video `8898`, optional MCP
HTTP `8891`. The standalone video Compose profile uses backend `8899`; resolve
that conflict before combining it with Image. Do not edit an active configuration
until you understand the running process and restart implications.

Keep raw APIs on loopback or an isolated container network. For native video,
explicitly set `FVL_HOST=127.0.0.1`; the shell launcher's fallback is a wildcard.
Docker may listen on `0.0.0.0` inside its container while publishing only host
loopback. Set `FVL_API_BIND`/`FVL_UI_BIND` for host publication, not `FVL_UI_HOST`.

The browser Studio has no built-in authentication. A private bind still gives
reachable peers access to prompts, generation, downloads and Trash actions. Use
the user's existing private access controls or SSH tunnel. MCP authentication
protects the companion only. Do not advertise an unauthenticated public URL.

## 5. Verify the actual user path

Record a result for each applicable row. Small generation probes use real GPU
resources, so keep them within the user's requested installation scope.

| Check | Required evidence |
| --- | --- |
| Source/config | Commit, interpreter/runtime versions, selected engine profile, paths and port map |
| Engine health | Expected model identity and ready state; explain any missing optional component |
| Studio | `/image` or `/video` loads; `/api/engines` shows the intended backends |
| Image | One small job reaches `done`; PNG opens/downloads; chosen reference/enhancer modes tested if installed |
| Video | One short supported job reaches `done`; MP4 opens; Scene Lab/identity only claimed if exercised |
| MCP | The actual client discovers 12 tools, calls `frosty_status` and can poll/fetch an authorized job |
| Network | Works from the intended client, not merely from the server itself |
| Persistence | Saved test output and metadata remain in the configured directory |
| Boundaries | Missing model, unsupported feature, skipped test and unverified capacity explicitly reported |

Use disposable output for any Trash/restore check, and verify it restores. Never
use the user's existing collection as test data. Protocol tests with fixtures do
not qualify a real model, benchmark speed, or establish GPU capacity.

## 6. Handoff and maintenance

Give the user a concise report with:

- installed components and source commit;
- exact browser URL, MCP transport/config location and configured engine IDs;
- model, output and virtual-environment paths, with credentials omitted;
- how to start, stop and inspect the **specific** processes or containers;
- successful checks and missing/untested optional features;
- backup/update procedure and the fact that in-memory queues do not survive restart.

Use [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#updating-an-installation) for upgrades.
Preserve galleries, Trash journals, private configuration, direction files and
source weights. Do not run the video updater as an Image/MCP installer. Do not
claim completion when only source files were copied or a listener appeared.
