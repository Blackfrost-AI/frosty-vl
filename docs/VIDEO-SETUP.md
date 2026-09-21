# Install a Frosty video backend

[Setup chooser](../GETTING-STARTED.md) · [Video workspace](../VIDEO-STUDIO.md) · [MCP](../MCP.md)

The existing Docker Compose deployment is a specific modular video profile. Image
has its own [native Windows setup](WINDOWS-IMAGE-SETUP.md). MCP can use an already
running Studio without installing any video model.

## Compatibility and prerequisites

The modular loader uses Diffusers `ModularPipeline`, `MiniMaxH3Ref2VABlocks` and
`MiniMaxH3Reference`. It expects the corresponding full diffusion pipeline layout,
including a compatible Qwen3-VL text encoder. A text-encoder family name alone is
not sufficient: arbitrary Qwen3-VL chat/VL checkpoints are **not** supported video
models. Inspect [pipeline_load.py](../server/pipeline_load.py) and the model index
before committing to a download. The user supplies the compatible base model.

Required components checked by `scripts/preflight.py`:

```text
modular_model_index.json
text_encoder/      tokenizer/       processor/
transformer/       transformer_ref/
vae/               audio_vae/
scheduler/         audio_scheduler/
```

The supplied topology expects Linux x86_64, two B200-class NVIDIA GPUs with at
least 175 GiB exposed per card, Docker Engine, Compose v2 and NVIDIA Container
Toolkit. Allow storage for the complete pipeline, container layers and outputs.
A full BF16 pipeline can occupy hundreds of GB. These are requirements of this
profile, not of Image, MCP, or every possible video backend.

### Separately supplied direction files

The public Git clone **does not include** the `.npy` direction artifacts.
Licensed deployment bundles may provide them separately. The default video
configuration expects both in `server/data/` before the image is built:

- `refusal_dir_L50.npy`
- `safety_dir_L50.npy`

Check the shape `(5120,)`, `float32` dtype and expected checksums in
[server/data/README.md](../server/data/README.md). Never download a random vector,
use Image's differently shaped DWM bank, or generate a placeholder. If you lack
the artifacts for the requested profile, report that prerequisite. The default
safety strength is `FVL_SAFETY_LAM=3.0`; preserve the user's established settings.

## 1. Provide your model and persistent storage

From the repository root, create a private `.env` from the supplied template:

```bash
test -e .env || cp env.docker.template .env
mkdir -p /srv/frosty-vl/outputs
```

Edit `.env` with absolute paths to your complete model and output directory:

```dotenv
FVL_MODEL_DIR=/srv/frosty-vl/models/base-pipeline
FVL_OUTPUT_DIR=/srv/frosty-vl/outputs
FVL_FL_GPU=0
FVL_REF_GPU=1
FVL_API_BIND=127.0.0.1
FVL_PORT=8899
FVL_UI_BIND=127.0.0.1
FVL_UI_PORT=8890
FVL_GEN_TIMEOUT=2400
FVL_ABL_LAM=3.0
FVL_SAFETY_LAM=3.0
```

Do not overwrite an existing `.env`. The model mount is read-only; outputs are
writable and shared with Studio. Keep unrelated data out of the output directory.
The optional `scripts/download-model.sh` helper requires your explicit
`FVL_MODEL_REPO` and `FVL_MODEL_DIR`; it chooses no model by default. A successful
file-tree check is not a checksum or full runtime compatibility test.

## 2. Build and preflight without loading weights

```bash
docker version
docker compose version
nvidia-smi
docker compose build
bash scripts/container-preflight.sh
```

The repository's Dockerfile records the existing runtime recipe, including its
Diffusers commit. Preflight checks required files, direction shape, imports and
GPU visibility/capacity. It does not validate every weight tensor or perform a
render. Resolve each failure before starting services. If changing the runtime,
check the current upstream model/runtime recipe instead of guessing new versions.

## 3. Start the selected services

```bash
docker compose up -d frosty-vl-server frosty-vl-studio
docker compose logs -f frosty-vl-server
```

FL2VA loads first, then the identity-reference pipeline warms on the second GPU.
Initialization can take a long time. Inspect:

```bash
curl --fail http://127.0.0.1:8899/health
curl --fail http://127.0.0.1:8890/api/engines
```

For the modular profile, wait for `ready: true` and `reference_ready: true` before
claiming both modes are ready. Studio starts after the backend container's health
check passes; it can still show identity warmup separately.

Open **http://127.0.0.1:8890/video**. The shipped `restart: unless-stopped` policy
can bring these containers back after Docker restarts. This is different from the
foreground Windows Image walkthrough; choose persistence deliberately.

## 4. Verify rendering through Studio

```bash
bash scripts/smoke-test.sh http://127.0.0.1:8899
```

This script checks service information, health and the API schema. **It does not
generate a clip.** Finish verification in the browser: select the backend, queue
one short supported clip, wait for `done`, then open/download the resulting MP4.
Test opening frames, identity and Scene Lab separately if you intend to use them.
A 1.2 workflow test with a synthetic backend is not a replacement for this check.

Default modular duration mappings at 24 fps:

| Selected | Frames | Actual duration |
| ---: | ---: | ---: |
| 5 s | 124 | 5.167 s |
| 8 s | 192 | 8.000 s |
| 10 s | 243 | 10.125 s |
| 12 s | 294 | 12.250 s |
| 14 s | 345 | 14.375 s |

The API uses a 2,400-second per-render proxy timeout by default. Long renders are
hardware-dependent. Video jobs have one FIFO worker per engine and at most eight
outstanding jobs overall. Cancellation of an active render waits for that render
to finish. Read [VIDEO-STUDIO.md](../VIDEO-STUDIO.md) for the complete behavior.

## Optional Wan engine

The optional service uses the local
[Wan 2.2 TI2V 5B Diffusers checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
and its own adapter, not the modular Qwen3-VL loader or directions. It exposes
text/image-to-video, 480p/720p presets and engine-specific settings. Native audio
and identity lock are not provided by this adapter.

For an existing modular Compose installation, set `FVL_WAN_MODEL_DIR` and a suitable
`FVL_WAN_GPU` in `.env`, verify capacity and start only the added service:

```bash
docker compose --profile wan up -d --build frosty-wan
```

Studio's Docker engine configuration already lists that optional backend. Recheck
`/api/engines` and generate a small supported clip. The unqualified command
`docker compose --profile wan up -d` starts the regular modular services too;
it is not a Wan-only installation shortcut.

For a separately provisioned native Wan environment, inspect
[config/wan.env.example](../config/wan.env.example) and
[scripts/start-wan-engine.sh](../scripts/start-wan-engine.sh). Set `FROST_MODEL`,
`FROST_VENV`, `FROST_DEVICE`, `FROST_OUTPUT_DIR`, `FROST_HOST=127.0.0.1` and an unused
`FROST_PORT`; then configure Studio to reach it. This repo has not established a
universal minimum VRAM or a Windows qualification for that native video path.

## Networking and combined Image/video setups

The raw API stays on host loopback; Docker's internal wildcard listener is behind
its port publication. For private remote access, change `FVL_UI_BIND` to your
actual VPN address or use an [SSH tunnel](../GETTING-STARTED.md#remote-use).
Studio and raw APIs have no built-in user authentication.

Native script users must explicitly set `FVL_HOST=127.0.0.1` for
`scripts/start-server.sh`: its fallback/example binds all interfaces.
Native Studio uses `FVL_UI_HOST`; Compose host publication uses `FVL_UI_BIND`.
Do not confuse the two.

For Image alongside video, resolve the default `8899` port conflict and use
[combined workspaces](../GETTING-STARTED.md#combined-workspaces). Container-to-host
addresses and shared video storage need configuration; the native example's
loopback addresses do not work unchanged inside Docker.

## Operations and preservation

```bash
docker compose ps
docker compose logs --tail=200 frosty-vl-server
docker compose logs --tail=200 frosty-vl-studio
```

For a planned stop, allow active jobs to finish, then `docker compose stop`.
`docker compose down` removes this stack's containers/network while leaving the
bind-mounted model and output directories. Never add volume-deletion or filesystem
cleanup commands as part of routine setup. Review pending jobs before any restart:
in-memory queues/history are lost, but saved MP4s, metadata and Trash persist.

Use [the update guide](DEVELOPMENT.md#updating-an-installation) and
[troubleshooting](TROUBLESHOOTING.md) for maintenance.
