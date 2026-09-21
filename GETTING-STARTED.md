# Start using Frosty Studio

[Project overview](README.md) · [Agent runbook](AGENT-INSTRUCTIONS.md)

Frosty has three independent installation paths. Start with one; add the others
when you need them. Clone the source before following any guide:

```sh
git clone https://github.com/Blackfrost-AI/frosty-vl.git
cd frosty-vl
```

All commands in the guides run from this repository root unless stated otherwise.
Use a persistent output directory and keep a record of your paths and engine ports.

## Pick a path

| I want to… | Follow | Prerequisites | Successful result |
| --- | --- | --- | --- |
| Create and edit with Qwen Image 2.1 on Windows | [Windows Image setup](docs/WINDOWS-IMAGE-SETUP.md) | Git, Python 3.12, NVIDIA driver/CUDA-compatible PyTorch, local model storage | `/image` renders and downloads a PNG |
| Let my assistant use an existing Studio | [MCP setup](MCP.md) | Python 3.10+, a reachable Studio URL, an MCP-capable client | Client lists 12 tools and `frosty_status` reports the expected engine |
| Run the existing modular video engine | [Video setup](docs/VIDEO-SETUP.md) | Linux NVIDIA host, Docker/Compose/Container Toolkit, a compatible complete video pipeline and separately supplied directions | Both pipeline health flags and a real clip succeed |
| Add Wan video | [Video setup](docs/VIDEO-SETUP.md#optional-wan-engine) | Its local checkpoint and enough GPU capacity for its configuration | Wan readiness and a supported clip succeed |
| Use Image and video together | [Combined workspaces](#combined-workspaces) | Each backend already works on its own | Both workspaces appear and route to their own engines |
| Ask an agent to install it | [Agent instructions](AGENT-INSTRUCTIONS.md) | The machine/connection and desired components | A verified setup report with URLs, paths and limits |

There is no all-in-one installer that downloads every model or chooses GPU
assignments for you. The existing Docker Compose stack is the **video** deployment,
not the native Windows Image environment. MCP can run on macOS, Linux or Windows
without a local GPU; this does not establish native model support on each OS.

## Before downloading

For Image, the tested snapshots contained about **33.1 GB** for the base model
and **37.7 GB** for both optional enhancers combined, in decimal units. Leave
additional room for the Python environment, download cache and outputs. These are
checkpoint sizes, not RAM or VRAM requirements. Inspect current file sizes when
choosing a different revision.

The Image NF4 path was tested with 16 GB VRAM on an RTX 4080. Large canvases and
many references can exceed capacity. Start with a single 512-pixel image and add
features incrementally. No minimum system-RAM guarantee has been established.

The existing modular video Compose profile is much larger: its preflight expects
two GPUs with at least 175 GiB each. Qwen Image and MCP do not need that topology.
A plain Qwen3-VL language/vision checkpoint is not a video diffusion pipeline.

## Combined workspaces

1. Verify each backend independently, including one small output.
2. Give them distinct ports and GPU allocations that fit your hardware. The
   example uses Image on loopback `8899` and video on loopback `8898`.
3. Copy `config/engines.combined.example.json` to your private configuration
   directory. Edit the engine IDs, addresses and capabilities to match reality.
4. Set `FVL_ENGINES_FILE` to that file before starting `ui/webui.py`.
5. Set `FVL_GALLERY_DIR` to the video backend's shared output folder. Image output
   storage remains `FVL_IMAGE_OUTPUT_DIR` on the image engine.
6. Open `/api/engines`, `/image` and `/video` on Studio. A menu entry alone does not
   establish backend readiness. Point MCP at the Studio URL on port `8890`.

If Studio runs in Docker, `127.0.0.1` means that container. Use addresses reachable
from inside it; the native combined example cannot be copied unchanged into Docker.
A remote video engine also needs a shared output strategy so gallery operations
see its files. Do not point two independent image workers at one output folder.

## Remote use

Start on loopback. For SSH access, run this on the client machine, replacing the
host and user with your actual SSH target:

```sh
ssh -N -L 8890:127.0.0.1:8890 user@your-host
```

Open `http://127.0.0.1:8890` on that client. For an existing WireGuard connection,
bind Studio to the host's actual WireGuard address and restrict access to intended
peers. The setup guides do not create a VPN or public firewall opening. Studio
and raw render APIs have no built-in authentication or tenant isolation.

MCP stdio needs no new port. Optional MCP HTTP on `8891` requires its own bearer
token for a non-loopback private bind. That token does not protect Studio or the
render engines; see [MCP networking](MCP.md#optional-http-listener-over-a-private-network).

## Let an agent help

Paste this into your assistant, supplying the bracketed values:

> Install Frosty Studio from https://github.com/Blackfrost-AI/frosty-vl on
> [machine/OS]. Read AGENTS.md and AGENT-INSTRUCTIONS.md first. I want
> [Image / video / MCP only / combined]. My existing Studio is [URL or none],
> models are [paths or download sources], and I connect via [local / VPN / SSH].
> Use [storage directory]. Inspect prerequisites, use isolated environments,
> preserve my current files and settings, and verify the browser or MCP client
> path. Report missing information before making dependent changes.

For a problem after setup, use [Troubleshooting](docs/TROUBLESHOOTING.md).
