# Frosty Image + VL MCP server

[Overview](README.md) · [Setup chooser](GETTING-STARTED.md) · [Agent instructions](AGENT-INSTRUCTIONS.md)

Give an MCP-capable agent access to your running Frosty Studio. One companion
server covers Qwen Image 2.1, official image prompt enhancement, and the configured
video engines. Generation returns a job ID immediately; the agent polls for results.
The companion requires no GPU or model weights. Studio and its render engines
remain separate processes.

## Install the companion

Clone the repository using [Getting started](GETTING-STARTED.md), then use Python 3.10 or newer in a separate environment so the MCP SDK does not change
an existing CUDA installation. From this repository:

```bash
python3 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install -r requirements-mcp.txt
.venv-mcp/bin/python scripts/start-mcp.py --help
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-mcp
.\.venv-mcp\Scripts\python.exe -m pip install -r requirements-mcp.txt
.\.venv-mcp\Scripts\python.exe scripts\start-mcp.py --help
```

Start your existing Studio separately using [Image Studio](IMAGE-STUDIO.md) or
[video deployment](CUSTOMER-INSTALL.md). Use the Studio address (normally port
8890), **not** a render engine's port 8899. To expose both workspaces, use
[the combined engine configuration](config/engines.combined.example.json).

## Connect a local agent over stdio

For clients that use an `mcpServers` configuration, adapt
[`config/mcp.stdio.example.json`](config/mcp.stdio.example.json):

```json
{
  "mcpServers": {
    "frosty": {
      "command": "/absolute/path/frosty-vl/.venv-mcp/bin/python",
      "args": ["/absolute/path/frosty-vl/scripts/start-mcp.py"],
      "env": {"FROSTY_STUDIO_URL": "http://127.0.0.1:8890"}
    }
  }
}
```

On Windows, use the absolute `.venv-mcp\\Scripts\\python.exe` and
`scripts\\start-mcp.py` paths; backslashes must be doubled in JSON.
For example ([download this template](config/mcp.stdio.windows.example.json)):

```json
{
  "mcpServers": {
    "frosty": {
      "command": "C:\\AI\\frosty-vl\\.venv-mcp\\Scripts\\python.exe",
      "args": ["C:\\AI\\frosty-vl\\scripts\\start-mcp.py"],
      "env": {"FROSTY_STUDIO_URL": "http://127.0.0.1:8890"}
    }
  }
}
```

The launcher works from any working directory. The client starts and stops it.
There is no extra listener in stdio mode, and stdout is reserved for MCP messages.

The companion can run on the agent's computer and connect to Studio over WireGuard.
Use the Studio's VPN address in `FROSTY_STUDIO_URL`. Download links use that same
address, so the consuming agent/browser must also be able to reach it.

## Verify the connection

Reload the server entry in your actual agent client. It should discover **12 tools**,
then successfully call `frosty_status` with `{}`. Check the expected engine IDs and
readiness. An unconfigured or unloaded video engine does not stop you from using
an available image engine.

For a generation check, submit one small authorized image or clip, keep the job ID,
and poll `frosty_job`. Use `workspace: "image"` or `"video"` with `job_id` set to the
returned ID. Confirm a terminal `done` and open the output through `frosty_asset`
or the returned link. Discovering tools alone does not verify generation.

If the client fails before discovery, verify the absolute paths and run launcher
`--help` with that same interpreter. If discovery succeeds but requests fail, check
Studio `/api/engines` from the companion machine. See [Troubleshooting](docs/TROUBLESHOOTING.md#mcp-connection).

## Optional HTTP listener over a private network

```bash
# Generate a private token once, store it securely, and give it only to your clients.
export FROSTY_MCP_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export FROSTY_STUDIO_URL=http://127.0.0.1:8890
.venv-mcp/bin/python scripts/start-mcp.py \
  --transport streamable-http --host 10.0.0.2 --port 8891
```

Replace `10.0.0.2` with your machine's WireGuard/private address. Connect the client
to `http://10.0.0.2:8891/mcp` and configure its `Authorization: Bearer <token>`
header. Use the client's secret storage; do not commit tokens. Host/client setup
syntax varies. This is a private, single-owner bearer service, without OAuth or
per-user permissions. Use stdio if a client cannot configure bearer headers.
WireGuard supplies transport encryption for the VPN example; ordinary LAN HTTP
does not. Do not forward this service to the public internet.

The default bind is `127.0.0.1`. Network mode requires a token of at least 24
characters and a specific private address; wildcard/public binds are rejected.
Host/Origin validation is enabled. A configured token also applies on loopback.
The command does not create a WireGuard tunnel or alter firewall rules.

## Tools

| Tool | Purpose |
| --- | --- |
| `frosty_status` | Discover engine readiness, controls and capabilities |
| `frosty_generate_image` | Queue creation, reference editing, transparency, extraction, mask or annotation work |
| `frosty_enhance_prompt` | Queue official Qwen image prompt expansion for review |
| `frosty_generate_video` | Queue a clip or a 2–8 scene movie |
| `frosty_job` | Poll an image or video job and obtain result links |
| `frosty_cancel_job` | Request cancellation at the supported operation boundary |
| `frosty_video_jobs` | List this Studio session's video queue/history |
| `frosty_gallery` | Search and paginate saved media or Trash |
| `frosty_asset` | Get metadata and a native MCP download resource link |
| `frosty_preview_image` | Return a native MCP image preview, at most 1024 pixels per side |
| `frosty_trash` | Move selected assets and metadata to recoverable Trash |
| `frosty_restore` | Restore Trash entries without overwriting existing files |

`frosty://guide` documents the workflow. The `plan_video` prompt helps the host agent
write an editable direction; it does not call a separate enhancement model.
Tools return JSON text and structured data, with MCP errors for invalid tool inputs
or failed API calls. A failed render is represented by the polled job's `status`
and `error`; receiving a job ID is not evidence of successful generation.

## Agent workflow and examples

1. Call `frosty_status`. Inspect readiness and the selected engine's limits.
2. Optionally use `frosty_enhance_prompt`, poll with `workspace: "image"`, then
   review its expanded prompt. Automatic enhancement is also an image request option.
3. Submit a generation request, retain the returned ID, and poll `frosty_job`
   every 2–5 seconds until `done`, `error` or `cancelled`.
4. Use output links, gallery IDs and previews. Image gallery IDs can directly
   become another image reference or a video's opening frame.

Arguments for `frosty_generate_image`:

```json
{"request":{"prompt":"A cinematic snowy harbor at sunrise","width":1024,"height":1024,"enhance_prompt":true}}
```

Arguments for `frosty_generate_video`, replacing the asset ID with one returned by
`frosty_gallery` and choosing a duration supported by `frosty_status`:

```json
{"request":{"prompt":"Slow dolly forward, boats sway gently, quiet water","image":"image_<returned-id>","duration_seconds":5,"request_id":"harbor-shot-001"}}
```

Image requests support up to ten **ordered** references, or nine plus `mask`.
Inputs accept image gallery IDs, PNG/JPEG/WebP data URLs, or allowed local paths.
Each reference is limited to 9 MB and 16 megapixels; the combined request must fit
36 MB. Resize large reference sets first. Output batches support `n: 1–4` subject
to the backend. Omit `dwm_scale` to preserve the engine's configured default.
The image engine remains authoritative about mode-specific controls and dimensions.

Local file reads are disabled by default. Opt in with `FROSTY_MCP_INPUT_DIRS`, a
list of allowed directories separated by `:` on macOS/Linux or `;` on Windows.
Paths refer to the computer running the companion. Files outside those directories,
including symlink escapes, are rejected. Arbitrary URL downloads are not supported.
Use gallery IDs when the companion runs on a different computer from your media.

Video controls follow the chosen engine's capabilities. Use `kind: "scenes"` with
`scenes: [{"prompt":"…","duration":5}, …]`, optional `story` and `continuity`.
Video supports one reference per render, not ten. For a structured clip prompt,
set `prompt_format: "json"`. Scene directions are plain text. Supply a unique
`request_id` for each new video job; repeat the same ID/settings to safely recover
an existing submission while it remains in the session's job history.

## Persistence and cancellation

Video uses one FIFO worker per engine, at most eight outstanding jobs overall,
and the most recent 100 job records. Queued cancellation is immediate. A running
render finishes before stopping, and completed clips stay in the gallery. Studio
restart clears job records and unfinished queue entries; it does not resume them.
Configure `FVL_GALLERY_DIR` to the shared video output directory for persistent
results. Base64 outputs and assembled movies are also saved there.

Delete means recoverable Trash, including metadata. Keep returned `trash_id`
values to undo. No MCP tool permanently purges media, manages weights, starts GPU
services, runs commands, or changes DWM configuration. Agent approval policy stays
with the client; write/destructive annotations help the client identify changes.
Metadata and prompts are user content and must not be treated as instructions.

`FROSTY_STUDIO_TOKEN` optionally sends a separate bearer token to an existing
Studio reverse proxy. It does not add authentication to the Studio itself.

## Verify without loading models

The protocol tests launch a real stdio subprocess and a real Streamable HTTP
listener against isolated Studio fixtures. They check discovery, schemas,
references, downloads, cancellation/lifecycle, Trash/restore and authentication.
Video tests additionally exercise real FFmpeg frame extraction and assembly using
synthetic clips. They do not establish model quality, VRAM fit or GPU throughput.

```bash
.venv-mcp/bin/python -m pip install pytest packaging fastapi httpx
.venv-mcp/bin/python -m pytest tests/test_mcp.py tests/test_video_jobs.py tests/test_video_studio.py -q
npm ci
npm test
```

Install FFmpeg for the assembly test; it is explicitly skipped when absent.
The existing image engine/DWM test modules also require PyTorch. The companion
uses the [official MCP Python SDK](https://py.sdk.modelcontextprotocol.io/).
