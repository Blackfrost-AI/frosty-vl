# Developing and maintaining Frosty

[Overview](../README.md) · [Agent instructions](../AGENT-INSTRUCTIONS.md)

## Source map

| Area | Files |
| --- | --- |
| Studio router and proxies | `ui/webui.py` |
| Shared Frosty colors and brand | `ui/studio_theme.css` |
| Image browser | `ui/image_studio.html`, `.css`, `.js` |
| Video browser | `ui/video_studio.html`, `.css`, `.js` |
| Video FIFO and lifecycle | `ui/video_jobs.py` |
| Image engine/API | `server/qwen_image_serve.py` |
| Official PE adapter | `server/qwen_prompt_enhance.py`, `server/vendor/qwen_pe/` |
| Optional Image DWM | `server/image_dwm.py` |
| File libraries and Trash | `server/image_library.py`, `server/video_library.py` |
| Modular video loader/API | `server/pipeline_load.py`, `server/serve.py` |
| Wan adapter | `server/wan_serve.py` |
| MCP client, schemas and tools | `frosty_mcp/`, `scripts/start-mcp.py` |
| Engine examples | `config/engines*.json`, `config/qwen-image-engines.json` |
| Documentation and visuals | Root Markdown, `docs/`, `assets/` |

Keep engine-specific behavior behind capabilities and each engine's loader. Use
`ui/studio_theme.css` for shared color changes; preserve workspace-specific layouts.
The Studio has no frontend bundle/build requirement: its static HTML/CSS/JS is
served directly. Node dependencies are for UI tests.

## Checks without model weights

Use a disposable test environment outside the checkout. Example for macOS/Linux:

```bash
python3 -m venv ../frosty-test-venv
../frosty-test-venv/bin/python -m pip install -r requirements-mcp.txt
../frosty-test-venv/bin/python -m pip install pytest packaging fastapi httpx
../frosty-test-venv/bin/python -m pytest -q tests/test_mcp.py tests/test_video_jobs.py tests/test_video_studio.py
npm ci
npm test
```

Use a current Node release supported by `package-lock.json`. On Windows substitute
the environment's `Scripts/python.exe` for `bin/python`.

The full Python suite also requires PyTorch (CPU is sufficient for the tensor
checks) and NumPy. Install a platform-compatible build in that disposable test
environment, then run:

```bash
../frosty-test-venv/bin/python -m pip install torch numpy
../frosty-test-venv/bin/python -m pytest -q tests
```

FFmpeg enables the real synthetic clip extraction/assembly check; that test skips
when FFmpeg is absent. MCP tests exercise actual stdio and HTTP transports against
local fixtures. They are not GPU benchmarks. An isolated test run may create
scratch output directories, so never point test environment variables at a user's
real model service or collection.

For a smaller change, run the relevant checks. For documentation-only edits,
verify links/anchors, example JSON and shell syntax, filenames and image rendering.
There is no reason to start a GPU model simply to refresh the README.

## Updating an installation

1. Record the current commit and configuration. Back up private launchers, env
   files and the output library/Trash using your normal backup process.
2. Inspect local changes with `git status --short`. Commit or preserve your work;
   do not reset it to force an upgrade.
3. Review the incoming changes. Drain active jobs before restarting an affected
   process; in-memory queues do not survive restart.
4. Update a clean checkout with `git pull --ff-only origin main`. If it diverges,
   resolve the history rather than discarding local changes.
5. Install only changed dependencies in the matching environment. Restart only
   the affected component and repeat its small consumer-path check.

**Native Image:** retain your model/output paths and DWM configuration. Update
`requirements-image.txt` only in the Image environment when needed; do not run the
video Compose updater. Keep the previous source/env available for rollback.

**MCP:** update `requirements-mcp.txt` in `.venv-mcp` when needed, then let the client
restart the companion. Confirm discovery and `frosty_status` from that client.
This does not require restarting a GPU backend.

**Video Docker:** build, preflight and recreate only the intended services using
the video guide. The legacy `scripts/update.sh` automates more: it checks `VERSION`,
can pull source, rebuilds/restarts Compose services, and can sync a ComfyUI node
when `FVL_COMFYUI_DIR` is set. It is an operational action, not a read-only update
check. It may rebuild even when versions match, and same-version documentation
changes need an explicit source pull. Read it before use.

Weights, direction files and output libraries are not framework update targets.
Never delete them to force an update or use a model conversion as a routine fix.

## Reporting and contributing

Open an [issue](https://github.com/Blackfrost-AI/frosty-vl/issues) with the affected
component, expected/actual behavior, source commit and a minimal sanitized example.
For a change, describe the user-visible behavior and relevant checks. Separate
implemented features from roadmap ideas and tested hardware from estimates.

Do not commit secrets, personal media, model weights, local receipts, calibration
pairs, per-pair captures or licensed video direction files. The explicitly
published `server/data/qwen-image21-dwm.safetensors` inference bank and its
manifest are the narrow exception for image DWM vectors. Review screenshots for
private browser tabs and account details.
Read the repository's existing [LICENSE](../LICENSE), [NOTICE](../NOTICE), and
[distribution notice](../LICENSE-NOTICE.md) before contributing or redistributing.
