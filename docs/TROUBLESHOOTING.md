# Troubleshooting Frosty Studio

[Setup chooser](../GETTING-STARTED.md) · [Image setup](WINDOWS-IMAGE-SETUP.md) · [Video setup](VIDEO-SETUP.md) · [MCP](../MCP.md)

Start with the component that owns the error: browser/Studio, MCP companion,
render engine, or model files. Preserve the failed job's ID and error. Do not
reinstall every environment or redownload every model as the first response.

## Image setup and generation

| Symptom | Check and next step |
| --- | --- |
| `QwenImage21Pipeline` import fails | Use the interpreter from the Image environment and its pinned `requirements-image.txt`; older Diffusers builds may lack this pipeline |
| CUDA is unavailable | Check `nvidia-smi` and `torch.cuda.is_available()` using the same interpreter that starts the engine; verify driver and CUDA wheel compatibility |
| Engine health stays loading or shows an error | Read the engine terminal's first failure; verify the full local model directory, checkpoint revision and memory availability |
| Enhancer is “still downloading” | Its sibling folder must have the exact PE-T2I/I2I name, `system_prompt.txt`, and a UTF-8 `download-status.json` with `state: completed` written after checksum verification |
| Base-only setup fails when generating | Turn off the browser's default automatic enhancement checkbox until the appropriate enhancer is installed |
| Enhancement takes much longer than expected | On the 16 GB path, the engine unloads Image, loads PE, runs enhancement, then reloads Image for rendering; include checkpoint-switching time |
| CUDA out of memory | Inspect other GPU use; reduce output size, reference count/detail and variation count; disable reference context caching if necessary. Do not kill an unrelated workload |
| Editing rejects references | Use valid PNG/JPEG/WebP; at most ten inputs, or nine plus a mask; stay within the request size and pixel limits |
| Mask edit does not preserve the surrounding pixels | Enable the preservation option; the mask must match the first reference's dimensions. Preservation is compositing against the resized original |
| Negative prompt or guidance rejected | Negative prompts require true CFG above 1, and a true-CFG request must include the paired negative prompt; check the returned validation detail |
| Unexpected DWM strength | Check health and the active environment. Image DWM is optional; preserve existing defaults and omit MCP `dwm_scale` when you want the server default |

## Video and gallery

| Symptom | Check and next step |
| --- | --- |
| Video preflight cannot find direction files | They are not in the public clone; supply the separately provided licensed artifacts for that profile and verify their checksums |
| Qwen3-VL model fails to load as video | Check the full modular pipeline contract. A standalone chat/VL model is insufficient |
| Studio waits for a backend container | Inspect `docker compose ps` and server logs; the UI depends on backend health |
| Same Face is unavailable | The selected engine must advertise/support it, and the modular reference pipeline must be ready |
| Wan profile unexpectedly starts other models | The broad Compose profile command includes ordinary services. Use the explicit `frosty-wan` service when adding it to a prepared deployment |
| FFmpeg is missing | Set `FVL_FFMPEG` to an existing executable (`/usr/bin/ffmpeg` is the default). Scene assembly and match-cut extraction require it |
| Saved videos do not appear | Studio's `FVL_GALLERY_DIR` must see the backend's real output files. A URL alone does not mount a remote directory |
| Queued video will not start | One render runs per engine; inspect its active job and readiness. The overall outstanding limit is eight |
| Cancel appears to wait | A running video render finishes before cancellation stops later work. Completed clips remain saved |
| Jobs disappeared after restart | Job records/queues are memory-only. Check the persistent gallery for completed outputs; do not assume unfinished work resumed |
| Restore reports a conflict | It will not overwrite an existing file. Preserve both the active file and Trash item, and resolve the naming conflict deliberately |

Image and video use separate Trash storage. Restore through the relevant workspace
or MCP tool. Do not manually remove journals or metadata to hide a library error.
Neither workspace automatically purges Trash or expires old outputs.

## MCP connection

| Symptom | Check and next step |
| --- | --- |
| Client cannot start the companion | Use absolute interpreter and launcher paths; confirm the interpreter has `requirements-mcp.txt`. Windows JSON needs escaped backslashes |
| Launcher waits with no output | Stdio is a protocol transport; the client must start/connect to it. Use `--help` for a human-readable CLI check |
| Tools exist, but engine discovery fails | `FROSTY_STUDIO_URL` must reach Studio `8890`, not the engine port. Test `/api/engines` from the companion machine |
| HTTP 401 | Match the client's Bearer token to `FROSTY_MCP_TOKEN`; a configured token also protects loopback |
| HTTP listener refuses to bind | Use a specific private address and a token of at least 24 characters. Wildcard/public binding is rejected |
| Download link fails on another device | The returned Studio URL must be reachable from that device as well as from the companion |
| Local image path is rejected | Enable only the needed folders in `FROSTY_MCP_INPUT_DIRS`. Paths refer to the companion's computer, with `;` separators on Windows and `:` on Unix |
| Image URL is rejected as a reference | Arbitrary URL fetching is unsupported. Use a gallery ID, data URL or allowed local path |
| Job ID returned but there is no image/video | Poll `frosty_job` and read the terminal status/error. Submission is not completion |

## Network and port map

Native Image: engine `127.0.0.1:8899`, Studio `127.0.0.1:8890`.
Standalone video Compose also publishes backend `8899`; combined installs must
resolve that conflict. Optional MCP HTTP uses `8891/mcp`; stdio uses no listener.
Inside a container, loopback means that container, not the host or another service.

For remote use, verify both the server's chosen bind and the client's actual route
or tunnel. A VPN connection alone does not confirm permission to reach a particular
port. Do not “fix” a reachability issue by exposing an unauthenticated public port.

## Useful checks

Windows Image (PowerShell):

```powershell
nvidia-smi
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 8890,8891,8898,8899 }
Invoke-RestMethod http://127.0.0.1:8890/api/engines | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8890/api/images/health | ConvertTo-Json -Depth 8
```

Video Compose:

```bash
docker compose ps
docker compose logs --tail=200 frosty-vl-server
docker compose logs --tail=200 frosty-vl-studio
curl --fail http://127.0.0.1:8890/api/engines
```

For a support report, include commit/version, OS, GPU/driver, interpreter versions,
selected component, sanitized error, steps to reproduce and whether a disposable
small job succeeds. Remove tokens, private addresses, prompts, personal media and
sensitive paths before publishing logs. Do not attach `.env` or an unreviewed
`docker compose config` dump.
