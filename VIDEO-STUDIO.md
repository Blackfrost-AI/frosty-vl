# Frosty VL video workspace

Open `/video` in Frosty Studio. Image and video share navigation, job-oriented
workflows and recoverable galleries. See [deployment](CUSTOMER-INSTALL.md) for an
existing VL engine, [Image Studio](IMAGE-STUDIO.md) for Qwen Image, and
[MCP setup](MCP.md) for agents. No model downloads are needed to run the companion.

## Create

Choose a video engine. Studio shows its readiness, capabilities and supported
controls. Write a natural-language direction or switch to structured JSON; the
editable JSON template includes subject, environment, camera, lighting and audio.
The **Shot builder** adds camera, light, motion and sound instructions to your
prompt. It is a local writing aid, not a separate AI prompt-enhancement model.

Upload, paste or drop one reference image. When Image Studio is configured,
**Choose from Frosty Image** brings a saved image into the video composer as an
opening frame. Same Face is available only on engines that support identity
references. Images and Same Face are not silently emulated on other engines.

Set the duration and optional seed. Settings reveal supported resolution, frame,
steps, prompt strength and negative prompt controls. Quality presets adjust steps
relative to the engine default: Draft 60%, Standard 100%, High 150%, clamped to its
limits. They are convenience settings, not a promise of speed or better quality.

**Queue clip** submits immediately. Keep editing or open Jobs while it runs.
Your direction, scene order and settings are saved in this browser. Reference
image bytes are not stored in browser drafts; reattach them after refreshing.

## Scene Lab

Build 2–8 scenes. Reorder, duplicate or remove shots and choose each duration.
Match-cut continuity uses the previous clip's final frame for the next scene.
Same Face, when supported, uses the original identity reference instead. Completed
clips are kept individually; the final movie is assembled with FFmpeg.

Scenes and clips share one FIFO queue per engine. Jobs shows queued/running states,
scene progress, failure details, cancellation and output links. Browser refresh
reconnects to watched job IDs while Studio remains running. Cancelling an active
movie waits for the current render, keeps finished clips and stops later scenes.
It does not interrupt GPU kernels or destroy a render process.

## Gallery

Search saved videos, download outputs, select one or many, and Delete to Trash.
Undo restores the most recent deletion. Trash supports later restoration with the
video and metadata kept together, and refuses to overwrite a conflicting file.
**Reuse settings** brings a saved clip's prompt, seed and supported controls back
to Create; original reference images must be reattached.

Point `FVL_GALLERY_DIR` at the render backend's shared output directory. The gallery
is rebuilt from saved files and metadata and survives Studio restart. Image and
video galleries use separate Trash directories. No permanent purge is provided.

## API and operating limits

- `POST /api/videos/jobs` accepts clip or scene requests and returns a job ID.
- `GET /api/videos/jobs` and `GET /api/videos/jobs/{id}` provide history/status.
- `POST /api/videos/jobs/{id}/cancel` requests cancellation.
- `GET /api/videos/gallery`, `/api/videos/gallery/trash` and `/api/videos/files/{name}` provide saved media.
- `POST /api/videos/gallery/trash` and `/api/videos/gallery/restore` take `{"ids":[...]}`.

Use the engine IDs and controls from `/api/engines`. A `request_id` reuses the
same video job when its normalized settings match; different settings with the
same ID are rejected. The limit is eight outstanding jobs across video engines,
with up to 100 job records. Job history and pending requests are memory-only;
Studio restart clears them. Saved outputs and Trash persist.

Legacy `/api/generate`, `/api/lab` and `/api/lab/status` remain available. The
blocking generation adapter can return HTTP 202 with a still-running job when
its wait expires; new clients should use `/api/videos/jobs` or MCP.

The 1.2 workflow is qualified with API, queue, DOM, MCP transport and synthetic
FFmpeg tests. Live VL inference still requires validation with your loaded model.
