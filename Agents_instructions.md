# Frosty VL — Agent Installation Instructions

**Audience: an AI agent installing Frosty VL on behalf of its operator.**
Read this whole file first, then follow it top to bottom. Do not skip the questions in
Step 0 — they are required for correctness and licensing.

---

## What Frosty VL is

Frosty VL is a **weights-free persona-steering plugin** for **Qwen3-VL-based** diffusion
pipelines. It ships **no model weights.** It contains:

- two small direction vectors (`server/data/refusal_dir_L50.npy`, `safety_dir_L50.npy`),
- a projection wrapper that applies them **in memory at load time** to the pipeline's
  Qwen3-VL text encoder (the on-disk model is **never modified**),
- a FastAPI backend (`server/serve.py`) that serves text/image→video+audio and identity
  (same-person) generation,
- a browser Studio UI (`ui/webui.py`).

The operator brings their **own** Qwen3-VL-based base model. Frosty VL steers it; it does not
replace, modify, or redistribute it.

---

## Step 0 — REQUIRED: ask the operator (do not assume a model)

Before doing anything, ask the operator and record the answers:

1. **"Which Qwen3-VL-based diffusion model do you want to run Frosty VL on, and where are its
   weights (local path or a model repo you have access to)?"**
   - Do **not** assume or auto-download any specific model. Do **not** default to any
     particular vendor's model. Frosty VL works with any pipeline whose text encoder is
     **Qwen3-VL**; the operator chooses and supplies it.
   - The operator is responsible for obtaining and licensing that base model. Frosty VL ships
     no weights and does not fetch a model unless the operator explicitly provides a source
     they control.
2. **"Is this being installed on the same machine you'll use it from (local), or on a remote
   host / VM you'll connect to?"** — this decides serving and routing (Step 4).
3. **"If remote: how do you normally reach that host — a private network (VPN / overlay /
   private VPC subnet), or an SSH connection?"** — this decides the bind address (Step 4).

If the operator cannot name a Qwen3-VL-based model they have rights to, **stop** and tell them
Frosty VL needs a compatible base model they supply. Do not proceed with a guess.

---

## Step 1 — Scan the environment

Detect and report before installing:

- **OS / arch**, and whether you have `sudo`.
- **GPU**: `nvidia-smi` — model, count, free VRAM. A full BF16 Qwen3-VL video pipeline needs
  substantial VRAM (typically two large-VRAM GPUs). Note free memory per card.
- **Runtime**: is Docker + `docker compose` available (preferred), or must you use a Python
  venv? Check `docker version` / `docker compose version`.
- **Disk**: free space where the base model will live (a full BF16 pipeline can be several
  hundred GB).
- **Locality** (critical for Step 4): determine whether you are running **on the operator's
  own workstation** or **on a remote/VM host**. Signals: is there a local display/desktop
  session? Is this a cloud instance (metadata service, cloud-init, headless server)? Are you
  connected in over SSH? Report your conclusion: `LOCAL` or `REMOTE`.
- **If REMOTE**: enumerate the host's network interfaces (`ip -brief addr`). Identify any
  **private** interface the operator's machine can also reach — e.g. an overlay/VPN interface
  or a private subnet address (RFC1918 `10.x`/`172.16-31.x`/`192.168.x`, CGNAT `100.64.x`,
  or a VPN interface). Record that **private** address as `<PRIVATE_IP>`. Do **not** use or
  expose the host's public address.

---

## Step 2 — Confirm the architecture is Qwen3-VL (abort if not)

Point at the operator's model directory and confirm its **text encoder is Qwen3-VL** before
installing. Check the pipeline index / component configs (e.g. `modular_model_index.json` and
the `text_encoder/` config) for a Qwen3-VL text encoder (architecture such as
`Qwen3VLForConditionalGeneration`, model type `qwen3_vl`).

Run the shipped validator, which checks artifacts, imports, and GPU **without loading weights**:

```
python scripts/preflight.py          # or: bash scripts/container-preflight.sh
```

If the base model's text encoder is **not** Qwen3-VL, **stop** — Frosty VL's direction vectors
are defined in the Qwen3-VL text-encoder space and will not apply to another architecture. Tell
the operator their model is incompatible.

---

## Step 3 — Provide the base model + configure

1. Place the operator's Qwen3-VL pipeline at a persistent path (set as `FVL_MODEL_DIR`). If the
   operator gave a repo they control, `scripts/download-model.sh` can fetch it — it has **no
   default source**; the operator must set `FVL_MODEL_REPO`. Frosty VL never ships or assumes a model.
2. Copy the env template and edit paths/ports:
   ```
   cp env.docker.template .env        # docker-compose
   # or edit config/server.env(.example) and config/ui.env(.example) for the script path
   ```
   Key settings (safe release defaults — leave the safety floor ON):
   - `FVL_MODEL` / `FVL_MODEL_DIR` → the operator's base pipeline path
   - `FVL_ABL_LAM=3.0` (persona direction) and `FVL_SAFETY_LAM=3.0` (**golden-safety floor ON**)
   - Ports: backend `FVL_PORT=8899`, Studio `FVL_UI_PORT=8890`

---

## Step 4 — Serve and route (LOCAL vs REMOTE)

**Security invariants — always:**
- The **backend API** (`serve.py`, port 8899) stays bound to **`127.0.0.1`** (loopback). It is
  never exposed directly. The Studio UI proxies to it locally.
- Never bind anything to a **public** interface / `0.0.0.0` on a public IP without
  authentication in front of it.
- Never hardcode a public address. Use `localhost`, a detected `<PRIVATE_IP>`, or a tunnel.

### Case A — LOCAL (installed on the machine the operator uses)
- Bind the Studio UI to loopback: `FVL_UI_HOST=127.0.0.1`, `FVL_UI_PORT=8890`.
- The operator opens **`http://localhost:8890`**.
- Nothing is exposed to the network. Done.

### Case B — REMOTE, reached over a private network (VPN / overlay / private subnet)
- Bind the Studio UI to the **private** interface only:
  `FVL_UI_HOST=<PRIVATE_IP>` (the private address you found in Step 1), `FVL_UI_PORT=8890`.
- The operator opens **`http://<PRIVATE_IP>:8890`** from a machine on that same private network.
- Backend stays on `127.0.0.1:8899`. Do not bind the UI to the public interface.

### Case C — REMOTE, reached over SSH (no shared private network)
- Keep **both** backend and UI on loopback: `FVL_UI_HOST=127.0.0.1`.
- Have the operator open an SSH local-forward from **their** machine:
  ```
  ssh -L 8890:127.0.0.1:8890 <operator's ssh target>
  ```
  then browse **`http://localhost:8890`** on their machine. Nothing is exposed on the host.

> Routing rule of thumb: **loopback by default**; only widen to a **private** address the
> operator explicitly reaches, and prefer an **SSH tunnel** when unsure. Never a public bind.

### Start it
Docker (preferred):
```
bash scripts/container-preflight.sh     # validate
docker compose up -d --build            # backend + Studio
```
Or scripts directly:
```
bash scripts/start-server.sh            # backend on 127.0.0.1:8899
bash scripts/start-ui.sh                # Studio on FVL_UI_HOST:8890
```

---

## Step 5 — Verify

```
bash scripts/smoke-test.sh              # hits the backend API
```
Then open the Studio at the URL from Step 4 (`http://localhost:8890` or
`http://<PRIVATE_IP>:8890`). Confirm it loads and the backend health shows ready. The
identity (same-person / Ref2VA) pipeline may finish loading a few moments after first start.

---

## Privacy & safety rules (do not violate)

- **No address leakage.** This document and your setup must contain **no** hardcoded public or
  third-party IP addresses. Use `localhost`, the operator's own `<PRIVATE_IP>`, or a tunnel.
- **Backend private.** Port 8899 stays on loopback; the Studio (8890) is the only front door,
  and only on loopback or a private address.
- **Weights untouched.** Frosty VL projects its vectors in memory; it never writes to or
  redistributes the operator's base-model weights.
- **Keep the safety floor ON** (`FVL_SAFETY_LAM=3.0`) unless the operator directs otherwise.
- **Operator owns the model.** You did not supply the base model; the operator obtained and
  licensed it. Do not download or assume a model on their behalf.

---

## Quick reference

| Setting | Default | Meaning |
|---|---|---|
| `FVL_MODEL` / `FVL_MODEL_DIR` | — | operator's Qwen3-VL base pipeline path (required) |
| `FVL_PORT` | `8899` | backend API — keep on `127.0.0.1` |
| `FVL_UI_PORT` | `8890` | Studio UI |
| `FVL_UI_HOST` | `127.0.0.1` | UI bind — `127.0.0.1` (local/SSH) or `<PRIVATE_IP>` (private net) |
| `FVL_ABL_LAM` | `3.0` | persona-steering strength |
| `FVL_SAFETY_LAM` | `3.0` | golden-safety floor (keep ON) |
