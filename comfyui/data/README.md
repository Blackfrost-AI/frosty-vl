# Frosty VL steering vectors

Frosty VL steering vectors are delivered to licensed customers, **not** included
in this public repo.

Place the following files here before using the node:

- `refusal_dir_L50.npy` — persona-steering (refusal-removal) direction
- `safety_dir_L50.npy` — golden safety-floor direction

Each is a `(5120,)` unit direction vector for a Qwen3-VL text encoder. Optional
`.json` sidecars carry extraction provenance. Without these vectors the node
will raise at load time.
