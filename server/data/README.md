# Bundled direction artifacts

The customer container embeds these Qwen3-VL text-encoder direction artifacts
(the Frosty VL plugin's steering vectors):

| File | Purpose | SHA-256 |
|---|---|---|
| `refusal_dir_L50.npy` | Frosty VL persona-steering direction enabled by default | `8205cbc660a366fb93c717e6dd3a5f06354416be2e1a3c12439240d669523f2e` |
| `safety_dir_L50.npy` | Golden-safety floor direction, ENABLED in this release | `b88170c3342e8d925338227baa6c7bfd825025b3fcfabed1a524cab9c04fbb99` |

The JSON files preserve extraction metadata. Both vectors have shape `(5120,)`
and dtype `float32`.

`serve.py` projects the configured directions onto the Qwen3-VL text encoder
at `text_encoder.model.language_model.layers[40:50].self_attn.o_proj` in memory.
It never writes modified weights to the customer's base pipeline.

Both bundled L50 directions are valid 5120-wide Qwen3-VL text-encoder
directions. Their checksums are recorded above and are not claimed to match any
other build's provenance; that distinction is labeled explicitly rather than
papered over.

This is the **safe release**: `FVL_SAFETY_LAM=3.0` is the default, so the
golden-safety floor is ENABLED. Both the persona-steering and golden-safety
directions are projected at load time.
