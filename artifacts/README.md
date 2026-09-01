# Release assets

Large model files are intentionally absent from ordinary Git. For jury review,
attach one of the following exact frozen assets:

- `adapter_model.safetensors`;
- the submitted `final_C_plus2_rationale.zip`;
- `checksums.sha256` generated from the exact uploaded bytes.

Never upload the Qwen base checkpoint. The runtime obtains it from
`$SHARED_MODELS_PATH/Qwen/Qwen3.5-4B`.

The submitted competition ZIP is the immutable release anchor. `make verify`
checks it directly; `make import-frozen` plus `make rebuild-submitted` must
produce the same archive SHA-256, not a newly normalized package.
