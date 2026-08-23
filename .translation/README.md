# Offline English Translation

This directory configures incremental Chinese-to-English translation for the
repository. Translation runs entirely inside a standard GitHub-hosted runner;
it does not call a paid translation API and does not require repository
secrets.

## Runtime

- Model: [`tencent/Hy-MT2-1.8B-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
- Quantization: `Q4_K_M` (about 1.13 GB)
- Inference: [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
- Model license: Apache-2.0

The model and inference binary are downloaded at workflow runtime and verified
against pinned SHA-256 digests. Neither file is committed to this repository.

## Safety Rules

- Existing hand-written files under `en/chXX` are not overwritten.
- Only Markdown files in configured source chapters are translated.
- Code fences, inline code, formulas, URLs, and HTML tags are protected.
- A translation is rejected if protected tokens are changed or removed.
- Generated changes are submitted as a pull request and are never auto-merged.
- Backfill is limited to a small batch per run.

`state.json` records the source content hash and target path. Unchanged files
are skipped, so routine runs only process new or modified documentation.
