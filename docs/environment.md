# Execution Environment

Probe date: 2026-10-02 UTC.

## Hardware

- 4 × NVIDIA H200 NVL, 143,771 MiB each.
- NVIDIA driver 580.82.07; `nvidia-smi` reports CUDA compatibility 13.0.
- GPU 0 has a small unrelated process. Experiment scripts must not stop or replace it.
- Model/cache storage is configured outside Git through `model_cache_root`.
- The active network interface negotiated 100 Mb/s. Model acquisition is therefore
  I/O-bound at approximately 11--12 MB/s even when Xet high-performance mode is used;
  concurrent downloads divide rather than increase the available bandwidth.

## Base software

- Ubuntu system Python 3.10.12.
- The system Python lacks `ensurepip`; the project environment is created with the user-installed `virtualenv` package.
- No TeX engine, Graphviz, or ImageMagick executable was initially available.

## Dependency policy

Audit, tests, and figure tooling are installable without vLLM. The GPU serving stack is pinned only after an H200 import and processor/config compatibility probe. Exact resolved direct versions are recorded in `requirements.lock`; complete transitive versions are captured in run manifests.

The initial resolver selected PyTorch 2.13.0+cu130, Transformers 5.18.0, vLLM
0.30.0, Accelerate 1.15.0, and Hugging Face Hub 1.33.0. Qwen, InternVL, and
Gemma smoke tests passed, but the Mistral/Pixtral import exposed an API mismatch:
vLLM imports `PixtralRotaryEmbedding`, which Transformers 5.18.0 renamed. The
reproducible serving environment is therefore pinned to Transformers 5.10.4 and
Tokenizers 0.22.2, the pair directly verified here to retain that class and satisfy
Transformers' declared tokenizer range. CUDA remains available and PyTorch
enumerates all four H200 GPUs.

Flash Attention is not installed as a separate `flash_attn` package. This is recorded
as an environment fact rather than treated as a failure: smoke tests determine whether
the PyTorch/vLLM kernels selected by each architecture are sufficient.

Pinned-revision configuration and processor probes passed for all seven checkpoints:

| Checkpoint | Config class | Processor/tokenizer class |
|---|---|---|
| Qwen3-VL-8B-Instruct | `Qwen3VLConfig` | `Qwen3VLProcessor` |
| InternVL3-8B | `InternVLChatConfig` | `Qwen2Tokenizer` (remote-code model path) |
| Gemma-4-12B-it | `Gemma4UnifiedConfig` | `Gemma4UnifiedProcessor` |
| Mistral Small 3.1 24B | `Mistral3Config` | `PixtralProcessor` |
| Qwen3-8B | `Qwen3Config` | `Qwen2Tokenizer` |
| Gemma-4-31B-it | `Gemma4Config` | `Gemma4Processor` |
| Qwen3-32B | `Qwen3Config` | `Qwen2Tokenizer` |

The Mistral probe emitted the upstream tokenizer-regex warning. Both C04 and C06
therefore set `fix_mistral_regex=true`, preserving the same preprocessing control in
the paired comparison.

## Cache policy

Run scripts use `/data/hsi-vqagen-cache/huggingface` for authentication metadata and
`/data/hsi-vqagen-cache/huggingface/hub` for model snapshots. No model weights or
Hugging Face cache data may be stored in either Git repository. The download script
pins revisions and excludes Mistral's duplicate sharded copy when using its
consolidated vLLM format.
