# Execution Environment

Probe date: 2026-10-02 UTC.

## Hardware

- 4 × NVIDIA H200 NVL, 143,771 MiB each.
- NVIDIA driver 580.82.07; `nvidia-smi` reports CUDA compatibility 13.0.
- GPU 0 has a small unrelated process. Experiment scripts must not stop or replace it.
- Model/cache storage is configured outside Git through `model_cache_root`.

## Base software

- Ubuntu system Python 3.10.12.
- The system Python lacks `ensurepip`; the project environment is created with the user-installed `virtualenv` package.
- No TeX engine, Graphviz, or ImageMagick executable was initially available.

## Dependency policy

Audit, tests, and figure tooling are installable without vLLM. The GPU serving stack is pinned only after an H200 import and processor/config compatibility probe. Exact resolved direct versions are recorded in `requirements.lock`; complete transitive versions are captured in run manifests.

The initial resolved serving stack is PyTorch 2.13.0+cu130, Transformers 5.18.0, vLLM 0.30.0, Accelerate 1.15.0, and Hugging Face Hub 1.33.0. Import checks passed, `pip check` reported no broken requirements, CUDA was available, and PyTorch enumerated all four H200 GPUs.

## Cache policy

Run scripts derive `HF_HOME` from the ignored local configuration. No model weights or Hugging Face cache data may be stored in either Git repository.
