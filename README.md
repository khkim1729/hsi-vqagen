# HSI-VQAGen

Reproducible feasibility study for generating visual question-answer pairs from HSI-derived RGB images and paired HSI-derived descriptions.

The experiment separates model architecture from input condition:

- **Multimodal condition:** HSI-derived RGB + description.
- **Text-only condition:** description only; no image payload is sent.

The approved study uses five fixed samples, seven unique open checkpoints, and eleven configurations. Four same-checkpoint pairs compare multimodal and text-only input while holding model weights fixed. Raw hyperspectral cubes are never sent to the VQA-generation models.

## Setup

```bash
python3 -m virtualenv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e .
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

Set the local paths in the ignored `configs/local_paths.yaml`. Keep model caches on high-capacity storage outside the repository.

## Workflow

```bash
.venv/bin/python -m hsi_vqagen.data.inspect_dataset --config configs/local_paths.yaml
scripts/run_smoke_tests.sh
scripts/run_feasibility.sh --config configs/experiments.yaml --samples configs/feasibility_samples.yaml --output outputs/feasibility --resume
.venv/bin/python scripts/generate_figures.py --outputs outputs/feasibility --artifacts artifacts --figures figures --paper-assets paper_assets
```

Full commands become available as their implementation tasks land. Generated model outputs, caches, local paths, datasets, and model weights are excluded from Git.

## Paper

The companion paper is titled *HSI-VQAGen: Comparing Vision-Language and Text-Only Models for Hyperspectral VQA Generation*. English and Korean entry points are maintained in the existing paper repository.

## Current status

The dataset and prior HSI-description system have been inspected. A preliminary
same-sample comparison has completed for Qwen3-VL-8B, InternVL3-8B,
Gemma-4-12B, and Mistral Small 3.1 under both input conditions. The Mistral
vLLM/Transformers incompatibility was reproduced and resolved with a pinned stack.

- [초기 4개 멀티모달 모델 비교 결과 (한국어)](docs/early_model_comparison_ko.md)
- [One-sample smoke-test execution record](docs/smoke_test_results.md)

The preliminary result is not the final eleven-configuration paper experiment.
Remaining checkpoints, blind human evaluation, figures, and paper assets continue
in the documented gated order.
