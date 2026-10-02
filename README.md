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

The dataset and prior HSI-description system have been inspected. Repository implementation, model smoke tests, the five-by-eleven run, and paper assets are being built in the documented gated order.
