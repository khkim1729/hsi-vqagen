# HSI-VQAGen Feasibility Study Design

**Date:** 2026-10-01  
**Paper:** `[BDOT_ICCE2027] HSI-VQAGen: Comparing Vision-Language and Text-Only Models for Hyperspectral VQA Generation`

## 1. Objective and Success Criteria

Build a reproducible feasibility study that compares multimodal and text-only VQA-generation conditions on the same five hyperspectral-derived samples. Model architecture and experimental input condition are separate concepts: a multimodal-capable checkpoint may be evaluated without an image.

The study is complete when:

1. The real shard layout and the existing HSI description-generation pipeline are documented from source code and artifacts.
2. All eleven experimental configurations successfully complete a one-sample smoke test with a shared validated output schema.
3. The same five samples are processed by all eleven configurations, yielding 55 inference cases.
4. Raw responses, normalized JSONL, run metadata, and manual-review worksheets are retained.
5. Publication-quality pipeline and qualitative-comparison figures are produced.
6. English and Korean paper entry points (`main.tex` and `main_ko.tex`) compile or have any local tool limitation documented.
7. P1 and P2 contain no credentials, large model files, dataset copies, or fabricated results/citations and are pushed to their existing remotes.

This feasibility study does not feed raw HSI cubes to any model. It compares RGB representations derived from HSI plus paired HSI-derived descriptions against the descriptions alone.

## 2. Verified Starting State

### Repositories

- P1: `/home/khkim/gnew_projects/01_hsi_vqa_gen_github/hsi-vqagen`
  - Existing clone with `origin=https://github.com/khkim1729/hsi-vqagen.git`.
  - Empty `main` branch at the start of the work.
- P2: `/home/khkim/gnew_projects/02_hsi_vqa_gen_overleaf/overleaf_hsi_vqagen`
  - Existing `main` clone, initially containing a minimal `main.tex`.
  - Existing remote includes user information and must be sanitized after credentials are moved to user-level storage.

### Dataset

The data root is `/data/jypark/neon40k_work/shards`. Read-only inspection established:

- 100 shard directories, each containing 100 rows in both `descriptions.jsonl` and `sft.jsonl`.
- 10,000 unique sample IDs and no duplicates.
- 10,000 non-empty paired descriptions.
- 10,000 present and readable RGB PNG files, all 384 x 384.
- 10,000 patch directories and metadata files.
- HSI shape metadata is 64 x 64 x 426 for every sample.
- No missing pair component or malformed JSON/PNG was found in the initial full scan.
- Description length is 425--1,326 characters, with a median of 750 characters.
- All samples record prompt set `core-config-prompts@2026-07-30` with hash `0e88f131955c0f41`.

The authoritative pairing records are the shard-level `sft.jsonl` and `descriptions.jsonl`. The audit tool must verify their consistency with patch-level artifacts instead of inferring paths from sample IDs alone.

### Runtime environment

- Four NVIDIA H200 NVL GPUs, each with approximately 143 GB memory.
- Driver 580.82.07 and driver-advertised CUDA 13.0 compatibility.
- GPU 0 has an unrelated process using approximately 818 MiB; it must not be stopped or modified.
- System Python is 3.10.12 and initially lacks PyTorch, Transformers, vLLM, Pillow, and related ML packages.
- Root storage has approximately 205 GB free; `/data` has approximately 8 TB free.
- Model weights and Hugging Face cache must therefore use a dedicated location under `/data`, not the repository or default home cache.
- No local TeX or Graphviz executable was found initially.

## 3. Verified Provenance of the Paired Descriptions

The two owner-supplied legacy references resolve to one repository plus its nested package:

- `/home/khkim/1_users/2_aurora/02_all_hsi_descriptive_system/Hyperspectral_Descriptive_System`
- `Hyperspectral_Descriptive_System/hsi_describer_v2`

Shard metadata additionally identifies `/data/jypark/CLI_Descriptive_System` as the batch system that generated the audited artifacts. Both codebases are read-only references for this project.

The realized 10,000-sample description pipeline was:

1. Load a NEON 426-band 64 x 64 reflectance patch with known wavelength grid and georeferencing.
2. Reject insufficiently valid patches before paid calls.
3. Render an HSI-derived RGB image.
4. Automatically select spectral indices, compute index arrays, and render monochrome index maps with a consistent direction and explicit no-data treatment.
5. Group selected indices by semantic catalog and run category-specific map agents. These agents see RGB plus only their relevant index maps.
6. Run spectral-angle K-medoids clustering in original reflectance space, with automatic elbow-based cluster-count selection. A cluster agent sees RGB, the cluster map, cluster spectra, proportions, and material candidates.
7. Acquire a separate real-time geographic/web context note where available.
8. Run catalog synthesis on the category-agent text outputs.
9. Run a reduce agent that sees RGB and text evidence from the preceding analyses and produces one Korean prose description.

For the audited production batch, pixel-level supervised SAM observation and Tetracorder material identification were explicitly disabled. The paper must not describe them as executed components of this dataset generation run. The pipeline used GPT-5.4-mini for catalog/cluster calls and GPT-5.4 for catalog synthesis/reduce, with web research outside the three Batch API waves.

## 4. Experiment Design

### Input conditions and terminology

- **Multimodal condition:** HSI-derived RGB image plus the paired description.
- **Text-only condition:** Paired description only. The RGB path is retained only as provenance and is never sent to the model.

The paper uses these condition names instead of treating `VLM` and `LLM` as mutually exclusive model classes. In particular, Gemma 4, Mistral Small 3.1, Qwen3-VL, and InternVL3 are multimodal-capable architectures that are also evaluated in a text-only condition.

All other controllable factors are held constant: sample IDs, prompt semantics, requested number of QA pairs, output schema, maximum generation length, and deterministic decoding where supported.

### Experimental configurations

| ID | Output directory | Checkpoint | Condition | Input |
|---|---|---|---|---|
| C01 | `qwen3_vl_8b_multimodal` | `Qwen/Qwen3-VL-8B-Instruct` | Multimodal | RGB + description |
| C02 | `internvl3_8b_multimodal` | `OpenGVLab/InternVL3-8B` | Multimodal | RGB + description |
| C03 | `gemma4_12b_multimodal` | `google/gemma-4-12B-it` | Multimodal | RGB + description |
| C04 | `mistral_small_3_1_24b_multimodal` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Multimodal | RGB + description |
| C05 | `qwen3_8b_text_only` | `Qwen/Qwen3-8B` | Text-only | Description only |
| C06 | `mistral_small_3_1_24b_text_only` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Text-only | Description only |
| C07 | `gemma4_31b_text_only` | `google/gemma-4-31B-it` | Text-only | Description only |
| C08 | `qwen3_32b_text_only` | `Qwen/Qwen3-32B` | Text-only | Description only |
| C09 | `gemma4_12b_text_only` | `google/gemma-4-12B-it` | Controlled text-only ablation | Description only |
| C10 | `qwen3_vl_8b_text_only` | `Qwen/Qwen3-VL-8B-Instruct` | Controlled text-only ablation | Description only |
| C11 | `internvl3_8b_text_only` | `OpenGVLab/InternVL3-8B` | Controlled text-only ablation | Description only |

The explicit list above contains seven unique checkpoints and eleven configurations because Gemma 4 12B, Mistral Small 3.1, Qwen3-VL 8B, and InternVL3 8B each appear under two input conditions. The paper reports checkpoint count and configuration count separately and must not call these eleven distinct models.

Mistral Small 3.1 remains the stronger-practical text-only selection rather than a parameter-matched 8B baseline. Its repository contains redundant weight formats, so only the format needed by the chosen serving path may be downloaded. C10 and C11 are added because the official Qwen3-VL documentation reports pure-text behavior and the official InternVL3 example explicitly supports a pure-text call without image tensors. Their inclusion in the full matrix is conditional on a real image-free smoke test.

### Controlled modality ablations

Four same-checkpoint pairs isolate image availability from model family and parameter count:

- C01 versus C10: `Qwen/Qwen3-VL-8B-Instruct`.
- C02 versus C11: `OpenGVLab/InternVL3-8B`.
- C03 versus C09: `google/gemma-4-12B-it`.
- C04 versus C06: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.

Each pair uses the exact same checkpoint revision, server process where practical, prompt semantics, requested four QA pairs, generation parameters, and output schema. The only intended difference is whether the HSI-derived RGB image is included. C03 versus C09 remains the primary preregistered Gemma analysis; the three additional pairs test whether the modality effect is consistent across architectures.

The ablation research question is:

> Does adding the HSI-derived RGB representation improve VQA generation when the underlying model is held fixed?

Its analysis covers correctness, grounding, hallucination, diversity, visual dependence, usefulness, and answer specificity. It also examines whether image input elicits details absent from the description, whether text-only questions collapse toward paraphrase, and whether image input changes hallucination frequency.

### Serving strategy

Use one vLLM OpenAI-compatible server per GPU as the primary path where the checkpoint is supported. This gives one request format and one client implementation across configurations. Each server uses a distinct localhost port and an explicit `CUDA_VISIBLE_DEVICES` assignment.

Four GPUs cannot host all seven checkpoints simultaneously, so execution occurs in documented waves. Up to four checkpoints are loaded concurrently, their assigned configuration(s) run, and then the next wave starts. C01/C10, C02/C11, C03/C09, and C04/C06 run back-to-back against the same loaded checkpoint where practical. GPU assignment belongs to run metadata rather than configuration identity.

Transformers direct inference is a per-checkpoint fallback when a documented vLLM incompatibility remains during the one-sample smoke test. This is especially relevant to the newer Gemma 4 12B Unified architecture, whose official model card currently documents Transformers loading but not a vLLM command, unlike Gemma 4 31B. If a fallback is used, the backend difference is recorded in every output and latency values are not directly compared without qualification. No checkpoint is silently replaced.

### Fixed feasibility samples

The authoritative manifest contains these five sample IDs in this order:

1. `NEON_D01_BART_DP3_312000_4875000_bidirectional_reflectance-468-532X660-724` -- forest canopy.
2. `NEON_D01_HARV_DP3_725000_4701000_bidirectional_reflectance-404-468X788-852` -- open-water/vegetation boundary.
3. `NEON_D06_KONZ_DP3_708000_4336000_bidirectional_reflectance-788-852X212-276` -- dry cropland.
4. `NEON_D14_SRER_DP3_506000_3520000_bidirectional_reflectance-788-852X532-596` -- shrub/bare surface with a bright linear feature.
5. `NEON_D19_DEJU_DP3_566000_7088000_bidirectional_reflectance-532-596X724-788` -- bright developed/exposed surface.

They were selected after inspecting their RGB images, descriptions, land-cover context, index diversity, and artifact completeness. Every model must reject a run manifest that changes or reorders this set unless an explicit new experiment ID is used.

## 5. Prompt and Output Contract

Each model generates exactly four QA pairs per sample. Questions must be answerable from the input available to that condition and should cover different categories where supported, such as content, attributes, spatial relations, scene interpretation, or description-grounded reasoning.

Prompts prohibit:

- unsupported wavelength, mineral, material, or geographic claims;
- references to inputs not supplied to the model;
- generic duplicates or simple paraphrases of the same fact;
- claiming that RGB is raw hyperspectral data.

Qwen3 text inference uses non-thinking mode so that hidden reasoning markers do not contaminate JSON. Thinking behavior is fixed explicitly and identically within every same-checkpoint pair. Generation uses the closest stable common strategy supported by all checkpoints; model-specific constraints, chat templates, image order, and preprocessing are captured rather than hidden. All configurations use the same maximum output budget.

One normalized JSONL line represents one QA pair and contains at least:

- `schema_version`, `experiment_id`, `case_id`, and `pair_index`;
- `sample_id`, shard, source hashes, and source paths;
- exact model ID, model revision, backend, input mode, and GPU;
- question, answer, question type, evidence, and grounded-input references;
- source description and RGB provenance path;
- prompt version, generation configuration, timing, and validation status.

Each configuration directory contains `raw/`, `normalized.jsonl`, `generation_config.json`, `run_manifest.json`, and `errors.jsonl`. The run manifest records checkpoint revision, prompt hash, load time, per-case latency, peak VRAM when measurable, backend version, and model-specific preprocessing/template settings.

Raw model responses are saved separately from normalized JSONL. A strict schema validator parses the response. On parse failure, one repair attempt may be made with the same model; the initial response and failure remain recorded. A second failure produces an explicit failed case instead of fabricated or hand-corrected data.

## 6. Repository Architecture

P1 will use focused modules with clear boundaries:

- `src/hsi_vqagen/data/`: shard discovery, dataset records, audit, and fixed sample manifest.
- `src/hsi_vqagen/schema/`: generation and evaluation data contracts.
- `src/hsi_vqagen/inference/`: shared vLLM client, model adapters, response normalization, smoke and feasibility orchestration.
- `src/hsi_vqagen/evaluation/`: deterministic checks, comparison tables, and manual-rating templates.
- `src/hsi_vqagen/figures/`: pipeline, description-pipeline, qualitative-grid, and rating-summary rendering.
- `configs/`: tracked model/run configs plus ignored `local_paths.yaml`.
- `prompts/`: versioned VLM, LLM, repair, and optional evaluation prompts.
- `scripts/`: auditable entry points for environment setup, model serving, smoke tests, complete feasibility, figures, and paper synchronization.
- `docs/`: required audit, pipeline, model-selection, sample-selection, environment, and experiment-log documents.
- `artifacts/`: small audit and tabular results suitable for Git.
- `outputs/`: ignored raw and normalized generation output by experiment/model.
- `figures/` and `paper_assets/`: versioned publication assets that do not expose private absolute paths.
- `tests/`: unit and integration tests using tiny fixtures and mocked serving responses.

The public README uses configurable placeholders instead of private server paths. `configs/local_paths.yaml`, model caches, virtual environments, raw outputs, and secrets are ignored.

## 7. Work Order and Gates

The implementation order is mandatory:

1. Configure user-level Git and Hugging Face authentication without modifying tracked files.
2. Implement and run the dataset audit; write JSON, CSV, and Markdown findings.
3. Document the verified description pipeline from legacy and production sources.
4. Document model selection, environment compatibility, and the fixed sample manifest.
5. Complete repository skeleton, shared schema, tests, prompts, and model configs.
6. Start each checkpoint individually and run all eleven one-sample configuration smoke tests, recording load time, peak VRAM, latency, revision, and failures.
7. Only after the smoke gate passes, run the 5 x 11 study in checkpoint waves across the four GPUs.
8. Validate and compare results; leave subjective manual ratings unfilled unless an actual reviewer supplies them.
9. Generate cross-model and Gemma 4 12B controlled-ablation reports from real outputs.
10. Generate publication figures and tables from real outputs.
11. Draft English and Korean papers with conservative claims and verified bibliography entries.
12. Run tests, schema checks, secret scans, repository-size checks, and available LaTeX validation.
13. Commit and push P1 and P2.

If one of the original nine configurations cannot pass its smoke test, the full study pauses. If exploratory C10 or C11 fails despite documented supported invocation attempts, the failure is retained as a compatibility result and the original 45-case matrix may proceed; neither checkpoint is silently replaced.

## 8. Evaluation and Figures

Deterministic checks cover schema validity, requested pair count, duplicate questions, empty answers/evidence, input-mode leakage, and unsupported spectral-claim indicators. These checks are diagnostics, not replacements for human judgment.

The manual worksheet covers grounding, correctness, hallucination, question diversity, visual dependence, downstream usefulness, and answer specificity. No rating is populated without actual review.

Publication assets include:

1. A vector end-to-end overview that clearly separates HSI-derived RGB from the raw HSI cube and splits multimodal and text-only conditions.
2. A vector diagram of the verified production description pipeline, explicitly showing disabled components only as out-of-scope notes if mentioned.
3. Figure A: RGB, description summary, and the four multimodal configurations for two or three representative samples in the main paper, with all five in the appendix.
4. Figure B: the same samples and the four original text-only baselines C05--C08, keeping labels and typography consistent with Figure A.
5. Figure C: same-checkpoint controlled-ablation panels for C01/C10, C02/C11, C03/C09, and C04/C06, with paired outputs and highlighted visual-only details or paraphrase behavior. The Gemma 4 12B pair receives a focused main-paper panel; all pairs and all five samples appear in the appendix.
6. A quantitative summary only if completed manual ratings exist; otherwise the paper states that rating results are pending and omits the plot.

No figure forces eleven long outputs into one unreadable panel. Figure text is truncated only for display, while full outputs remain in appendix tables and machine-readable artifacts. Vector PDF/SVG is preferred.

## 9. Paper Structure

P2 retains the existing author information and adds:

- `main.tex`: English IEEE-style paper entry point.
- `main_ko.tex`: Korean companion paper for team review.
- `references.bib`: verified BibTeX entries only.
- `sections/en/` and `sections/ko/`: mirrored section structure.
- `figures/` and `tables/`: synchronized generated assets and LaTeX tables.

Both versions distinguish the previously generated HSI descriptions from the new VQA-generation experiment. Claims about performance are written only after the associated results exist.

The experiment and results sections separate two analytical axes:

1. **Cross-model comparison:** quality differences among open checkpoints under multimodal or text-only input conditions.
2. **Controlled modality ablation:** four same-checkpoint pairs hold model weights fixed while changing only image availability, with C03 versus C09 as the primary Gemma analysis.

The paper includes an eleven-row configuration/setup table and a dedicated controlled-ablation subsection and table.

## 10. Credential, Storage, and Safety Design

Git credentials are stored at user level with `credential.useHttpPath=true` so GitHub and the specific Overleaf project can coexist. Credential files and parent directories use restrictive permissions. Repository remotes contain no token or password.

The provided Hugging Face token is stored only in the standard user-level Hugging Face credential location if authentication is needed. Public availability and any gated-license acceptance are checked per checkpoint; the token is not placed in environment files merely for convenience.

Model downloads and Hugging Face caches use a dedicated `/data` directory. P1 and P2 never contain model weights, dataset copies, virtual environments, secret files, or large raw output. Legacy repositories and the shard tree remain read-only.

Before either push, staged content and repository history created by this work are scanned for GitHub, Overleaf, Hugging Face, and common secret patterns. Remote URLs are checked in sanitized form, and generated logs are checked for authentication headers or private tokens.

## 11. Error Handling and Reproducibility

- Every command records an experiment ID and machine-readable status without recording secrets.
- Server readiness, model revision, GPU assignment, and prompt hash are captured before inference.
- Runs are resumable at case granularity and do not overwrite successful prior cases.
- Failed cases retain error type and raw response where safe.
- Absolute paths are allowed only in ignored local config and local outputs; tracked artifacts use sample IDs and repository-relative references.
- Tests cover malformed JSONL, missing components, duplicate IDs, mismatched fixed samples, text-only image leakage, partial model responses, repair failure, and interrupted/resumed execution.
- Tests enforce exactly eleven unique configuration IDs, exactly five ordered sample IDs per configuration, no image payload in C05--C11, and identical checkpoint revision and generation settings within C01/C10, C02/C11, C03/C09, and C04/C06.

## 12. Known Risks

- Installing a current vLLM/PyTorch stack on the initially empty Python environment may expose CUDA-wheel or model-version incompatibilities.
- InternVL uses model-specific code and may behave differently from native Transformers architectures despite current vLLM support.
- Gemma 4 12B Unified may require Transformers direct inference if vLLM support is incomplete; this backend difference must be documented before the full run.
- Gemma 4 and Qwen3 expose configurable thinking behavior, so an implicit model default could invalidate fairness unless the run manifest pins it.
- Mistral's repository contains duplicate formats and can waste tens of gigabytes if downloaded indiscriminately.
- Structured JSON compliance may vary across models; raw retention and strict validation prevent silent data corruption.
- The 384 x 384 RGB renderings contain spatial appearance but not the full 426-band information; visual-dependence claims must be restricted to RGB-visible evidence.
- The local machine initially lacks TeX tooling, so paper compilation may require installing a toolchain or relying on Overleaf after source-level checks.
