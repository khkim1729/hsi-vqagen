# HSI-VQAGen Feasibility Study Design

**Date:** 2026-10-01  
**Paper:** `[BDOT_ICCE2027] HSI-VQAGen: Comparing Vision-Language and Text-Only Models for Hyperspectral VQA Generation`

## 1. Objective and Success Criteria

Build a reproducible feasibility study that compares vision-language and text-only generation of VQA pairs from the same five hyperspectral-derived samples.

The study is complete when:

1. The real shard layout and the existing HSI description-generation pipeline are documented from source code and artifacts.
2. Four models successfully complete a one-sample smoke test with a shared validated output schema.
3. The same five samples are processed by all four models, yielding 20 inference cases.
4. Raw responses, normalized JSONL, run metadata, and manual-review worksheets are retained.
5. Publication-quality pipeline and qualitative-comparison figures are produced.
6. English and Korean paper entry points (`main.tex` and `main_ko.tex`) compile or have any local tool limitation documented.
7. P1 and P2 contain no credentials, large model files, dataset copies, or fabricated results/citations and are pushed to their existing remotes.

This feasibility study does not feed raw HSI cubes to the VLMs. It compares RGB representations derived from HSI plus paired HSI-derived descriptions against the descriptions alone.

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

### Input conditions

- **VLM:** HSI-derived RGB image plus the paired description.
- **LLM:** Paired description only. The RGB path is retained only as provenance and is never sent to the model.

All other controllable factors are held constant: sample IDs, prompt semantics, requested number of QA pairs, output schema, maximum generation length, and deterministic decoding where supported.

### Models and GPU mapping

| GPU | Model | Mode | Rationale |
|---|---|---|---|
| 0 | `Qwen/Qwen3-VL-8B-Instruct` | RGB + text | Current Qwen VLM; official Transformers and vLLM paths; Apache-2.0. |
| 1 | `OpenGVLab/InternVL3-8B` | RGB + text | Independent VLM family; vLLM support; MIT project with Apache-2.0 Qwen2.5 component. |
| 2 | `Qwen/Qwen3-8B` | Text only | Parameter-matched Qwen language baseline; Apache-2.0. |
| 3 | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Text only | Stronger practical open model that fits easily on one H200; Apache-2.0 and vLLM-recommended. |

Mistral Small 3.1 intentionally represents the strongest-practical axis rather than a parameter-matched axis. The model-selection document must state this trade-off. Its repository contains redundant weight formats, so only the format needed by the chosen serving path may be downloaded.

### Serving strategy

Use one vLLM OpenAI-compatible server per GPU as the primary path. This gives one request format and one client implementation across all models. Each server uses a distinct localhost port and an explicit `CUDA_VISIBLE_DEVICES` assignment.

Transformers direct inference is a per-model fallback only when a documented vLLM incompatibility remains after the one-sample smoke test. If a fallback is used, the backend difference is recorded in every output and latency values are not directly compared without qualification.

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

Qwen3 text inference uses non-thinking mode so that hidden reasoning markers do not contaminate JSON. Generation is greedy/deterministic where the backend supports it, with a common output budget. Any backend-specific deviation is recorded.

One normalized JSONL line represents one QA pair and contains at least:

- `schema_version`, `experiment_id`, `case_id`, and `pair_index`;
- `sample_id`, shard, source hashes, and source paths;
- exact model ID, model revision, backend, input mode, and GPU;
- question, answer, question type, evidence, and grounded-input references;
- source description and RGB provenance path;
- prompt version, generation configuration, timing, and validation status.

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
6. Start each model individually and run one-sample smoke tests, recording VRAM, latency, revision, and failures.
7. Only after all four smoke tests pass, start four single-GPU servers and run the 5 x 4 study.
8. Validate and compare results; leave subjective manual ratings unfilled unless an actual reviewer supplies them.
9. Generate publication figures and tables from real outputs.
10. Draft English and Korean papers with conservative claims and verified bibliography entries.
11. Run tests, schema checks, secret scans, repository-size checks, and available LaTeX validation.
12. Commit and push P1 and P2.

If one model cannot pass its smoke test, the full study pauses. The failure and attempted workarounds are documented, and no replacement model is silently substituted.

## 8. Evaluation and Figures

Deterministic checks cover schema validity, requested pair count, duplicate questions, empty answers/evidence, input-mode leakage, and unsupported spectral-claim indicators. These checks are diagnostics, not replacements for human judgment.

The manual worksheet covers grounding, correctness, hallucination, question diversity, visual dependence, and downstream usefulness. No rating is populated without actual review.

Publication assets include:

1. A vector end-to-end overview that clearly separates HSI-derived RGB from the raw HSI cube and splits VLM and LLM conditions.
2. A vector diagram of the verified production description pipeline, explicitly showing disabled components only as out-of-scope notes if mentioned.
3. A readable five-sample qualitative comparison: an overview RGB contact sheet plus per-sample panels or appendix pages containing representative VQA text from all four models.
4. A quantitative summary only if completed manual ratings exist; otherwise the paper states that rating results are pending and omits the plot.

## 9. Paper Structure

P2 retains the existing author information and adds:

- `main.tex`: English IEEE-style paper entry point.
- `main_ko.tex`: Korean companion paper for team review.
- `references.bib`: verified BibTeX entries only.
- `sections/en/` and `sections/ko/`: mirrored section structure.
- `figures/` and `tables/`: synchronized generated assets and LaTeX tables.

Both versions distinguish the previously generated HSI descriptions from the new VQA-generation experiment. Claims about performance are written only after the associated results exist.

## 10. Credential, Storage, and Safety Design

Git credentials are stored at user level with `credential.useHttpPath=true` so GitHub and the specific Overleaf project can coexist. Credential files and parent directories use restrictive permissions. Repository remotes contain no token or password.

The provided Hugging Face token is stored only in the standard user-level Hugging Face credential location if authentication is needed. The four selected repositories are currently public, so the token is not placed in environment files merely for convenience.

Model downloads and Hugging Face caches use a dedicated `/data` directory. P1 and P2 never contain model weights, dataset copies, virtual environments, secret files, or large raw output. Legacy repositories and the shard tree remain read-only.

Before either push, staged content and repository history created by this work are scanned for GitHub, Overleaf, Hugging Face, and common secret patterns. Remote URLs are checked in sanitized form, and generated logs are checked for authentication headers or private tokens.

## 11. Error Handling and Reproducibility

- Every command records an experiment ID and machine-readable status without recording secrets.
- Server readiness, model revision, GPU assignment, and prompt hash are captured before inference.
- Runs are resumable at case granularity and do not overwrite successful prior cases.
- Failed cases retain error type and raw response where safe.
- Absolute paths are allowed only in ignored local config and local outputs; tracked artifacts use sample IDs and repository-relative references.
- Tests cover malformed JSONL, missing components, duplicate IDs, mismatched fixed samples, text-only image leakage, partial model responses, repair failure, and interrupted/resumed execution.

## 12. Known Risks

- Installing a current vLLM/PyTorch stack on the initially empty Python environment may expose CUDA-wheel or model-version incompatibilities.
- InternVL uses model-specific code and may behave differently from native Transformers architectures despite current vLLM support.
- Mistral's repository contains duplicate formats and can waste tens of gigabytes if downloaded indiscriminately.
- Structured JSON compliance may vary across models; raw retention and strict validation prevent silent data corruption.
- The 384 x 384 RGB renderings contain spatial appearance but not the full 426-band information; visual-dependence claims must be restricted to RGB-visible evidence.
- The local machine initially lacks TeX tooling, so paper compilation may require installing a toolchain or relying on Overleaf after source-level checks.
