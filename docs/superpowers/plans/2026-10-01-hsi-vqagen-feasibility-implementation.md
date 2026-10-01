# HSI-VQAGen Nine-Configuration Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, document, and publish a reproducible five-sample feasibility study covering nine multimodal/text-only configurations and a controlled Gemma 4 12B modality ablation.

**Architecture:** A typed Python package reads the audited shard records, locks a five-sample manifest, builds condition-aware prompts, invokes either an OpenAI-compatible vLLM server or a documented Transformers fallback, and writes raw plus normalized outputs with complete run metadata. Deterministic evaluation and figure modules consume only normalized artifacts; a separate synchronization step copies verified tables, figures, and conservative English/Korean prose into the existing Overleaf repository.

**Tech Stack:** Python 3.10+, pytest, Pydantic 2, PyYAML, Pillow, pandas, matplotlib, httpx/OpenAI client, PyTorch, Transformers, Accelerate, vLLM where supported, NVIDIA H200 NVL, IEEE LaTeX.

**Spec:** `docs/superpowers/specs/2026-10-01-hsi-vqagen-feasibility-design.md`

## Global Constraints

- Use P1 only at `/home/khkim/gnew_projects/01_hsi_vqa_gen_github/hsi-vqagen`; do not create a duplicate clone at the obsolete path.
- Reuse P2 at `/home/khkim/gnew_projects/02_hsi_vqa_gen_overleaf/overleaf_hsi_vqagen`.
- Treat `/data/jypark/neon40k_work/shards`, the legacy repository, and `/data/jypark/CLI_Descriptive_System` as read-only.
- Never put raw HSI cubes into model requests; multimodal input is HSI-derived RGB plus description.
- Use exactly the five ordered sample IDs in the approved spec for every configuration.
- Use exactly C01--C09; this is seven unique checkpoints and nine configurations.
- C03 and C09 must share the exact Gemma 4 12B revision and generation settings; image availability is their only intended input difference.
- C05--C09 must never send an image payload, even though some checkpoints are multimodal-capable.
- Run one sample through every configuration before any 45-case run.
- Do not silently replace an incompatible checkpoint; document cause and alternatives first.
- Do not invent results, ratings, citations, dataset details, or legacy-pipeline components.
- Keep credentials, datasets, model weights, virtual environments, caches, and raw output out of Git.
- Preserve the unrelated process already occupying approximately 818 MiB on GPU 0.
- Store Hugging Face/model cache under `/data/hsi-vqagen-cache`, not the home filesystem or repository.
- Generate four QA pairs per sample with one shared normalized JSONL schema and a common maximum output budget.
- Record raw output, normalized output, generation settings, revision, prompt version, load time, latency, peak VRAM when measurable, and errors for every configuration.

## Review Focus

1. **Text-only leakage:** a multimodal-capable C06/C07/C09 checkpoint must receive no image object, image token, base64 data, or path in its request; Task 5 adds a payload-level test.
2. **Ablation drift:** C03/C09 must reject differing revision, prompt version, QA count, output budget, or decoding settings; Task 4 adds a registry invariant test.
3. **Partial/resumed runs:** an interrupted 45-case run must preserve completed raw/normalized cases and resume only missing case IDs; Task 6 adds a resume test.
4. **Shard disagreement:** duplicate IDs, missing paired rows, stale RGB paths, or mismatched descriptions must appear in audit errors rather than being silently repaired; Task 2 adds fixture tests.
5. **Backend divergence:** vLLM and Transformers must emit the same request/result contract while recording backend-specific preprocessing and timing; Task 5 adds a contract test and Task 8 verifies real smoke manifests.

---

### Task 1: Secure repository and reproducible project bootstrap

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `requirements.in`
- Create: `requirements-dev.in`
- Create: `requirements.lock`
- Create: `configs/local_paths.example.yaml`
- Create locally but ignore: `configs/local_paths.yaml`
- Create: `src/hsi_vqagen/__init__.py`
- Create: `tests/test_project_layout.py`
- Create: `docs/environment.md`

**Interfaces:**
- Consumes: approved spec and the two existing Git working trees.
- Produces: installable package `hsi_vqagen`, ignored local-path configuration, secure user-level credential setup, and a healthy local `main` ready for first push.

- [ ] **Step 1: Verify both working trees and remotes without exposing user information**

Run sanitized `git status --short --branch`, `git remote -v`, `git branch -vv`, and `git ls-remote --heads origin` in P1 and P2. Record in the experiment log that P1's remote is an empty repository, so `[origin/main: gone]` is expected until the first push; do not reset or delete either working tree.

- [ ] **Step 2: Configure persistent credentials outside both repositories**

Set `credential.useHttpPath=true`; use a user-level Git credential store under `~/.config/git/` with parent mode `700` and credential file mode `600`. Insert GitHub and Overleaf credentials through `git credential approve` using interactive/no-echo input, then replace P2's remote with a credential-free HTTPS URL. Store the Hugging Face token only through the standard `hf auth login` credential location if access is required. Never place literal tokens in a command log or tracked file.

- [ ] **Step 3: Write the failing layout/security test**

`tests/test_project_layout.py` asserts that required source/config directories exist, `.gitignore` covers `.venv/`, `configs/local_paths.yaml`, `outputs/`, caches, model formats, `.env`, and credential-like files, and tracked remote/config examples contain no private absolute server path or token prefix.

- [ ] **Step 4: Create a minimal test environment and verify the layout test fails**

Run: `python3 -m venv .venv && .venv/bin/pip install pytest pyyaml && .venv/bin/pytest tests/test_project_layout.py -v`  
Expected: pytest starts successfully and assertions FAIL because the skeleton and ignore rules do not yet exist.

- [ ] **Step 5: Create the minimal package and configuration skeleton**

Use Python `>=3.10`; declare runtime dependencies separately from GPU-serving dependencies so audit/tests can run without installing vLLM. Put the real five local roots only in ignored `configs/local_paths.yaml`; use neutral placeholders in the tracked example and README.

- [ ] **Step 6: Create the environment and lock resolved versions**

Create `.venv`, install the audit/test dependencies, then install the GPU stack after resolving a compatible PyTorch/Transformers/vLLM combination. Record exact resolved versions in `requirements.lock` and `docs/environment.md`; set `HF_HOME=/data/hsi-vqagen-cache/huggingface` in ignored local configuration or run scripts, never globally in tracked secrets.

- [ ] **Step 7: Run the layout test and package import check**

Run: `.venv/bin/pytest tests/test_project_layout.py -v && .venv/bin/python -c "import hsi_vqagen"`  
Expected: PASS and exit code 0.

- [ ] **Step 8: Commit the bootstrap**

```bash
git add .gitignore README.md pyproject.toml requirements.in requirements-dev.in requirements.lock configs/local_paths.example.yaml src/hsi_vqagen/__init__.py tests/test_project_layout.py docs/environment.md
git commit -m "chore: bootstrap secure research project"
```

### Task 2: Dataset records, full audit, and fixed sample manifest

**Files:**
- Create: `src/hsi_vqagen/data/__init__.py`
- Create: `src/hsi_vqagen/data/records.py`
- Create: `src/hsi_vqagen/data/audit.py`
- Create: `src/hsi_vqagen/data/samples.py`
- Create: `src/hsi_vqagen/data/inspect_dataset.py`
- Create: `configs/feasibility_samples.yaml`
- Create: `tests/fixtures/shards/` minimal synthetic fixture
- Create: `tests/data/test_records.py`
- Create: `tests/data/test_audit.py`
- Create: `tests/data/test_samples.py`
- Create: `artifacts/dataset_audit.json`
- Create: `artifacts/dataset_audit.csv`
- Create: `docs/dataset_audit.md`
- Create: `docs/feasibility_samples.md`

**Interfaces:**
- Consumes: `s*/descriptions.jsonl`, `s*/sft.jsonl`, patch `meta.json`, and referenced RGB files.
- Produces: `DatasetRecord`, `DatasetAudit`, `load_dataset_records(root: Path) -> list[DatasetRecord]`, `audit_dataset(root: Path) -> DatasetAudit`, and `load_fixed_samples(records, manifest_path) -> list[DatasetRecord]`.

- [ ] **Step 1: Write failing record and audit tests**

Test exact pairing by `sample_id`, duplicate detection across shards, malformed JSON reporting, missing RGB/description/meta reporting, PNG dimension parsing, description-length statistics, HSI-shape distribution from metadata, and disagreement between description and SFT rows. Assert the fixed manifest returns the approved five IDs in approved order and rejects a missing or reordered ID.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/data -v`  
Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Implement typed record loading and auditing**

Implement `DatasetRecord` and `DatasetAudit` as Pydantic models. Stream JSONL instead of loading patch artifacts wholesale; verify files and image headers, calculate counts/distributions, and retain structured errors without mutating source data.

- [ ] **Step 4: Implement the locked five-sample selector**

`configs/feasibility_samples.yaml` contains the exact five IDs, semantic reason, and shard. `load_fixed_samples` validates uniqueness, order, RGB readability, non-empty descriptions, and manifest hash.

- [ ] **Step 5: Run data tests**

Run: `.venv/bin/pytest tests/data -v`  
Expected: PASS.

- [ ] **Step 6: Run the real full audit and inspect outputs**

Run: `.venv/bin/python -m hsi_vqagen.data.inspect_dataset --config configs/local_paths.yaml --json artifacts/dataset_audit.json --csv artifacts/dataset_audit.csv --markdown docs/dataset_audit.md`  
Expected: 100 shards, 10,000 unique/fully paired records, zero duplicate IDs, zero missing required components, 384 x 384 RGB distribution, and 64 x 64 x 426 HSI metadata distribution. If actual results differ from the preliminary read-only scan, the generated results take precedence.

- [ ] **Step 7: Commit audit code and evidence**

```bash
git add src/hsi_vqagen/data configs/feasibility_samples.yaml tests/fixtures tests/data artifacts/dataset_audit.json artifacts/dataset_audit.csv docs/dataset_audit.md docs/feasibility_samples.md
git commit -m "feat: audit dataset and lock feasibility samples"
```

### Task 3: Description-pipeline provenance documentation

**Files:**
- Create: `docs/hsi_description_pipeline.md`
- Create: `artifacts/hsi_description_pipeline_evidence.json`
- Create: `tests/docs/test_pipeline_claims.py`

**Interfaces:**
- Consumes: read-only legacy source, production batch source, five sample `meta.json`/`report.json`, call directories, and prompt version metadata.
- Produces: citation-ready factual account and a machine-readable evidence map from each Method claim to source file/artifact fields.

- [ ] **Step 1: Write the failing documentation evidence test**

Assert every required section—input, preprocessing, RGB, index selection/maps, SAM K-medoids clustering, map agents, web context, catalog synthesis, reduce synthesis, output, disabled pixel SAM/Tetracorder, limitations—has at least one source path and symbol/artifact key. Assert the text does not claim raw HSI was sent to the VQA models or that disabled components generated the 10,000 descriptions.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/docs/test_pipeline_claims.py -v`  
Expected: FAIL because the evidence files do not exist.

- [ ] **Step 3: Trace and document the production path**

Use the production wrapper as realized provenance and legacy core as the algorithm reference. Record the exact prompt version/hash, enabled agents, three batch waves plus real-time web call, index-map rendering behavior, cluster algorithm, GPT-5.4/GPT-5.4-mini roles, and disabled components. Distinguish code capabilities from components active in the audited run.

- [ ] **Step 4: Run the evidence test**

Run: `.venv/bin/pytest tests/docs/test_pipeline_claims.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit provenance documentation**

```bash
git add docs/hsi_description_pipeline.md artifacts/hsi_description_pipeline_evidence.json tests/docs/test_pipeline_claims.py
git commit -m "docs: document verified HSI description pipeline"
```

### Task 4: Nine-configuration registry and model selection

**Files:**
- Create: `src/hsi_vqagen/config.py`
- Create: `configs/experiments.yaml`
- Create: `tests/test_config.py`
- Create: `docs/model_selection.md`

**Interfaces:**
- Consumes: YAML configuration and fixed sample manifest hash.
- Produces: `GenerationSettings`, `ExperimentConfig`, `ExperimentRegistry`, and `load_experiment_registry(path: Path) -> ExperimentRegistry`.

- [ ] **Step 1: Write the failing registry invariant tests**

Assert IDs are exactly C01--C09, output directories are unique and match the spec, C01--C04 are multimodal, C05--C09 are text-only, checkpoint count is seven, and C03/C09 match on checkpoint, revision, prompt version, QA count, max tokens, temperature/top-p/top-k/min-p, seed, and thinking mode. Assert C05--C09 set `allow_image=false`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`  
Expected: FAIL with missing registry.

- [ ] **Step 3: Implement strict configuration models and YAML**

Pin explicit configuration IDs, checkpoint IDs, revisions after download resolution, preferred backend, fallback backend, condition, output directory, common settings, model-specific chat-template controls, and documented preprocessing exceptions. Registry validation raises before inference on any ablation drift or count mismatch.

- [ ] **Step 4: Write the model-selection document from official sources and environment probes**

Cover parameters, license, weight size, modality, current Transformers/vLLM support, expected BF16 footprint, one-H200 fit, serving command, prompt/thinking controls, and selection rationale. State that Gemma 4 12B's official card currently lacks a vLLM recipe while Gemma 4 31B and Qwen3 provide one, so smoke testing decides its backend. Explain the seven-checkpoint/nine-configuration count and the deliberate C03/C09 ablation.

- [ ] **Step 5: Run registry tests**

Run: `.venv/bin/pytest tests/test_config.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit registry and model decision**

```bash
git add src/hsi_vqagen/config.py configs/experiments.yaml tests/test_config.py docs/model_selection.md
git commit -m "feat: define nine experiment configurations"
```

### Task 5: Shared schema, condition-aware prompts, and backend contract

**Files:**
- Create: `src/hsi_vqagen/schema/__init__.py`
- Create: `src/hsi_vqagen/schema/generation.py`
- Create: `src/hsi_vqagen/inference/__init__.py`
- Create: `src/hsi_vqagen/inference/prompts.py`
- Create: `src/hsi_vqagen/inference/backends.py`
- Create: `src/hsi_vqagen/inference/normalize.py`
- Create: `prompts/vqa_generation_v1.txt`
- Create: `prompts/json_repair_v1.txt`
- Create: `tests/inference/test_prompts.py`
- Create: `tests/inference/test_backends.py`
- Create: `tests/inference/test_normalize.py`

**Interfaces:**
- Consumes: `DatasetRecord`, `ExperimentConfig`, and prompt templates.
- Produces: `GenerationRequest`, `RawGeneration`, `GeneratedPair`, `NormalizedCase`, `GenerationBackend.generate(request: GenerationRequest) -> RawGeneration`, `build_messages(record, config) -> list[dict]`, and `normalize_response(raw, request) -> NormalizedCase`.

- [ ] **Step 1: Write failing schema and prompt tests**

Assert every request asks for exactly four QA objects, keeps answer style/category guidance constant, and includes description text. Inspect serialized payloads to prove C05--C09 contain no image content/token/base64/path while C01--C04 contain exactly one RGB input before the description. Assert C03/C09 differ only by the image content element after canonicalization.

- [ ] **Step 2: Write failing normalization tests**

Cover clean JSON, fenced JSON, extra prose, missing pair fields, fewer/more than four pairs, duplicate questions, empty evidence, one repair attempt, repair failure, and unsupported manual mutation. Require each normalized line to include schema/experiment/case IDs, pair index, model/revision/backend, input condition, provenance, prompt hash, generation settings, latency, load time, peak VRAM, and validation status.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/inference/test_prompts.py tests/inference/test_backends.py tests/inference/test_normalize.py -v`  
Expected: FAIL with missing modules.

- [ ] **Step 4: Implement schema and one shared semantic prompt**

Use one prompt body with a condition-specific sentence describing available evidence. Questions may use spectral/material facts only when explicitly stated in the description; RGB is described as HSI-derived representation, never raw HSI. Store prompt SHA-256 and version.

- [ ] **Step 5: Implement backend adapters**

Implement `VLLMBackend` against an OpenAI-compatible local endpoint and `TransformersBackend` for documented fallback. Both return the same `RawGeneration` contract. Backend metadata records server/library versions, preprocessing class, chat-template controls, image order, and timing; the Transformers adapter resets and reads CUDA peak memory around load/generation where possible.

- [ ] **Step 6: Implement strict normalization and repair**

Parse without altering semantic content, validate exactly four pairs, save initial raw output before an optional single repair request, and emit explicit failed case metadata after a second failure.

- [ ] **Step 7: Run inference-unit tests**

Run: `.venv/bin/pytest tests/inference -v`  
Expected: PASS.

- [ ] **Step 8: Commit generation contracts**

```bash
git add src/hsi_vqagen/schema src/hsi_vqagen/inference prompts tests/inference
git commit -m "feat: add condition-safe generation contract"
```

### Task 6: Resumable orchestration and runtime measurements

**Files:**
- Create: `src/hsi_vqagen/inference/runner.py`
- Create: `src/hsi_vqagen/inference/telemetry.py`
- Create: `src/hsi_vqagen/utils/__init__.py`
- Create: `src/hsi_vqagen/utils/io.py`
- Create: `src/hsi_vqagen/utils/logging.py`
- Create: `tests/inference/test_runner.py`
- Create: `tests/inference/test_telemetry.py`
- Create: `scripts/run_configuration.py`

**Interfaces:**
- Consumes: registry, fixed `DatasetRecord` list, and a `GenerationBackend`.
- Produces: `run_configuration(config_id, sample_ids, output_root, resume=True) -> RunSummary`, atomic raw/normalized case files, `run_manifest.json`, `generation_config.json`, and `errors.jsonl`.

- [ ] **Step 1: Write failing resume and telemetry tests**

Test atomic case writes, stable case IDs, preservation of successful cases, rerun of failed/missing cases only, refusal to mix prompt/config hashes in one output directory, load-time recording, per-case latency, peak-VRAM null-with-reason fallback, and error redaction for authorization headers/tokens.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/inference/test_runner.py tests/inference/test_telemetry.py -v`  
Expected: FAIL with missing runner.

- [ ] **Step 3: Implement atomic case-oriented orchestration**

Write each raw response and normalized case to a temporary file followed by atomic rename. Aggregate `normalized.jsonl` deterministically by approved sample order and pair index. Refuse output-directory reuse when model revision, prompt hash, sample-manifest hash, or generation settings differ.

- [ ] **Step 4: Implement telemetry and redacted logging**

Measure wall-clock load time, request latency, and NVIDIA memory using backend CUDA statistics or `nvidia-smi`; store measurement method. Redact credential patterns and never serialize request headers.

- [ ] **Step 5: Run orchestration tests**

Run: `.venv/bin/pytest tests/inference/test_runner.py tests/inference/test_telemetry.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/hsi_vqagen/inference/runner.py src/hsi_vqagen/inference/telemetry.py src/hsi_vqagen/utils tests/inference scripts/run_configuration.py
git commit -m "feat: add resumable measured inference runner"
```

### Task 7: Deterministic evaluation and controlled-ablation analysis

**Files:**
- Create: `src/hsi_vqagen/evaluation/__init__.py`
- Create: `src/hsi_vqagen/evaluation/checks.py`
- Create: `src/hsi_vqagen/evaluation/tables.py`
- Create: `src/hsi_vqagen/evaluation/ablation.py`
- Create: `tests/evaluation/test_checks.py`
- Create: `tests/evaluation/test_ablation.py`
- Create: `prompts/evaluation_prompt.txt`
- Create: `docs/evaluation_protocol.md`

**Interfaces:**
- Consumes: normalized cases from C01--C09.
- Produces: `evaluate_deterministic(cases) -> EvaluationSummary`, `build_manual_review_sheet(cases) -> DataFrame`, and `compare_gemma_ablation(c03, c09) -> AblationReport`.

- [ ] **Step 1: Write failing evaluation tests**

Cover duplicate/paraphrase warnings, empty fields, pair-count mismatch, unsupported spectral keywords, text-only visual-source leakage, missing configuration/sample matrix cells, and exact C03/C09 pairing by sample/pair index. Assert the manual worksheet has blank fields for correctness, grounding, hallucination, diversity, visual dependence, usefulness, and specificity.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/evaluation -v`  
Expected: FAIL with missing evaluation modules.

- [ ] **Step 3: Implement deterministic checks and review tables**

Diagnostics flag issues but never assign subjective quality scores. The ablation report pairs C03/C09, summarizes only observable output properties before human rating, and provides columns for description-absent visual detail, paraphrase tendency, hallucination direction, diversity change, and specificity change. Add a `main()` entry point to `checks.py` accepting `--outputs`, `--require-configs`, `--require-samples`, and `--require-pairs` for the smoke and full-run gates.

- [ ] **Step 4: Run evaluation tests**

Run: `.venv/bin/pytest tests/evaluation -v`  
Expected: PASS.

- [ ] **Step 5: Commit evaluation framework**

```bash
git add src/hsi_vqagen/evaluation tests/evaluation prompts/evaluation_prompt.txt docs/evaluation_protocol.md
git commit -m "feat: add feasibility and ablation evaluation"
```

### Task 8: Model serving scripts and nine one-sample smoke tests

**Files:**
- Create: `scripts/serve_model.sh`
- Create: `scripts/run_smoke_tests.sh`
- Create: `scripts/check_server.py`
- Create: `docs/smoke_test_results.md`
- Modify: `docs/environment.md`
- Modify: `docs/model_selection.md`
- Modify after resolved download: `configs/experiments.yaml`

**Interfaces:**
- Consumes: installed runtime, checkpoint cache, registry, first approved sample, and GPU availability.
- Produces: nine smoke output directories with real raw/normalized results and a compatibility decision per configuration.

- [ ] **Step 1: Probe package/model compatibility before full downloads**

Record `nvidia-smi`, Python, PyTorch CUDA, Transformers, vLLM, Accelerate, Flash Attention, and processor/config loading. Resolve and pin checkpoint revisions. Verify available disk against all required unique weight sets and download only required Mistral format.

- [ ] **Step 2: Implement server lifecycle commands**

`serve_model.sh <config-id> <gpu> <port>` reads the registry, exports the `/data` cache, starts one checkpoint on one GPU, writes a PID/status log without secrets, and waits for model readiness. It must not kill unrelated processes. `check_server.py` verifies model ID/revision and performs a minimal health request.

- [ ] **Step 3: Run C01--C09 smoke tests one checkpoint at a time**

Use only the first fixed sample. For Gemma 4 12B, run C03 then C09 against the same load if possible. For Mistral, run C04 then C06 against the same load if possible. Each smoke test must produce four schema-valid pairs plus load time, latency, revision, prompt hash, backend, peak VRAM or a reason it is unavailable, and error log.

- [ ] **Step 4: Diagnose failures without substituting models**

For each failed configuration, capture the exact exception and smallest reproducer, test supported version/backend changes, and update `docs/smoke_test_results.md`. If Gemma 4 12B vLLM fails, test the approved Transformers fallback. Do not proceed to Task 9 while any configuration remains failed; document any genuine blocker for owner decision.

- [ ] **Step 5: Validate the smoke gate**

Run: `.venv/bin/python -m hsi_vqagen.evaluation.checks --outputs outputs/smoke --require-configs C01,C02,C03,C04,C05,C06,C07,C08,C09 --require-samples 1 --require-pairs 4`  
Expected: PASS, 9 complete cases and 36 normalized QA rows.

- [ ] **Step 6: Commit compatibility evidence, not raw output**

```bash
git add scripts/serve_model.sh scripts/run_smoke_tests.sh scripts/check_server.py docs/smoke_test_results.md docs/environment.md docs/model_selection.md configs/experiments.yaml
git commit -m "docs: validate nine configuration smoke tests"
```

### Task 9: Execute and validate the five-by-nine study

**Files:**
- Create: `scripts/run_feasibility.sh`
- Create: `docs/experiment_log.md`
- Create: `artifacts/feasibility_summary.json`
- Create: `artifacts/feasibility_summary.csv`
- Create: `artifacts/manual_review.csv`
- Create: `artifacts/gemma4_12b_ablation.csv`
- Create: `docs/gemma4_12b_ablation.md`

**Interfaces:**
- Consumes: passing smoke configurations and all five fixed samples.
- Produces: complete 9 x 5 case matrix, 180 normalized QA rows when all cases return four pairs, telemetry summaries, and unfilled manual-review/ablation worksheets.

- [ ] **Step 1: Define and log execution waves**

Choose up to four concurrent checkpoint loads based on measured smoke VRAM and backend compatibility. Record GPU, port, checkpoint revision, PID, start/end time, and configuration IDs. Reuse Gemma 4 12B and Mistral loads for their paired conditions.

- [ ] **Step 2: Run all five samples through C01--C09**

Run: `scripts/run_feasibility.sh --config configs/experiments.yaml --samples configs/feasibility_samples.yaml --output outputs/feasibility --resume`  
Expected: 45 complete cases. If interrupted, rerun the same command and confirm completed case files are untouched.

- [ ] **Step 3: Normalize, aggregate, and validate the matrix**

Run deterministic validation requiring nine configurations, the exact five ordered sample IDs, and four pairs per successful case. Verify text-only payload audit records show zero image inputs and C03/C09 manifest settings/revision match.

- [ ] **Step 4: Generate factual summaries and blank human-review sheets**

Write counts, latency, load time, peak VRAM, validation errors, duplicate warnings, and backend/revision data to JSON/CSV. Do not populate subjective scores. Generate the Gemma paired worksheet and Markdown report with actual outputs and empty reviewer fields.

- [ ] **Step 5: Commit small summaries and scripts**

```bash
git add scripts/run_feasibility.sh docs/experiment_log.md docs/gemma4_12b_ablation.md artifacts/feasibility_summary.json artifacts/feasibility_summary.csv artifacts/manual_review.csv artifacts/gemma4_12b_ablation.csv
git commit -m "feat: record nine configuration feasibility run"
```

### Task 10: Publication-quality figures and tables

**Files:**
- Create: `src/hsi_vqagen/figures/__init__.py`
- Create: `src/hsi_vqagen/figures/pipeline.py`
- Create: `src/hsi_vqagen/figures/description_pipeline.py`
- Create: `src/hsi_vqagen/figures/qualitative.py`
- Create: `src/hsi_vqagen/figures/ablation.py`
- Create: `tests/figures/test_figures.py`
- Create: `scripts/generate_figures.py`
- Create: `figures/study_overview.pdf` and `.svg`
- Create: `figures/hsi_description_pipeline.pdf` and `.svg`
- Create: `figures/qualitative_multimodal.pdf`
- Create: `figures/qualitative_text_only.pdf`
- Create: `figures/gemma4_12b_ablation.pdf`
- Create: `paper_assets/model_setup.tex`
- Create: `paper_assets/feasibility_summary.tex`
- Create: `paper_assets/gemma4_12b_ablation.tex`

**Interfaces:**
- Consumes: evidence map, normalized real outputs, factual summaries, and completed ratings only if available.
- Produces: readable vector figures and LaTeX tables with no fabricated value.

- [ ] **Step 1: Write failing figure tests**

Assert every renderer rejects missing real input, produces nonempty PDF/SVG, labels input condition, keeps all five samples in appendix mode, limits main-paper mode to two or three representative samples, and keeps C03/C09 adjacent in ablation output. Test that a quantitative rating plot is omitted when rating cells are blank.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/figures -v`  
Expected: FAIL with missing renderers.

- [ ] **Step 3: Implement vector pipeline figures**

Use matplotlib vector patches/text with consistent type scale and color-safe palette. The study overview must show HSI branching to RGB and prior description system, then multimodal/text-only conditions. The description figure must reflect only verified active components.

- [ ] **Step 4: Implement split qualitative and ablation figures**

Figure A shows RGB, description summary, and C01--C04; Figure B shows the same samples and C05--C08; Figure C pairs C03/C09. Truncate display text deterministically with a visible marker and retain full outputs in tables. Produce main-paper and all-five appendix variants where needed.

- [ ] **Step 5: Run tests and generate real assets**

Run: `.venv/bin/pytest tests/figures -v && .venv/bin/python scripts/generate_figures.py --outputs outputs/feasibility --artifacts artifacts --figures figures --paper-assets paper_assets`  
Expected: PASS and all declared nonempty assets.

- [ ] **Step 6: Visually inspect every rasterized PDF page**

Render PDFs to temporary PNG previews, inspect for clipping, unreadable fonts, overlap, incorrect labels, or tiny text, and regenerate until readable at intended IEEE column width.

- [ ] **Step 7: Commit figures and tables**

```bash
git add src/hsi_vqagen/figures tests/figures scripts/generate_figures.py figures paper_assets
git commit -m "feat: add publication feasibility figures"
```

### Task 11: English/Korean Overleaf paper and verified bibliography

**Files in P2:**
- Modify: `main.tex`
- Create: `main_ko.tex`
- Create: `references.bib`
- Create: `sections/en/01_introduction.tex`
- Create: `sections/en/02_related_work.tex`
- Create: `sections/en/03_method.tex`
- Create: `sections/en/04_experiments.tex`
- Create: `sections/en/05_results.tex`
- Create: `sections/en/06_conclusion.tex`
- Create: `sections/en/appendix.tex`
- Create: matching `sections/ko/*.tex`
- Create/modify: `figures/` synchronized P1 assets
- Create/modify: `tables/` synchronized P1 tables
- Create in P1: `scripts/sync_overleaf.py`
- Create in P1: `tests/paper/test_paper_sources.py`

**Interfaces:**
- Consumes: verified pipeline docs, official citations, real run summaries, real figures, and tables.
- Produces: parallel English/Korean manuscripts with identical factual content and separate language prose.

- [ ] **Step 1: Write failing paper-source tests in P1**

Assert both main files exist, use `references.bib`, include mirrored section sets, reference all required figures/tables, use `Multimodal condition`/`Text-only condition`, contain a controlled-ablation subsection, do not claim raw HSI input to VQA models, and contain no placeholder result or uncited model claim.

- [ ] **Step 2: Run the paper test to verify it fails**

Run: `.venv/bin/pytest tests/paper/test_paper_sources.py -v`  
Expected: FAIL against the minimal P2 paper.

- [ ] **Step 3: Create verified bibliography entries**

Use primary papers/model reports only. Verify title, author list, year, identifier, and URL/DOI against the paper/model card before adding BibTeX. Never invent a citation for HSI pipeline implementation details; cite the inspected local system as the study's prior processing method where no publication exists.

- [ ] **Step 4: Draft Method and Experiments first**

Describe active HSI preprocessing/index/SAM-clustering/agent/reduce stages precisely. Add the nine-row setup table, exact five samples, shared task/schema/settings, backend exceptions, 45-case scope, and separate cross-model versus C03/C09 ablation axes.

- [ ] **Step 5: Draft Results without overclaiming**

Insert only real outputs and factual telemetry. If manual ratings remain blank, present qualitative observations as explicitly manual pending-review material and omit quantitative quality plots. Keep C03/C09 as a dedicated subsection/table.

- [ ] **Step 6: Draft Introduction, Related Work, Conclusion, and Korean mirror**

Keep contributions proportional to a feasibility study. Translate factual claims and table/figure references consistently; preserve model IDs and condition terminology.

- [ ] **Step 7: Synchronize assets and run source/compile validation**

Run P1 paper tests and, if a TeX engine is available, compile both entry points with bibliography. Otherwise run brace/include/reference/static checks, push source to Overleaf, and record that cloud compilation remains the validation route.

- [ ] **Step 8: Commit P1 sync/test tooling and P2 manuscript**

Commit P1 with `feat: add reproducible Overleaf synchronization`; commit P2 with `docs: draft bilingual nine-configuration paper`.

### Task 12: Final verification, secure push, and handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment_log.md`
- Create: `docs/final_status.md`

**Interfaces:**
- Consumes: complete P1/P2 worktrees and all verification commands.
- Produces: clean pushed `origin/main` branches and concise owner handoff.

- [ ] **Step 1: Run complete code and data verification**

Run all pytest suites, dataset audit assertions, configuration invariants, smoke gate, 45-case matrix validation, figure generation checks, and paper source/compile checks. Save command, timestamp, exit code, and concise outcome in `docs/experiment_log.md`.

- [ ] **Step 2: Audit secrets and repository size before staging/push**

Search tracked/staged files and new Git history for GitHub/Overleaf/Hugging Face token patterns, authorization headers, embedded remote credentials, `.env`, private local config, model extensions, and unexpectedly large blobs. Verify both `git remote -v` outputs are credential-free in sanitized reporting and credential files are mode `600`.

- [ ] **Step 3: Verify repository state and first-push behavior**

Confirm P1 `git ls-remote --heads origin` is still empty or reconcile non-destructively if another branch appeared. Never reset/delete existing work. Push with `git push -u origin main`; this creates and restores normal tracking for the previously empty remote. Pull/rebase only if a remote commit genuinely appeared and preserve both histories.

- [ ] **Step 4: Push P2 safely**

Fetch and inspect P2 first. If remote advanced, integrate non-destructively and re-run paper validation. Push the bilingual paper and assets to its existing `main` remote.

- [ ] **Step 5: Write and commit final status**

`docs/final_status.md` summarizes completed items, discovered data structure, final checkpoints/configurations, failures/workarounds, exact next commands, and decisions still required from the owner. Include the accurate seven-checkpoint/nine-configuration count.

- [ ] **Step 6: Confirm remote commits and hand off**

Verify P1 and P2 remote `main` SHAs match local HEAD, both working trees are clean, and the user-level credentials support noninteractive `git fetch` without printing secrets.
