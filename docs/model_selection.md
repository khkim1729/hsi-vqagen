# Model Selection and H200 Compatibility

## Decision

The feasibility study uses seven pinned checkpoints in eleven input configurations. The second text-only baseline is **Mistral Small 3.1 24B Instruct**. It offers a materially larger text model than the 8B baseline, an Apache-2.0 license, official vLLM instructions, and a usable BF16 checkpoint of about 44.73 GiB that fits on one 143,771 MiB H200 NVL. Reusing that checkpoint for multimodal C04 and text-only C06 also adds a controlled modality comparison without another model download.

The architecture and the input condition are recorded separately. In particular, C07, C09, C10, and C11 are multimodal-capable architectures run under a strict description-only condition; they are not relabelled as text-only architectures.

## 2026-10-02 Korean-run availability update

Qwen3-VL-8B, InternVL3-8B, Gemma-4-12B, Mistral Small 3.1 24B, Qwen3-8B,
and Gemma-4-31B have completed download, one-sample Korean smoke testing, and the
fixed-five-sample Korean run for their configured conditions. Gemma-4-31B loaded in
BF16 on one H200 and C07 completed 5/5 cases. Qwen3-32B is still downloading, so C08
is explicitly absent from the current 10-configuration result and has not been
replaced. See `docs/korean_vqa_evaluation.md` for measured outputs and limitations.

## Audited runtime

The server has four NVIDIA H200 NVL GPUs, each reporting 143,771 MiB, with driver
580.82.07. The isolated environment contains Python 3.10.12, PyTorch 2.13.0+cu130,
Transformers 5.10.4, Tokenizers 0.22.2, and vLLM 0.30.0. Transformers was pinned
after an actual Mistral smoke-load failure demonstrated that 5.18.0 had removed the
Pixtral rotary class imported by this vLLM release. CUDA enumeration succeeds for
all four GPUs. The installed vLLM registry recognizes all configured architecture names:

- `Qwen3VLForConditionalGeneration`
- `InternVLChatModel`
- `Gemma4UnifiedForConditionalGeneration`
- `Gemma4ForConditionalGeneration`
- `Mistral3ForConditionalGeneration`
- `Qwen3ForCausalLM`

Registry recognition is necessary but not sufficient; the one-sample smoke test remains the gate for chat-template, processor, kernel, and image-free invocation compatibility.

## Checkpoint comparison

The sizes below are repository BF16 SafeTensors payloads observed from Hugging Face metadata on 2026-10-02. They are storage-size indicators, not peak-VRAM predictions; processor state, CUDA graphs, activations, and KV cache require additional memory. All seven repositories declare Apache-2.0. A 1,024-token output cap and short feasibility prompts leave substantial headroom on one H200, but measured peak VRAM is still recorded during smoke and full runs.

| Checkpoint | Nominal scale | Modality capability | BF16 weight payload | Preferred serving path | One H200 assessment |
|---|---:|---|---:|---|---|
| [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8B | text + image | 16.33 GiB | vLLM, Transformers fallback | Fits comfortably |
| [`OpenGVLab/InternVL3-8B`](https://huggingface.co/OpenGVLab/InternVL3-8B) | 8B | text + image | 14.80 GiB | vLLM, Transformers fallback | Fits comfortably; remote-code and dynamic-tiling path must be tested |
| [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | 12B | multimodal unified model | 22.28 GiB | vLLM probe, Transformers fallback | Fits; backend decided by smoke test |
| [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | 24B | text + image | 44.73 GiB per usable format | vLLM | Fits; avoid downloading both duplicate weight formats |
| [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | 8B | text | 15.26 GiB | vLLM | Fits comfortably |
| [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | 31B | text + image | 58.25 GiB | vLLM, Transformers fallback | Fits; vLLM documentation lists one 80 GB GPU for BF16 |
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | 32B | text | 61.02 GiB | vLLM | Fits with reduced context/KV reservation |

Mistral's repository contains a 44.73 GiB consolidated file and another approximately 44.73 GiB sharded copy. The download stage must select only the format used by the backend. This prevents a misleading 89.45 GiB cache estimate and unnecessary transfer.

## Eleven configurations

| ID | Output directory | Checkpoint | Experimental input condition |
|---|---|---|---|
| C01 | `qwen3_vl_8b_multimodal` | Qwen3-VL-8B-Instruct | RGB + description |
| C02 | `internvl3_8b_multimodal` | InternVL3-8B | RGB + description |
| C03 | `gemma4_12b_multimodal` | Gemma-4-12B-it | RGB + description |
| C04 | `mistral_small_3_1_24b_multimodal` | Mistral Small 3.1 24B | RGB + description |
| C05 | `qwen3_8b_text_only` | Qwen3-8B | description only |
| C06 | `mistral_small_3_1_24b_text_only` | Mistral Small 3.1 24B | description only |
| C07 | `gemma4_31b_text_only` | Gemma-4-31B-it | description only |
| C08 | `qwen3_32b_text_only` | Qwen3-32B | description only |
| C09 | `gemma4_12b_text_only` | Gemma-4-12B-it | description only |
| C10 | `qwen3_vl_8b_text_only` | Qwen3-VL-8B-Instruct | description only |
| C11 | `internvl3_8b_text_only` | InternVL3-8B | description only |

There are four weight-controlled modality pairs: C01/C10, C02/C11, C03/C09, and C04/C06. C03/C09 is the preregistered primary Gemma ablation. Qwen3-VL's official card reports pure-text behavior, and InternVL3's official usage shows `model.chat(tokenizer, None, question, ...)`; these facts motivate C10 and C11. Their actual image-free payloads are verified in the adapter tests and smoke runs rather than assumed from model labels.

Every pair shares its exact checkpoint revision, prompt version, requested four QA pairs, 1,024-token maximum, temperature 0, top-p 1, seed 20261001, and disabled thinking mode. Only the presence of the RGB image changes. Backend-specific parameters that do not change semantic generation—such as InternVL dynamic tiling or Mistral serialization—are recorded as preprocessing exceptions.

## Backend and serving assessment

The preferred path is an OpenAI-compatible vLLM server with the pinned revision, BF16, a deliberately bounded context length, and one process per occupied GPU. Representative launch form:

```bash
CUDA_VISIBLE_DEVICES=<gpu> vllm serve <checkpoint> \
  --revision <revision> \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port <port>
```

Qwen3 text runs explicitly disable thinking through chat-template arguments. Mistral uses its official `tokenizer_mode=mistral`, `config_format=mistral`, and `load_format=mistral` controls; its processor probe additionally required the upstream-recommended `fix_mistral_regex=true`, applied identically to C04 and C06. InternVL uses remote code and its native dynamic image tiling when the Transformers fallback is needed.

At the pinned revision, the Gemma-4-12B repository card supplies a Transformers `AutoModelForMultimodalLM` example but no vLLM launch recipe. The Gemma-4-31B Hugging Face integration and the official [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html) do provide a serving path, including a text-only mode that disables image profiling. Although local vLLM 0.30.0 recognizes the newer `Gemma4UnifiedForConditionalGeneration` architecture used by 12B, its real backend is intentionally unresolved until C03/C09 smoke tests. A Transformers fallback is predeclared; no checkpoint substitution is allowed silently.

## Execution waves

Four GPUs cannot retain all seven checkpoints simultaneously. Checkpoints are therefore loaded in waves of at most four. Each same-checkpoint pair runs back-to-back while its server remains loaded, reducing load overhead and preventing accidental revision or runtime drift. GPU ID, load time, per-case latency, backend/package versions, and peak allocated/reserved VRAM are run metadata rather than properties of a configuration.

The original nine conditions C01–C09 are mandatory. C10 and C11 are exploratory same-checkpoint ablations: if a verified image-free invocation still fails, the error and attempted backend are retained, while the original 45-case matrix may proceed. No failed model is replaced without documenting the incompatibility and proposed alternative first.

## Reproducibility decision record

- Checkpoint revisions are immutable 40-character commits in `configs/experiments.yaml`.
- The fixed sample manifest hash is validated when the registry loads.
- All text-only configurations set `allow_image: false`; request construction rejects image payloads independently in the next implementation stage.
- Model cards and the local vLLM registry establish plausibility only. A one-sample generation, schema validation, and telemetry capture are required before the five-sample run.
