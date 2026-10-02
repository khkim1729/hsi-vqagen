# One-Sample Smoke-Test Results

Probe date: 2026-10-02 UTC. This file is an execution record, not an estimate of model quality.

## Gate and fixed input

All configurations use the first locked sample,
`NEON_D01_BART_DP3_312000_4875000_bidirectional_reflectance-468-532X660-724`,
and request exactly four pairs. A configuration passes only when its pinned revision
loads, the request completes, strict normalization yields four pairs, telemetry is
written, and deterministic evaluation accepts the cell. C10 and C11 additionally
inherit the tested request invariant that serialized text-only payloads contain no
image element, base64 data, or RGB path.

## Pre-download compatibility

- Four H200 NVL devices and CUDA were visible to PyTorch.
- vLLM 0.30.0 listed all six distinct architecture names used by the seven checkpoints;
  actual import/runtime loading is still checked by this smoke gate.
- Pinned `AutoConfig` and processor/tokenizer loading passed for all seven checkpoints.
- Mistral tokenizer inspection required `fix_mistral_regex=true`; the control is
  identical in C04/C06. Its initial vLLM/Transformers Pixtral API mismatch was
  reproduced and resolved with the pinned dependency pair recorded below.
- GPU 0 contained unrelated processes and is excluded from the smoke schedule.
- Weight acquisition is constrained by the host's 100 Mb/s network link. Downloads
  use a resumable `/data` cache and are not copied into Git.

## Configuration outcomes

The table below is populated only from real case manifests. Until a row has been run,
its state remains `not run`; this is not a passing result.

| ID | Condition | Checkpoint | State | Backend | Valid pairs | Load / latency / peak VRAM | Evidence |
|---|---|---|---|---|---:|---|---|
| C01 | RGB + description | Qwen3-VL-8B | pass | vLLM 0.30.0 | 4 | 129 s / 1.881 s / 116.76 GiB | `outputs/smoke_early_v2/qwen3_vl_8b_multimodal/run_manifest.json` |
| C02 | RGB + description | InternVL3-8B | pass | vLLM 0.30.0 | 4 | 128 s / 2.132 s / 118.01 GiB | `outputs/smoke_early_v2/internvl3_8b_multimodal/run_manifest.json` |
| C03 | RGB + description | Gemma-4-12B | pass | vLLM 0.30.0 | 4 | 218 s / 4.446 s / 118.42 GiB | `outputs/smoke_early_v2/gemma4_12b_multimodal/run_manifest.json` |
| C04 | RGB + description | Mistral Small 3.1 24B | pass | vLLM 0.30.0 | 4 | 114 s / 5.077 s / 117.60 GiB | `outputs/smoke_early_v2/mistral_small_3_1_24b_multimodal/run_manifest.json` |
| C05 | description only | Qwen3-8B | not run | — | — | — | — |
| C06 | description only | Mistral Small 3.1 24B | pass | vLLM 0.30.0 | 4 | 114 s / 4.822 s / 117.60 GiB | `outputs/smoke_early_v2/mistral_small_3_1_24b_text_only/run_manifest.json` |
| C07 | description only | Gemma-4-31B | not run | — | — | — | — |
| C08 | description only | Qwen3-32B | not run | — | — | — | — |
| C09 | description only | Gemma-4-12B | pass | vLLM 0.30.0 | 4 | 218 s / 4.586 s / 118.42 GiB | `outputs/smoke_early_v2/gemma4_12b_text_only/run_manifest.json` |
| C10 | description only | Qwen3-VL-8B | pass | vLLM 0.30.0 | 4 | 129 s / 2.125 s / 116.76 GiB | `outputs/smoke_early_v2/qwen3_vl_8b_text_only/run_manifest.json` |
| C11 | description only | InternVL3-8B | pass | vLLM 0.30.0 | 4 | 128 s / 2.159 s / 118.01 GiB | `outputs/smoke_early_v2/internvl3_8b_text_only/run_manifest.json` |

The first C04 server attempt failed before an inference request because vLLM 0.30.0
imports `PixtralRotaryEmbedding`, which is absent from Transformers 5.18.0. Both the
native Mistral/Pixtral loader and forced Hugging Face `Mistral3ForConditionalGeneration`
path reached the same import. An isolated wheel check verified the API in Transformers
5.10.4; pinning it with Tokenizers 0.22.2 restored loading. Mistral tokenizers also
reject `chat_template_kwargs`, so that serving-only option is omitted identically for
C04/C06. Both smoke tests then passed without substituting the checkpoint.

## Failure policy

No checkpoint is substituted silently. A failed row retains its server log, raw/error
artifact, pinned revision, attempted backend, and smallest reproducer. Gemma-4-12B is
the only checkpoint with a predeclared Transformers fallback if its unified architecture
fails under vLLM. The five-sample run remains gated on C01--C09; C10/C11 may be retained
as documented exploratory compatibility failures under the approved design.
