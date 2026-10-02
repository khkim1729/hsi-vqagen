# HSI-VQAGen

HSI에서 추출한 RGB와 기존 HSI 분석 시스템이 만든 한국어 description을 이용해
한국어 visual question answering(VQA) 데이터를 생성하는 재현 가능한 feasibility study다.
일반 VLM에 raw hyperspectral cube를 직접 입력하지 않는다.

## 한눈에 보는 현재 상태

| 항목 | 현재 상태 |
|---|---|
| 고정 sample | 5개, 모든 configuration에서 동일 ID·순서 사용 |
| 등록 checkpoint | 7개 open checkpoint |
| 실험 configuration | 11개: RGB + description 4개, description only 7개 |
| 완료 | 10개 configuration, 50 case, 한국어 VQA 200개 |
| 대기 | C08 Qwen3-32B 다운로드 후 smoke/full run |
| 한국어 형식 통과 | 완료된 200/200 VQA |
| 주요 controlled ablation | Gemma-4-12B C03 vs C09 |
| 원시 HSI 입력 여부 | 없음; HSI-derived RGB와 paired description만 사용 |

현재 수치는 동일한 고정 5개 sample로 수행한 예비 결과다. 전체 데이터셋 결과나 통계적
우월성 주장이 아니며, correctness는 자동 cosine만으로 판정하지 않는다.

## 모델 및 실험 조건

모델 architecture와 실제 input condition을 분리한다. 예를 들어 Gemma-4-31B는
multimodal-capable architecture지만 C07에서는 image를 전달하지 않는 text-only condition이다.

| ID | Checkpoint | 규모 | Architecture capability | Input condition | 완료 VQA | Source cosine | 평균 latency | 상태 |
|---|---|---:|---|---|---:|---:|---:|---|
| C01 | Qwen/Qwen3-VL-8B-Instruct | 8B | Multimodal | RGB + description | 20 | 0.632 | 3.15 s | 완료 |
| C02 | OpenGVLab/InternVL3-8B | 8B | Multimodal | RGB + description | 20 | 0.821 | 4.09 s | 완료 |
| C03 | google/gemma-4-12B-it | 12B | Multimodal | RGB + description | 20 | 0.559 | 5.30 s | 완료 |
| C04 | Mistral-Small-3.1-24B-Instruct-2503 | 24B | Multimodal | RGB + description | 20 | 0.537 | 6.10 s | 완료 |
| C05 | Qwen/Qwen3-8B | 8B | Text | Description only | 20 | 0.625 | 3.06 s | 완료 |
| C06 | Mistral-Small-3.1-24B-Instruct-2503 | 24B | Multimodal | Description only | 20 | 0.585 | 5.48 s | 완료 |
| C07 | google/gemma-4-31B-it | 31B | Multimodal | Description only | 20 | 0.544 | 10.28 s | 완료 |
| C08 | Qwen/Qwen3-32B | 32B | Text | Description only | -- | -- | -- | 다운로드 중 |
| C09 | google/gemma-4-12B-it | 12B | Multimodal | Description only | 20 | 0.572 | 5.29 s | 완료 |
| C10 | Qwen/Qwen3-VL-8B-Instruct | 8B | Multimodal | Description only | 20 | 0.557 | 2.88 s | 완료 |
| C11 | OpenGVLab/InternVL3-8B | 8B | Multimodal | Description only | 20 | 0.652 | 3.73 s | 완료 |

`Source cosine`은 answer와 evidence가 source description의 문장과 얼마나 어휘적으로
겹치는지 나타내는 character 2--4 gram TF-IDF proxy다. 원문을 길게 복사하면 높아질 수
있으므로 정확도나 모델 순위로 해석하지 않는다.

### 동일 checkpoint modality 비교

| Multimodal | Text-only | 통제되는 항목 |
|---|---|---|
| C01 Qwen3-VL-8B | C10 Qwen3-VL-8B | checkpoint, revision, prompt, decoding, sample |
| C02 InternVL3-8B | C11 InternVL3-8B | checkpoint, revision, prompt, decoding, sample |
| C03 Gemma-4-12B | C09 Gemma-4-12B | primary controlled ablation |
| C04 Mistral Small 3.1 24B | C06 Mistral Small 3.1 24B | checkpoint, revision, prompt, decoding, sample |

각 쌍에서 의도적으로 바뀌는 것은 HSI-derived RGB의 전달 여부뿐이다.

## VQA 생성 방법

### 입력

- Multimodal condition: `HSI-derived RGB + 한국어 paired description`
- Text-only condition: `한국어 paired description only`
- Description은 기존 HSI pipeline의 spectral/index map, clustering, region statistics,
  specialized agent 분석을 거쳐 만들어진 source artifact다.
- `Source/GT description`은 gold QA annotation이 아니라 VQA 생성의 근거 텍스트다.

### 한국어 prompt

실제 template은 [`prompts/vqa_generation_ko_v1.txt`](prompts/vqa_generation_ko_v1.txt)에
고정하고 SHA-256을 각 run manifest에 기록한다. 핵심 지시는 다음과 같다.

```text
[입력 근거 조건: {RGB + description 또는 description only}]

아래 원격탐사 sample에 대해 유용한 VQA 객체를 정확히 4개 생성하라.

- 질문, 답변, 근거를 모두 자연스러운 한국어로 작성한다.
- 네 질문은 서로 달라야 하고 제공된 입력만으로 답할 수 있어야 한다.
- description에 명시된 경우에만 분광·재질 정보를 사용한다.
- RGB만으로 보이지 않는 파장대나 확정적인 재질 정체를 추론하지 않는다.
- text-only 조건에서는 image/RGB를 직접 보았다고 표현하지 않는다.
- 답변은 짧고 구체적으로, evidence는 답을 지지하는 사실로 작성한다.
- 설명문이나 Markdown 없이 strict JSON 객체 하나만 반환한다.

Paired HSI-derived source description (한국어):
{description}
```

요청 schema는 다음과 같다. JSON key와 `question_type` 값만 영어를 유지한다.

```json
{
  "pairs": [
    {
      "question": "장면에서 가장 넓은 식생대의 비율은 얼마인가요?",
      "answer": "약 43%입니다.",
      "evidence": "가장 넓은 습윤 식생대가 약 43%를 차지합니다.",
      "question_type": "counting"
    }
  ]
}
```

허용하는 `question_type`은 `spatial`, `counting`, `comparison`, `attribute`,
`interpretive`다. 출력은 정확히 4개 pair인지, 필수 필드가 있는지, 질문이 중복되지
않는지를 strict validator로 검사한다. 검증 실패 시 선택적으로 seed를 1 증가시켜 최대
한 번만 재시도하며, 원시 시도와 validation status를 기록한다.

### 공통 generation setting

| 설정 | 값 |
|---|---:|
| VQA 수 | sample당 4개 |
| Temperature | 0.0 |
| Top-p | 1.0 |
| Seed | 20261001 |
| Max new tokens | 1,024 |
| Thinking mode | 비활성화 |
| Prompt version | `vqa-generation-ko-v1` |

모델별 chat template, image processor, InternVL dynamic tiling, Mistral serialization처럼
통일할 수 없는 항목은 `configs/experiments.yaml`과 generation metadata에 기록한다.

## 데이터와 기존 HSI description pipeline

고정 sample은 shard에서 직접 검증한 RGB `384 × 384`, HSI metadata `64 × 64 × 426`,
비어 있지 않은 한국어 description을 갖는다. 모든 configuration이
[`configs/feasibility_samples.yaml`](configs/feasibility_samples.yaml)의 동일 ID를 사용한다.

- [Dataset audit](docs/dataset_audit.md)
- [고정 5개 sample과 선정 근거](docs/feasibility_samples.md)
- [기존 HSI description pipeline 조사](docs/hsi_description_pipeline.md)
- [모델 선정 및 H200 호환성](docs/model_selection.md)
- [실행 환경](docs/environment.md)

Legacy description-generation 프로젝트는 read-only reference로 조사했으며 수정하지 않았다.

## 현재 결과

### 정량·정성 평가

- [한국어 VQA 평가 방법, 전체 지표와 원문 대조 결과](docs/korean_vqa_evaluation.md)
- [초기 영어 결과를 보존한 한국어 선비교 문서](docs/early_model_comparison_ko.md)
- [Machine-readable metrics JSON](artifacts/korean_vqa_metrics.json)
- [Metrics CSV](artifacts/korean_vqa_metrics.csv)

완료된 200개 VQA는 한국어 형식 검사를 모두 통과했다. 다만 수동 원문 대조에서
Qwen3-8B KONZ 한 답변과 InternVL text-only SRER 한 답변의 의미 오류를 확인했다.
따라서 최종 논문 평가는 자동 지표와 평가자 2인의 독립 human evaluation을 함께 사용한다.

### Qualitative grids

| 비교 | 대표 2개 sample | 전체 5개 sample |
|---|---|---|
| RGB + description | [PNG](figures/korean_multimodal_grid.png) · [PDF](figures/korean_multimodal_grid.pdf) | [PNG](figures/korean_multimodal_grid_all.png) · [PDF](figures/korean_multimodal_grid_all.pdf) |
| Description only | [PNG](figures/korean_text_only_grid.png) · [PDF](figures/korean_text_only_grid.pdf) | [PNG](figures/korean_text_only_grid_all.png) · [PDF](figures/korean_text_only_grid_all.pdf) |
| Gemma-4-12B ablation | [PNG](figures/korean_gemma4_12b_ablation.png) · [PDF](figures/korean_gemma4_12b_ablation.pdf) | [PNG](figures/korean_gemma4_12b_ablation_all.png) · [PDF](figures/korean_gemma4_12b_ablation_all.pdf) |

논문 본문은 대표 2개 sample을 사용하고 전체 5개 비교는 Appendix에 배치한다.

## 설치와 재현

```bash
python3 -m virtualenv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e .
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

로컬 경로는 Git에서 제외되는 `configs/local_paths.yaml`에 설정한다. 모델 weight와 cache는
repository 밖의 대용량 저장소에 둔다.

### 모델 다운로드와 smoke test

```bash
.venv/bin/python scripts/download_models.py
scripts/run_smoke_tests.sh
```

Smoke workflow는 GPU 0의 unrelated process를 피하고 GPU 1--3을 사용하며, 한국어 prompt로
각 configuration의 첫 sample을 검증한다.

### 한 configuration 실행 예시

```bash
scripts/serve_model.sh C01 1 8101 outputs/runtime/c01-8101

.venv/bin/python scripts/run_configuration.py C01 \
  --base-url http://127.0.0.1:8101/v1 \
  --output-root outputs/korean_comparison \
  --prompt-version vqa-generation-ko-v1 \
  --gpu-index 1 \
  --retry-invalid-once
```

### 평가와 figure 재생성

```bash
.venv/bin/python scripts/evaluate_korean_outputs.py \
  --outputs outputs/korean_comparison \
  --json artifacts/korean_vqa_metrics.json \
  --csv artifacts/korean_vqa_metrics.csv

.venv/bin/python scripts/generate_korean_figures.py \
  --outputs outputs/korean_comparison
```

## Output 구조

각 configuration은 다음 파일을 로컬에 남긴다.

```text
outputs/korean_comparison/<configuration>/
├── generation_config.json   # checkpoint/revision, prompt hash, decoding setting
├── run_manifest.json        # 완료/실패 ID, latency, load time, peak VRAM
├── raw/                     # 원시 model response
├── cases/                   # validation 결과와 raw attempts
├── normalized.jsonl         # 공통 VQA schema
└── errors.jsonl             # 실패와 오류 로그
```

Dataset, raw output, model cache, weight, token, local path는 Git에 넣지 않는다.

## 논문과 다음 단계

논문 제목은 *HSI-VQAGen: Comparing Vision-Language and Text-Only Models for
Hyperspectral VQA Generation*이다. Overleaf에는 영문 `main.tex`와 한국어 `main_ko.tex`,
`references.bib`, publication-quality PDF figure를 함께 관리한다.

남은 순서는 다음과 같다.

1. Qwen3-32B 다운로드 완료 후 C08 한국어 smoke test
2. 동일 5개 sample의 C08 full run 및 11-configuration 표 갱신
3. 평가자 2인의 독립 1--5점 human evaluation과 불일치 조정
4. 대표 2개 qualitative figure는 본문, 전체 5개는 Appendix에 배치
5. 5-sample feasibility 결론과 향후 전체 데이터 실행을 명확히 분리

## 보안과 재현성

- Token은 tracked file, README, source, log, remote URL에 기록하지 않는다.
- Git/Hugging Face credential은 repository 밖의 사용자 홈 credential store를 사용한다.
- Checkpoint는 40-character revision으로 고정한다.
- Text-only request에는 image payload와 RGB path를 모두 전달하지 않는다.
- 실패한 checkpoint를 문서화 없이 다른 모델로 교체하지 않는다.
