# 한국어 VQA 평가 방법과 5-sample 예비 결과

실행일: 2026-10-02 UTC
범위: 7개 checkpoint, 11개 configuration, 고정 5개 sample

## 무엇을 평가했는가

각 configuration은 같은 5개 sample에서 정확히 4개의 한국어 VQA를 생성했다. 현재 평가
대상은 총 55 case, 220 VQA이다. `Source/GT description`은 기존 HSI 분석 pipeline이 만든
paired description을 뜻한다. **정답 VQA annotation을 뜻하지 않는다.** 따라서 description과의
유사도는 grounding proxy일 뿐 correctness 점수가 아니다.

모든 C01--C11 조건은 같은 sample manifest와 한국어 prompt로 완료했다.

## VQA를 생성한 prompt와 검증 과정

모든 조건은 [`prompts/vqa_generation_ko_v1.txt`](../prompts/vqa_generation_ko_v1.txt)의
동일 template을 사용한다. `{evidence_mode}`만 `RGB + description` 또는
`description only`로 바뀌고, `{description}`에는 해당 sample의 한국어 paired description이
들어간다. Multimodal 조건에서는 HSI-derived RGB를 text보다 먼저 전달하고, text-only
조건에서는 image content와 RGB path를 모두 제거한다.

Prompt는 정확히 4개의 서로 다른 VQA, 자연스러운 한국어 question/answer/evidence, 짧고
구체적인 answer, 입력에 근거한 evidence, strict JSON 출력을 요구한다. Description에 없는
spectral/material 정보를 RGB만으로 추론하지 못하게 하고, text-only 모델이 image를 직접
보았다고 표현하는 것도 금지한다. JSON key와 `spatial`, `counting`, `comparison`,
`attribute`, `interpretive`의 `question_type` 값만 영어로 유지한다.

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

Prompt file의 SHA-256, prompt version, checkpoint revision, generation settings는
`generation_config.json`과 각 normalized row에 기록한다. Validator는 pair 수, schema,
provenance와 중복 질문을 검사한다. 선택적 validation retry는 최초 실패 시 seed를
20261001에서 20261002로 바꾸는 최대 1회이며, 원시 응답과 retry 여부를 남긴다.

## 자동 평가

- 한국어 완결률: 질문과 evidence에 한글이 있고, 답변은 한글 또는 `43%` 같은 완결된
  숫자·기호 응답인지 검사한다.
- source-description cosine: `answer + evidence`와 source description의 각 문장 사이에서
  character 2--4 gram TF-IDF cosine 최댓값을 구한다. 복사에 가까울수록 높아질 수 있으므로
  의미 정확도나 VQA 품질 순위로 해석하면 안 된다.
- 질문 다양성: 한 case의 질문 4개 간 평균 cosine을 1에서 뺀 값이다. 높을수록 표면형이
  덜 겹치지만, 의미 중복을 완전히 판별하지는 못한다.
- 경고 검사: description-only 요청의 직접 이미지 접근 주장과 source에 없는 spectral term을
  탐지한다.
- latency: 5개 case의 순차 생성 wall time 평균이다. load time과 vLLM 예약 메모리는 별도
  run manifest에 기록한다.

| ID | checkpoint / condition | 한국어 완결률 | source cosine | 질문 다양성 | 평균 latency (s) |
|---|---|---:|---:|---:|---:|
| C01 | Qwen3-VL-8B / RGB + description | 1.00 | 0.632 | 0.966 | 3.15 |
| C02 | InternVL3-8B / RGB + description | 1.00 | 0.821 | 0.926 | 4.09 |
| C03 | Gemma-4-12B / RGB + description | 1.00 | 0.559 | 0.980 | 5.30 |
| C04 | Mistral Small 3.1 24B / RGB + description | 1.00 | 0.537 | 0.900 | 6.10 |
| C05 | Qwen3-8B / description only | 1.00 | 0.625 | 0.869 | 3.06 |
| C06 | Mistral Small 3.1 24B / description only | 1.00 | 0.585 | 0.923 | 5.48 |
| C07 | Gemma-4-31B / description only | 1.00 | 0.544 | 0.978 | 10.28 |
| C08 | Qwen3-32B / description only | 1.00 | 0.504 | 0.968 | 10.64 |
| C09 | Gemma-4-12B / description only | 1.00 | 0.572 | 0.980 | 5.29 |
| C10 | Qwen3-VL-8B / description only | 1.00 | 0.557 | 0.943 | 2.88 |
| C11 | InternVL3-8B / description only | 1.00 | 0.652 | 0.921 | 3.73 |

모든 조건의 direct-image-access 경고와 unsupported-spectral-term 경고는 0건이었다.
InternVL3의 높은 source cosine은 긴 source 구절을 답변에 그대로 사용하는 경향의 영향이
크므로 최고 correctness로 해석하지 않는다. C03/C09의 질문 다양성이 거의 같고 latency도
비슷하며, 실제 출력도 상당 부분 source description 중심이었다. 5개 표본만으로 RGB의 유의미한
기여를 주장할 근거는 아직 없다.

CSV와 계산 전 JSON은 각각 `artifacts/korean_vqa_metrics.csv`와
`artifacts/korean_vqa_metrics.json`에 저장했다.

## 원문 대조 정성 점검

220개 VQA를 configuration별로 펼쳐 보고 source description과 대표 RGB를 대조했다.
확인된 핵심 사항은 다음과 같다.

- 수치 표현은 한국어 prompt에서 정상화되었다. 이전 영어 실험에서 InternVL3가 `4/3`,
  `5/1`로 잘못 썼던 BART 비율을 이번에는 `4분의 3`, `5분의 1`로 생성했다.
- Gemma-4-31B(C07)는 20개 모두 source에 직접 근거한 질문·답변을 만들었고, KONZ의
  건물/포장/열린 물 부재, HARV의 43%·16%, DEJU의 34%·32%를 보존했다.
- Qwen3-32B(C08)는 20개 모두 schema와 grounding 경고 검사를 통과했고 수치·공간 관계를
  대체로 보존했다. 다만 HARV의 한 질문에 `보입니다가?`라는 조사 결합 오류가 있어 한국어
  완결성 검사가 문법적 자연스러움까지 보장하지 않음을 확인했다.
- Qwen3-8B(C05)의 KONZ 문답 중 “식생 밀도가 가장 낮은 지역”을 “화면 가장자리의 어두운
  부분”이라고 답한 것은 source의 “더 촘촘하거나 그늘진 건조 식생”과 맞지 않는다.
- InternVL3 text-only(C11)의 SRER 첫 답변은 `연속적인 수면 반사`가 주를 이룬다고 썼지만,
  source는 그러한 반사가 넓게 이어지지 않는다고 설명한다. 어휘 유사도만으로는 잡기 어려운
  의미 반전 사례다.
- InternVL3 multimodal(C02)의 BART 네 번째 문답은 “수관의 분포” 질문에 수관 그림자나
  물이 밴 낮은 식생이라고 답해 질문-답변 초점이 불명확하다.
- Mistral C04 최초 실행 한 건은 같은 질문을 두 번 생성해 strict validator가 차단했다.
  해당 failed case만 재실행했으며 최종 5건은 모두 서로 다른 질문 4개를 통과했다. 이 이력은
  실패를 숨기지 않고 문서화하며, 향후 실행은 검증 실패 시 seed+1의 최대 1회 재시도를
  manifest에 기록할 수 있다.

이 점검은 연구자 1인의 비맹검 예비 검토다. 논문용 correctness·grounding·hallucination·
usefulness 점수에는 최소 2인의 독립 평가와 불일치 조정이 필요하다.

## 확정한 human evaluation rubric

평가자 2인이 서로의 점수를 보지 않고 각 VQA를 1--5점으로 평가한다.

1. correctness: answer가 source/RGB와 모순되지 않는가
2. grounding: answer와 evidence가 제공된 입력에서 확인 가능한가
3. hallucination: 입력에 없는 객체·수치·분광 주장을 만들지 않았는가
4. usefulness: HSI 장면 이해에 유용한 질문인가
5. answer specificity: 모호한 일반론이 아니라 확인 가능한 답인가

추가로 question diversity와 visual dependence를 case 단위 1--5점으로 평가한다.
description-only 평가자에게는 RGB를 보여주지 않고, multimodal 평가에는 RGB와 description을
함께 제공해야 입력 조건이 섞이지 않는다. Gemma-4-12B C03/C09는 같은 checkpoint의 controlled
modality ablation으로 별도 비교한다. 두 평가자의 점수가 2점 이상 차이나면 근거를 기록해
조정하고, 원점수·조정점수·일치도를 모두 보존한다.

## 재현 명령

```bash
.venv/bin/python scripts/evaluate_korean_outputs.py \
  --outputs outputs/korean_comparison \
  --json artifacts/korean_vqa_metrics.json \
  --csv artifacts/korean_vqa_metrics.csv

.venv/bin/python scripts/generate_korean_figures.py \
  --outputs outputs/korean_comparison
```

원시 응답과 normalized JSONL은 용량·데이터 정책 때문에 Git에서 제외되며, 각 실행 디렉터리의
`generation_config.json`, `run_manifest.json`, `raw/`, `cases/`, `normalized.jsonl`,
`errors.jsonl`에 로컬 보존된다.
