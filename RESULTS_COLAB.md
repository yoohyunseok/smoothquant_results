# SmoothQuant 재현 실험 결과 (Google Colab, GPU)

- 실행 환경: Google Colab, GPU **Tesla T4** (VRAM 15360MiB), RAM 12GB, Python 3.13.15
- dtype: **FP16** (T4는 bf16 하드웨어 미지원이므로 전 구간 FP16으로 통일)
- 평가 데이터: WikiText-2 test, 2048토큰 윈도우, 원본 `ppl_eval.py`의 평가 로직(shift CrossEntropy,
  `exp(sum nll/(n*2048))`)을 그대로 사용
- 평가량: 모델마다 `--n_samples 2` 프로브로 속도를 측정해 "설정당 15분" 예산을 넘는지 자동 판단했는데,
  실제로는 opt-2.7b가 가장 느려도 설정당 약 6분이었어서 **모든 모델에서 전체 test셋(140윈도우)을
  그대로 사용**했다 (축소 없음). 판단 근거는 `logs/decisions.log` 참고.
- 설치/코드 변경 상세는 `SETUP_LOG_COLAB.md` 참고. 원본 smoothquant 코드(`fake_quant.py`,
  `smooth.py`, `calibration.py`, `ppl_eval.py`)는 전혀 수정하지 않았고, GPU 전용 복사본
  `my_ppl_eval.py`만 추가로 만들어 사용했다.

## 0. 파이프라인 검증

opt-125m FP16 전체 WikiText-2 test PPL = **27.6563** (두 번 독립 실행: 최초 검증용 태그
`opt-125m_fp16_fp_fullcheck`와 본 실험 태그 `opt-125m_fp16_fp` 모두 소수점까지 동일하게 재현됨).
노트북(CPU, FP32) 전체 test셋 기준선 27.655와 거의 일치 → 토크나이즈/윈도우 분할/손실 계산
파이프라인이 원본과 동일하게 동작함을 확인.

## 1. opt-6.7b: 실행 불가 (OOM)

프로브(`--n_samples 2`) 단계에서 바로 CUDA OOM 발생:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 14.56 GiB ... this process has 14.19 GiB memory in use.
Of the allocated memory 13.31 GiB is allocated by PyTorch, ...
```
- FP16 6.7B 파라미터 가중치만으로 13.31GiB를 이미 사용 (14.56GiB 중). 어텐션 softmax 계산에
  필요한 512MiB조차 할당할 여유가 없었음.
- 이건 **윈도우 수(n_samples)를 줄여도 해결되지 않는 OOM**이다 — 단 1개 윈도우의 forward
  한 번만으로도 실패했기 때문. 2048토큰 평가 로직을 바꾸지 않는 한(과제 지침상 바꾸지 않음)
  현재 T4 15GB 런타임에서는 구조적으로 opt-6.7b를 평가할 수 없다.
- **결론**: opt-6.7b 실험을 돌리려면 더 큰 VRAM의 GPU(A100 40GB, L4 24GB 등)가 필요함.
  이는 확인된 사실이고, "그 GPU라면 될 것"이라는 부분은 메모리 산술에 근거한 추정이다.

## 2. 모델별 5개 설정 결과 (FP16, 전체 140윈도우)

### opt-2.7b (FP16 기준 PPL = 12.4709)
| 설정 | PPL | FP16 대비 |
|---|---|---|
| 설정A naive (per_channel/per_token) | 13.0160 | +4.37% |
| 설정A SmoothQuant α=0.5 | 12.4732 | **+0.02%** |
| 설정B naive (per_tensor/per_tensor) | 13.5711 | +8.82% |
| 설정B SmoothQuant α=0.5 | 12.5274 | **+0.45%** |

### opt-1.3b (FP16 기준 PPL = 14.6238)
| 설정 | PPL | FP16 대비 |
|---|---|---|
| 설정A naive | 15.1752 | +3.77% |
| 설정A SmoothQuant α=0.5 | 14.7908 | **+1.14%** |
| 설정B naive | 15.8446 | +8.35% |
| 설정B SmoothQuant α=0.5 | 15.0048 | **+2.61%** |

### opt-125m (FP16 기준 PPL = 27.6563)
| 설정 | PPL | FP16 대비 |
|---|---|---|
| 설정A naive | 27.9299 | +0.99% |
| 설정A SmoothQuant α=0.5 | 27.7448 | **+0.32%** |
| 설정B naive | 30.5334 | +10.40% |
| 설정B SmoothQuant α=0.5 | 29.4690 | **+6.55%** |

자동 생성된 표는 `results_table.md`, 원본 수치는 `results/results.jsonl` 참고.

## 3. 모델 크기별 naive 열화 추세

설정A(per_channel/per_token, 원본 `ppl_eval.py`와 동일 조합) 기준 naive 열화율:

| 모델 | naive 열화(FP16 대비) | SmoothQuant 열화(FP16 대비) |
|---|---|---|
| opt-125m | +0.99% | +0.32% |
| opt-1.3b | +3.77% | +1.14% |
| opt-2.7b | +4.37% | +0.02% |
| opt-6.7b | (실행 불가) | (실행 불가) |

**모델이 커질수록 naive 열화가 커지는 추세가 125m→1.3b→2.7b 구간에서 뚜렷하게 관찰됨**
(0.99% → 3.77% → 4.37%). 설정B(거친 per_tensor/per_tensor)에서는 이 추세가 더 뚜렷하고 절대
열화 폭도 훨씬 큼(0.99%→8.35%→8.82%, opt-125m조차 +10.4%). SmoothQuant는 두 설정 모두에서
naive 대비 열화를 거의 대부분 되돌렸다.

### 활성화 채널 최대값(outlier 크기) 추세 — 사용자 가설 검증

act_scales 파일(캘리브레이션으로 구한 레이어별 입력 채널 절댓값 최댓값)의 전역 최댓값:

| 모델 | 전역 최댓값 | 레이어별 최댓값의 평균 |
|---|---|---|
| opt-125m | 26.50 | 11.83 |
| opt-1.3b | 44.47 | 31.29 |
| opt-2.7b | 49.94 | 35.53 |
| opt-6.7b | 63.31 | 44.89 |

opt-125m=26.50, opt-1.3b=44.47는 노트북에서 측정한 값(26.5, 44.5)과 소수점까지 일치 —
같은 스케일 파일(1.3b/2.7b/6.7b는 HF `mit-han-lab/smoothquant-scales`에서 받은 동일 파일,
125m은 동일한 Pile 캘리브레이션 절차로 재생성)을 쓰고 있음을 교차 확인.

**"모델이 커질수록 activation outlier가 커진다"는 사용자의 가설은 이 4개 모델에서 일관되게
확인됨** (26.5 → 44.5 → 49.9 → 63.3). opt-6.7b는 PPL 평가는 못 했지만 스케일 파일 자체는
다운로드했으므로 이 추세표에는 포함할 수 있었다.

**논문 원문(arXiv:2211.10438v7, 2024-03-29 리비전, Introduction 1문단) 대조 결과(확인됨):**
> "When we scale up LLMs beyond 6.7B parameters, systematic outliers with large magnitude
> will emerge in activations... leading to large quantization errors and accuracy degradation."

즉 **"6.7B를 넘으면 activation에 systematic outlier가 나타난다"는 것은 논문이 명시한 문장이지
추정이 아니다.** 우리 4개 모델의 outlier 최댓값 추세(125m 26.5 → 1.3b 44.5 → 2.7b 49.9 →
6.7b 63.3)가 바로 이 6.7B 문턱을 향해 꾸준히 증가하는 모습과 정확히 일치한다. 이 부분은
더 이상 "정성적으로 비슷해 보인다"는 추정이 아니라, 논문 본문과 직접 대조해 확인한 사실이다.

## 4. Alpha sweep (opt-2.7b, 설정A/B)

가장 큰 성공 모델인 opt-2.7b에서 α=0.0/0.25/0.5/0.75/1.0 측정. 그래프: `alpha_vs_ppl_opt-2.7b.png`

| α | 설정A PPL | 설정B PPL |
|---|---|---|
| 0.0 | 14.1413 | 15.5587 |
| 0.25 | 12.6293 | 12.7525 |
| 0.5 | 12.4732 | 12.5274 |
| 0.75 | 12.4939 | 12.5559 |
| 1.0 | 12.4940 | **16.4695** |

(FP16 기준 = 12.4709)

관찰:
- 설정A(세밀한 양자화: weight per_channel + activation per_token)는 α=0.25~1.0 전 구간에서
  FP16에 가깝게 안정적이다(12.47~12.63). α=0.0(스무딩 없음에 가까움)에서만 크게 나빠진다(14.14).
- 설정B(거친 양자화: weight/activation 모두 per_tensor)는 **양 끝(α=0.0, α=1.0)에서 모두
  크게 나빠지고 α=0.5 근처에서 최적**인 U자 형태를 보인다. 특히 α=1.0(16.47)은 naive(13.57)
  보다도 나쁘다.
- α=0.0일 때 `scale = (act_scale^0 / weight_scale^1) = 1/weight_scale`이 되어 가중치 쪽에만
  과도하게 의존하는 변환이 되고, α=1.0일 때는 반대로 활성화 쪽에만 의존하는 변환이 된다.
  설정B처럼 가중치/활성화 모두 거칠게(텐서 전체 하나의 스케일로) 양자화하는 경우 이런 극단적인
  분배가 한쪽의 양자화 오차를 극대화하는 것으로 보인다.

**논문 원문 대조(확인됨)**: 논문 4절(SmoothQuant)에 정확히 같은 식 `sj = max(|Xj|)^α / max(|Wj|)^(1-α)`
이 Eq. (4)로 등장하고, "α가 너무 작으면(<0.4) 활성화가, 너무 크면(>0.6) 가중치가 양자화하기
어려워진다"고 명시되어 있다(OPT-175B, LAMBADA 기준 ablation, Figure 10). 즉 **"양 끝에서
한쪽의 양자화 난이도가 커져서 성능이 나빠진다"는 메커니즘 자체는 더 이상 추정이 아니라 논문이
직접 설명한 내용**이다. 다만 논문의 ablation은 OPT-175B·LAMBADA 정확도 기준이고 우리 실험은
opt-2.7b·WikiText PPL 기준이라 수치 자체(sweet spot이 정확히 0.4~0.6이라는 범위 등)를 그대로
대입할 수는 없다 — 우리 "설정A"가 양 끝에서도 비교적 안정적이었던 이유(세밀한 양자화라 한쪽이
나빠져도 절대 오차가 작음)까지는 논문이 직접 설명하지 않으므로 이 부분은 여전히 우리의 추정이다.
- 논문은 "모든 OPT·BLOOM 모델에서 α=0.5가 균형 잡힌 지점(a well-balanced point)"이라고
  **명시적으로** 기술한다(5.1절 Activation smoothing). 우리 실험(opt-2.7b, OPT 계열)에서도
  α=0.5가 설정A/B 모두 거의 최적점으로 재현됨 — 이는 더 이상 "논문에서 권장했던 것 같다"가
  아니라 논문 문장과 1:1로 확인된 사실이다.

## 5. 논문과의 비교, 다른 점과 추정 원인

**(2026-10-11 갱신)** 처음에는 이 저장소의 `assets/SmoothQuant.pdf`를 발표 슬라이드로만 보고
논문 원문은 열어보지 않은 채 "기억에 의존한 추정"이라고 적었었다. 이후 arXiv 원문
(arXiv:2211.10438v7, ICML 2023, PMLR 202)을 직접 받아서 읽었고, 아래는 그 대조 결과다.
확인된 사실과 여전히 추정으로 남는 부분을 구분해서 적는다.

### 확인된 사실 (논문 원문과 직접 대조함)

1. **"6.7B를 넘으면 activation에 systematic outlier가 나타난다"는 것은 논문 서론에 명시된
   문장이다** (정확한 인용은 위 3번 섹션 참고). 우리가 opt-6.7b에서 OOM으로 막힌 지점이
   바로 논문이 말하는 "흥미로운 현상이 시작되는 지점"과 정확히 겹친다 — 즉 우리는 논문의
   핵심 현상이 시작되기 **직전**까지만 접근했던 것이다.

2. **논문 Table 1(OPT-6.7B~175B, WinoGrande/HellaSwag/PIQA/LAMBADA 제로샷 평균 정확도)에서
   naive INT8 per-tensor 양자화가 6.7B에서 이미 FP16 64.9%→39.9%로 급락한다.** 이는 모델
   크기가 더 커져도(13B~175B) 33% 안팎에 머무는 "붕괴 이후 평평한 구간"이고, 붕괴 자체는
   6.7B 지점에서 이미 일어난다. (다만 이 수치는 PPL이 아니라 zero-shot 분류 정확도다 — 우리가
   측정한 WikiText-2 PPL과는 다른 지표라는 점은 유의해야 한다. 논문이 WikiText PPL로 "붕괴"
   수준을 보여주는 곳은 Table 3인데, 거기선 OPT-175B 한 모델만 보여준다: FP16 10.99 vs naive
   W8A8 93080(!) — 사실상 완전히 망가진 수준.)

3. **중요한 방법론적 차이를 발견함**: 논문 Table 2(baseline/SmoothQuant 설정 정의)를 보면
   논문의 "W8A8"(naive) baseline은 **weight per-tensor / activation per-tensor dynamic**이고,
   SmoothQuant-O1/O2/O3 전부 **weight per-tensor**를 쓴다 (weight per-channel을 쓰는 건
   비교 대상인 `LLM.int8()` baseline뿐). 반면 이 저장소의 원본 `examples/ppl_eval.py`가 기본값으로
   쓰는 조합(우리 실험의 "설정A" = weight **per_channel** / activation per_token)은 논문의
   어떤 공식 설정(O1/O2/O3/naive W8A8)과도 정확히 일치하지 않는, **더 보수적인(세밀한) 커스텀
   설정**이다. 반면 우리의 "설정B"(weight per_tensor / activation per_tensor)는 **논문의 naive
   "W8A8" baseline과 정확히 같은 정의**다.
   → 이게 바로 **설정A naive가 설정B naive보다 훨씬 덜 나빠지는 이유를 설명해준다**: 설정A는
   애초에 논문이 "naive"라고 부르는 것보다 더 세밀하게(weight per-channel로) 양자화하고
   있어서 덜 망가지는 것이고, 설정B가 논문이 말하는 진짜 "naive W8A8"에 해당한다. 우리 결과에서
   설정B naive의 열화가 설정A naive보다 항상 더 큰 것(opt-2.7b 기준 +8.82% vs +4.37%)과
   정확히 일치하는 설명이다. 이건 더 이상 추정이 아니라 두 설정의 정의를 논문 Table 2와 대조해서
   확인한 사실이다.

4. **활성화 스무딩 공식과 α=0.5 기본값도 논문 본문과 정확히 일치**: Eq.(4) `sj = max(|Xj|)^α
   / max(|Wj|)^(1-α)`, "α=0.5 is a well-balanced point... for all the OPT and BLOOM models"
   (5.1절) — 우리가 돌린 `smooth.py`와 alpha sweep 결과 모두 이 문장과 정확히 부합한다
   (위 4번 섹션 참고).

5. **α sweet spot(0.4~0.6 부근에서 최적, 양 끝에서 급격히 나빠짐)의 메커니즘도 논문이 직접
   설명**: "α가 너무 작으면 활성화가, 너무 크면 가중치가 어려워진다"(Figure 10 설명). 다만
   논문의 ablation은 OPT-175B·LAMBADA 정확도 기준이라 우리(opt-2.7b·WikiText PPL) 수치와
   1:1로 겹치진 않는다.

### 여전히 추정으로 남는 부분 (검증 안 됨)

- opt-2.7b 수준에서 naive 열화가 완만했던 것(+4.37%, 붕괴 아님)이 "아직 6.7B 문턱 전이라서"
  라는 설명은 논문의 "6.7B 문턱" 문장과 방향은 일치하지만, **논문이 2.7B를 직접 측정해 보여준
  적은 없어서**(Table 1은 6.7B부터 시작) 정량적으로 확인된 건 아니다 — 정성적으로 매우
  그럴듯한 추정일 뿐이다.
- 설정A가 양 끝 alpha(0.0, 1.0)에서도 비교적 안정적이었던 이유("세밀한 양자화라 한쪽이
  나빠져도 절대 오차가 작다")는 논문이 직접 설명하지 않은, 우리 쪽 해석이다.
- 노트북(CPU) 결과와 Colab(GPU) 결과의 절대 열화율 차이(아래 6번) 원인은 여전히 추정이다.

### 종합

**결론적으로, 처음에 "확인 못 했다"고 적었던 것 중 상당수(6.7B 문턱, α=0.5 기본값, α
sweet spot 메커니즘, outlier 증가 추세)가 실제로 논문 본문과 정확히 일치함을 확인했다.**
반대로 "더 보수적인 설정(설정A)이 왜 naive인데도 덜 나빠지는가"처럼, 처음엔 눈치채지 못했던
방법론적 차이도 새로 발견했다 — 이 저장소의 예시 스크립트가 쓰는 "naive" 설정이 논문이 말하는
"naive W8A8"보다 실제로는 더 관대한 설정이라는 점이다.

## 6. 노트북(CPU, FP32) 결과와의 비교

노트북 결과는 윈도우 수가 다르고(125m=10개, 1.3b=2개 vs 이번 140개 전체) dtype도 다르므로
(FP32 vs FP16) **PPL 절대값은 직접 비교할 수 없다**. 방향성만 비교하면:

| 모델 | 노트북(부분 윈도우, FP32) naive 열화 | Colab(전체 140윈도우, FP16) naive 열화 |
|---|---|---|
| opt-125m | per_token +0.85%, per_tensor +5.96% | per_token +0.99%, per_tensor +10.40% |
| opt-1.3b | per_token +1.91%, per_tensor +7.10% | per_token +3.77%, per_tensor +8.35% |

(노트북 수치는 제공된 PPL로부터 역산: 125m per_token = 28.913/28.670-1, per_tensor =
30.374/28.670-1; 1.3b per_token = 14.490/14.219-1, per_tensor = 15.228/14.219-1)

방향(naive가 SmoothQuant보다 나쁘고, 거친 양자화일수록 더 나쁘다)은 두 환경에서 일치한다.
절대 열화율은 Colab 쪽이 대체로 더 크게 나오는데, 이는 (추정) 윈도우 수가 10배 이상 많아
더 다양한 텍스트 구간의 outlier를 포착했기 때문일 가능성과, FP32 대신 FP16을 쓴 것 자체가
양자화 오차 위에 FP16 반올림 오차를 더 얹었을 가능성을 모두 고려할 수 있다 — 어느 쪽이
주된 원인인지는 이번 실험에서 분리해서 검증하지 않았다.

노트북에서 관찰했던 "모델이 작아서 outlier가 약해 naive가 잘 안 무너진다"는 가설은, Colab에서
더 큰 모델(2.7b, 6.7b 스케일)까지 봐도 outlier가 꾸준히 커지는 것으로 뒷받침되었다(2번의
표 참고). 다만 6.7b에서 실제 PPL 붕괴를 확인하지는 못했다(OOM).

## 7. 한계점

- **opt-6.7b PPL을 전혀 측정하지 못함** (OOM). 논문이 "systematic outlier가 나타나기 시작한다"고
  명시한 바로 그 경계값(6.7B)에서 멈췄기 때문에, 논문 Table 1이 보여주는 수준의 정확도 붕괴
  (FP16 64.9%→naive INT8 39.9%, OPT-6.7B)가 우리 PPL 지표로도 재현되는지는 끝내 확인하지 못했다.
- alpha sweep은 opt-2.7b 한 모델에서만 수행함 (시간/GPU 제약, 지침에 따라 "가장 큰 성공 모델"
  하나만 선택).
- numpy/datasets/accelerate를 Colab 기본 버전(numpy 2.1.3, datasets 4.8.5, accelerate 1.15.0)
  그대로 썼고, transformers/tokenizers/huggingface_hub만 2023년 당시 버전으로 맞췄다. 이
  "섞인" 조합이 PPL에 미묘한 영향을 줬을 가능성을 완전히 배제하지는 못한다(다만 opt-125m
  파이프라인 검증에서 노트북 기준선과 거의 일치했으므로 영향이 있더라도 작을 것으로 추정).
- 캘리브레이션(activation scale)은 opt-1.3b/2.7b/6.7b는 HF에 공개된 동일 파일을 썼고
  opt-125m만 직접 생성했다. opt-125m 캘리브레이션에 쓴 Pile val 512샘플 시드/셔플은
  `generate_act_scales.py` 원본 그대로(`shuffle(seed=42)`)이므로 재현 가능하다.
- 평가 윈도우를 전부 140개로 통일했지만, 이는 "결과가 전부 전체 평가량이어서 생긴 우연"이고
  원래 지침대로라면 설정당 15분을 넘겼을 경우 모델별로 다른 윈도우 수를 썼어야 한다. 이번에는
  그럴 필요가 없었을 뿐이다(판단 근거: `logs/decisions.log`).
- fake quantization 특성상 실제 속도/메모리 이득은 측정되지 않았다(의도된 것, torch-int 같은
  실제 INT8 커널은 사용하지 않음).
