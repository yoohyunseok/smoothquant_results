# SmoothQuant Colab 결과표 (자동 생성)

각 설정이 끝날 때마다 run_all.sh가 이 파일을 다시 만든다. 직접 수정하지 말 것.


## opt-2.7b

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | 12.4709 | ok | 140 | +0.00% | 228.77 |
| 설정A naive (per_channel/per_token) | 13.0160 | ok | 140 | +4.37% | 363.74 |
| 설정A SmoothQuant a=0.5 | 12.4732 | ok | 140 | +0.02% | 363.82 |
| 설정B naive (per_tensor/per_tensor) | 13.5711 | ok | 140 | +8.82% | 358.88 |
| 설정B SmoothQuant a=0.5 | 12.5274 | ok | 140 | +0.45% | 358.82 |

## opt-1.3b

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | 14.6238 | ok | 140 | +0.00% | 129.11 |
| 설정A naive (per_channel/per_token) | - | 미실행 | - | - | - |
| 설정A SmoothQuant a=0.5 | - | 미실행 | - | - | - |
| 설정B naive (per_tensor/per_tensor) | - | 미실행 | - | - | - |
| 설정B SmoothQuant a=0.5 | - | 미실행 | - | - | - |

## opt-125m

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | - | 미실행 | - | - | - |
| 설정A naive (per_channel/per_token) | - | 미실행 | - | - | - |
| 설정A SmoothQuant a=0.5 | - | 미실행 | - | - | - |
| 설정B naive (per_tensor/per_tensor) | - | 미실행 | - | - | - |
| 설정B SmoothQuant a=0.5 | - | 미실행 | - | - | - |

## 모델 크기별 naive 열화 추세 (설정A 기준)

| model | 설정 | PPL | FP16 PPL | 변화(%) |
|---|---|---|---|---|
| opt-2.7b | A_naive | 13.0160 | 12.4709 | +4.37% |
| opt-2.7b | A_smooth_a0.5 | 12.4732 | 12.4709 | +0.02% |
