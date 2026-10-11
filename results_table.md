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

### opt-2.7b alpha sweep

| 설정 | alpha | PPL | 상태 |
|---|---|---|---|
| A_smooth_a0.0 | 0.0 | 14.1413 | ok |
| A_smooth_a0.25 | 0.25 | 12.6293 | ok |
| A_smooth_a0.5 | 0.5 | 12.4732 | ok |
| A_smooth_a0.75 | 0.75 | 12.4939 | ok |
| A_smooth_a1.0 | 1.0 | 12.4940 | ok |
| B_smooth_a0.0 | 0.0 | 15.5587 | ok |
| B_smooth_a0.25 | 0.25 | 12.7525 | ok |
| B_smooth_a0.5 | 0.5 | 12.5274 | ok |
| B_smooth_a0.75 | 0.75 | 12.5559 | ok |
| B_smooth_a1.0 | 1.0 | 16.4695 | ok |

## opt-1.3b

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | 14.6238 | ok | 140 | +0.00% | 129.11 |
| 설정A naive (per_channel/per_token) | 15.1752 | ok | 140 | +3.77% | 196.54 |
| 설정A SmoothQuant a=0.5 | 14.7908 | ok | 140 | +1.14% | 197.92 |
| 설정B naive (per_tensor/per_tensor) | 15.8446 | ok | 140 | +8.35% | 192.72 |
| 설정B SmoothQuant a=0.5 | 15.0048 | ok | 140 | +2.61% | 193.43 |

## opt-125m

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | 27.6563 | ok | 140 | +0.00% | 31.07 |
| 설정A naive (per_channel/per_token) | 27.9299 | ok | 140 | +0.99% | 37.83 |
| 설정A SmoothQuant a=0.5 | 27.7448 | ok | 140 | +0.32% | 36.7 |
| 설정B naive (per_tensor/per_tensor) | 30.5334 | ok | 140 | +10.40% | 37.84 |
| 설정B SmoothQuant a=0.5 | 29.4690 | ok | 140 | +6.55% | 37.08 |

## 모델 크기별 naive 열화 추세 (설정A 기준)

| model | 설정 | PPL | FP16 PPL | 변화(%) |
|---|---|---|---|---|
| opt-2.7b | A_naive | 13.0160 | 12.4709 | +4.37% |
| opt-2.7b | A_smooth_a0.5 | 12.4732 | 12.4709 | +0.02% |
| opt-1.3b | A_naive | 15.1752 | 14.6238 | +3.77% |
| opt-1.3b | A_smooth_a0.5 | 14.7908 | 14.6238 | +1.14% |
| opt-125m | A_naive | 27.9299 | 27.6563 | +0.99% |
| opt-125m | A_smooth_a0.5 | 27.7448 | 27.6563 | +0.32% |
