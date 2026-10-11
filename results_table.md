# SmoothQuant Colab 결과표 (자동 생성)

각 설정이 끝날 때마다 run_all.sh가 이 파일을 다시 만든다. 직접 수정하지 말 것.


## opt-2.7b

| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |
|---|---|---|---|---|---|
| FP16 기준 | 12.4709 | ok | 140 | +0.00% | 228.77 |
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
