# smoothquant_results

SmoothQuant 논문(Xiao et al., ICML 2023)의 실험을 공식 코드([mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant))로 재현한 결과를 모아두는 저장소입니다. Google Colab(Tesla T4 GPU)에서 실행했습니다.

## 다 끝났는지 / 결과가 어디 있는지

실험은 완료됐습니다. 결과는 이 저장소와 Google Drive(`smoothquant_vlm/`)에 동일하게 저장되어 있습니다.

- **[`RESULTS_COLAB.md`](RESULTS_COLAB.md)** — 최종 분석 보고서. 가장 먼저 볼 파일.
- **[`results_table.md`](results_table.md)** — 설정별 PPL 자동 생성 표
- **[`results/results.jsonl`](results/results.jsonl)** — 원본 수치 데이터 (설정 하나당 한 줄)
- **[`alpha_vs_ppl_opt-2.7b.png`](alpha_vs_ppl_opt-2.7b.png)** — alpha sweep 그래프
- **[`SETUP_LOG_COLAB.md`](SETUP_LOG_COLAB.md)** — 설치/실행 기록 (환경, 패키지 버전, 만난 에러와 해결 방법, 재현 방법)
- **`logs/`** — 전체 실행 로그

커밋 이력은 설정 하나가 끝날 때마다 자동으로 쌓인 것이라, 실험이 어떤 순서로 진행됐는지도 git log로 추적할 수 있습니다.

## 실험 결과 핵심 요약

실행 환경: Google Colab, GPU Tesla T4(VRAM 15GB), FP16, WikiText-2 test 전체(140윈도우).

- **opt-6.7b**: GPU 메모리 부족(OOM)으로 실행 불가. FP16 가중치만으로 13.3GB를 차지해 T4 15GB로는 2048토큰 forward 한 번도 돌릴 여유가 없었음 — 더 큰 VRAM의 GPU가 필요함.
- **opt-2.7b / opt-1.3b / opt-125m**: 전부 성공. SmoothQuant가 naive W8A8의 PPL 열화를 FP16 수준으로 거의 완전히 회복시킴.

| 모델 | FP16 PPL | naive 열화(설정A) | SmoothQuant 열화(설정A) |
|---|---|---|---|
| opt-2.7b | 12.4709 | +4.37% | +0.02% |
| opt-1.3b | 14.6238 | +3.77% | +1.14% |
| opt-125m | 27.6563 | +0.99% | +0.32% |

(설정A = weight per_channel / activation per_token, 원본 `ppl_eval.py`와 동일 조합)

- 모델이 커질수록 naive 열화와 activation outlier 크기(스케일 파일 최댓값)가 함께 커지는 추세를 확인함 (outlier 최댓값: opt-125m 26.5 → opt-1.3b 44.5 → opt-2.7b 49.9 → opt-6.7b 63.3).
- alpha sweep(opt-2.7b)에서 α=0.5가 대체로 최적점(또는 최적점 근방)임을 재현함. 거친 양자화 설정(설정B)에서는 α가 0 또는 1에 가까울수록 오히려 naive보다 나빠지는 U자 형태를 보임.

자세한 수치, 노트북(CPU) 결과와의 비교, 논문과 다른 점(확인된 사실과 추정을 구분), 한계점은 [`RESULTS_COLAB.md`](RESULTS_COLAB.md)에 전부 정리되어 있습니다.
