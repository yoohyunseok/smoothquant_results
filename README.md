# smoothquant_results

SmoothQuant 논문(Xiao et al., ICML 2023)의 실험을 공식 코드([mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant))로 재현한 결과를 모아두는 저장소입니다. Google Colab(Tesla T4 GPU)에서 실행했습니다.

## 다 끝났는지 / 결과가 어디 있는지

실험은 완료됐습니다. 결과는 이 저장소와 Google Drive(`smoothquant_vlm/`)에 동일하게 저장되어 있습니다.

- **[`RESULTS_COLAB.md`](RESULTS_COLAB.md)** — 최종 분석 보고서. 가장 먼저 볼 파일.
- **[`results_table.md`](results_table.md)** — 설정별 PPL 자동 생성 표
- **[`results/results.jsonl`](results/results.jsonl)** — 원본 수치 데이터 (설정 하나당 한 줄)
- **[`alpha_vs_ppl_opt-2.7b.png`](alpha_vs_ppl_opt-2.7b.png)** — alpha sweep 그래프
- **[`SETUP_LOG_COLAB.md`](SETUP_LOG_COLAB.md)** — 설치/실행 기록 (환경, 패키지 버전, 만난 에러와 해결 방법, 재현 방법)
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — 저장소 구조와 각 스크립트/파일의 역할 설명
- **[`scripts/`](scripts)** — 실제로 실행한 스크립트 원본 (`setup.sh`, `run_all.sh`, `my_ppl_eval.py` 등, 재현용)
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

## 다른 환경에서 돌릴 때 고쳐야 할 부분

`scripts/`를 그대로 `bash setup.sh && bash run_all.sh`로 실행하면 설치와 실험 로직 자체는
동일하게 재현되지만, 아래 항목들은 **이 저장소 소유자(yoohyunseok) 전용으로 박혀있는 값**이라
다른 사람이 쓰려면 고쳐야 합니다.

1. **이 저장소가 비공개(private)입니다.** 다른 사람이 `scripts/`나 결과를 보려면 저장소를
   public으로 바꾸거나 그 사람을 collaborator로 추가해야 합니다.

2. **결과를 올릴 GitHub 저장소 주소가 하드코딩되어 있습니다.**
   `scripts/setup.sh`의 `RESULTS_REMOTE`(11번째 줄)와 `scripts/sync_results.sh`의
   `RESULTS_REMOTE`(10번째 줄)가 둘 다 `https://github.com/yoohyunseok/smoothquant_results.git`
   로 고정되어 있습니다. 자기 저장소에 결과를 올리고 싶다면 이 두 곳을 자기 저장소 주소로
   바꿔야 합니다. **안 바꾸고 그냥 돌려도 실험 자체는 멈추지 않습니다** — `/content/.gh_token`
   파일이 없으면 `sync_results.sh`가 GitHub 푸시만 건너뛰고 경고만 남기도록 짜여 있어서,
   Google Drive에만 저장하고 싶다면 토큰 파일을 아예 안 만들면 됩니다.

3. **GitHub 토큰 파일 경로**: `/content/.gh_token`에 자기 Personal Access Token(repo 쓰기 권한)을
   직접 만들어 넣어야 합니다. 이 파일은 저장소에 포함되어 있지 않고(당연히), `setup.sh`/
   `sync_results.sh` 둘 다 `GH_TOKEN_FILE` 변수로 이 경로를 참조합니다.

4. **Google Drive 경로**: 모든 스크립트가 `/content/drive/MyDrive/smoothquant_vlm`
   (`DRIVE_DIR` 변수)을 기준으로 동작합니다. Colab에서 `from google.colab import drive;
   drive.mount('/content/drive')`로 Drive를 마운트한 뒤 써야 하고, 다른 경로를 쓰고 싶다면
   세 스크립트(`setup.sh`, `run_all.sh`, `sync_results.sh`) 안의 `DRIVE_DIR`을 전부 같은
   값으로 바꿔야 합니다.

5. **GPU 종류/VRAM에 따라 결과가 달라질 수 있습니다.** 이 실험은 Tesla T4(15GB)에서 돌려서
   opt-6.7b가 OOM으로 실행되지 못했습니다. A100/L4처럼 VRAM이 더 큰 GPU라면 opt-6.7b까지
   전부 성공해서 **이 저장소보다 더 완전한 결과**를 얻을 수 있습니다. 또한 PPL은 GPU/CUDA/
   드라이버 버전에 따라 소수점 아래 미세한 차이가 날 수 있습니다(방향성과 큰 수치는 동일할
   것으로 예상).

6. **Colab 기본 제공 패키지 버전이 다를 수 있습니다.** `setup.sh`는 Colab에 미리 설치된
   CUDA torch는 건드리지 않고 `transformers`/`tokenizers`/`huggingface_hub`만 특정 버전으로
   고정합니다. Colab의 기본 Python/torch 버전이 바뀌면(실행 당시 Python 3.13.15,
   torch 2.11.0+cu130) 이 고정 설치가 실패할 수 있습니다 — 그럴 때 겪을 수 있는 문제와
   해결 과정은 [`SETUP_LOG_COLAB.md`](SETUP_LOG_COLAB.md)에 상세히 적어뒀습니다.

요약하면: **코드/로직은 그대로 재현 가능**하지만, 1~4번은 "내 환경에 맞게 값을 바꾸는" 설정
작업이 필요하고, 5~6번은 환경 차이로 인해 **결과가 완전히 동일하지는 않을 수 있다**는 뜻입니다.
