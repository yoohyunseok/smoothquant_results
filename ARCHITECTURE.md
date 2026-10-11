# 저장소/작업 디렉토리 구조와 각 파일의 역할

이 문서는 SmoothQuant 재현 실험에서 만든 모든 스크립트와 산출물이 각각 무슨 역할을 하는지
정리한 것이다. 작업 디렉토리(Google Drive, 실행 중 쓰던 곳)와 결과 저장소(GitHub,
재현 가능하도록 스크립트+결과를 함께 보관하는 곳)의 구조를 모두 다룬다.

## 1. 실행 흐름 한눈에 보기

```
bash setup.sh          # 1회성: 환경 준비
      │
      ▼
bash run_all.sh         # 메인 오케스트레이터
      │
      ├─ (파이프라인 검증) my_ppl_eval.py 1회 실행 → 노트북 기준선과 대조
      │
      ├─ 모델별 반복 (opt-6.7b → opt-2.7b → opt-1.3b → opt-125m):
      │     ├─ activation scale 준비 (HF hub 다운로드 또는 생성)
      │     ├─ decide_n_samples(): my_ppl_eval.py를 n_samples=2로 프로브 실행
      │     │     → 속도 측정 → 15분 예산에 맞는 윈도우 수 결정
      │     ├─ run_config() × 5: my_ppl_eval.py를 설정(FP/A-naive/A-smooth/B-naive/B-smooth)별로
      │     │     새 프로세스로 실행 (in-place 양자화 때문에 프로세스 재사용 금지)
      │     │     각 설정이 끝날 때마다:
      │     │        gen_table.py  → results_table.md 갱신
      │     │        sync_results.sh → Drive/GitHub에 즉시 저장
      │     └─ (OOM 등 실패 시) 해당 모델 전체를 건너뛰고 decisions.log에 사유 기록
      │
      └─ alpha sweep (성공한 모델 중 가장 큰 모델 1개):
            run_config() × (최대 10, 이미 한 결과는 재사용) → plot_alpha.py → png 생성 → sync
```

`my_ppl_eval.py`가 "한 번의 실험"을 담당하는 실행 단위이고, `run_all.sh`가 이것을 반복
호출하는 지휘자다. 나머지 파이썬 스크립트(gen_table.py, plot_alpha.py)는 결과를 사람이
보기 좋은 형태로 만드는 보조 도구, `sync_results.sh`는 끊김에 대비해 저장을 보장하는
안전장치다.

## 2. 스크립트별 역할 (핵심)

### `my_ppl_eval.py` — 실제 실험을 실행하는 파일
원본 `smoothquant/ppl_eval.py`를 GPU 환경에 맞게 고친 복사본. 원본은 전혀 건드리지 않았다.
- **인자**: `--model_path`, `--dtype`(fp32/bf16/fp16), `--act_quant`(per_token/per_tensor),
  `--weight_quant`(per_channel/per_tensor), `--smooth`, `--quantize`, `--alpha`,
  `--act_scales_path`, `--n_samples`, `--tag`, `--results_jsonl`
- **모델 로드**: `device_map={"": 0}, low_cpu_mem_usage=True`로 CPU RAM을 거의 거치지 않고
  바로 GPU에 올림 (opt-6.7b가 Colab RAM 12GB를 초과할 수 있어서 필요했던 조치)
- **양자화 분기**: `--smooth`면 `smooth_lm()` 적용, `--quantize`면 `quantize_model()` 적용 →
  이 두 플래그 조합으로 FP16 기준 / naive W8A8 / SmoothQuant W8A8을 전부 표현
- **평가**: 2048토큰 윈도우 단위 WikiText-2 test PPL (원본 `Evaluator` 로직 그대로, 손대지 않음)
- **출력**: 끝나면 설정+PPL+소요시간을 `results.jsonl`에 한 줄 append, 그리고 `run_all.sh`가
  파싱할 수 있도록 `EVAL_META tag=... status=... ppl=... n_used=... load_sec=... eval_sec=...`
  한 줄을 stdout에 출력
- 매 실행마다 **새 프로세스**로 띄워야 한다 — `fake_quant.py`의 양자화 함수들이 가중치/활성화
  텐서를 in-place로 수정하기 때문에 같은 프로세스에서 설정을 바꿔가며 재사용하면 안 됨

### `setup.sh` — 환경 준비 (1회성, 재실행해도 안전)
- Colab 기본 CUDA torch는 그대로 두고, `transformers==4.36.0` / `tokenizers==0.15.2` /
  `huggingface_hub==0.23.4`만 정확히 고정 설치
- `accelerate`/`datasets`/`zstandard`/`matplotlib`는 Colab 기본 버전 그대로 사용하되,
  이 설치 과정에서 `huggingface_hub`가 1.x로 끌어올려지는 부작용이 있어 **마지막에 다시
  0.23.4로 되돌리는 보정 단계**를 둠 (이게 없으면 transformers import가 깨짐)
- `mit-han-lab/smoothquant` 원본 저장소를 `/content/repo`에 clone하고 `pip install -e .`
- 결과 저장용 GitHub 비공개 저장소를 `/content/results_repo`에 clone, git 사용자 정보를
  로컬(`colab-runner`)로만 설정
- 패키지 버전을 `logs/package_versions.txt`에 스냅샷으로 남김

### `run_all.sh` — 전체 실험의 지휘자
- `tag_exists()`: `results.jsonl`에 이미 `status: ok`로 기록된 설정은 건너뜀 → 런타임이
  끊겼다 재시작해도 이어서 진행되는 핵심 로직(resume)
- `decide_n_samples()`: 모델마다 `n_samples=2` 프로브를 한 번 돌려 로드시간·윈도우당 시간을
  측정하고, 전체 윈도우를 다 돌렸을 때 15분(900초)을 넘을지 추정해서 평가량을 자동 결정.
  결정 파일(`logs/eval_n_<model>.txt`)에 캐시해서 재실행 시 프로브를 또 하지 않음
- `prepare_act_scales_hub()` / `prepare_act_scales_125m()`: activation scale 파일 준비
  (1.3b/2.7b/6.7b는 HF hub 다운로드, 125m은 Pile val로 직접 생성)
- `run_model()`: 한 모델에 대해 FP16/설정A naive·smooth/설정B naive·smooth 5개 설정을 순서대로
  실행, 실패해도 다음 설정은 계속 시도
- `run_alpha_sweep()`: 가장 큰 성공 모델에서 alpha 5개 값 × 설정A/B 실행 (이미 있는 α=0.5
  결과는 재사용)
- 설정 하나가 끝날 때마다 `gen_table.py`와 `sync_results.sh`를 호출

### `sync_results.sh` — 저장 안전장치
`run_all.sh`가 설정 하나 끝날 때마다 호출. 두 단계:
1. Drive `sync` 호출 (이미 같은 파일시스템이라 즉시 반영되지만 한 번 더 보장)
2. GitHub 토큰(`/content/.gh_token`, 절대 로그/커밋에 노출 안 함)이 있으면 결과 파일
   (jsonl, md, png, log, **scripts/**)만 `/content/results_repo`로 복사 → commit → push.
   push 실패는 경고만 남기고 실험을 멈추지 않음 (다음 설정 끝나고 다시 시도)
3. 모델 가중치·HF 캐시처럼 큰 파일은 `.gitignore`로 원천 차단 (`*.pt`, `*.bin`,
   `*.safetensors`, `*.zst`, `hf_cache/`, `models/`)

### `gen_table.py` — 결과표 자동 생성
`results.jsonl`을 읽어 모델별/설정별 PPL, FP16 대비 변화율, alpha sweep 표를 담은
`results_table.md`를 매번 처음부터 다시 생성한다(직접 수정하지 않는 파일).

### `plot_alpha.py` — alpha sweep 그래프
`results.jsonl`에서 특정 모델의 `*_smooth_a*` 태그만 뽑아 설정A/B 두 선으로
alpha-PPL 그래프(`alpha_vs_ppl_<model>.png`)를 그린다.

## 3. Google Drive 작업 디렉토리 구조 (전체)

```
smoothquant_vlm/
├── setup.sh                 # 환경 준비
├── run_all.sh                # 실험 오케스트레이터
├── sync_results.sh           # Drive+GitHub 저장
├── my_ppl_eval.py             # 실제 평가 스크립트 (실험의 핵심)
├── gen_table.py               # results_table.md 생성기
├── plot_alpha.py               # alpha sweep 그래프 생성기
├── RESULTS_COLAB.md            # 최종 분석 보고서
├── SETUP_LOG_COLAB.md           # 설치/실행 기록, 에러와 해결
├── ARCHITECTURE.md               # 이 문서
├── results_table.md               # 자동 생성 결과표
├── alpha_vs_ppl_opt-2.7b.png       # alpha sweep 그래프
├── results/
│   └── results.jsonl                # 설정별 원본 측정치 (한 줄 = 한 설정)
├── act_scales/                        # activation scale 캐시 (작은 .pt, Drive에 보관)
│   ├── opt-125m.pt   (직접 생성)
│   ├── opt-1.3b.pt    (HF hub 다운로드)
│   ├── opt-2.7b.pt     (HF hub 다운로드)
│   └── opt-6.7b.pt      (HF hub 다운로드, PPL 평가 자체는 OOM으로 못함)
└── logs/
    ├── run_all.log                     # run_all.sh 전체 진행 로그
    ├── setup.log                        # setup.sh 실행 로그
    ├── sync.log                          # 매 동기화 성공/실패 기록
    ├── decisions.log                      # 자동 판단 근거 (평가량 축소/스킵 사유 등)
    ├── model_status.txt                    # 모델별 최종 상태 (ok/skip/partial_fail)
    ├── eval_n_<model>.txt                   # 모델별로 결정된 평가 윈도우 수 캐시
    ├── probe_results.jsonl                   # n_samples=2 프로브 전용 결과 (본 결과와 분리)
    ├── <tag>.log                              # 설정별 my_ppl_eval.py stdout/stderr 전체
    ├── package_versions.txt                    # 패키지 버전 스냅샷
    ├── act_scales_download.log, pile_download.log, gen_act_scales_125m.log
    └── (기타 다운로드/생성 로그)
```

`act_scales/`와 `logs/eval_n_*.txt`, `logs/probe_results.jsonl`, `logs/model_status.txt`는
**재실행 시 중복 작업을 피하기 위한 캐시/상태 파일**이지, 그 자체가 최종 결과물은 아니다.

## 4. GitHub 결과 저장소 구조 (재현 가능하도록 스크립트 + 결과만 보관)

```
smoothquant_results/
├── README.md                 # 저장소 첫 화면 — 결과 요약 + 핵심 수치
├── RESULTS_COLAB.md            # 최종 분석 보고서 (Drive와 동일)
├── SETUP_LOG_COLAB.md            # 설치/실행 기록 (Drive와 동일)
├── ARCHITECTURE.md                 # 이 문서 (Drive와 동일)
├── results_table.md                 # 자동 생성 결과표 (Drive와 동일)
├── alpha_vs_ppl_opt-2.7b.png          # alpha sweep 그래프
├── scripts/                             # 실행했던 스크립트 원본 (재현용)
│   ├── setup.sh
│   ├── run_all.sh
│   ├── sync_results.sh
│   ├── my_ppl_eval.py
│   ├── gen_table.py
│   └── plot_alpha.py
├── results/
│   └── results.jsonl                      # Drive와 동일
└── logs/
    └── *.log                                # Drive logs/의 .log 파일들만 (Drive와 동일)
```

Drive와의 차이: **모델 가중치·HF 캐시·activation scale(.pt)·Pile 데이터(.zst) 등 큰 파일은
GitHub에 절대 올리지 않는다**(`.gitignore`로 차단). 그래서 `act_scales/`, `hf_cache/`,
`data/` 같은 디렉토리는 GitHub 쪽에는 존재하지 않는다. 대신 어떤 파일을 어떻게 받았는지는
`scripts/run_all.sh`의 `prepare_act_scales_*` 함수와 `SETUP_LOG_COLAB.md`에 전부 기록되어
있어서, `scripts/` 안의 스크립트들로 `bash setup.sh && bash run_all.sh`를 실행하면 처음부터
다시 재현할 수 있다.

## 5. 문서 파일 역할 정리

| 파일 | 역할 |
|---|---|
| `README.md` | GitHub 저장소 첫 화면. 결과가 끝났는지, 핵심 수치, 어디를 봐야 하는지 안내 |
| `RESULTS_COLAB.md` | 실험 결과 본문 — 모델별 표, 크기별 추세, alpha sweep, 논문/노트북 비교, 한계점 |
| `SETUP_LOG_COLAB.md` | 설치 과정에서 만난 문제(huggingface_hub 충돌 등)와 해결, 패키지 버전, OOM 상세, 소요시간 |
| `ARCHITECTURE.md` | 이 문서 — 구조와 각 파일의 역할 |
| `results_table.md` | 사람이 보기 좋은 자동 생성 표 (매 설정 끝날 때마다 다시 생성됨, 직접 수정 금지) |
| `results/results.jsonl` | 모든 분석의 원천 데이터. 한 줄 = 한 설정의 측정 결과(모델, 설정, PPL, 소요시간, 상태 등) |
