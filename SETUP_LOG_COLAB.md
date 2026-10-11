# SmoothQuant Colab 재현 - 설치/실행 기록

## 환경 (최초 확인, 2026-10-11 13:57 UTC)
- GPU: Tesla T4, VRAM 15360MiB (bf16 하드웨어 미지원 → FP16 사용)
- RAM: 12Gi total
- 디스크(`/`, `/content` 포함): 113G 중 70G 여유
- Python: 3.13.15 (Colab 기본 제공, 매우 최신이라 2023년산 패키지와 호환성 이슈가 있었음 — 아래 참고)
- torch: 2.11.0+cu130 (Colab 사전 설치, 건드리지 않음)

## 실행 명령 (요약)
```
bash setup.sh && bash run_all.sh
```
두 스크립트 모두 idempotent하게 작성해서 런타임이 끊겨도 그대로 재실행하면 이어서 진행된다.

## 패키지 설치 관련 결정 및 이슈

### 1. transformers==4.36.0 설치는 Python 3.13에서도 가능했음
`tokenizers==0.15.2`가 `cp313` 휠을 이미 제공하고 있어서(`pip download`로 확인) 소스 빌드 없이 바로 설치됨.
`numpy<2`로 내릴 필요는 없었음 — numpy 1.26.4는 Python 3.13을 공식 지원하지 않아서 강제하면
소스 빌드(meson+ninja, 매우 느림)로 빠지는 것을 확인하고 중단함. transformers 4.36.0의
`dependency_versions_table.py`에는 `numpy>=1.17`만 명시되어 있어 numpy 2.1.3을 그대로 써도
import/동작에 문제없음을 확인.

### 2. huggingface_hub 버전 충돌 (가장 까다로웠던 부분)
- `transformers==4.36.0`, `tokenizers==0.15.2` 모두 `huggingface_hub<1.0,>=0.19.3` 를 요구.
- Colab 기본 환경의 huggingface_hub는 1.33.0 (너무 최신).
- `huggingface_hub==0.23.4`로 맞추니 import, `AutoModelForCausalLM.from_pretrained`,
  `hf_hub_download`, `load_dataset` 모두 정상 동작 확인.
- **함정**: 이후 `pip install accelerate datasets zstandard matplotlib`를 버전 고정 없이
  실행하면 pip 의존성 리졸버가 datasets의 요구사항을 만족시키려고 huggingface_hub를
  다시 1.33.0으로 끌어올려 버림 (transformers/tokenizers 요구사항과 충돌 경고 발생).
  → `setup.sh`에서 이 설치 **다음에** `pip install --no-deps "huggingface_hub==0.23.4"`를
  한 번 더 실행해서 되돌리도록 고정함. 재실행해도 항상 같은 순서로 동작하므로 idempotent.
  (pip 의존성 경고(ERROR: pip's dependency resolver...)는 뜨지만 실제 런타임 동작은
  정상임을 import 테스트와 opt-125m 파이프라인 검증으로 확인함.)

### 3. numpy, datasets, accelerate는 다운그레이드하지 않음
- numpy 2.1.3, datasets 4.8.5, accelerate 1.15.0 (Colab 기본값 그대로) 사용.
- `load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`가 datasets 4.8.5에서도
  정상 동작함을 확인 (스크립트 기반 로딩이 폐지된 최신 datasets에서도 wikitext는 자동으로
  parquet 미러를 사용하는 것으로 보임).

### 4. 최종 확정 버전 조합 (Colab, Python 3.13.15, Tesla T4)
| 패키지 | 버전 |
|---|---|
| torch | 2.11.0+cu130 (사전 설치, 변경 안 함) |
| transformers | 4.36.0 |
| tokenizers | 0.15.2 |
| huggingface_hub | 0.23.4 |
| numpy | 2.1.3 (사전 설치 그대로) |
| datasets | 4.8.5 (사전 설치 그대로) |
| accelerate | 1.15.0 (사전 설치 그대로) |
| matplotlib | 3.10.0 (사전 설치 그대로) |
| zstandard | 0.25.0 (사전 설치 그대로) |

노트북(CPU)에서 쓴 조합(torch 2.2.2+cpu, transformers 4.36.0, tokenizers 0.15.2,
numpy 1.26.4, datasets 5.0.1)과는 numpy/datasets 버전이 다르지만, transformers/tokenizers/
huggingface_hub 조합은 동일하게 맞췄다.

### 5. 원본 코드 호환성 확인
- `from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP` → OK
- `from transformers import OPTForCausalLM, AutoTokenizer` → OK
- `smoothquant/fake_quant.py`, `smoothquant/smooth.py`, `smoothquant/calibration.py`는
  원본 그대로 두고 수정하지 않음. `pip install --no-deps -e /content/repo`로 설치.

## my_ppl_eval.py (복사본, Drive에 위치)
원본 `smoothquant/ppl_eval.py` 대비 변경점:
- `--dtype` (fp32/bf16/fp16, 기본 fp16), `--act_quant`, `--weight_quant`, `--tag`,
  `--results_jsonl` 인자 추가.
- 모델을 `device_map={"": 0}, low_cpu_mem_usage=True`로 바로 GPU에 로드 (CPU RAM 경유 최소화,
  opt-6.7b가 Colab RAM 12GB를 넘길 수 있어서).
- `quantize_model(..., quantize_bmm_input=True)` 유지 (원본 ppl_eval.py와 동일 값).
- 평가 로직(2048 윈도우, shift CrossEntropy, `exp(sum nll/(n*2048))`)은 원본 그대로.
- 끝에서 설정/PPL/소요시간을 jsonl 한 줄로 append, `EVAL_META ...` 한 줄을 stdout에 출력해서
  `run_all.sh`가 평가량(윈도우 수) 자동 결정에 파싱해서 사용.
- 이 수정본은 opt-125m에서 1회 실행 후 문제없이 동작 확인함 (아래 파이프라인 검증 참고).

## 파이프라인 검증 (2026-10-11 14:12 UTC)
`opt-125m_fp16_fp_fullcheck` 태그로 FP16, 전체 WikiText-2 test(140윈도우) 실행:
- 결과 PPL = **27.6563**
- 노트북(CPU, FP32) 전체 test셋 기준선 27.655와 거의 일치 (FP16이라 소수점 셋째 자리 수준의
  차이는 있을 수 있음, 여기선 거의 동일) → 평가 파이프라인(토크나이즈, 윈도우 분할, 손실 계산)이
  원본과 동일하게 동작함을 확인.
- 소요시간: 모델 로드 5.32s + 평가 18.93s = 총 44.99s (140윈도우 기준). 이 속도라면 opt-6.7b도
  설정당 15분 예산에 충분히 들어갈 것으로 예상(실제 결과는 RESULTS_COLAB.md 참고).

## 저장 파이프라인 검증 (2026-10-11 14:14 UTC)
- `sync_results.sh` 최초 실행 → Drive(`results/results.jsonl`, `results_table.md`, `logs/*`)
  갱신 확인.
- `/content/results_repo`에 동일 파일 복사 → `git commit` → `git push origin HEAD` 성공
  (`git log --oneline`: `77b07ea sync results ...` 커밋 확인, push 성공 로그 확인).
- GitHub 토큰은 `/content/.gh_token`에서 읽어 `http.extraheader`로만 사용, 변수/로그/커밋에
  남기지 않음. 커밋 작성자는 `colab-runner <noreply@example.com>` (로컬 설정, 실제 이메일 아님).

## 평가량(윈도우 수) 자동 결정 로직
`run_all.sh`의 `decide_n_samples`가 모델별로 `--n_samples 2`짜리 프로브를 한 번 실행해서
(load_sec, eval_sec, 전체 윈도우 수)를 측정하고, 전체 윈도우를 다 돌렸을 때 예상 소요시간이
15분(900초)을 넘으면 윈도우 수를 줄인다. 결정 내역은 `logs/decisions.log`에 모두 기록됨.
opt-125m 기준 140윈도우에 45초가 채 안 걸렸으므로, 더 큰 모델도 대부분 전체 test셋을
그대로 사용할 수 있을 것으로 예상했고, 실제 결정 내역은 `logs/decisions.log` 참고.

## opt-6.7b OOM (2026-10-11 14:23 UTC)
프로브(n_samples=2)에서 OOM 발생, 그 즉시 opt-6.7b 전체 설정을 건너뛰고 opt-2.7b로 진행함.
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 383.81 MiB is free.
Including non-PyTorch memory, this process has 14.19 GiB memory in use.
Of the allocated memory 13.31 GiB is allocated by PyTorch, ...
```
- 모델 가중치 로드만으로 13.31GiB를 이미 사용 (FP16 6.7B 파라미터 ≈ 13.4GB 추정과 일치).
- OOM은 attention softmax 계산 중(`nn.functional.softmax(attn_weights, ..., dtype=torch.float32)`)
  512MiB를 추가로 요청하다 실패한 것 — 즉 **윈도우 수(n_samples)를 줄여도 해결되지 않는 종류의
  OOM**이다(단 1개 윈도우짜리 forward 한 번만으로도 실패). 2048 토큰 윈도우 평가 로직은
  과제 지침에 따라 변경하지 않았으므로, 이 조합(T4 15GB + opt-6.7b + 2048 윈도우 FP16 평가)은
  현재 런타임에서는 구조적으로 불가능하다고 판단하고 opt-6.7b는 전체 건너뜀.
- **결론: opt-6.7b 실험을 돌리려면 더 큰 VRAM의 GPU 런타임(A100 40GB/L4 24GB 등)이 필요함.**

## 아직 남은 실험/이슈
(런타임이 끊기면 이 섹션을 이어서 갱신할 것)
