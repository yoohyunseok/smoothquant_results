#!/bin/bash
# 전체 실험을 순차 실행. 재실행해도 이미 완료된 설정(tag)은 건너뛴다(resume).
# 사용법: bash setup.sh && bash run_all.sh
set -uo pipefail

DRIVE_DIR="/content/drive/MyDrive/smoothquant_vlm"
REPO_DIR="/content/repo"
DATA_DIR="/content/data"
ACT_SCALES_DIR="$DRIVE_DIR/act_scales"
LOG_DIR="$DRIVE_DIR/logs"
RESULTS_JSONL="$DRIVE_DIR/results/results.jsonl"
DECISIONS_LOG="$LOG_DIR/decisions.log"
MODEL_STATUS_FILE="$LOG_DIR/model_status.txt"
MY_EVAL="$DRIVE_DIR/my_ppl_eval.py"
GEN_TABLE="$DRIVE_DIR/gen_table.py"
PLOT_ALPHA="$DRIVE_DIR/plot_alpha.py"
SYNC_SH="$DRIVE_DIR/sync_results.sh"

TIME_BUDGET_SEC=900  # 설정 하나당 15분

mkdir -p "$LOG_DIR" "$DRIVE_DIR/results" "$ACT_SCALES_DIR"
touch "$RESULTS_JSONL" "$MODEL_STATUS_FILE" "$DECISIONS_LOG"

cd "$REPO_DIR" || { echo "[run_all] $REPO_DIR 없음. setup.sh를 먼저 실행하세요."; exit 1; }
export HF_HOME="/content/hf_cache"
export HF_DATASETS_CACHE="/content/hf_cache/datasets"
export PYTHONUNBUFFERED=1

log() { echo "[run_all] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

tag_exists() {
  local tag="$1"
  python3 -c "
import json, sys
tag = sys.argv[1]
path = sys.argv[2]
found = False
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('tag') == tag and rec.get('status') == 'ok':
                found = True
except FileNotFoundError:
    pass
sys.exit(0 if found else 1)
" "$tag" "$RESULTS_JSONL"
}

run_config() {
  # $1=tag $2=model_path $3=dtype, 나머지는 my_ppl_eval.py 추가 인자
  local tag="$1"; shift
  local model_path="$1"; shift
  local dtype="$1"; shift
  if tag_exists "$tag"; then
    log "SKIP (이미 성공 기록 있음): $tag"
    return 0
  fi
  log "START: $tag"
  local logfile="$LOG_DIR/${tag}.log"
  python "$MY_EVAL" --model_path "$model_path" --dtype "$dtype" --tag "$tag" \
    --results_jsonl "$RESULTS_JSONL" "$@" > "$logfile" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "FAIL (rc=$rc): $tag - 로그: $logfile ($(tail -n 1 "$logfile"))"
  else
    log "DONE: $tag ($(grep EVAL_META "$logfile" | tail -n 1))"
  fi
  python "$GEN_TABLE" "$RESULTS_JSONL" "$DRIVE_DIR/results_table.md"
  bash "$SYNC_SH"
  return $rc
}

decide_n_samples() {
  # 모델별로 한 번만 프로브(n_samples=2)를 돌려 전체 평가에 걸리는 시간을 추정하고,
  # 설정당 15분을 넘으면 윈도우 수를 줄인다. 결과는 파일에 캐시해서 재실행 시 재사용.
  local model_key="$1" model_path="$2" dtype="$3"
  local cache_file="$LOG_DIR/eval_n_${model_key}.txt"
  if [ -s "$cache_file" ]; then
    cat "$cache_file"
    return 0
  fi

  local probe_tag="${model_key}_probe_n2"
  local probe_log="$LOG_DIR/${probe_tag}.log"
  local probe_jsonl="$LOG_DIR/probe_results.jsonl"
  log "PROBE START: $model_key (n_samples=2 로 속도 측정)" >&2
  python "$MY_EVAL" --model_path "$model_path" --dtype "$dtype" --tag "$probe_tag" \
    --results_jsonl "$probe_jsonl" --n_samples 2 > "$probe_log" 2>&1
  local probe_rc=$?

  local meta
  meta=$(grep "EVAL_META" "$probe_log" | tail -n 1)

  if [ $probe_rc -ne 0 ] || [ -z "$meta" ]; then
    echo "[decide] $(date -u) $model_key: 프로브 실패(rc=$probe_rc) - 모델 로드/추론 자체가 불가능한 것으로 보고 건너뜀 후보로 기록" >> "$DECISIONS_LOG"
    echo "0" | tee "$cache_file"
    return 0
  fi

  local n_total load_sec eval_sec
  n_total=$(echo "$meta" | grep -oP 'n_total=\K[0-9]+')
  load_sec=$(echo "$meta" | grep -oP 'load_sec=\K[0-9.]+')
  eval_sec=$(echo "$meta" | grep -oP 'eval_sec=\K[0-9.]+')

  if [ -z "$n_total" ] || [ -z "$load_sec" ] || [ -z "$eval_sec" ]; then
    echo "[decide] $(date -u) $model_key: EVAL_META 파싱 실패 - 기본값 20으로 폴백" >> "$DECISIONS_LOG"
    echo "20" | tee "$cache_file"
    return 0
  fi

  local decision
  decision=$(python3 -c "
import math
load_sec = $load_sec
eval_sec = $eval_sec
n_total = $n_total
budget = $TIME_BUDGET_SEC
per_window = eval_sec / 2.0
full_est = load_sec + per_window * n_total
if full_est <= budget:
    print(n_total)
else:
    n = max(1, math.floor((budget - load_sec) / per_window))
    print(min(n, n_total))
")
  local full_est
  full_est=$(python3 -c "print(round($load_sec + ($eval_sec/2.0)*$n_total, 1))")

  if [ "$decision" -ge "$n_total" ]; then
    echo "[decide] $(date -u) $model_key: 전체 $n_total 윈도우 사용 (예상 소요 ${full_est}s <= ${TIME_BUDGET_SEC}s)" >> "$DECISIONS_LOG"
  else
    echo "[decide] $(date -u) $model_key: 전체 예상 소요 ${full_est}s > ${TIME_BUDGET_SEC}s -> n_samples=$decision 로 축소 (전체 $n_total 중)" >> "$DECISIONS_LOG"
  fi

  echo "$decision" | tee "$cache_file"
}

prepare_act_scales_hub() {
  local model_key="$1" hub_filename="$2"
  local out="$ACT_SCALES_DIR/${model_key}.pt"
  if [ -f "$out" ]; then
    echo "$out"
    return 0
  fi
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='mit-han-lab/smoothquant-scales', filename='$hub_filename')
shutil.copy(p, '$out')
print('downloaded:', p)
" >> "$LOG_DIR/act_scales_download.log" 2>&1
  if [ -f "$out" ]; then
    echo "$out"
  else
    echo "[decide] $(date -u) $model_key act_scales 다운로드 실패" >> "$DECISIONS_LOG"
    echo ""
  fi
}

prepare_act_scales_125m() {
  local out="$ACT_SCALES_DIR/opt-125m.pt"
  if [ -f "$out" ]; then
    echo "$out"
    return 0
  fi
  local pile_path="$DATA_DIR/val.jsonl.zst"
  if [ ! -f "$pile_path" ]; then
    log "Pile val 다운로드 시도 (mit-han-lab/pile-val-backup)" >&2
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='mit-han-lab/pile-val-backup', filename='val.jsonl.zst', repo_type='dataset')
shutil.copy(p, '$pile_path')
print('downloaded:', p)
" >> "$LOG_DIR/pile_download.log" 2>&1
  fi

  local cal_path="$pile_path"
  if [ ! -f "$pile_path" ]; then
    echo "[decide] $(date -u) opt-125m: Pile val 다운로드 실패 -> WikiText-2 train 일부로 캘리브레이션 데이터 대체" >> "$DECISIONS_LOG"
    cal_path="$DATA_DIR/wikitext_train_calib.jsonl"
    python3 -c "
from datasets import load_dataset
import json
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
n = 0
with open('$cal_path', 'w') as f:
    for row in ds:
        t = row['text'].strip()
        if len(t) > 20:
            f.write(json.dumps({'text': t}) + '\n')
            n += 1
        if n >= 2000:
            break
print('wrote', n, 'lines')
" >> "$LOG_DIR/wikitext_fallback.log" 2>&1
  fi

  if [ -f "$cal_path" ]; then
    log "opt-125m activation scale 생성 중 (generate_act_scales.py)" >&2
    python "$REPO_DIR/examples/generate_act_scales.py" \
      --model-name facebook/opt-125m \
      --output-path "$out" \
      --dataset-path "$cal_path" \
      --num-samples 512 --seq-len 512 >> "$LOG_DIR/gen_act_scales_125m.log" 2>&1
  fi

  if [ -f "$out" ]; then
    echo "$out"
  else
    echo "[decide] $(date -u) opt-125m act_scales 생성 실패 (로그: gen_act_scales_125m.log)" >> "$DECISIONS_LOG"
    echo ""
  fi
}

run_model() {
  local model_key="$1" model_path="$2" act_scales_path="$3"
  local dtype="fp16"

  log "==== $model_key 시작 ===="
  local n
  n=$(decide_n_samples "$model_key" "$model_path" "$dtype")
  if [ "$n" = "0" ]; then
    log "$model_key: 프로브 실패로 전체 설정을 건너뜁니다 (OOM 등 - decisions.log 참고)"
    echo "$model_key: skip (probe failed)" >> "$MODEL_STATUS_FILE"
    return 1
  fi
  log "$model_key: n_samples=$n 사용"

  local ok=1

  run_config "${model_key}_fp16_fp" "$model_path" "$dtype" --n_samples "$n" || ok=0

  run_config "${model_key}_fp16_A_naive" "$model_path" "$dtype" --n_samples "$n" \
    --quantize --weight_quant per_channel --act_quant per_token || ok=0

  if [ -n "$act_scales_path" ]; then
    run_config "${model_key}_fp16_A_smooth_a0.5" "$model_path" "$dtype" --n_samples "$n" \
      --smooth --quantize --alpha 0.5 --act_scales_path "$act_scales_path" \
      --weight_quant per_channel --act_quant per_token || ok=0
  else
    log "$model_key: act_scales 없음 -> SmoothQuant 설정 건너뜀"
  fi

  run_config "${model_key}_fp16_B_naive" "$model_path" "$dtype" --n_samples "$n" \
    --quantize --weight_quant per_tensor --act_quant per_tensor || ok=0

  if [ -n "$act_scales_path" ]; then
    run_config "${model_key}_fp16_B_smooth_a0.5" "$model_path" "$dtype" --n_samples "$n" \
      --smooth --quantize --alpha 0.5 --act_scales_path "$act_scales_path" \
      --weight_quant per_tensor --act_quant per_tensor || ok=0
  fi

  if [ "$ok" = "1" ]; then
    echo "$model_key: ok (n_samples=$n)" >> "$MODEL_STATUS_FILE"
  else
    echo "$model_key: partial_fail (n_samples=$n, 일부 설정 실패 - 로그 확인)" >> "$MODEL_STATUS_FILE"
  fi
  log "==== $model_key 종료 ===="
}

run_alpha_sweep() {
  local model_key="$1" model_path="$2" act_scales_path="$3"
  local dtype="fp16"
  if [ -z "$act_scales_path" ]; then
    log "alpha sweep: $model_key act_scales 없음 -> 건너뜀"
    return 0
  fi
  local n
  n=$(decide_n_samples "$model_key" "$model_path" "$dtype")
  if [ "$n" = "0" ]; then
    log "alpha sweep: $model_key 프로브 실패 -> 건너뜀"
    return 0
  fi
  log "==== $model_key alpha sweep 시작 (n_samples=$n) ===="
  for alpha in 0.0 0.25 0.5 0.75 1.0; do
    run_config "${model_key}_fp16_A_smooth_a${alpha}" "$model_path" "$dtype" --n_samples "$n" \
      --smooth --quantize --alpha "$alpha" --act_scales_path "$act_scales_path" \
      --weight_quant per_channel --act_quant per_token
    run_config "${model_key}_fp16_B_smooth_a${alpha}" "$model_path" "$dtype" --n_samples "$n" \
      --smooth --quantize --alpha "$alpha" --act_scales_path "$act_scales_path" \
      --weight_quant per_tensor --act_quant per_tensor
  done
  python "$PLOT_ALPHA" "$RESULTS_JSONL" "$model_key" "$DRIVE_DIR/alpha_vs_ppl_${model_key}.png"
  bash "$SYNC_SH"
  log "==== $model_key alpha sweep 종료 ===="
}

############################################
# 0. 파이프라인 검증: opt-125m FP16 전체 PPL
############################################
log "파이프라인 검증: opt-125m FP16 전체 WikiText-2 test PPL"
run_config "opt-125m_fp16_fp_fullcheck" "facebook/opt-125m" "fp16"

############################################
# 본 실험: opt-6.7b -> opt-2.7b -> opt-1.3b -> opt-125m
############################################
act67=$(prepare_act_scales_hub "opt-6.7b" "opt-6.7b.pt")
run_model "opt-6.7b" "facebook/opt-6.7b" "$act67"

act27=$(prepare_act_scales_hub "opt-2.7b" "opt-2.7b.pt")
run_model "opt-2.7b" "facebook/opt-2.7b" "$act27"

act13=$(prepare_act_scales_hub "opt-1.3b" "opt-1.3b.pt")
run_model "opt-1.3b" "facebook/opt-1.3b" "$act13"

act125m=$(prepare_act_scales_125m)
run_model "opt-125m" "facebook/opt-125m" "$act125m"

############################################
# alpha sweep: 성공한 모델 중 가장 큰 모델 하나
############################################
ALPHA_MODEL_KEY=""
ALPHA_MODEL_PATH=""
ALPHA_ACT_SCALES=""
for cand in "opt-6.7b:facebook/opt-6.7b:$act67" "opt-2.7b:facebook/opt-2.7b:$act27" \
            "opt-1.3b:facebook/opt-1.3b:$act13" "opt-125m:facebook/opt-125m:$act125m"; do
  key="${cand%%:*}"
  rest="${cand#*:}"
  path="${rest%%:*}"
  scales="${rest#*:}"
  if grep -q "^${key}: ok" "$MODEL_STATUS_FILE" 2>/dev/null && [ -n "$scales" ]; then
    ALPHA_MODEL_KEY="$key"
    ALPHA_MODEL_PATH="$path"
    ALPHA_ACT_SCALES="$scales"
    break
  fi
done

if [ -n "$ALPHA_MODEL_KEY" ]; then
  log "alpha sweep 대상 모델: $ALPHA_MODEL_KEY"
  run_alpha_sweep "$ALPHA_MODEL_KEY" "$ALPHA_MODEL_PATH" "$ALPHA_ACT_SCALES"
else
  log "alpha sweep: 완전히 성공한 모델이 없어 건너뜀"
  echo "[decide] $(date -u) alpha sweep: 성공한 모델이 없어 건너뜀" >> "$DECISIONS_LOG"
fi

python "$GEN_TABLE" "$RESULTS_JSONL" "$DRIVE_DIR/results_table.md"
bash "$SYNC_SH"
log "run_all.sh 전체 완료"
