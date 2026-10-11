#!/bin/bash
# idempotent: 몇 번을 다시 실행해도 안전하다 (런타임이 새로 할당된 뒤 재실행 전제).
set -uo pipefail

DRIVE_DIR="/content/drive/MyDrive/smoothquant_vlm"
REPO_DIR="/content/repo"
RESULTS_REPO_DIR="/content/results_repo"
HF_CACHE_DIR="/content/hf_cache"
DATA_DIR="/content/data"
GH_TOKEN_FILE="/content/.gh_token"
RESULTS_REMOTE="https://github.com/yoohyunseok/smoothquant_results.git"

mkdir -p "$DRIVE_DIR/results" "$DRIVE_DIR/logs" "$DRIVE_DIR/act_scales"
mkdir -p "$HF_CACHE_DIR" "$DATA_DIR"

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

SETUP_LOG="$DRIVE_DIR/logs/setup.log"
echo "[setup] $(date -u) 시작" | tee -a "$SETUP_LOG"

# 1. 패키지 설치
# Colab 기본 CUDA torch 는 절대 건드리지 않는다 (torch/torchvision/torchaudio 미지정).
echo "[setup] transformers==4.36.0 / tokenizers==0.15.2 / huggingface_hub==0.23.4 설치" | tee -a "$SETUP_LOG"
pip install --no-deps "transformers==4.36.0" "tokenizers==0.15.2" "huggingface_hub==0.23.4" 2>&1 | tee -a "$SETUP_LOG"

echo "[setup] 나머지 의존 패키지 확인/설치 (버전 고정 없이 - 이미 있으면 스킵됨)" | tee -a "$SETUP_LOG"
pip install -q accelerate datasets zstandard matplotlib 2>&1 | tee -a "$SETUP_LOG"

# datasets 최신판이 huggingface_hub를 1.x로 끌어올려 transformers==4.36.0/tokenizers==0.15.2
# 의 "huggingface_hub<1.0" 요구와 충돌하는 것을 확인함(2026-10-11). 기능적으로는 0.23.4로도
# datasets/huggingface_hub 사용 API(hf_hub_download, load_dataset 등)가 전부 동작하므로 되돌림.
echo "[setup] huggingface_hub 를 0.23.4로 재고정 (datasets가 1.x로 끌어올리는 것을 방지)" | tee -a "$SETUP_LOG"
pip install --no-deps "huggingface_hub==0.23.4" 2>&1 | tee -a "$SETUP_LOG"

# 2. smoothquant 원본 저장소
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[setup] smoothquant 저장소 clone" | tee -a "$SETUP_LOG"
  git clone https://github.com/mit-han-lab/smoothquant.git "$REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"
else
  echo "[setup] $REPO_DIR 이미 존재, clone 생략" | tee -a "$SETUP_LOG"
fi
pip install --no-deps -e "$REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"

if [ ! -f "$DRIVE_DIR/my_ppl_eval.py" ]; then
  echo "[setup][경고] my_ppl_eval.py 가 Drive에 없습니다. 실험을 실행하기 전에 생성되어야 합니다." | tee -a "$SETUP_LOG"
fi

# 3. 결과 저장용 GitHub 비공개 저장소 clone
if [ -f "$GH_TOKEN_FILE" ]; then
  if [ ! -d "$RESULTS_REPO_DIR/.git" ]; then
    echo "[setup] 결과 저장소 clone -> $RESULTS_REPO_DIR" | tee -a "$SETUP_LOG"
    git -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$(cat "$GH_TOKEN_FILE")" | base64 -w0)" \
      clone "$RESULTS_REMOTE" "$RESULTS_REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"
  else
    echo "[setup] $RESULTS_REPO_DIR 이미 존재, clone 생략" | tee -a "$SETUP_LOG"
  fi
  if [ -d "$RESULTS_REPO_DIR/.git" ]; then
    git -C "$RESULTS_REPO_DIR" config user.name "colab-runner"
    git -C "$RESULTS_REPO_DIR" config user.email "noreply@example.com"
    if [ ! -f "$RESULTS_REPO_DIR/.gitignore" ]; then
      cat > "$RESULTS_REPO_DIR/.gitignore" <<'EOF'
*.pt
*.bin
*.safetensors
*.zst
hf_cache/
models/
__pycache__/
EOF
    fi
    mkdir -p "$RESULTS_REPO_DIR/results" "$RESULTS_REPO_DIR/logs"
  fi
else
  echo "[setup][경고] $GH_TOKEN_FILE 없음 - GitHub 결과 동기화는 비활성화됩니다." | tee -a "$SETUP_LOG"
fi

# 4. 패키지 버전 스냅샷 기록
VERS_FILE="$DRIVE_DIR/logs/package_versions.txt"
{
  echo "기록 시각(UTC): $(date -u)"
  python3 --version
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
  python3 -c "import transformers; print('transformers', transformers.__version__)"
  python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
  python3 -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
  python3 -c "import numpy; print('numpy', numpy.__version__)"
  python3 -c "import datasets; print('datasets', datasets.__version__)"
  python3 -c "import accelerate; print('accelerate', accelerate.__version__)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$VERS_FILE" 2>&1
echo "[setup] 패키지 버전 기록 -> $VERS_FILE" | tee -a "$SETUP_LOG"

echo "[setup] $(date -u) 완료" | tee -a "$SETUP_LOG"
