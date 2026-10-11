#!/bin/bash
# 설정 한 건이 끝날 때마다 run_all.sh가 호출한다.
# 1) Drive는 이미 같은 파일 시스템이라 쓰기 즉시 반영되지만 sync로 한번 더 보장.
# 2) GitHub 토큰이 있으면 결과 파일만 복사해 commit/push. 실패해도 실험은 멈추지 않는다.
set -uo pipefail

DRIVE_DIR="/content/drive/MyDrive/smoothquant_vlm"
RESULTS_REPO_DIR="/content/results_repo"
GH_TOKEN_FILE="/content/.gh_token"
RESULTS_REMOTE="https://github.com/yoohyunseok/smoothquant_results.git"
SYNC_LOG="$DRIVE_DIR/logs/sync.log"

mkdir -p "$DRIVE_DIR/logs"

sync 2>&1 | tee -a "$SYNC_LOG" >/dev/null
echo "[sync] $(date -u) Drive sync 호출 완료" >> "$SYNC_LOG"

if [ ! -f "$GH_TOKEN_FILE" ]; then
  echo "[sync] $(date -u) $GH_TOKEN_FILE 없음 - GitHub 푸시 건너뜀" | tee -a "$SYNC_LOG"
  exit 0
fi

if [ ! -d "$RESULTS_REPO_DIR/.git" ]; then
  echo "[sync][경고] $(date -u) $RESULTS_REPO_DIR 가 없음 - setup.sh를 먼저 실행해야 함" | tee -a "$SYNC_LOG"
  exit 0
fi

mkdir -p "$RESULTS_REPO_DIR/results" "$RESULTS_REPO_DIR/logs" "$RESULTS_REPO_DIR/scripts"
cp -f "$DRIVE_DIR"/results/*.jsonl "$RESULTS_REPO_DIR/results/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/results_table.md "$RESULTS_REPO_DIR/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/logs/*.log "$RESULTS_REPO_DIR/logs/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/*.png "$RESULTS_REPO_DIR/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/RESULTS_COLAB.md "$RESULTS_REPO_DIR/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/SETUP_LOG_COLAB.md "$RESULTS_REPO_DIR/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/ARCHITECTURE.md "$RESULTS_REPO_DIR/" 2>/dev/null || true
# 재현 가능하도록 스크립트 자체도 결과 저장소에 같이 올림 (코드, 텍스트 파일만 - 모델/캐시는 제외)
cp -f "$DRIVE_DIR"/*.sh "$RESULTS_REPO_DIR/scripts/" 2>/dev/null || true
cp -f "$DRIVE_DIR"/*.py "$RESULTS_REPO_DIR/scripts/" 2>/dev/null || true

cd "$RESULTS_REPO_DIR" || exit 0
git add -A
if git commit -m "sync results $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SYNC_LOG" 2>&1; then
  echo "[sync] $(date -u) commit 생성" >> "$SYNC_LOG"
else
  echo "[sync] $(date -u) 커밋할 변경 없음 (또는 커밋 실패, 위 로그 참고)" >> "$SYNC_LOG"
fi

if git -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$(cat "$GH_TOKEN_FILE")" | base64 -w0)" \
     push origin HEAD >> "$SYNC_LOG" 2>&1; then
  echo "[sync] $(date -u) push 성공" | tee -a "$SYNC_LOG"
else
  echo "[sync][경고] $(date -u) push 실패 - 네트워크/인증 문제일 수 있음. 다음 동기화 때 재시도." | tee -a "$SYNC_LOG"
fi
