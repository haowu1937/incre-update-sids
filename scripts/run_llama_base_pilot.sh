#!/usr/bin/env bash
# Extract this exact script with git show before switching the server worktree.
set -euo pipefail
DELIVERY_SHA=${1:?Pass the full delivery commit SHA}
MODE=${2:-}
[[ $DELIVERY_SHA =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full commit SHA'; exit 1; }
[[ -z $MODE || $MODE == --sync-only ]] || { echo 'Unknown mode'; exit 1; }
cd "$(git rev-parse --show-toplevel)"
git branch --show-current
git rev-parse HEAD
git status --short
git cat-file -e "${DELIVERY_SHA}^{commit}"
git merge-base --is-ancestor "$DELIVERY_SHA" research/vibe-dev
if git show-ref --verify --quiet refs/heads/vibe-dev; then
  git merge-base --is-ancestor vibe-dev "$DELIVERY_SHA" || {
    echo 'Local vibe-dev diverges or is ahead; no files changed.'; exit 1;
  }
fi

STATE=$(git status --porcelain=v1 --untracked-files=all)
CONFLICT=verify_incremental_embeddings.py
if [[ $STATE == "?? $CONFLICT" ]]; then
  [[ -f $CONFLICT && ! -L $CONFLICT ]] || { echo 'Expected a regular file; stopped.'; exit 1; }
  mkdir -p results/incremental/Beauty
  BACKUP=$(mktemp -d "results/incremental/Beauty/local_verifier_backup_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  git check-ignore -q "$BACKUP/target_verifier.py" || { echo 'Backup must be ignored by Git; stopped.'; exit 1; }
  git show "$DELIVERY_SHA:$CONFLICT" > "$BACKUP/target_verifier.py"
  if ! cmp -s "$CONFLICT" "$BACKUP/target_verifier.py"; then
    echo 'Verifier differs from the delivered file; original file and branch preserved.'
    sha256sum "$CONFLICT" "$BACKUP/target_verifier.py"
    exit 1
  fi
  git branch --show-current > "$BACKUP/original_branch.txt"
  git rev-parse HEAD > "$BACKUP/original_commit.txt"
  sha256sum "$CONFLICT" > "$BACKUP/original_sha256.txt"
  mv -- "$CONFLICT" "$BACKUP/$CONFLICT"
  printf 'Identical untracked verifier preserved at: %s/%s\n' "$BACKUP" "$CONFLICT"
elif [[ -n $STATE ]]; then
  echo 'Other local changes exist; original files and branch preserved.'
  exit 1
fi
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Worktree changed during synchronization; stopped.'; exit 1;
}
if git show-ref --verify --quiet refs/heads/vibe-dev; then
  git switch vibe-dev
  git merge --ff-only "$DELIVERY_SHA"
else
  git switch -c vibe-dev "$DELIVERY_SHA"
fi
test "$(git rev-parse HEAD)" = "$DELIVERY_SHA"
git diff --check
printf 'workspace_sync = completed; branch=vibe-dev; commit=%s\n' "$DELIVERY_SHA"
[[ $MODE != --sync-only ]] || exit 0

eval "$(conda shell.bash hook)"
conda activate frec-gpu
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FEATURES_DIR=cache/incremental/Beauty/llama7b_features_20261016T132127Z
mkdir -p results/incremental/Beauty ckpt/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_base_pilot_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
BASE_DIR="ckpt/incremental/Beauty/$(basename "$RUN")_50"
git rev-parse HEAD > "$RUN/commit.txt"
git show "$DELIVERY_SHA:scripts/run_llama_base_pilot.sh" > "$RUN/run_command.sh"
printf 'FEATURES_DIR=%s\nBASE_DIR=%s\nDELIVERY_SHA=%s\n' "$FEATURES_DIR" "$BASE_DIR" "$DELIVERY_SHA" > "$RUN/paths.txt"
python -B train_incremental_rqvae.py --help > "$RUN/cli_help.txt"
printf 'Training log: %s/train.log\n' "$RUN"
if ! python -u -B train_incremental_rqvae.py \
  --features-dir "$FEATURES_DIR" --output "$BASE_DIR" \
  --device cuda:0 --max-steps 50 --batch-size 256 --lr 0.001 --weight-decay 0 \
  --seed 2026 --num-emb-list 256 256 256 --e-dim 32 --layers 512 256 128 64 \
  --quant-loss-weight 0.25 --kmeans-iters 50 --init-samples 4096 \
  --eval-batch-size 256 --save-every 25 --log-every 10 \
  > "$RUN/train.log" 2>&1; then
  tail -n 100 "$RUN/train.log"
  exit 1
fi
python -B report_content_rqvae.py --features-dir "$FEATURES_DIR" \
  --rqvae-dir "$BASE_DIR" --output "$RUN/report" 2>&1 | tee "$RUN/report.log"
printf 'RESULT_DIR=%s\n' "$RUN"
