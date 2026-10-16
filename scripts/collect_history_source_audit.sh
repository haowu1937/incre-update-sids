#!/usr/bin/env bash
# Run from the server repository after fetching the exact delivery commit.
set -euo pipefail
DELIVERY_SHA=${1:?Pass the full delivery commit SHA}
[[ $DELIVERY_SHA =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full commit SHA'; exit 1; }
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
git cat-file -e "${DELIVERY_SHA}^{commit}"
git merge-base --is-ancestor "$DELIVERY_SHA" research/vibe-dev
eval "$(conda shell.bash hook)"
conda activate frec-gpu
mkdir -p results/incremental/Beauty
RUN=$(mktemp -d "$REPO/results/incremental/Beauty/source_audit_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
git show "$DELIVERY_SHA:scripts/audit_history_sources.py" > "$RUN/audit_history_sources.py"
git show "$DELIVERY_SHA:scripts/collect_history_source_audit.sh" > "$RUN/run_command.sh"
printf '%s\n' "$DELIVERY_SHA" > "$RUN/delivery_commit.txt"
python -B "$RUN/audit_history_sources.py" --help > "$RUN/cli_help.txt"
python -u -B "$RUN/audit_history_sources.py" \
  --repo "$REPO" --delivery-sha "$DELIVERY_SHA" \
  --output "$RUN/report" 2>&1 | tee "$RUN/collection.log"

# Archive even if synchronization is blocked, preserving the original source.
finish() {
  tar -czf "$RUN.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
  cat "$RUN/report/compact.txt"
  cat "$RUN/sync_status.txt"
  printf 'RETURN_ARCHIVE=%s.tar.gz\n' "$RUN"
}
if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
  printf '%s\n' 'workspace_sync = blocked_dirty; original files and branch preserved' > "$RUN/sync_status.txt"
  git status --short
  finish
  exit 0
fi
if git show-ref --verify --quiet refs/heads/vibe-dev; then
  if ! git merge-base --is-ancestor vibe-dev "$DELIVERY_SHA"; then
    printf '%s\n' 'workspace_sync = blocked_diverged; original branch preserved' > "$RUN/sync_status.txt"
    finish
    exit 0
  fi
  git switch vibe-dev
  git merge --ff-only "$DELIVERY_SHA"
else
  git switch -c vibe-dev "$DELIVERY_SHA"
fi
test "$(git rev-parse HEAD)" = "$DELIVERY_SHA"
git branch --set-upstream-to=research/vibe-dev vibe-dev
git diff --check
python -B tests/incremental/test_history_source_audit.py 2>&1 | tee "$RUN/light_checks.log"
printf '%s\n' "workspace_sync = completed; branch=vibe-dev; commit=$DELIVERY_SHA" > "$RUN/sync_status.txt"
git status --short
finish
