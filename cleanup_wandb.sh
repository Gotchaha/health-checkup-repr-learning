#!/usr/bin/env bash
# Purpose: Record and fully clean local W&B traces (project dir + user cache/config + API creds),
#          with a tamper-evident session log via `script`.
# Notes:
# - This removes: ./wandb, ~/.cache/wandb, ~/.config/wandb, and (backup then remove) ~/.netrc.
# - We assume ~/.netrc only contains the W&B (api.wandb.ai) entry (as you confirmed).
# - The session log is named wandb_cleanup_<UTC_TIMESTAMP>.log in the current directory.

set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="wandb_cleanup_${STAMP}.log"

# Create a temporary command file that will be executed under `script`
TMP="$(mktemp -t wandb_cleanup.XXXXXX)"

cat > "$TMP" <<'REC'
set -euo pipefail

# Add a clear UTC timestamp to the recorded session
date -u

echo "=== BEFORE SNAPSHOT ==="
# List current state; suppress 'No such file' noise if paths are absent
ls -la ./wandb ~/.cache/wandb ~/.config/wandb ~/.netrc 2>/dev/null || true

echo "=== DELETE PHASE ==="
# 1) Remove project-local W&B logs and pending offline runs
rm -rf ./wandb

# 2) Remove user-level caches (downloaded artifacts, staging, etc.)
rm -rf ~/.cache/wandb

# 3) Remove user-level W&B config (prevents auto-pointing to old projects)
rm -rf ~/.config/wandb

# 4) Backup then remove API credentials (ensures this machine can't upload)
if [ -f ~/.netrc ]; then
  cp ~/.netrc ~/.netrc.bak.$(date -u +%Y%m%dT%H%M%SZ)
  rm -f ~/.netrc
fi

echo "=== AFTER SNAPSHOT ==="
# Re-list to prove removal
ls -la ./wandb ~/.cache/wandb ~/.config/wandb ~/.netrc 2>/dev/null || true

echo "Check ~/.netrc content (should be absent):"
[ -f ~/.netrc ] && sed -n '1,200p' ~/.netrc || echo "~/.netrc not present (removed)"

REC

# Record the whole operation with `script`.
# util-linux `script` prefers: script -q -c "CMD" LOGFILE
# BSD/macOS `script` supports: script -q LOGFILE CMD [args...]
if script -V >/dev/null 2>&1; then
  # util-linux flavor
  script -q -c "bash '$TMP'" "$LOG"
else
  # BSD/macOS flavor
  script -q "$LOG" bash "$TMP"
fi

rm -f "$TMP"

echo "Done. Session log written to: $LOG"
