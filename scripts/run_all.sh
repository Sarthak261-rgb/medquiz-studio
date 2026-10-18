#!/usr/bin/env bash
# Full pipeline on a local GPU. Every step resumes where it stopped, so rerunning
# after an interruption is safe.
set -euo pipefail
export PATH="/usr/bin:/bin:$PATH"   # a detached Git Bash on Windows starts without coreutils
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

if [ -x .venv/Scripts/medquiz ]; then MQ=.venv/Scripts/medquiz; else MQ=.venv/bin/medquiz; fi

step() { echo; echo "=== $* ($(date '+%H:%M:%S'))"; }

step ingest;   $MQ ingest --check-urls
step generate; $MQ generate --run-id gen-v1
step check;    $MQ check gen-v1
step control;  $MQ control
step stress;   $MQ stress --from-run gen-v1 --n 30
step report;   $MQ report
echo "PIPELINE DONE $(date '+%H:%M:%S')"
