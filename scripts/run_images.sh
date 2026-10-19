#!/usr/bin/env bash
# Image track on a local GPU: figures, base SD 1.5 images, LoRA training, LoRA images,
# automatic checks. Every step resumes where it stopped.
set -euo pipefail
export PATH="/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_DISABLE_PROGRESS_BARS=1

if [ -x .venv/Scripts/medquiz ]; then MQ=.venv/Scripts/medquiz; else MQ=.venv/bin/medquiz; fi
step() { echo; echo "=== $* ($(date '+%H:%M:%S'))"; }

step figures;       $MQ images figures
step base images;   $MQ images generate --arm base
step train lora;    $MQ images train
step lora images;   $MQ images generate --arm lora
step check images;  $MQ images check
echo "IMAGES DONE $(date '+%H:%M:%S')"
