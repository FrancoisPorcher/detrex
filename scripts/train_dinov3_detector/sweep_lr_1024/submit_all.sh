#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/private/home/francoisporcher/detrex"
SCRIPT_DIR="${REPO_ROOT}/scripts/train_dinov3_detector/sweep_lr_1024"

sbatch "${SCRIPT_DIR}/train_lr_2e-5.sh"
sbatch "${SCRIPT_DIR}/train_lr_5e-5.sh"
sbatch "${SCRIPT_DIR}/train_lr_1e-4.sh"
sbatch "${SCRIPT_DIR}/train_lr_2e-4.sh"
sbatch "${SCRIPT_DIR}/train_lr_5e-4.sh"
