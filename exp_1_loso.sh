#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TARGET_MODE="${TARGET_MODE:-wh}"
export ENCODER_CHOICES="${ENCODER_CHOICES:-mgcn}"

exec bash "${SCRIPT_DIR}/exp_v1_loso.sh" "$@"
