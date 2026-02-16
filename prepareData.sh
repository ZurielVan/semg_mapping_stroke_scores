#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m semg_mapping_stroke_scores.prepare_data \
  --ensembled_info_csv "${SCRIPT_DIR}/dataExplore/ensembledInfo.csv" \
  --data_info_csv "${SCRIPT_DIR}/dataExplore/dataInfo.csv" \
  --data_root "${REPO_ROOT}/data/Logan_16_subjects/Data" \
  --out_dir "${REPO_ROOT}/dataset" \
  --convert_to_npy \
  --emg_cols 1,2,3,4 \
  --time_col 0 \
  --expected_emg_ch 4 \
  --drop_unmatched_subjects
