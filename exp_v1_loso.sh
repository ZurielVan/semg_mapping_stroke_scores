#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MANIFEST_CSV="${MANIFEST_CSV:-${REPO_ROOT}/dataset/manifest.csv}"
LABELS_CSV="${LABELS_CSV:-${REPO_ROOT}/dataset/labels.csv}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/experiments/run_001}"

N_TRIALS="${N_TRIALS:-20}"
N_VAL_SUBJECTS="${N_VAL_SUBJECTS:-1}"
SEED="${SEED:-19990514}"
DEVICE="${DEVICE:-cuda}"
SSL_EPOCHS="${SSL_EPOCHS:-100}"
SSL_WINDOWS_PER_EPOCH="${SSL_WINDOWS_PER_EPOCH:-20000}"

if [[ ! -f "${MANIFEST_CSV}" || ! -f "${LABELS_CSV}" ]]; then
  echo "Missing dataset CSVs. Run: bash ${SCRIPT_DIR}/prepareData.sh"
  exit 1
fi

cd "${REPO_ROOT}"

# For exhaustive grid traversal, print search-space summary and require confirmation.
if [[ "${N_TRIALS}" == "-1" ]]; then
  echo "[INFO] N_TRIALS=-1 -> exhaustive grid traversal mode."
  "${PYTHON_BIN}" - <<'PY'
from semg_mapping_stroke_scores.grid_search import search_space_summary, grid_size

encoder_choices = ["tcn", "rescnn", "itransformer", "transformer", "gcn"]
from semg_mapping_stroke_scores.grid_search import check_mamba_ssm_available
mamba_ok, mamba_reason = check_mamba_ssm_available()
if mamba_ok:
    encoder_choices.append("mamba")
    encoder_choices.append("mgcn")
else:
    print(
        "[WARN] Disable encoder_type in {'mamba','mgcn'} for this run because "
        f"mamba-ssm is not usable: {mamba_reason}"
    )

total = grid_size(encoder_choices)

print(f"[GRID] Total combinations per fold: {total}")
print("[GRID] Search space table:")
for info in search_space_summary(encoder_choices):
    print(f"  * encoder_type={info['encoder_type']} (combos={info['combos']})")
    for k, vals in info["space"].items():
        vals_text = ", ".join(str(v) for v in vals)
        print(f"    - {k} ({len(vals)}): {vals_text}")
PY

  if [[ -t 0 ]]; then
    read -r -p "Proceed with exhaustive grid traversal? [y/N]: " CONFIRM_GRID
    case "${CONFIRM_GRID}" in
      y|Y) ;;
      *)
        echo "Cancelled by user."
        exit 0
        ;;
    esac
  else
    echo "N_TRIALS=-1 requires interactive confirmation (tty). Aborting."
    exit 1
  fi
fi

mkdir -p "${OUTDIR}"

"${PYTHON_BIN}" -m semg_mapping_stroke_scores.grid_search \
  --manifest_csv "${MANIFEST_CSV}" \
  --labels_csv "${LABELS_CSV}" \
  --outdir "${OUTDIR}" \
  --n_trials "${N_TRIALS}" \
  --n_val_subjects "${N_VAL_SUBJECTS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --ssl_epochs "${SSL_EPOCHS}" \
  --ssl_windows_per_epoch "${SSL_WINDOWS_PER_EPOCH}"
