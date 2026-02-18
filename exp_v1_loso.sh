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
SSL_USE_MASKED_RECON="${SSL_USE_MASKED_RECON:-true}"
SSL_RECON_WEIGHT="${SSL_RECON_WEIGHT:-0.5}"
SSL_MASK_RATIO="${SSL_MASK_RATIO:-0.5}"
SSL_MASK_BLOCK_LEN="${SSL_MASK_BLOCK_LEN:-16}"
SSL_MASK_MODE="${SSL_MASK_MODE:-time}"
SSL_RECON_DECODER_HIDDEN="${SSL_RECON_DECODER_HIDDEN:-256}"
SSL_RECON_LOSS_TYPE="${SSL_RECON_LOSS_TYPE:-smoothl1}"
SSL_RECON_SMOOTHL1_BETA="${SSL_RECON_SMOOTHL1_BETA:-0.02}"
TARGET_MODE="${TARGET_MODE:-whse}"
ENCODER_CHOICES="${ENCODER_CHOICES:-itransformer}"
WINDOW_AGG_MODE="${WINDOW_AGG_MODE:-temporal_transformer}"
WINDOW_TEMPORAL_LAYERS="${WINDOW_TEMPORAL_LAYERS:-2}"
WINDOW_TEMPORAL_HEADS="${WINDOW_TEMPORAL_HEADS:-4}"
WINDOW_TEMPORAL_FFN_RATIO="${WINDOW_TEMPORAL_FFN_RATIO:-2}"
WINDOW_TEMPORAL_DROPOUT="${WINDOW_TEMPORAL_DROPOUT:-0.1}"

if [[ ! -f "${MANIFEST_CSV}" || ! -f "${LABELS_CSV}" ]]; then
  echo "Missing dataset CSVs. Run: bash ${SCRIPT_DIR}/prepareData.sh"
  exit 1
fi

cd "${REPO_ROOT}"

# For exhaustive grid traversal, print search-space summary and require confirmation.
if [[ "${N_TRIALS}" == "-1" ]]; then
  echo "[INFO] N_TRIALS=-1 -> exhaustive grid traversal mode."
  "${PYTHON_BIN}" - <<'PY'
import os
from semg_mapping_stroke_scores.grid_search import search_space_summary, grid_size

req = os.environ.get("ENCODER_CHOICES", "").strip()
if req:
    encoder_choices = [x.strip().lower() for x in req.split(",") if x.strip()]
else:
    encoder_choices = ["tcn", "rescnn", "itransformer", "transformer", "gcn"]
from semg_mapping_stroke_scores.grid_search import check_mamba_ssm_available
mamba_ok, mamba_reason = check_mamba_ssm_available()
if not mamba_ok and any(x in {"mamba", "mgcn"} for x in encoder_choices):
    print(
        "[WARN] Requested encoder_type in {'mamba','mgcn'} but "
        f"mamba-ssm is not usable: {mamba_reason}"
    )
    raise SystemExit(1)
if mamba_ok and not req:
    encoder_choices.append("mamba")
    encoder_choices.append("mgcn")

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
  --encoder_choices "${ENCODER_CHOICES}" \
  --ssl_epochs "${SSL_EPOCHS}" \
  --ssl_windows_per_epoch "${SSL_WINDOWS_PER_EPOCH}" \
  --ssl_use_masked_recon "${SSL_USE_MASKED_RECON}" \
  --ssl_recon_weight "${SSL_RECON_WEIGHT}" \
  --ssl_mask_ratio "${SSL_MASK_RATIO}" \
  --ssl_mask_block_len "${SSL_MASK_BLOCK_LEN}" \
  --ssl_mask_mode "${SSL_MASK_MODE}" \
  --ssl_recon_decoder_hidden "${SSL_RECON_DECODER_HIDDEN}" \
  --ssl_recon_loss_type "${SSL_RECON_LOSS_TYPE}" \
  --ssl_recon_smoothl1_beta "${SSL_RECON_SMOOTHL1_BETA}" \
  --target_mode "${TARGET_MODE}" \
  --window_agg_mode "${WINDOW_AGG_MODE}" \
  --window_temporal_layers "${WINDOW_TEMPORAL_LAYERS}" \
  --window_temporal_heads "${WINDOW_TEMPORAL_HEADS}" \
  --window_temporal_ffn_ratio "${WINDOW_TEMPORAL_FFN_RATIO}" \
  --window_temporal_dropout "${WINDOW_TEMPORAL_DROPOUT}"
