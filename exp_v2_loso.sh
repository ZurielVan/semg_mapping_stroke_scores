#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MANIFEST_CSV="${MANIFEST_CSV:-${REPO_ROOT}/dataset/manifest.csv}"
LABELS_CSV="${LABELS_CSV:-${REPO_ROOT}/dataset/labels.csv}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/experiments/run_002_itransformer_temporal_grid}"
TRIAL_GRID_JSON="${TRIAL_GRID_JSON:-${OUTDIR}/preset_trials_v2.json}"

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
mkdir -p "${OUTDIR}"

export TRIAL_GRID_JSON
"${PYTHON_BIN}" - <<'PY'
import json
import os

path = os.environ["TRIAL_GRID_JSON"]

common = {
    "encoder_type": "itransformer",
    "dropout": 0.1,
    "overlap_ratio": 0.5,
    "windows_per_trial": 32,
    "trials_per_axis": 3,
    "huber_delta": 0.04,
    "lambda_ue": 0.05,
    "ema_decay": 0.999,
    "window_dropout_p": 0.1,
    "trial_dropout_p": 0.05,
    "consistency_weight": 0.1,
    "consistency_rampup_epochs": 20,
    "consistency_loss_type": "smoothl1",
    "heads": 4,
    "ffn_ratio": 4,
}

variants = [
    {"emb_dim": 128, "layers": 4, "Tw_samples": 512, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 128, "layers": 4, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 128, "layers": 6, "Tw_samples": 512, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 128, "layers": 6, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 4, "Tw_samples": 512, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 4, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 512, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 640, "lr_head": 1e-4, "weight_decay": 1e-3, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-4, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 640, "lr_head": 1e-4, "weight_decay": 1e-4, "lr_encoder_scale": 0.3},
    {"emb_dim": 256, "layers": 6, "Tw_samples": 640, "lr_head": 3e-5, "weight_decay": 1e-3, "lr_encoder_scale": 1.0},
]

grid = []
for v in variants:
    row = dict(common)
    row.update(v)
    grid.append(row)

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(grid, f, ensure_ascii=False, indent=2)
print(f"[INFO] Preset trial grid saved: {path} (n={len(grid)})")
PY

"${PYTHON_BIN}" -m semg_mapping_stroke_scores.grid_search \
  --manifest_csv "${MANIFEST_CSV}" \
  --labels_csv "${LABELS_CSV}" \
  --outdir "${OUTDIR}" \
  --n_trials 12 \
  --trial_grid_json "${TRIAL_GRID_JSON}" \
  --n_val_subjects "${N_VAL_SUBJECTS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --encoder_choices "itransformer" \
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

