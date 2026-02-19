#!/usr/bin/env bash
set -euo pipefail

# One-click setup for Project_sEMG_FMA GPU env (Python 3.10 + CUDA + MGCN/Mamba stack)
# Usage:
#   bash setup_mgcn_gpu_env.sh /path/to/Project_sEMG_FMA
# If no path is provided, default is: /mnt/z/Project_sEMG_FMA

PROJECT_DIR="${1:-/mnt/z/Project_sEMG_FMA}"
VENV_DIR="${PROJECT_DIR}/.venv310"

log() { echo -e "\n[INFO] $*"; }
err() { echo -e "\n[ERROR] $*" >&2; }

if ! command -v sudo >/dev/null 2>&1; then
  err "sudo not found. Please install sudo or run as root."
  exit 1
fi

log "Step 0/9: Check GPU visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "nvidia-smi not found. Please install NVIDIA driver first."
  exit 1
fi
nvidia-smi || { err "GPU not available"; exit 1; }

log "Step 1/9: Install Python 3.10 (Ubuntu 24.04 path)"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils

log "Step 2/9: Install CUDA toolkit (nvcc)"
sudo apt-get install -y nvidia-cuda-toolkit
nvcc --version

log "Step 3/9: Validate project path"
if [[ ! -d "${PROJECT_DIR}" ]]; then
  err "Project directory not found: ${PROJECT_DIR}"
  exit 1
fi

log "Step 4/9: Create and activate venv (.venv310)"
python3.10 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel

log "Step 5/9: Install PyTorch (CUDA 12.1)"
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

log "Step 6/9: Install base scientific stack"
pip install numpy==1.26.4 pandas==2.2.3 tqdm==4.66.5 matplotlib==3.9.2

log "Step 7/9: Install DGL (torch2.3 + cu121)"
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

log "Step 8/9: Install Mamba dependencies"
pip install --no-build-isolation causal-conv1d==1.4.0
pip install --no-build-isolation mamba-ssm==2.2.2
pip install --force-reinstall transformers==4.38.2

log "Step 9/9: Verify full environment"
python - << 'PY'
import torch, dgl, causal_conv1d, mamba_ssm, transformers, numpy, pandas
print('torch', torch.__version__)
print('dgl', dgl.__version__)
print('causal', causal_conv1d.__version__)
print('mamba', mamba_ssm.__version__)
print('transformers', transformers.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
print('cuda', torch.cuda.is_available())
PY

log "DONE ✅"
log "Activate with: source ${VENV_DIR}/bin/activate"
log "Run smoke test:"
echo "python -m semg_mapping_stroke_scores.model_matrix_smoke_test --manifest_csv ${PROJECT_DIR}/dataset_smoke/manifest.csv --labels_csv ${PROJECT_DIR}/dataset_smoke/labels.csv --outdir ${PROJECT_DIR}/runs/smoke_gpu --device cuda"
