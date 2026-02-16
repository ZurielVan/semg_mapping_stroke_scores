# 从 0 到 LOSO 结果（WSL/Bash 最短手册）

## 0. 安装/确认 WSL（Windows PowerShell，建议管理员）
```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

如果提示需要重启，先重启 Windows 再继续。

## 1. 进入 Bash（Ubuntu）
```powershell
wsl -d Ubuntu
```

## 2. 初始化 Python 环境（Ubuntu 内）
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

cd /mnt/z/Project_sEMG_FMA
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas tqdm
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

如需 CUDA 版 PyTorch，请改用官方 CUDA wheel 安装命令。

## 3. 生成训练数据索引和 npy
```bash
cd /mnt/z/Project_sEMG_FMA
bash semg_mapping_stroke_scores/prepareData.sh
```

产物：
- `semg_mapping_stroke_scores/dataset/manifest.csv`
- `semg_mapping_stroke_scores/dataset/labels.csv`
- `semg_mapping_stroke_scores/dataset/npy/`

## 4. 先跑一个快速冒烟（CPU，几分钟级）
```bash
cd /mnt/z/Project_sEMG_FMA
DEVICE=cpu N_TRIALS=1 SSL_EPOCHS=1 SSL_WINDOWS_PER_EPOCH=2000 \
OUTDIR=semg_mapping_stroke_scores/experiments/smoke_run \
bash semg_mapping_stroke_scores/exp_v1_loso.sh
```

## 5. 正式 LOSO（按默认配置）
```bash
cd /mnt/z/Project_sEMG_FMA
DEVICE=cuda bash semg_mapping_stroke_scores/exp_v1_loso.sh
```

可调参数（环境变量）：
- `N_TRIALS`（默认 `20`）
- `N_VAL_SUBJECTS`（默认 `1`）
- `SEED`（默认 `19990514`）
- `SSL_EPOCHS`（默认 `100`）
- `SSL_WINDOWS_PER_EPOCH`（默认 `20000`）
- `OUTDIR`（默认 `semg_mapping_stroke_scores/experiments/run_001`）
- `PYTHON_BIN`（默认 `python`）

## 6. 查看结果
- 折叠汇总：`semg_mapping_stroke_scores/experiments/run_001/summary.jsonl`
- 每折详情：`semg_mapping_stroke_scores/experiments/run_001/fold_*/fold_record.json`
