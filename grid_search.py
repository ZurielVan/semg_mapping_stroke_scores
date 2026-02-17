from __future__ import annotations

import argparse
import csv
import itertools
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .splits import make_loso_folds, split_train_val_subjects
from .utils import set_seed, ensure_dir, append_jsonl, seed_worker, save_json, torch_load_compat
from .baseModels import EncoderConfig, build_encoder, HierarchicalMILRegressor
from .train_ssl import pretrain_ssl_moco, SSLTrainConfig
from .train_mil import train_mil_supervised, MILTrainConfig, evaluate_mil
from .dataset import SessionBagDataset


def parse_int_list(s: Optional[str]) -> Optional[list[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",")]


HPARAM_ORDER = [
    "encoder_type",
    "emb_dim",
    "dropout",
    "Tw_samples",
    "overlap_ratio",
    "Ts_samples",
    "windows_per_trial",
    "trials_per_axis",
    "lr_head",
    "lr_encoder_scale",
    "weight_decay",
    "huber_delta",
    "lambda_ue",
    "ema_decay",
    "window_dropout_p",
    "trial_dropout_p",
    "consistency_weight",
    "consistency_rampup_epochs",
    "consistency_loss_type",
    "patch_size",
    "layers",
    "heads",
    "ffn_ratio",
    "tcn_blocks",
    "tcn_kernel",
    "tcn_width",
    "tcn_stem_stride",
    "gcn_layers",
    "gcn_learn_adj",
]
TEST_METRIC_KEYS = [
    "MAE_WH",
    "MAE_SE",
    "MAE_UE",
    "R2_WH",
    "R2_SE",
    "R2_UE",
    "Corr_WH",
    "Corr_SE",
    "Corr_UE",
    "RMSE_WH",
    "RMSE_SE",
    "within1_WH",
    "within2_WH",
    "within1_SE",
    "within2_SE",
    "val_score",
]

PARAM_CANDIDATES: Dict[str, list[Any]] = {
    "emb_dim": [64, 128, 256],
    "dropout": [0.05, 0.1, 0.2, 0.3],
    "Tw_samples": [256, 384, 512, 640],
    "overlap_ratio": [0.0, 0.25, 0.5],
    "windows_per_trial": [8, 16, 32],
    "trials_per_axis": [3, 4, 6],
    "lr_head": [3e-4, 1e-4, 3e-5],
    "lr_encoder_scale": [0.1, 0.3, 1.0],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "huber_delta": [0.02, 0.04, 0.08],
    "lambda_ue": [0.0, 0.05, 0.1, 0.2],
    "ema_decay": [0.99, 0.995, 0.999],
    "window_dropout_p": [0.0, 0.1, 0.2, 0.3],
    "trial_dropout_p": [0.0, 0.05, 0.1, 0.15],
    "consistency_weight": [0.0, 0.05, 0.1, 0.2, 0.4],
    "consistency_rampup_epochs": [0, 10, 20, 40],
    "consistency_loss_type": ["mse", "smoothl1"],
    "patch_size": [4, 8, 16],
    "layers": [2, 4, 6],
    "heads": [2, 4, 8],
    "ffn_ratio": [2, 4],
    "tcn_blocks": [3, 5, 7],
    "tcn_kernel": [3, 5, 7],
    "tcn_width": [32, 64, 96],
    "tcn_stem_stride": [1, 2],
    "gcn_layers": [1, 2, 3],
    "gcn_learn_adj": [False, True],
}

COMMON_PARAM_NAMES = [
    "emb_dim",
    "dropout",
    "Tw_samples",
    "overlap_ratio",
    "windows_per_trial",
    "trials_per_axis",
    "lr_head",
    "lr_encoder_scale",
    "weight_decay",
    "huber_delta",
    "lambda_ue",
    "ema_decay",
    "window_dropout_p",
    "trial_dropout_p",
    "consistency_weight",
    "consistency_rampup_epochs",
    "consistency_loss_type",
]

MODEL_PARAM_NAMES: Dict[str, list[str]] = {
    "tcn": ["tcn_blocks", "tcn_kernel", "tcn_width", "tcn_stem_stride"],
    "rescnn": ["tcn_blocks", "tcn_kernel", "tcn_width", "tcn_stem_stride"],
    "transformer": ["patch_size", "layers", "heads", "ffn_ratio"],
    "itransformer": ["layers", "heads", "ffn_ratio"],
    "gcn": ["gcn_layers", "gcn_learn_adj"],
    "mamba": ["patch_size", "layers"],
    "mgcn": [],
}


def active_param_names(encoder_type: str) -> list[str]:
    et = str(encoder_type).lower()
    if et not in MODEL_PARAM_NAMES:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
    return COMMON_PARAM_NAMES + MODEL_PARAM_NAMES[et]


def active_hparam_space_for_encoder(encoder_type: str) -> Dict[str, list[Any]]:
    names = active_param_names(encoder_type)
    return {k: PARAM_CANDIDATES[k] for k in names}


def _first_defaults_for_all_params() -> Dict[str, Any]:
    return {k: v[0] for k, v in PARAM_CANDIDATES.items()}


def build_hparams_for_encoder(encoder_type: str, selected_params: Dict[str, Any]) -> Dict[str, Any]:
    params = _first_defaults_for_all_params()
    params.update(selected_params)
    return _build_hparams(
        encoder_type=str(encoder_type),
        emb_dim=int(params["emb_dim"]),
        dropout=float(params["dropout"]),
        Tw_samples=int(params["Tw_samples"]),
        overlap_ratio=float(params["overlap_ratio"]),
        windows_per_trial=int(params["windows_per_trial"]),
        trials_per_axis=int(params["trials_per_axis"]),
        lr_head=float(params["lr_head"]),
        lr_encoder_scale=float(params["lr_encoder_scale"]),
        weight_decay=float(params["weight_decay"]),
        huber_delta=float(params["huber_delta"]),
        lambda_ue=float(params["lambda_ue"]),
        ema_decay=float(params["ema_decay"]),
        window_dropout_p=float(params["window_dropout_p"]),
        trial_dropout_p=float(params["trial_dropout_p"]),
        consistency_weight=float(params["consistency_weight"]),
        consistency_rampup_epochs=int(params["consistency_rampup_epochs"]),
        consistency_loss_type=str(params["consistency_loss_type"]),
        patch_size=int(params["patch_size"]),
        layers=int(params["layers"]),
        heads=int(params["heads"]),
        ffn_ratio=int(params["ffn_ratio"]),
        tcn_blocks=int(params["tcn_blocks"]),
        tcn_kernel=int(params["tcn_kernel"]),
        tcn_width=int(params["tcn_width"]),
        tcn_stem_stride=int(params["tcn_stem_stride"]),
        gcn_layers=int(params["gcn_layers"]),
        gcn_learn_adj=bool(params["gcn_learn_adj"]),
    )


def search_space_summary(encoder_choices: list[str]) -> list[Dict[str, Any]]:
    out = []
    for et in encoder_choices:
        space = active_hparam_space_for_encoder(et)
        combos = 1
        for vals in space.values():
            combos *= len(vals)
        out.append({"encoder_type": et, "space": space, "combos": int(combos)})
    return out


def _build_hparams(
    *,
    encoder_type: str,
    emb_dim: int,
    dropout: float,
    Tw_samples: int,
    overlap_ratio: float,
    windows_per_trial: int,
    trials_per_axis: int,
    lr_head: float,
    lr_encoder_scale: float,
    weight_decay: float,
    huber_delta: float,
    lambda_ue: float,
    ema_decay: float,
    window_dropout_p: float,
    trial_dropout_p: float,
    consistency_weight: float,
    consistency_rampup_epochs: int,
    consistency_loss_type: str,
    patch_size: int,
    layers: int,
    heads: int,
    ffn_ratio: int,
    tcn_blocks: int,
    tcn_kernel: int,
    tcn_width: int,
    tcn_stem_stride: int,
    gcn_layers: int,
    gcn_learn_adj: bool,
) -> Dict[str, Any]:
    Ts_samples = max(1, int(round(Tw_samples * (1.0 - overlap_ratio))))
    return dict(
        encoder_type=encoder_type,
        emb_dim=emb_dim,
        dropout=dropout,
        Tw_samples=Tw_samples,
        overlap_ratio=overlap_ratio,
        Ts_samples=Ts_samples,
        windows_per_trial=windows_per_trial,
        trials_per_axis=trials_per_axis,
        lr_head=lr_head,
        lr_encoder_scale=lr_encoder_scale,
        weight_decay=weight_decay,
        huber_delta=huber_delta,
        lambda_ue=lambda_ue,
        ema_decay=ema_decay,
        window_dropout_p=window_dropout_p,
        trial_dropout_p=trial_dropout_p,
        consistency_weight=consistency_weight,
        consistency_rampup_epochs=consistency_rampup_epochs,
        consistency_loss_type=consistency_loss_type,
        patch_size=patch_size,
        layers=layers,
        heads=heads,
        ffn_ratio=ffn_ratio,
        tcn_blocks=tcn_blocks,
        tcn_kernel=tcn_kernel,
        tcn_width=tcn_width,
        tcn_stem_stride=tcn_stem_stride,
        gcn_layers=gcn_layers,
        gcn_learn_adj=gcn_learn_adj,
    )


def grid_size(encoder_choices: list[str]) -> int:
    total = 0
    for info in search_space_summary(encoder_choices):
        total += int(info["combos"])
    return total


def iter_grid_hparams(encoder_choices: list[str]):
    for et in encoder_choices:
        space = active_hparam_space_for_encoder(et)
        keys = list(space.keys())
        values_list = [space[k] for k in keys]
        for combo in itertools.product(*values_list):
            selected = dict(zip(keys, combo))
            yield build_hparams_for_encoder(et, selected)


def check_mamba_ssm_available() -> tuple[bool, Optional[str]]:
    try:
        from mamba_ssm import Mamba  # noqa: F401
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sample_hparams(rng: np.random.RandomState, encoder_choices: list[str]) -> Dict[str, Any]:
    if len(encoder_choices) == 0:
        raise ValueError("encoder_choices must not be empty")
    et = str(rng.choice(encoder_choices))
    space = active_hparam_space_for_encoder(et)
    selected = {}
    for k, vals in space.items():
        selected[k] = rng.choice(vals)
    return build_hparams_for_encoder(et, selected)


def hparams_key(hps: Dict[str, Any]) -> tuple:
    return tuple(hps[k] for k in HPARAM_ORDER)


def sample_unique_hparams(
    rng: np.random.RandomState,
    encoder_choices: list[str],
    n_trials: int,
) -> list[Dict[str, Any]]:
    if n_trials <= 0:
        return []
    total = grid_size(encoder_choices)
    target = min(int(n_trials), int(total))
    if target < n_trials:
        print(
            f"[WARN] Requested n_trials={n_trials} but effective search space has only {total} "
            f"unique combinations. Use {target}."
        )
    picked = []
    seen = set()
    while len(picked) < target:
        hps = sample_hparams(rng, encoder_choices)
        key = hparams_key(hps)
        if key in seen:
            continue
        seen.add(key)
        picked.append(hps)
    return picked


def mean_numeric(values: list[float], default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def mean_test_metrics(rows: list[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in TEST_METRIC_KEYS:
        vals = [float(r[k]) for r in rows if k in r]
        out[k] = mean_numeric(vals, default=float("nan"))
    return out


def to_encoder_cfg(obj: Any) -> EncoderConfig:
    if isinstance(obj, EncoderConfig):
        return obj
    if isinstance(obj, dict):
        return EncoderConfig(**obj)
    raise TypeError(f"Unsupported encoder_cfg type: {type(obj)}")


def to_mil_cfg(obj: Any) -> MILTrainConfig:
    if isinstance(obj, MILTrainConfig):
        return obj
    if isinstance(obj, dict):
        return MILTrainConfig(**obj)
    raise TypeError(f"Unsupported mil_cfg type: {type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_val_subjects", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--emg_cols", type=str, default="", help="comma-separated indices for EMG columns when reading .lvm (default auto)")
    ap.add_argument("--time_col", type=int, default=0)
    ap.add_argument("--expected_emg_ch", type=int, default=4)

    ap.add_argument("--ssl_epochs", type=int, default=100)
    ap.add_argument("--ssl_windows_per_epoch", type=int, default=20000)
    args = ap.parse_args()
    if args.n_trials == 0 or args.n_trials < -1:
        raise ValueError("n_trials must be > 0 for random search, or -1 for exhaustive grid search.")

    ensure_dir(args.outdir)
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")
    encoder_choices = ["tcn", "rescnn", "itransformer", "transformer", "gcn"]
    mamba_ok, mamba_reason = check_mamba_ssm_available()
    if mamba_ok:
        encoder_choices.append("mamba")
        encoder_choices.append("mgcn")
    else:
        print(
            "[WARN] Disable encoder_type in {'mamba','mgcn'} for this run because "
            f"mamba-ssm is not usable: {mamba_reason}"
        )
    print(f"[INFO] Encoder search space: {encoder_choices}")
    summaries = search_space_summary(encoder_choices)
    total_effective = 0
    for info in summaries:
        et = str(info["encoder_type"])
        combos = int(info["combos"])
        total_effective += combos
        print(f"[INFO] Active params for {et} (combos={combos}):")
        for k, vals in info["space"].items():
            vals_text = ", ".join(str(v) for v in vals)
            print(f"  - {k} ({len(vals)}): {vals_text}")
    print(f"[INFO] Total effective combinations per fold: {total_effective}")
    search_mode = "grid" if args.n_trials == -1 else "random"
    print(f"[INFO] Search mode: {search_mode}")
    if search_mode == "grid":
        total_grid = grid_size(encoder_choices)
        print(f"[INFO] Grid size per fold: {total_grid}")
        if total_grid > 100000:
            print(
                "[WARN] Grid size is very large. This run may take a very long time. "
                "Use N_TRIALS>0 for random search if needed."
            )

    manifest_df = pd.read_csv(args.manifest_csv)
    labels_df = pd.read_csv(args.labels_csv)

    emg_cols = parse_int_list(args.emg_cols)

    subjects = sorted(list({str(s) for s in manifest_df["subject_id"].unique().tolist()}))
    folds = make_loso_folds(subjects)

    summary_path = os.path.join(args.outdir, "summary.jsonl")
    if os.path.exists(summary_path):
        os.remove(summary_path)
    trial_summary_jsonl = os.path.join(args.outdir, "loso_trial_summary.jsonl")
    if os.path.exists(trial_summary_jsonl):
        os.remove(trial_summary_jsonl)
    trial_summary_csv_path = os.path.join(args.outdir, "loso_trial_summary.csv")
    if os.path.exists(trial_summary_csv_path):
        os.remove(trial_summary_csv_path)

    trial_summary_fields = (
        ["trial_id", "search_mode", "n_folds", "mean_fold_val_score"]
        + [f"mean_test_{k}" for k in TEST_METRIC_KEYS]
        + HPARAM_ORDER
    )

    grid_csv_writer = None
    grid_csv_file = None
    if search_mode == "grid":
        grid_csv_path = os.path.join(args.outdir, "grid_traversal_results.csv")
        if os.path.exists(grid_csv_path):
            os.remove(grid_csv_path)
        grid_csv_file = open(grid_csv_path, "w", newline="", encoding="utf-8")
        grid_csv_writer = csv.DictWriter(grid_csv_file, fieldnames=trial_summary_fields)
        grid_csv_writer.writeheader()

    trial_csv_file = open(trial_summary_csv_path, "w", newline="", encoding="utf-8")
    trial_csv_writer = csv.DictWriter(trial_csv_file, fieldnames=trial_summary_fields)
    trial_csv_writer.writeheader()

    fold_infos = []
    fold_records: Dict[int, Dict[str, Any]] = {}
    for fold_id, (test_subject, train_pool) in enumerate(folds):
        fold_dir = os.path.join(args.outdir, f"fold_{fold_id:02d}_test_{test_subject}")
        ensure_dir(fold_dir)
        inner_train, val_subjects = split_train_val_subjects(train_pool, n_val=args.n_val_subjects, seed=args.seed + fold_id)
        fold_infos.append(
            {
                "fold_id": fold_id,
                "test_subject": str(test_subject),
                "train_subjects": [str(s) for s in inner_train],
                "val_subjects": [str(s) for s in val_subjects],
                "fold_dir": fold_dir,
            }
        )
        fold_records[fold_id] = {
            "fold_id": fold_id,
            "test_subject": str(test_subject),
            "val_subjects": [str(s) for s in val_subjects],
            "inner_train_subjects": [str(s) for s in inner_train],
            "search_mode": search_mode,
            "trials": [],
        }

    if search_mode == "grid":
        trial_iter = enumerate(iter_grid_hparams(encoder_choices))
    else:
        rng = np.random.RandomState(args.seed + 1000)
        fixed_trials = sample_unique_hparams(rng, encoder_choices, args.n_trials)
        print(f"[INFO] Fixed unique trial count: {len(fixed_trials)}")
        trial_iter = enumerate(fixed_trials)

    try:
        for t, hps in trial_iter:
            trial_fold_vals: list[float] = []
            trial_fold_tests: list[Dict[str, Any]] = []

            for fold_info in fold_infos:
                fold_id = int(fold_info["fold_id"])
                test_subject = str(fold_info["test_subject"])
                inner_train = list(fold_info["train_subjects"])
                val_subjects = list(fold_info["val_subjects"])
                fold_dir = str(fold_info["fold_dir"])

                trial_dir = os.path.join(fold_dir, f"trial_{t:03d}")
                ensure_dir(trial_dir)

                enc_cfg = EncoderConfig(
                    encoder_type=hps["encoder_type"],
                    emb_dim=hps["emb_dim"],
                    dropout=hps["dropout"],
                    tcn_width=hps["tcn_width"],
                    tcn_blocks=hps["tcn_blocks"],
                    tcn_kernel=hps["tcn_kernel"],
                    tcn_stem_stride=hps["tcn_stem_stride"],
                    patch_size=hps["patch_size"],
                    layers=hps["layers"],
                    heads=hps["heads"],
                    ffn_ratio=hps["ffn_ratio"],
                    gcn_layers=hps["gcn_layers"],
                    gcn_learn_adj=hps["gcn_learn_adj"],
                )

                ssl_cfg = SSLTrainConfig(
                    seed=args.seed + fold_id * 100 + t,
                    Tw_samples=hps["Tw_samples"],
                    windows_per_epoch=args.ssl_windows_per_epoch,
                    epochs=args.ssl_epochs,
                    batch_size=256,
                    lr=3e-4,
                    weight_decay=1e-3,
                    aug_strength="medium",
                    emg_cols=emg_cols,
                    time_col=args.time_col,
                    expected_emg_ch=args.expected_emg_ch,
                )

                mil_cfg = MILTrainConfig(
                    seed=args.seed + fold_id * 100 + t,
                    Tw_samples=hps["Tw_samples"],
                    Ts_samples=hps["Ts_samples"],
                    windows_per_trial=hps["windows_per_trial"],
                    trials_per_axis=hps["trials_per_axis"],
                    batch_sessions=8,
                    epochs=200,
                    patience=40,
                    lr_head=hps["lr_head"],
                    lr_encoder_scale=hps["lr_encoder_scale"],
                    weight_decay=hps["weight_decay"],
                    huber_delta=hps["huber_delta"],
                    lambda_ue=hps["lambda_ue"],
                    ema_decay=hps["ema_decay"],
                    attn_hidden=128,
                    fusion_hidden=256,
                    dropout=hps["dropout"],
                    window_dropout_p=hps["window_dropout_p"],
                    trial_dropout_p=hps["trial_dropout_p"],
                    use_mean_teacher=True,
                    consistency_weight=hps["consistency_weight"],
                    consistency_rampup_epochs=hps["consistency_rampup_epochs"],
                    consistency_loss_type=hps["consistency_loss_type"],
                    sup_aug_strength="weak",
                    emg_cols=emg_cols,
                    time_col=args.time_col,
                    expected_emg_ch=args.expected_emg_ch,
                )

                ssl_out = os.path.join(trial_dir, "ssl")
                ssl_ckpt = pretrain_ssl_moco(
                    manifest_df=manifest_df,
                    train_subjects=inner_train,
                    encoder_cfg=enc_cfg,
                    ssl_cfg=ssl_cfg,
                    out_dir=ssl_out,
                    device=device,
                )

                mil_out = os.path.join(trial_dir, "mil")
                mil_ckpt = train_mil_supervised(
                    manifest_df=manifest_df,
                    labels_df=labels_df,
                    train_subjects=inner_train,
                    val_subjects=val_subjects,
                    encoder_cfg=enc_cfg,
                    mil_cfg=mil_cfg,
                    out_dir=mil_out,
                    device=device,
                    ssl_encoder_ckpt=ssl_ckpt
                )

                ckpt = torch_load_compat(mil_ckpt, map_location="cpu")
                val_metrics = ckpt["val_metrics"]
                fold_val_score = float(val_metrics.get("val_score", float("inf")))
                trial_fold_vals.append(fold_val_score)

                enc_cfg_best = to_encoder_cfg(ckpt["encoder_cfg"])
                mil_cfg_best = to_mil_cfg(ckpt["mil_cfg"])
                test_ds = SessionBagDataset(
                    manifest_df, labels_df, subjects=[test_subject],
                    Tw_samples=mil_cfg_best.Tw_samples, Ts_samples=mil_cfg_best.Ts_samples,
                    windows_per_trial=mil_cfg_best.windows_per_trial,
                    trials_per_axis=mil_cfg_best.trials_per_axis,
                    mode="test",
                    supervised_aug=None,
                    return_two_views=False,
                    emg_cols=mil_cfg_best.emg_cols,
                    time_col=mil_cfg_best.time_col,
                    expected_emg_ch=mil_cfg_best.expected_emg_ch,
                    cache_size=64,
                )
                test_dl = DataLoader(
                    test_ds,
                    batch_size=mil_cfg_best.batch_sessions,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                )

                teacher = HierarchicalMILRegressor(
                    encoder=build_encoder(enc_cfg_best),
                    emb_dim=enc_cfg_best.emb_dim,
                    attn_hidden=mil_cfg_best.attn_hidden,
                    fusion_hidden=mil_cfg_best.fusion_hidden,
                    dropout=mil_cfg_best.dropout,
                    window_dropout_p=0.0,
                    trial_dropout_p=0.0,
                ).to(device)
                teacher.load_state_dict(ckpt["teacher"], strict=True)
                test_metrics = evaluate_mil(teacher, test_dl, device)
                trial_fold_tests.append(test_metrics)

                split_info = dict(ckpt.get("split_info", {}))
                split_info.update(
                    {
                        "trial_id": int(t),
                        "fold_id": fold_id,
                        "test_subject": str(test_subject),
                        "val_subjects": [str(s) for s in val_subjects],
                        "train_subjects": [str(s) for s in inner_train],
                    }
                )
                ckpt["split_info"] = split_info
                ckpt["test_metrics"] = test_metrics
                torch.save(ckpt, mil_ckpt)

                fold_records[fold_id]["trials"].append(
                    {
                        "trial_id": int(t),
                        "hparams": hps,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "mil_ckpt": mil_ckpt,
                        "ssl_ckpt": ssl_ckpt,
                    }
                )

            if len(trial_fold_tests) == 0:
                raise RuntimeError(f"No fold result produced for trial {t}.")

            mean_tests = mean_test_metrics(trial_fold_tests)
            trial_row: Dict[str, Any] = {
                "trial_id": int(t),
                "search_mode": search_mode,
                "n_folds": len(trial_fold_tests),
                "mean_fold_val_score": mean_numeric(trial_fold_vals, default=float("inf")),
            }
            for k in TEST_METRIC_KEYS:
                trial_row[f"mean_test_{k}"] = float(mean_tests.get(k, float("nan")))
            for k in HPARAM_ORDER:
                trial_row[k] = hps.get(k)

            trial_csv_writer.writerow(trial_row)
            trial_csv_file.flush()
            append_jsonl(trial_summary_jsonl, trial_row)
            if grid_csv_writer is not None:
                grid_csv_writer.writerow(trial_row)
                grid_csv_file.flush()

        for fold_info in fold_infos:
            fold_id = int(fold_info["fold_id"])
            fold_dir = str(fold_info["fold_dir"])
            fold_record = fold_records[fold_id]
            trials = fold_record.get("trials", [])
            if len(trials) == 0:
                raise RuntimeError(
                    f"No valid trial checkpoint found in fold {fold_id} "
                    f"(test_subject={fold_info['test_subject']}). Check n_trials and training logs."
                )
            best_trial = min(
                trials,
                key=lambda tr: float(tr.get("val_metrics", {}).get("val_score", float("inf")))
            )
            best_ckpt = best_trial["mil_ckpt"]
            fold_record["best"] = {
                "best_val": float(best_trial.get("val_metrics", {}).get("val_score", float("inf"))),
                "best_trial_id": int(best_trial["trial_id"]),
                "best_trial_dir": os.path.dirname(os.path.dirname(best_ckpt)),
                "best_hparams": best_trial["hparams"],
                "best_ckpt": best_ckpt,
                "test_metrics": best_trial["test_metrics"],
            }
            save_json(os.path.join(fold_dir, "fold_record.json"), fold_record)
            append_jsonl(summary_path, {
                "fold_id": fold_id,
                "test_subject": fold_record["test_subject"],
                "best_val": fold_record["best"]["best_val"],
                "test_metrics": fold_record["best"]["test_metrics"],
                "best_hparams": fold_record["best"]["best_hparams"],
                "best_trial_id": fold_record["best"]["best_trial_id"],
            })
    finally:
        trial_csv_file.close()
        if grid_csv_file is not None:
            grid_csv_file.close()


if __name__ == "__main__":
    main()
