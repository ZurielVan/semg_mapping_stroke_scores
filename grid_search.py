from __future__ import annotations

import argparse
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


def sample_hparams(rng: np.random.RandomState) -> Dict[str, Any]:
    encoder_type = rng.choice(["tcn", "rescnn", "itransformer", "transformer", "gcn", "mgcn"])
    emb_dim = int(rng.choice([64, 128, 256]))
    dropout = float(rng.choice([0.05, 0.1, 0.2, 0.3]))

    Tw_samples = int(rng.choice([256, 384, 512, 640]))
    # Search overlap ratio in {0%, 25%, 50%}
    # overlap = 1 - Ts/Tw  => Ts = Tw * (1 - overlap)
    overlap_ratio = float(rng.choice([0.0, 0.25, 0.5]))
    Ts_samples = max(1, int(round(Tw_samples * (1.0 - overlap_ratio))))
    windows_per_trial = int(rng.choice([8, 16, 32]))
    trials_per_axis = int(rng.choice([3, 4, 6]))

    lr_head = float(rng.choice([3e-4, 1e-4, 3e-5]))
    lr_encoder_scale = float(rng.choice([0.1, 0.3, 1.0]))
    weight_decay = float(rng.choice([1e-4, 1e-3, 1e-2]))

    huber_delta = float(rng.choice([0.02, 0.04, 0.08]))
    lambda_ue = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    ema_decay = float(rng.choice([0.99, 0.995, 0.999]))

    window_dropout_p = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
    trial_dropout_p = float(rng.choice([0.0, 0.05, 0.1, 0.15]))

    consistency_weight = float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.4]))
    consistency_rampup_epochs = int(rng.choice([0, 10, 20, 40]))
    consistency_loss_type = rng.choice(["mse", "smoothl1"])

    patch_size = int(rng.choice([4, 8, 16]))
    layers = int(rng.choice([2, 4, 6]))
    heads = int(rng.choice([2, 4, 8]))
    ffn_ratio = int(rng.choice([2, 4]))

    tcn_blocks = int(rng.choice([3, 5, 7]))
    tcn_kernel = int(rng.choice([3, 5, 7]))
    tcn_width = int(rng.choice([32, 64, 96]))
    tcn_stem_stride = int(rng.choice([1, 2]))

    gcn_layers = int(rng.choice([1, 2, 3]))
    gcn_learn_adj = bool(rng.choice([0, 1]))

    return dict(
        encoder_type=encoder_type,
        emb_dim=emb_dim,
        dropout=dropout,
        Tw_samples=Tw_samples,
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

    ensure_dir(args.outdir)
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")

    manifest_df = pd.read_csv(args.manifest_csv)
    labels_df = pd.read_csv(args.labels_csv)

    emg_cols = parse_int_list(args.emg_cols)

    subjects = sorted(list({str(s) for s in manifest_df["subject_id"].unique().tolist()}))
    folds = make_loso_folds(subjects)

    summary_path = os.path.join(args.outdir, "summary.jsonl")
    if os.path.exists(summary_path):
        os.remove(summary_path)

    for fold_id, (test_subject, train_pool) in enumerate(folds):
        fold_dir = os.path.join(args.outdir, f"fold_{fold_id:02d}_test_{test_subject}")
        ensure_dir(fold_dir)

        inner_train, val_subjects = split_train_val_subjects(train_pool, n_val=args.n_val_subjects, seed=args.seed + fold_id)

        fold_record = {
            "fold_id": fold_id,
            "test_subject": test_subject,
            "val_subjects": val_subjects,
            "inner_train_subjects": inner_train,
            "trials": []
        }

        rng = np.random.RandomState(args.seed + 1000 * fold_id)

        best_val = float("inf")
        best_ckpt = None
        best_hparams = None
        best_trial_dir = None

        for t in range(args.n_trials):
            hps = sample_hparams(rng)
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
            val_score = float(val_metrics["val_score"])

            trial_rec = {
                "trial_id": t,
                "hparams": hps,
                "val_metrics": val_metrics,
                "mil_ckpt": mil_ckpt,
                "ssl_ckpt": ssl_ckpt,
            }
            fold_record["trials"].append(trial_rec)

            # Fallback: keep the first successful trial checkpoint to avoid
            # best_ckpt staying None when all val_score are inf.
            if best_ckpt is None:
                best_ckpt = mil_ckpt
                best_hparams = hps
                best_trial_dir = trial_dir

            if val_score < best_val:
                best_val = val_score
                best_ckpt = mil_ckpt
                best_hparams = hps
                best_trial_dir = trial_dir

        if best_ckpt is None:
            raise RuntimeError(
                f"No valid trial checkpoint found in fold {fold_id} "
                f"(test_subject={test_subject}). Check n_trials and training logs."
            )

        # Evaluate best teacher model on test subject
        ckpt = torch_load_compat(best_ckpt, map_location="cpu")
        enc_cfg = to_encoder_cfg(ckpt["encoder_cfg"])
        mil_cfg = to_mil_cfg(ckpt["mil_cfg"])

        test_ds = SessionBagDataset(
            manifest_df, labels_df, subjects=[test_subject],
            Tw_samples=mil_cfg.Tw_samples, Ts_samples=mil_cfg.Ts_samples,
            windows_per_trial=mil_cfg.windows_per_trial,
            trials_per_axis=mil_cfg.trials_per_axis,
            mode="test",
            supervised_aug=None,
            return_two_views=False,
            emg_cols=mil_cfg.emg_cols,
            time_col=mil_cfg.time_col,
            expected_emg_ch=mil_cfg.expected_emg_ch,
            cache_size=64,
        )
        test_dl = DataLoader(test_ds, batch_size=mil_cfg.batch_sessions, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)

        teacher = HierarchicalMILRegressor(
            encoder=build_encoder(enc_cfg),
            emb_dim=enc_cfg.emb_dim,
            attn_hidden=mil_cfg.attn_hidden,
            fusion_hidden=mil_cfg.fusion_hidden,
            dropout=mil_cfg.dropout,
            window_dropout_p=0.0,
            trial_dropout_p=0.0,
        ).to(device)
        teacher.load_state_dict(ckpt["teacher"], strict=True)

        test_metrics = evaluate_mil(teacher, test_dl, device)

        # Enrich best checkpoint with full split metadata for single-file traceability.
        split_info = dict(ckpt.get("split_info", {}))
        split_info.update(
            {
                "fold_id": fold_id,
                "test_subject": str(test_subject),
                "val_subjects": [str(s) for s in val_subjects],
                "train_subjects": [str(s) for s in inner_train],
            }
        )
        ckpt["split_info"] = split_info
        ckpt["test_metrics"] = test_metrics
        torch.save(ckpt, best_ckpt)

        fold_record["best"] = {
            "best_val": best_val,
            "best_trial_dir": best_trial_dir,
            "best_hparams": best_hparams,
            "best_ckpt": best_ckpt,
            "test_metrics": test_metrics,
        }

        save_json(os.path.join(fold_dir, "fold_record.json"), fold_record)
        append_jsonl(summary_path, {
            "fold_id": fold_id,
            "test_subject": test_subject,
            "best_val": best_val,
            "test_metrics": test_metrics,
            "best_hparams": best_hparams,
        })


if __name__ == "__main__":
    main()
