from __future__ import annotations

import argparse
import json
import os
import traceback

import pandas as pd
import torch

from .baseModels import EncoderConfig
from .train_ssl import pretrain_ssl_moco, SSLTrainConfig
from .train_mil import train_mil_supervised, MILTrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    labels = pd.read_csv(args.labels_csv)

    subjects = sorted(manifest["subject_id"].astype(str).unique().tolist())
    if len(subjects) < 4:
        raise ValueError("Need at least 4 subjects for smoke test (3 train + 1 val).")
    train_subjects = subjects[:3]
    val_subjects = subjects[3:4]

    models = ["tcn", "rescnn", "transformer", "itransformer", "gcn", "mgcn", "mamba"]
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    results = []

    for model_name in models:
        rec = {"model": model_name, "ok": False, "stage": None, "error": None}
        try:
            enc = EncoderConfig(
                encoder_type=model_name,
                emb_dim=64,
                dropout=0.1,
                tcn_width=32,
                tcn_blocks=3,
                tcn_kernel=5,
                tcn_stem_stride=1,
                patch_size=8,
                layers=2,
                heads=2,
                ffn_ratio=2,
                gcn_layers=1,
                gcn_learn_adj=True,
            )

            ssl_cfg = SSLTrainConfig(
                seed=42,
                Tw_samples=256,
                windows_per_epoch=32,
                epochs=1,
                batch_size=8,
                num_workers=0,
                lr=3e-4,
                weight_decay=1e-3,
                aug_strength="light",
                use_amp=False,
                expected_emg_ch=4,
            )

            mil_cfg = MILTrainConfig(
                seed=42,
                Tw_samples=256,
                Ts_samples=128,
                windows_per_trial=8,
                trials_per_axis=3,
                batch_sessions=2,
                num_workers=0,
                epochs=1,
                patience=1,
                lr_head=1e-4,
                lr_encoder_scale=0.3,
                weight_decay=1e-3,
                grad_clip=1.0,
                use_amp=False,
                use_cosine=False,
                attn_hidden=64,
                fusion_hidden=128,
                dropout=0.1,
                window_dropout_p=0.1,
                trial_dropout_p=0.05,
                huber_delta=0.04,
                lambda_ue=0.1,
                use_ue=True,
                ema_decay=0.99,
                use_mean_teacher=True,
                consistency_weight=0.1,
                consistency_loss_type="mse",
                consistency_huber_delta=0.02,
                consistency_rampup_epochs=1,
                sup_aug_strength="weak",
                expected_emg_ch=4,
            )

            out_dir = os.path.join(args.outdir, model_name)
            os.makedirs(out_dir, exist_ok=True)

            rec["stage"] = "ssl"
            ssl_ckpt = pretrain_ssl_moco(
                manifest_df=manifest,
                train_subjects=train_subjects,
                encoder_cfg=enc,
                ssl_cfg=ssl_cfg,
                out_dir=os.path.join(out_dir, "ssl"),
                device=device,
            )

            rec["stage"] = "mil"
            mil_ckpt = train_mil_supervised(
                manifest_df=manifest,
                labels_df=labels,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                encoder_cfg=enc,
                mil_cfg=mil_cfg,
                out_dir=os.path.join(out_dir, "mil"),
                device=device,
                ssl_encoder_ckpt=ssl_ckpt,
            )

            rec["ok"] = True
            rec["ssl_ckpt"] = ssl_ckpt
            rec["mil_ckpt"] = mil_ckpt
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=2)

        results.append(rec)
        print(
            json.dumps(
                {
                    "model": rec["model"],
                    "ok": rec["ok"],
                    "stage": rec["stage"],
                    "error": rec["error"],
                },
                ensure_ascii=False,
            )
        )

    out_json = os.path.join(args.outdir, "model_test_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("summary_file", out_json)


if __name__ == "__main__":
    main()
