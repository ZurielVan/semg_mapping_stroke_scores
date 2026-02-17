from __future__ import annotations

import os
import math
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .dataset import SessionBagDataset
from .augmentations import build_sup_augmentations
from .baseModels import build_encoder, EncoderConfig, HierarchicalMILRegressor
from .loss import MainFmaLoss, ConsistencyLoss
from .ema import clone_as_ema_teacher, update_ema_model
from .metrics import mae, rmse, normalized_mae, within_tolerance, r2_score, correlation
from .utils import set_seed, seed_worker, ensure_dir, to_device, torch_load_compat


@dataclass
class MILTrainConfig:
    seed: int = 0
    Tw_samples: int = 512
    Ts_samples: int = 256
    windows_per_trial: int = 16
    trials_per_axis: int = 3
    batch_sessions: int = 8
    num_workers: int = 4

    epochs: int = 200
    patience: int = 40
    lr_head: float = 1e-4
    lr_encoder_scale: float = 0.3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    use_amp: bool = True
    use_cosine: bool = True

    # MIL module sizes
    attn_hidden: int = 128
    fusion_hidden: int = 256
    dropout: float = 0.1

    # instance dropout (model-level mask dropout)
    window_dropout_p: float = 0.2
    trial_dropout_p: float = 0.1

    # supervised loss
    huber_delta: float = 0.04
    lambda_ue: float = 0.1
    use_ue: bool = True

    # EMA (teacher weights)
    ema_decay: float = 0.999

    # Mean-Teacher consistency regularization
    use_mean_teacher: bool = True
    consistency_weight: float = 0.2
    consistency_loss_type: str = "mse"   # mse | smoothl1
    consistency_huber_delta: float = 0.02
    consistency_rampup_epochs: int = 20
    sup_aug_strength: str = "weak"       # none | weak | medium

    # raw loader config
    emg_cols: Optional[list[int]] = None
    time_col: Optional[int] = 0
    expected_emg_ch: int = 4


def _sigmoid_rampup(current: int, rampup_length: int) -> float:
    if rampup_length <= 0:
        return 1.0
    current = max(0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def _safe_mean(total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(total / count)


def _save_loss_plot(history_df: pd.DataFrame, out_png: str) -> None:
    """
    Plot MIL losses and validation metrics in one figure.
    Rules:
      - Same target group keeps same color family (WH/SE/UE).
      - Different series in the same group use different line styles.
      - Validation metrics are plotted with high transparency.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Skip loss plot: matplotlib unavailable ({e})")
        return

    if len(history_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax_val = ax.twinx()
    x = history_df["epoch"].to_numpy()

    # Loss series (left axis)
    if "train_loss_total" in history_df.columns:
        ax.plot(
            x,
            history_df["train_loss_total"].to_numpy(),
            label="loss_total",
            linewidth=2.0,
            color="black",
            linestyle="-",
            alpha=0.9,
        )
    if "train_loss_sup" in history_df.columns:
        ax.plot(
            x,
            history_df["train_loss_sup"].to_numpy(),
            label="loss_sup",
            linewidth=1.8,
            color="black",
            linestyle="--",
            alpha=0.9,
        )
    if "train_loss_cons" in history_df.columns:
        ax.plot(
            x,
            history_df["train_loss_cons"].to_numpy(),
            label="loss_cons",
            linewidth=1.8,
            color="black",
            linestyle=":",
            alpha=0.9,
        )
    if "train_loss_wh" in history_df.columns:
        ax.plot(
            x,
            history_df["train_loss_wh"].to_numpy(),
            label="loss_WH",
            linewidth=1.8,
            color="tab:blue",
            linestyle="-",
            alpha=0.9,
        )
    if "train_loss_se" in history_df.columns:
        ax.plot(
            x,
            history_df["train_loss_se"].to_numpy(),
            label="loss_SE",
            linewidth=1.8,
            color="tab:orange",
            linestyle="-",
            alpha=0.9,
        )

    # Validation metrics (right axis) with high transparency.
    group_color = {
        "WH": "tab:blue",
        "SE": "tab:orange",
        "UE": "tab:green",
        "SCORE": "tab:gray",
    }
    series_style = {
        "MAE": "--",
        "RMSE": ":",
        "R2": (0, (3, 1, 1, 1)),
        "Corr": "-.",
        "within1": (0, (1, 1)),
        "within2": (0, (5, 2)),
        "score": "-",
    }

    def _plot_val_metric(col: str, metric_name: str) -> None:
        parts = metric_name.split("_")
        if len(parts) >= 2 and parts[-1] in {"WH", "SE", "UE"}:
            group = parts[-1]
            series = "_".join(parts[:-1])
        else:
            group = "SCORE"
            series = metric_name
        color = group_color.get(group, "tab:gray")
        linestyle = series_style.get(series, "--")
        ax_val.plot(
            x,
            history_df[col].to_numpy(),
            label=f"val_{metric_name}",
            linewidth=1.4,
            color=color,
            linestyle=linestyle,
            alpha=0.35,
        )

    for col in history_df.columns:
        if col.startswith("val_"):
            _plot_val_metric(col, col[4:])
    if "val_score" in history_df.columns:
        _plot_val_metric("val_score", "score")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax_val.set_ylabel("Validation Metrics")
    ax.set_title("MIL Loss + Validation Metrics")
    ax.grid(True, alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_val.get_legend_handles_labels()
    if len(h1) + len(h2) > 0:
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_mil(model, dl, device) -> Dict[str, float]:
    model.eval()
    ys_wh, ys_se, yh_wh, yh_se = [], [], [], []

    for batch in dl:
        batch = to_device(batch, device)
        windows = batch["windows"]
        window_mask = batch["window_mask"]
        trial_mask = batch["trial_mask"]
        y_wh = batch["y_wh"]
        y_se = batch["y_se"]
        label_mask = batch["label_mask"]

        yhat_wh, yhat_se = model(windows, window_mask, trial_mask)

        m_wh = label_mask[:, 0]
        m_se = label_mask[:, 1]
        if m_wh.any():
            ys_wh.append(y_wh[m_wh].detach().cpu())
            yh_wh.append(yhat_wh[m_wh].detach().cpu())
        if m_se.any():
            ys_se.append(y_se[m_se].detach().cpu())
            yh_se.append(yhat_se[m_se].detach().cpu())

    if len(ys_wh) == 0 or len(ys_se) == 0:
        return {"val_score": float("inf")}

    ys_wh = torch.cat(ys_wh)
    yh_wh = torch.cat(yh_wh)
    ys_se = torch.cat(ys_se)
    yh_se = torch.cat(yh_se)

    mae_wh = mae(ys_wh, yh_wh)
    mae_se = mae(ys_se, yh_se)
    mae_ue = mae(ys_wh + ys_se, yh_wh + yh_se)
    r2_wh = r2_score(ys_wh, yh_wh)
    r2_se = r2_score(ys_se, yh_se)
    r2_ue = r2_score(ys_wh + ys_se, yh_wh + yh_se)
    corr_wh = correlation(ys_wh, yh_wh)
    corr_se = correlation(ys_se, yh_se)
    corr_ue = correlation(ys_wh + ys_se, yh_wh + yh_se)

    nmae_wh = normalized_mae(ys_wh, yh_wh, 24.0)
    nmae_se = normalized_mae(ys_se, yh_se, 42.0)
    val_score = 0.5 * (nmae_wh + nmae_se)

    return {
        "MAE_WH": mae_wh,
        "MAE_SE": mae_se,
        "MAE_UE": mae_ue,
        "R2_WH": r2_wh,
        "R2_SE": r2_se,
        "R2_UE": r2_ue,
        "Corr_WH": corr_wh,
        "Corr_SE": corr_se,
        "Corr_UE": corr_ue,
        "RMSE_WH": rmse(ys_wh, yh_wh),
        "RMSE_SE": rmse(ys_se, yh_se),
        "within1_WH": within_tolerance(ys_wh, yh_wh, tol=1.0),
        "within2_WH": within_tolerance(ys_wh, yh_wh, tol=2.0),
        "within1_SE": within_tolerance(ys_se, yh_se, tol=1.0),
        "within2_SE": within_tolerance(ys_se, yh_se, tol=2.0),
        "val_score": val_score,
    }


def train_mil_supervised(
    manifest_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_subjects: list[str],
    val_subjects: list[str],
    encoder_cfg: EncoderConfig,
    mil_cfg: MILTrainConfig,
    out_dir: str,
    device: torch.device,
    ssl_encoder_ckpt: Optional[str] = None,
) -> str:
    """
    Supervised MIL regression with:
      - bounded outputs
      - normalized Huber loss
      - EMA teacher model for (a) Mean-Teacher consistency, (b) stable validation/testing
      - optional Mean-Teacher consistency regularization

    Returns:
      path to best checkpoint (contains student + teacher weights)
    """
    ensure_dir(out_dir)
    set_seed(mil_cfg.seed)

    sup_aug = build_sup_augmentations(mil_cfg.sup_aug_strength)
    return_two_views = mil_cfg.use_mean_teacher and mil_cfg.consistency_weight > 0.0

    ds_train = SessionBagDataset(
        manifest_df, labels_df, subjects=train_subjects,
        Tw_samples=mil_cfg.Tw_samples, Ts_samples=mil_cfg.Ts_samples,
        windows_per_trial=mil_cfg.windows_per_trial,
        trials_per_axis=mil_cfg.trials_per_axis,
        mode="train",
        supervised_aug=sup_aug,
        return_two_views=return_two_views,
        emg_cols=mil_cfg.emg_cols,
        time_col=mil_cfg.time_col,
        expected_emg_ch=mil_cfg.expected_emg_ch,
        cache_size=64,
    )
    ds_val = SessionBagDataset(
        manifest_df, labels_df, subjects=val_subjects,
        Tw_samples=mil_cfg.Tw_samples, Ts_samples=mil_cfg.Ts_samples,
        windows_per_trial=mil_cfg.windows_per_trial,
        trials_per_axis=mil_cfg.trials_per_axis,
        mode="val",
        supervised_aug=None,
        return_two_views=False,
        emg_cols=mil_cfg.emg_cols,
        time_col=mil_cfg.time_col,
        expected_emg_ch=mil_cfg.expected_emg_ch,
        cache_size=64,
    )

    dl_train = DataLoader(
        ds_train, batch_size=mil_cfg.batch_sessions, shuffle=True,
        num_workers=mil_cfg.num_workers, pin_memory=True,
        # Avoid 0-step epochs when train sessions < batch size.
        drop_last=(len(ds_train) >= mil_cfg.batch_sessions),
        worker_init_fn=seed_worker
    )
    dl_val = DataLoader(
        ds_val, batch_size=mil_cfg.batch_sessions, shuffle=False,
        num_workers=mil_cfg.num_workers, pin_memory=True,
        worker_init_fn=seed_worker
    )

    enc_cfg = encoder_cfg
    if encoder_cfg.encoder_type.lower() == "itransformer":
        enc_cfg = replace(encoder_cfg, itransformer_seq_len=mil_cfg.Tw_samples)
    elif encoder_cfg.encoder_type.lower() == "mgcn":
        enc_cfg = replace(encoder_cfg, mgcn_seq_len=mil_cfg.Tw_samples)

    encoder = build_encoder(enc_cfg).to(device)
    if ssl_encoder_ckpt is not None and os.path.exists(ssl_encoder_ckpt):
        ckpt = torch_load_compat(ssl_encoder_ckpt, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"], strict=True)

    student = HierarchicalMILRegressor(
        encoder=encoder,
        emb_dim=enc_cfg.emb_dim,
        attn_hidden=mil_cfg.attn_hidden,
        fusion_hidden=mil_cfg.fusion_hidden,
        dropout=mil_cfg.dropout,
        window_dropout_p=mil_cfg.window_dropout_p,
        trial_dropout_p=mil_cfg.trial_dropout_p,
    ).to(device)

    teacher = clone_as_ema_teacher(student).to(device)

    sup_loss_fn = MainFmaLoss(
        delta=mil_cfg.huber_delta,
        lambda_ue=mil_cfg.lambda_ue,
        use_ue=mil_cfg.use_ue,
    )
    cons_loss_fn = ConsistencyLoss(
        loss_type=mil_cfg.consistency_loss_type,
        huber_delta=mil_cfg.consistency_huber_delta,
    )

    encoder_lr = mil_cfg.lr_head * mil_cfg.lr_encoder_scale
    opt = torch.optim.AdamW(
        [
            {"params": student.encoder.parameters(), "lr": encoder_lr},
            {"params": [p for n, p in student.named_parameters() if not n.startswith("encoder.")], "lr": mil_cfg.lr_head},
        ],
        weight_decay=mil_cfg.weight_decay
    )

    total_steps = mil_cfg.epochs * len(dl_train)
    scheduler = None
    if mil_cfg.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    scaler = GradScaler(enabled=mil_cfg.use_amp and device.type == "cuda")

    # Initialize teacher = student at step 0
    update_ema_model(teacher, student, decay=0.0)

    best_score = float("inf")
    best_path = os.path.join(out_dir, "best_mil.pth")
    loss_csv_path = os.path.join(out_dir, "mil_loss_history.csv")
    loss_png_path = os.path.join(out_dir, "mil_loss_curves.png")
    bad_epochs = 0
    global_step = 0
    loss_history_rows = []

    for epoch in range(mil_cfg.epochs):
        student.train()
        pbar = tqdm(dl_train, desc=f"[MIL] epoch {epoch+1}/{mil_cfg.epochs}", leave=False)
        ramp = _sigmoid_rampup(epoch, mil_cfg.consistency_rampup_epochs)

        ep_total_sum = 0.0
        ep_sup_sum = 0.0
        ep_cons_sum = 0.0
        ep_wcons_sum = 0.0
        ep_wh_sum = 0.0
        ep_se_sum = 0.0
        ep_steps = 0
        ep_wh_steps = 0
        ep_se_steps = 0

        for batch in pbar:
            global_step += 1
            batch = to_device(batch, device)

            if return_two_views:
                w1 = batch["windows_1"]
                wm1 = batch["window_mask_1"]
                tm1 = batch["trial_mask_1"]
                w2 = batch["windows_2"]
                wm2 = batch["window_mask_2"]
                tm2 = batch["trial_mask_2"]
            else:
                w1 = batch["windows"]
                wm1 = batch["window_mask"]
                tm1 = batch["trial_mask"]
                w2 = wm2 = tm2 = None

            y_wh = batch["y_wh"]
            y_se = batch["y_se"]
            label_mask = batch["label_mask"]  # (B,2)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                # student forward
                yhat_wh_s, yhat_se_s = student(w1, wm1, tm1)
                L_sup = sup_loss_fn(y_wh, y_se, yhat_wh_s, yhat_se_s, label_mask)

                wh_mask = label_mask[:, 0]
                se_mask = label_mask[:, 1]
                L_wh = None
                L_se = None
                if wh_mask.any():
                    L_wh = sup_loss_fn.huber(y_wh[wh_mask], yhat_wh_s[wh_mask], scale=24.0)
                if se_mask.any():
                    L_se = sup_loss_fn.huber(y_se[se_mask], yhat_se_s[se_mask], scale=42.0)

                L_cons = torch.tensor(0.0, device=device)
                if return_two_views and mil_cfg.consistency_weight > 0.0:
                    with torch.no_grad():
                        teacher.eval()
                        yhat_wh_t, yhat_se_t = teacher(w2, wm2, tm2)

                    p_s = torch.stack([yhat_wh_s / 24.0, yhat_se_s / 42.0], dim=1)
                    p_t = torch.stack([yhat_wh_t / 24.0, yhat_se_t / 42.0], dim=1)
                    L_cons = cons_loss_fn(p_s, p_t)

                w_cons = mil_cfg.consistency_weight * ramp if (return_two_views and mil_cfg.use_mean_teacher) else 0.0
                L = L_sup + w_cons * L_cons

            ep_steps += 1
            ep_total_sum += float(L.item())
            ep_sup_sum += float(L_sup.item())
            ep_cons_sum += float(L_cons.item())
            ep_wcons_sum += float(w_cons)
            if L_wh is not None:
                ep_wh_steps += 1
                ep_wh_sum += float(L_wh.item())
            if L_se is not None:
                ep_se_steps += 1
                ep_se_sum += float(L_se.item())

            scaler.scale(L).backward()
            if mil_cfg.grad_clip is not None and mil_cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), mil_cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            # EMA teacher update after optimizer step
            update_ema_model(teacher, student, decay=mil_cfg.ema_decay)

            pbar.set_postfix(
                L=float(L.item()),
                Lsup=float(L_sup.item()),
                Lcons=float(L_cons.item()),
                wcons=float(w_cons),
                lr=float(opt.param_groups[0]["lr"]),
            )

        # Validation using teacher weights (EMA) for stability
        val_metrics = evaluate_mil(teacher, dl_val, device)
        val_score = float(val_metrics["val_score"])

        epoch_row = {
            "epoch": int(epoch + 1),
            "train_loss_total": _safe_mean(ep_total_sum, ep_steps),
            "train_loss_sup": _safe_mean(ep_sup_sum, ep_steps),
            "train_loss_cons": _safe_mean(ep_cons_sum, ep_steps),
            "train_cons_weight": _safe_mean(ep_wcons_sum, ep_steps),
            "train_loss_wh": _safe_mean(ep_wh_sum, ep_wh_steps),
            "train_loss_se": _safe_mean(ep_se_sum, ep_se_steps),
        }
        for k, v in val_metrics.items():
            if k == "val_score":
                epoch_row["val_score"] = float(v)
            else:
                epoch_row[f"val_{k}"] = float(v)
        loss_history_rows.append(epoch_row)
        loss_df = pd.DataFrame(loss_history_rows)
        loss_df.to_csv(loss_csv_path, index=False)
        _save_loss_plot(loss_df, loss_png_path)

        if val_score < best_score:
            best_score = val_score
            bad_epochs = 0
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                # Store plain dicts for cross-version safety.
                "encoder_cfg": asdict(enc_cfg),
                "mil_cfg": asdict(mil_cfg),
                "split_info": {
                    "train_subjects": [str(s) for s in train_subjects],
                    "val_subjects": [str(s) for s in val_subjects],
                },
                "val_metrics": val_metrics,
                "loss_history_csv": loss_csv_path,
                "loss_history_plot": loss_png_path,
                "epoch": epoch,
            }, best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= mil_cfg.patience:
            break

    return best_path
