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
    # window aggregator:
    #   set_attn: current order-invariant MIL pooling (default)
    #   temporal_transformer: order-aware over sampled window tokens
    window_agg_mode: str = "set_attn"
    window_temporal_layers: int = 2
    window_temporal_heads: int = 4
    window_temporal_ffn_ratio: int = 2
    window_temporal_dropout: float = 0.1

    # supervised loss
    huber_delta: float = 0.04
    lambda_ue: float = 0.1
    use_ue: bool = True
    # target mode:
    #   whse -> default current behavior (joint WH+SE, optional UE consistency)
    #   wh   -> only optimize/evaluate WH
    #   se   -> only optimize/evaluate SE
    target_mode: str = "whse"

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


SUPPORTED_TARGET_MODES = {"whse", "wh", "se"}
SUPPORTED_WINDOW_AGG_MODES = {"set_attn", "temporal_transformer"}


def _normalize_target_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    alias = {
        "default": "whse",
        "wh+se": "whse",
        "wh_se": "whse",
        "both": "whse",
    }
    m = alias.get(m, m)
    if m not in SUPPORTED_TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={mode}. Supported: {sorted(SUPPORTED_TARGET_MODES)}")
    return m


def _use_wh(mode: str) -> bool:
    return mode in {"whse", "wh"}


def _use_se(mode: str) -> bool:
    return mode in {"whse", "se"}


def _apply_target_label_mask(label_mask: torch.Tensor, target_mode: str) -> torch.Tensor:
    out = label_mask.clone()
    if not _use_wh(target_mode):
        out[:, 0] = False
    if not _use_se(target_mode):
        out[:, 1] = False
    return out


def _consistency_tensor(yhat_wh: torch.Tensor, yhat_se: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "wh":
        return (yhat_wh / 24.0).unsqueeze(1)
    if target_mode == "se":
        return (yhat_se / 42.0).unsqueeze(1)
    return torch.stack([yhat_wh / 24.0, yhat_se / 42.0], dim=1)


def _normalize_window_agg_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    alias = {
        "default": "set_attn",
        "set": "set_attn",
        "attn": "set_attn",
        "temporal": "temporal_transformer",
        "temporal_attn": "temporal_transformer",
    }
    m = alias.get(m, m)
    if m not in SUPPORTED_WINDOW_AGG_MODES:
        raise ValueError(
            f"Unsupported window_agg_mode={mode}. "
            f"Supported: {sorted(SUPPORTED_WINDOW_AGG_MODES)}"
        )
    return m


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


def _all_finite_tensors(*vals: Optional[torch.Tensor]) -> bool:
    for v in vals:
        if v is None:
            continue
        if torch.is_tensor(v) and not bool(torch.isfinite(v).all()):
            return False
    return True


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
        if col.startswith("val_") and col != "val_score":
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


class _RunningRegressionStats:
    """
    Streaming regression statistics (O(1) memory).
    """
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_yhat = 0.0
        self.sum_yhat2 = 0.0
        self.sum_yyhat = 0.0
        self.cnt_within1 = 0
        self.cnt_within2 = 0

    def update(self, y: torch.Tensor, yhat: torch.Tensor, tol1: Optional[float] = None, tol2: Optional[float] = None) -> None:
        if y.numel() == 0:
            return
        y = y.detach().double().view(-1)
        yhat = yhat.detach().double().view(-1)
        err = y - yhat

        n = int(y.numel())
        self.n += n
        self.sum_abs += float(torch.abs(err).sum().item())
        self.sum_sq += float(torch.square(err).sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_y2 += float(torch.square(y).sum().item())
        self.sum_yhat += float(yhat.sum().item())
        self.sum_yhat2 += float(torch.square(yhat).sum().item())
        self.sum_yyhat += float((y * yhat).sum().item())
        if tol1 is not None:
            self.cnt_within1 += int((torch.abs(err) <= float(tol1)).sum().item())
        if tol2 is not None:
            self.cnt_within2 += int((torch.abs(err) <= float(tol2)).sum().item())

    def mae(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(self.sum_abs / self.n)

    def rmse(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(math.sqrt(self.sum_sq / self.n))

    def r2(self) -> float:
        if self.n == 0:
            return float("nan")
        ss_tot = self.sum_y2 - (self.sum_y * self.sum_y / self.n)
        if ss_tot <= 1e-12:
            return 1.0 if self.sum_sq <= 1e-12 else 0.0
        return float(1.0 - (self.sum_sq / ss_tot))

    def corr(self) -> float:
        if self.n < 2:
            return 0.0
        var_y = self.sum_y2 - (self.sum_y * self.sum_y / self.n)
        var_yhat = self.sum_yhat2 - (self.sum_yhat * self.sum_yhat / self.n)
        if var_y <= 1e-12 or var_yhat <= 1e-12:
            return 0.0
        cov = self.sum_yyhat - (self.sum_y * self.sum_yhat / self.n)
        corr = cov / math.sqrt(var_y * var_yhat)
        return float(max(-1.0, min(1.0, corr)))

    def within1(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(self.cnt_within1 / self.n)

    def within2(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(self.cnt_within2 / self.n)

    def nmae(self, scale: float) -> float:
        if self.n == 0:
            return float("nan")
        return float((self.sum_abs / self.n) / float(scale))


@torch.no_grad()
def evaluate_mil(model, dl, device, target_mode: str = "whse") -> Dict[str, float]:
    target_mode = _normalize_target_mode(target_mode)
    model.eval()

    wh_stats = _RunningRegressionStats()
    se_stats = _RunningRegressionStats()
    ue_stats = _RunningRegressionStats()

    for batch in dl:
        batch = to_device(batch, device)
        windows = batch["windows"]
        window_mask = batch["window_mask"]
        trial_mask = batch["trial_mask"]
        window_starts = batch.get("window_starts")
        y_wh = batch["y_wh"]
        y_se = batch["y_se"]
        label_mask = batch["label_mask"]

        yhat_wh, yhat_se = model(windows, window_mask, trial_mask, window_starts=window_starts)

        m_wh = label_mask[:, 0] & _use_wh(target_mode)
        m_se = label_mask[:, 1] & _use_se(target_mode)
        if bool(m_wh.any()):
            wh_stats.update(y_wh[m_wh], yhat_wh[m_wh], tol1=1.0, tol2=2.0)
        if bool(m_se.any()):
            se_stats.update(y_se[m_se], yhat_se[m_se], tol1=1.0, tol2=2.0)

        m_ue = m_wh & m_se
        if bool(m_ue.any()):
            ue_stats.update(y_wh[m_ue] + y_se[m_ue], yhat_wh[m_ue] + yhat_se[m_ue])

    if _use_wh(target_mode) and wh_stats.n == 0:
        return {"val_score": float("inf")}
    if _use_se(target_mode) and se_stats.n == 0:
        return {"val_score": float("inf")}

    mae_wh = wh_stats.mae() if wh_stats.n > 0 else float("nan")
    mae_se = se_stats.mae() if se_stats.n > 0 else float("nan")
    mae_ue = ue_stats.mae() if ue_stats.n > 0 else float("nan")
    r2_wh = wh_stats.r2() if wh_stats.n > 0 else float("nan")
    r2_se = se_stats.r2() if se_stats.n > 0 else float("nan")
    r2_ue = ue_stats.r2() if ue_stats.n > 0 else float("nan")
    corr_wh = wh_stats.corr() if wh_stats.n > 0 else float("nan")
    corr_se = se_stats.corr() if se_stats.n > 0 else float("nan")
    corr_ue = ue_stats.corr() if ue_stats.n > 0 else float("nan")

    nmae_wh = wh_stats.nmae(24.0) if wh_stats.n > 0 else float("nan")
    nmae_se = se_stats.nmae(42.0) if se_stats.n > 0 else float("nan")
    if target_mode == "wh":
        val_score = nmae_wh
    elif target_mode == "se":
        val_score = nmae_se
    else:
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
        "RMSE_WH": wh_stats.rmse() if wh_stats.n > 0 else float("nan"),
        "RMSE_SE": se_stats.rmse() if se_stats.n > 0 else float("nan"),
        "within1_WH": wh_stats.within1() if wh_stats.n > 0 else float("nan"),
        "within2_WH": wh_stats.within2() if wh_stats.n > 0 else float("nan"),
        "within1_SE": se_stats.within1() if se_stats.n > 0 else float("nan"),
        "within2_SE": se_stats.within2() if se_stats.n > 0 else float("nan"),
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
    target_mode = _normalize_target_mode(mil_cfg.target_mode)
    window_agg_mode = _normalize_window_agg_mode(mil_cfg.window_agg_mode)
    mil_cfg = replace(mil_cfg, target_mode=target_mode, window_agg_mode=window_agg_mode)

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
        require_both_main_labels=(target_mode == "whse"),
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
        require_both_main_labels=(target_mode == "whse"),
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
    if ssl_encoder_ckpt is not None:
        if os.path.exists(ssl_encoder_ckpt):
            ckpt = torch_load_compat(ssl_encoder_ckpt, map_location="cpu")
            encoder.load_state_dict(ckpt["encoder"], strict=True)
            print(f"[INFO] MIL encoder initialized from SSL checkpoint: {ssl_encoder_ckpt}")
        else:
            print(f"[WARN] SSL checkpoint not found, MIL encoder uses random init: {ssl_encoder_ckpt}")

    student = HierarchicalMILRegressor(
        encoder=encoder,
        emb_dim=enc_cfg.emb_dim,
        attn_hidden=mil_cfg.attn_hidden,
        fusion_hidden=mil_cfg.fusion_hidden,
        dropout=mil_cfg.dropout,
        window_dropout_p=mil_cfg.window_dropout_p,
        trial_dropout_p=mil_cfg.trial_dropout_p,
        window_agg_mode=mil_cfg.window_agg_mode,
        window_temporal_layers=mil_cfg.window_temporal_layers,
        window_temporal_heads=mil_cfg.window_temporal_heads,
        window_temporal_ffn_ratio=mil_cfg.window_temporal_ffn_ratio,
        window_temporal_dropout=mil_cfg.window_temporal_dropout,
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
    last_ckpt_payload = None

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
        ep_skipped_nonfinite = 0

        for batch in pbar:
            global_step += 1
            batch = to_device(batch, device)

            if return_two_views:
                w1 = batch["windows_1"]
                wm1 = batch["window_mask_1"]
                tm1 = batch["trial_mask_1"]
                ws1 = batch.get("window_starts_1")
                w2 = batch["windows_2"]
                wm2 = batch["window_mask_2"]
                tm2 = batch["trial_mask_2"]
                ws2 = batch.get("window_starts_2")
            else:
                w1 = batch["windows"]
                wm1 = batch["window_mask"]
                tm1 = batch["trial_mask"]
                ws1 = batch.get("window_starts")
                w2 = wm2 = tm2 = ws2 = None

            y_wh = batch["y_wh"]
            y_se = batch["y_se"]
            label_mask = batch["label_mask"]  # (B,2)
            label_mask_eff = _apply_target_label_mask(label_mask, target_mode)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                # student forward
                yhat_wh_s, yhat_se_s = student(w1, wm1, tm1, window_starts=ws1)
                L_sup = sup_loss_fn(y_wh, y_se, yhat_wh_s, yhat_se_s, label_mask_eff)

                wh_mask = label_mask_eff[:, 0]
                se_mask = label_mask_eff[:, 1]
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
                        yhat_wh_t, yhat_se_t = teacher(w2, wm2, tm2, window_starts=ws2)

                    p_s = _consistency_tensor(yhat_wh_s, yhat_se_s, target_mode)
                    p_t = _consistency_tensor(yhat_wh_t, yhat_se_t, target_mode)
                    L_cons = cons_loss_fn(p_s, p_t)

                w_cons = mil_cfg.consistency_weight * ramp if (return_two_views and mil_cfg.use_mean_teacher) else 0.0
                L = L_sup + w_cons * L_cons

            if not _all_finite_tensors(yhat_wh_s, yhat_se_s, L_sup, L_cons, L):
                ep_skipped_nonfinite += 1
                if ep_skipped_nonfinite <= 3:
                    print(
                        "[WARN] Skip non-finite MIL batch: "
                        f"epoch={epoch + 1}, step={global_step}, "
                        f"L={float(L.detach().cpu())}, "
                        f"L_sup={float(L_sup.detach().cpu())}, "
                        f"L_cons={float(L_cons.detach().cpu())}"
                    )
                pbar.set_postfix(
                    skip_nonfinite=int(ep_skipped_nonfinite),
                    lr=float(opt.param_groups[0]["lr"]),
                )
                continue

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
                nan_skip=int(ep_skipped_nonfinite),
                lr=float(opt.param_groups[0]["lr"]),
            )

        if ep_steps == 0:
            raise RuntimeError(
                "All MIL train batches were non-finite in this epoch. "
                "Check data for NaN/Inf and hyperparameters (especially lr/amp)."
            )

        # Validation using teacher weights (EMA) for stability
        val_metrics = evaluate_mil(teacher, dl_val, device, target_mode=target_mode)
        val_score = float(val_metrics["val_score"])

        epoch_row = {
            "epoch": int(epoch + 1),
            "train_loss_total": _safe_mean(ep_total_sum, ep_steps),
            "train_loss_sup": _safe_mean(ep_sup_sum, ep_steps),
            "train_loss_cons": _safe_mean(ep_cons_sum, ep_steps),
            "train_cons_weight": _safe_mean(ep_wcons_sum, ep_steps),
            "train_loss_wh": _safe_mean(ep_wh_sum, ep_wh_steps),
            "train_loss_se": _safe_mean(ep_se_sum, ep_se_steps),
            "train_skipped_nonfinite": int(ep_skipped_nonfinite),
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

        ckpt_payload = {
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
        }
        last_ckpt_payload = ckpt_payload

        if val_score < best_score:
            best_score = val_score
            bad_epochs = 0
            torch.save(ckpt_payload, best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= mil_cfg.patience:
            break

    if not os.path.exists(best_path):
        if last_ckpt_payload is None:
            raise RuntimeError("MIL training ended without producing any checkpoint payload.")
        fallback_payload = dict(last_ckpt_payload)
        fallback_payload["fallback_used"] = True
        fallback_payload["fallback_reason"] = "no_epoch_improved_val_score"
        torch.save(fallback_payload, best_path)
        print(
            "[WARN] No epoch improved validation score; saved last-epoch fallback checkpoint "
            f"to {best_path}"
        )

    return best_path
