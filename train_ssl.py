from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, replace
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

from .dataset import SSLWindowDataset
from .augmentations import build_ssl_augmentations
from .baseModels import build_encoder, EncoderConfig
from .ssl_models import MoCo, MoCoConfig
from .utils import set_seed, seed_worker, ensure_dir


@dataclass
class SSLTrainConfig:
    seed: int = 0
    Tw_samples: int = 512
    windows_per_epoch: int = 20000
    epochs: int = 100
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-3
    aug_strength: str = "medium"
    use_amp: bool = True

    # raw loader config
    emg_cols: Optional[list[int]] = None
    time_col: Optional[int] = 0
    expected_emg_ch: int = 4

    # MoCo
    moco: MoCoConfig = field(default_factory=MoCoConfig)
    use_cosine: bool = True


def _safe_mean(total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(total / count)


def _save_ssl_loss_plot(history_df: pd.DataFrame, out_png: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Skip SSL loss plot: matplotlib unavailable ({e})")
        return

    if len(history_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = history_df["epoch"].to_numpy()
    ax.plot(x, history_df["ssl_loss"].to_numpy(), label="ssl_loss", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SSL Training Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def pretrain_ssl_moco(
    manifest_df: pd.DataFrame,
    train_subjects: list[str],
    encoder_cfg: EncoderConfig,
    ssl_cfg: SSLTrainConfig,
    out_dir: str,
    device: torch.device,
) -> str:
    """
    Fold-wise SSL pretraining. Only uses TRAIN subjects (no val/test leakage).

    Returns:
      path to saved encoder_q checkpoint.
    """
    ensure_dir(out_dir)
    set_seed(ssl_cfg.seed)

    ssl_aug = build_ssl_augmentations(ssl_cfg.aug_strength)
    ds = SSLWindowDataset(
        manifest_df=manifest_df,
        subjects=train_subjects,
        Tw_samples=ssl_cfg.Tw_samples,
        windows_per_epoch=ssl_cfg.windows_per_epoch,
        ssl_aug=ssl_aug,
        emg_cols=ssl_cfg.emg_cols,
        time_col=ssl_cfg.time_col,
        expected_emg_ch=ssl_cfg.expected_emg_ch,
        cache_size=256,
    )
    dl = DataLoader(
        ds,
        batch_size=ssl_cfg.batch_size,
        shuffle=True,
        num_workers=ssl_cfg.num_workers,
        pin_memory=True,
        # Avoid 0-step epochs when windows_per_epoch < batch_size.
        drop_last=(len(ds) >= ssl_cfg.batch_size),
        worker_init_fn=seed_worker,
    )

    enc_cfg = encoder_cfg
    if encoder_cfg.encoder_type.lower() == "itransformer":
        enc_cfg = replace(encoder_cfg, itransformer_seq_len=ssl_cfg.Tw_samples)
    elif encoder_cfg.encoder_type.lower() == "mgcn":
        enc_cfg = replace(encoder_cfg, mgcn_seq_len=ssl_cfg.Tw_samples)

    encoder_q = build_encoder(enc_cfg).to(device)
    encoder_k = build_encoder(enc_cfg).to(device)

    moco = MoCo(encoder_q, encoder_k, emb_dim=enc_cfg.emb_dim, cfg=ssl_cfg.moco).to(device)
    opt = torch.optim.AdamW(moco.parameters(), lr=ssl_cfg.lr, weight_decay=ssl_cfg.weight_decay)

    total_steps = ssl_cfg.epochs * len(dl)
    scheduler = None
    if ssl_cfg.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    scaler = GradScaler(enabled=ssl_cfg.use_amp and device.type == "cuda")
    loss_csv_path = os.path.join(out_dir, "ssl_loss_history.csv")
    loss_png_path = os.path.join(out_dir, "ssl_loss_curve.png")
    loss_history_rows = []

    moco.train()
    global_step = 0
    for epoch in range(ssl_cfg.epochs):
        pbar = tqdm(dl, desc=f"[SSL] epoch {epoch+1}/{ssl_cfg.epochs}", leave=False)
        ep_loss_sum = 0.0
        ep_steps = 0
        for x1, x2 in pbar:
            global_step += 1
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss = moco(x1, x2)
            ep_steps += 1
            ep_loss_sum += float(loss.item())

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

        loss_history_rows.append(
            {
                "epoch": int(epoch + 1),
                "ssl_loss": _safe_mean(ep_loss_sum, ep_steps),
                "lr_last": float(opt.param_groups[0]["lr"]),
                "steps": int(ep_steps),
            }
        )
        loss_df = pd.DataFrame(loss_history_rows)
        loss_df.to_csv(loss_csv_path, index=False)
        _save_ssl_loss_plot(loss_df, loss_png_path)

    ckpt_path = os.path.join(out_dir, "ssl_encoder_q.pth")
    torch.save(
        {
            "encoder": encoder_q.state_dict(),
            # Store plain dicts for cross-version safety.
            "encoder_cfg": asdict(enc_cfg),
            "ssl_cfg": asdict(ssl_cfg),
            "ssl_loss_history_csv": loss_csv_path,
            "ssl_loss_plot": loss_png_path,
        },
        ckpt_path
    )
    return ckpt_path
