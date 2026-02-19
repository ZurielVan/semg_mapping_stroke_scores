from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class MoCoConfig:
    proj_hidden: int = 256
    proj_dim: int = 128
    queue_size: int = 8192
    temperature: float = 0.1
    momentum: float = 0.999


@dataclass
class MaskedReconConfig:
    enabled: bool = False
    weight: float = 0.5
    mask_ratio: float = 0.5
    mask_block_len: int = 16
    mask_mode: str = "time"  # time | channel_time
    decoder_hidden: int = 256
    loss_type: str = "smoothl1"  # mse | smoothl1
    smoothl1_beta: float = 0.02


class MoCo(nn.Module):
    """
    MoCo-style contrastive learning:
      - Query encoder: updated by gradients
      - Key encoder: EMA (momentum) update
      - Queue: large set of negative keys for stable training with small batches
    """
    def __init__(self, encoder_q: nn.Module, encoder_k: nn.Module, emb_dim: int, cfg: MoCoConfig):
        super().__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.cfg = cfg

        self.proj_q = MLPHead(emb_dim, hidden=cfg.proj_hidden, out_dim=cfg.proj_dim)
        self.proj_k = MLPHead(emb_dim, hidden=cfg.proj_hidden, out_dim=cfg.proj_dim)

        # initialize key params = query params
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        self.register_buffer("queue", torch.randn(cfg.proj_dim, cfg.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        m = self.cfg.momentum
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)
        for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor):
        keys = keys.detach()
        batch_size = keys.shape[0]
        K = self.cfg.queue_size
        ptr = int(self.queue_ptr.item())

        if batch_size > K:
            keys = keys[:K]
            batch_size = keys.shape[0]

        end = ptr + batch_size
        if end <= K:
            self.queue[:, ptr:end] = keys.T
        else:
            first = K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - K] = keys[first:].T

        ptr = (ptr + batch_size) % K
        self.queue_ptr[0] = ptr

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, return_outputs: bool = False):
        q_backbone = self.encoder_q(x_q)
        q = self.proj_q(q_backbone)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(x_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)
        # Important: snapshot queue to avoid in-place version bumps before backward.
        # We enqueue new keys in this same forward pass, so using self.queue directly
        # can trigger: "variable needed for gradient computation has been modified..."
        queue_snapshot = self.queue.detach().clone()
        l_neg = torch.einsum("bd,dk->bk", [q, queue_snapshot])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.cfg.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            self.dequeue_and_enqueue(k)

        if not return_outputs:
            return loss

        return {
            "loss": loss,
            "q_backbone": q_backbone,
            "pos_sim": float(l_pos.detach().mean().item()),
            "neg_sim": float(l_neg.detach().mean().item()),
        }


class MoCoWithMaskedReconstruction(nn.Module):
    """
    Joint SSL objective:
      - MoCo contrastive loss
      - Masked reconstruction on the query branch
    """
    def __init__(
        self,
        moco: MoCo,
        emb_dim: int,
        in_ch: int,
        seq_len: int,
        recon_cfg: MaskedReconConfig,
    ):
        super().__init__()
        self.moco = moco
        self.emb_dim = int(emb_dim)
        self.in_ch = int(in_ch)
        self.seq_len = int(seq_len)
        self.recon_cfg = recon_cfg
        self.recon_cfg.mask_ratio = max(0.0, min(1.0, float(self.recon_cfg.mask_ratio)))
        self.recon_cfg.weight = max(0.0, float(self.recon_cfg.weight))
        self.recon_cfg.mask_block_len = max(1, int(self.recon_cfg.mask_block_len))
        self.recon_cfg.decoder_hidden = max(8, int(self.recon_cfg.decoder_hidden))
        self.recon_cfg.smoothl1_beta = max(1e-6, float(self.recon_cfg.smoothl1_beta))

        if self.recon_cfg.enabled:
            if self.recon_cfg.mask_mode not in {"time", "channel_time"}:
                raise ValueError(
                    f"Unsupported mask_mode={self.recon_cfg.mask_mode}. "
                    "Supported: {'time', 'channel_time'}"
                )
            if self.recon_cfg.loss_type not in {"mse", "smoothl1"}:
                raise ValueError(
                    f"Unsupported recon loss_type={self.recon_cfg.loss_type}. "
                    "Supported: {'mse', 'smoothl1'}"
                )
            self.decoder = nn.Sequential(
                nn.Linear(self.emb_dim, int(self.recon_cfg.decoder_hidden)),
                nn.GELU(),
                nn.Linear(int(self.recon_cfg.decoder_hidden), self.in_ch * self.seq_len),
            )
        else:
            self.decoder = None

    def _sample_mask(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.recon_cfg.enabled) or self.recon_cfg.mask_ratio <= 0.0:
            return torch.zeros_like(x, dtype=torch.bool)

        B, C, T = x.shape
        block_len = max(1, int(self.recon_cfg.mask_block_len))
        max_start = max(1, T - block_len + 1)
        device = x.device

        if self.recon_cfg.mask_mode == "time":
            target_tokens = max(1, int(round(float(self.recon_cfg.mask_ratio) * T)))
            n_blocks = max(1, (target_tokens + block_len - 1) // block_len)
            starts = torch.randint(0, max_start, (B, n_blocks), device=device)
            offsets = torch.arange(block_len, device=device).view(1, 1, block_len)
            idx_t = (starts.unsqueeze(-1) + offsets).clamp(max=T - 1)  # (B, n_blocks, L)
            mask_t = torch.zeros((B, T), dtype=torch.bool, device=device)
            mask_t.scatter_(1, idx_t.view(B, -1), True)
            mask = mask_t.unsqueeze(1).expand(B, C, T)
            return mask

        # channel_time mode
        target_tokens = max(1, int(round(float(self.recon_cfg.mask_ratio) * C * T)))
        n_blocks = max(1, (target_tokens + block_len - 1) // block_len)
        starts = torch.randint(0, max_start, (B, n_blocks), device=device)
        chans = torch.randint(0, C, (B, n_blocks), device=device)
        offsets = torch.arange(block_len, device=device).view(1, 1, block_len)
        idx_t = (starts.unsqueeze(-1) + offsets).clamp(max=T - 1)

        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_blocks, block_len)
        c_idx = chans.unsqueeze(-1).expand(B, n_blocks, block_len)

        mask = torch.zeros((B, C, T), dtype=torch.bool, device=device)
        mask[b_idx.reshape(-1), c_idx.reshape(-1), idx_t.reshape(-1)] = True
        return mask

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.recon_cfg.loss_type == "mse":
            return F.mse_loss(pred[mask], target[mask])
        return F.smooth_l1_loss(pred[mask], target[mask], beta=float(self.recon_cfg.smoothl1_beta))

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, return_stats: bool = False):
        if not self.recon_cfg.enabled:
            moco_out = self.moco(x_q, x_k, return_outputs=True)
            loss_total = moco_out["loss"]
            if not return_stats:
                return loss_total
            return loss_total, {
                "loss_total": float(loss_total.detach().item()),
                "loss_moco": float(loss_total.detach().item()),
                "loss_recon": 0.0,
                "masked_fraction": 0.0,
                "pos_sim": float(moco_out["pos_sim"]),
                "neg_sim": float(moco_out["neg_sim"]),
            }

        mask = self._sample_mask(x_q)
        x_q_masked = x_q.masked_fill(mask, 0.0)

        moco_out = self.moco(x_q_masked, x_k, return_outputs=True)
        loss_moco = moco_out["loss"]

        q_emb = moco_out["q_backbone"]
        recon = self.decoder(q_emb).view(x_q.shape[0], self.in_ch, self.seq_len)

        if bool(mask.any()):
            loss_recon = self._recon_loss(recon, x_q, mask)
        else:
            loss_recon = recon.sum() * 0.0

        loss_total = loss_moco + float(self.recon_cfg.weight) * loss_recon
        if not return_stats:
            return loss_total

        return loss_total, {
            "loss_total": float(loss_total.detach().item()),
            "loss_moco": float(loss_moco.detach().item()),
            "loss_recon": float(loss_recon.detach().item()),
            "masked_fraction": float(mask.float().mean().detach().item()),
            "pos_sim": float(moco_out["pos_sim"]),
            "neg_sim": float(moco_out["neg_sim"]),
        }
