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

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        q = self.encoder_q(x_q)
        q = self.proj_q(q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(x_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("bd,dk->bk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.cfg.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            self.dequeue_and_enqueue(k)
        return loss
