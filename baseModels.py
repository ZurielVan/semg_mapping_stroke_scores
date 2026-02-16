from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .itransformer_official import OfficialITransformerEncoder
from .utils import masked_softmax


# --------------------------
# Encoder blocks
# --------------------------

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, dropout=0.0):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, ch, k=5, dilation=1, dropout=0.1):
        super().__init__()
        p = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.drop1(self.act1(self.norm1(self.conv1(x))))
        x = self.drop2(self.act2(self.norm2(self.conv2(x))))
        return x + res


class TCNEncoder(nn.Module):
    """
    Strong, parameter-efficient baseline.
    Input: (B, C=4, T)
    Output: (B, d)
    """
    def __init__(self, in_ch=4, d=128, width=64, blocks=5, k=5, dropout=0.1, stem_stride=1):
        super().__init__()
        self.stem = ConvNormAct(in_ch, width, k=7, s=stem_stride, dropout=dropout)
        dilations = [2**i for i in range(blocks)]
        self.tcn = nn.Sequential(*[TCNBlock(width, k=k, dilation=di, dropout=dropout) for di in dilations])
        self.proj = nn.Linear(width, d)

    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x)
        x = x.mean(dim=-1)
        x = self.proj(x)
        return x


class Patchify1D(nn.Module):
    def __init__(self, in_ch: int, d: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_ch, d, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)         # (B,d,L)
        x = x.transpose(1, 2)    # (B,L,d)
        return x


class TransformerEncoder1D(nn.Module):
    """
    Time-token Transformer with patchify.
    """
    def __init__(self, in_ch=4, d=128, patch_size=8, layers=4, heads=4, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.patch = Patchify1D(in_ch, d, patch_size=patch_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d*ffn_ratio,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        tok = self.patch(x)      # (B,L,d)
        tok = self.enc(tok)
        tok = self.norm(tok)
        emb = tok.mean(dim=1)
        return emb


class iTransformerEncoder(nn.Module):
    """
    Adapter to the official iTransformer backbone.
    """
    def __init__(self, d=128, layers=4, heads=4, ffn_ratio=4, dropout=0.1, temporal_width=64, seq_len=256):
        super().__init__()
        del temporal_width  # kept for backward-compatibility with existing config
        self.backbone = OfficialITransformerEncoder(
            in_ch=4,
            seq_len=seq_len,
            d_model=d,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            use_norm=False,
        )

    def forward(self, x):
        return self.backbone(x)


class SimpleGCNLayer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A_hat):
        x = torch.matmul(A_hat, H)   # (B,N,d_in)
        x = self.lin(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class GCNEncoder(nn.Module):
    """
    Temporal encoding per channel node + GCN over channels.
    """
    def __init__(self, d=128, gcn_layers=2, dropout=0.1, learn_adj=True, temporal_width=64, num_nodes=4):
        super().__init__()
        self.temporal = nn.Sequential(
            ConvNormAct(1, temporal_width, k=9, s=2, dropout=dropout),
            ConvNormAct(temporal_width, temporal_width, k=7, s=2, dropout=dropout),
        )
        self.temporal_proj = nn.Linear(temporal_width, d)
        self.learn_adj = learn_adj
        N = num_nodes
        if learn_adj:
            self.A = nn.Parameter(torch.eye(N))
        else:
            self.register_buffer("A", torch.ones(N, N))

        self.gcn = nn.ModuleList([SimpleGCNLayer(d, d, dropout=dropout) for _ in range(gcn_layers)])
        self.out_norm = nn.LayerNorm(d)

    def _normalize_adj(self):
        A = self.A
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_tilde = A + I
        deg = A_tilde.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, x):
        B, C, T = x.shape
        nodes = []
        for c in range(C):
            xc = x[:, c:c+1, :]
            hc = self.temporal(xc).mean(dim=-1)
            vc = self.temporal_proj(hc)
            nodes.append(vc)
        H = torch.stack(nodes, dim=1)     # (B,C,d)

        A_hat = self._normalize_adj()
        for layer in self.gcn:
            H = layer(H, A_hat)
        H = self.out_norm(H)
        emb = H.mean(dim=1)
        return emb


class MambaEncoder(nn.Module):
    """
    Optional: requires `mamba-ssm`.
    """
    def __init__(self, in_ch=4, d=128, patch_size=8, layers=4, dropout=0.1):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError("MambaEncoder requires `mamba-ssm`. Install it to use encoder_type='mamba'.") from e
        self.patch = Patchify1D(in_ch, d, patch_size=patch_size)
        self.blocks = nn.ModuleList([Mamba(d_model=d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        tok = self.patch(x)  # (B,L,d)
        for blk in self.blocks:
            tok = tok + self.drop(blk(tok))
        tok = self.norm(tok)
        emb = tok.mean(dim=1)
        return emb


# --------------------------
# Hierarchical MIL + bounded regression
# --------------------------

class GatedAttnScore(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.V = nn.Linear(d, hidden)
        self.U = nn.Linear(d, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        s = self.w(h).squeeze(-1)
        return s


def _apply_dropout_to_mask(mask: torch.Tensor, p: float) -> torch.Tensor:
    """
    mask: bool tensor
    Returns: mask with random drops (only dropping True -> False).
    Ensures not all entries are dropped along last dim for any row:
      if all dropped, revert that row to original mask.
    """
    if p <= 0:
        return mask
    if not mask.any():
        return mask
    keep = torch.rand_like(mask.float()) > p
    dropped = mask & keep.to(dtype=torch.bool)
    # handle all-dropped rows along last dim
    # compute where original had any True and dropped has none
    orig_any = mask.any(dim=-1, keepdim=True)
    dropped_any = dropped.any(dim=-1, keepdim=True)
    need_fix = orig_any & (~dropped_any)
    # if need_fix, revert to original
    dropped = torch.where(need_fix, mask, dropped)
    return dropped


class HierarchicalMILRegressor(nn.Module):
    """
    Inputs:
      windows:     (B, A=2, R, K, C, Tw)
      window_mask: (B,2,R,K)
      trial_mask:  (B,2,R)
    Outputs:
      yhat_wh, yhat_se in score units (bounded to valid ranges)
    """
    def __init__(
        self,
        encoder: nn.Module,
        emb_dim: int,
        attn_hidden: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.1,
        window_dropout_p: float = 0.2,
        trial_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.win_score = GatedAttnScore(emb_dim, attn_hidden)
        self.trial_score = GatedAttnScore(emb_dim, attn_hidden)

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_wh = nn.Linear(fusion_hidden, 1)
        self.head_se = nn.Linear(fusion_hidden, 1)

        self.window_dropout_p = float(window_dropout_p)
        self.trial_dropout_p = float(trial_dropout_p)

    def forward(self, windows, window_mask, trial_mask):
        B, A, R, K, C, Tw = windows.shape
        x = windows.view(B * A * R * K, C, Tw)
        h = self.encoder(x)  # (B*A*R*K, d)
        d = h.shape[-1]
        h = h.view(B, A, R, K, d)

        # stochastic instance dropout (mask-level) for robustness
        if self.training:
            window_mask = _apply_dropout_to_mask(window_mask, self.window_dropout_p)
            trial_mask = _apply_dropout_to_mask(trial_mask, self.trial_dropout_p)

        # window -> trial attention
        win_logits = self.win_score(h)  # (B,A,R,K)
        alpha = masked_softmax(win_logits, window_mask, dim=-1)  # over K
        trial_emb = torch.sum(alpha.unsqueeze(-1) * h, dim=-2)   # (B,A,R,d)

        # trial -> axis attention
        tr_logits = self.trial_score(trial_emb)                  # (B,A,R)
        beta = masked_softmax(tr_logits, trial_mask, dim=-1)     # over R
        axis_emb = torch.sum(beta.unsqueeze(-1) * trial_emb, dim=-2)  # (B,A,d)

        # fuse y/z
        z = torch.cat([axis_emb[:, 0, :], axis_emb[:, 1, :]], dim=-1)  # (B,2d)
        z = self.fusion(z)

        u_wh = self.head_wh(z).squeeze(-1)
        u_se = self.head_se(z).squeeze(-1)

        yhat_wh = 24.0 * torch.sigmoid(u_wh)
        yhat_se = 42.0 * torch.sigmoid(u_se)
        return yhat_wh, yhat_se


# --------------------------
# Factory
# --------------------------

@dataclass
class EncoderConfig:
    encoder_type: str = "tcn"  # tcn | transformer | itransformer | gcn | mamba
    emb_dim: int = 128
    dropout: float = 0.1
    # TCN
    tcn_width: int = 64
    tcn_blocks: int = 5
    tcn_kernel: int = 5
    tcn_stem_stride: int = 1
    # Transformer/Mamba
    patch_size: int = 8
    layers: int = 4
    heads: int = 4
    ffn_ratio: int = 4
    # iTransformer
    itransformer_seq_len: int = 256
    # GCN
    gcn_layers: int = 2
    gcn_learn_adj: bool = True


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    et = cfg.encoder_type.lower()
    if et == "tcn":
        return TCNEncoder(
            in_ch=4, d=cfg.emb_dim, width=cfg.tcn_width,
            blocks=cfg.tcn_blocks, k=cfg.tcn_kernel,
            dropout=cfg.dropout, stem_stride=cfg.tcn_stem_stride
        )
    if et == "transformer":
        return TransformerEncoder1D(
            in_ch=4, d=cfg.emb_dim, patch_size=cfg.patch_size,
            layers=cfg.layers, heads=cfg.heads, ffn_ratio=cfg.ffn_ratio,
            dropout=cfg.dropout
        )
    if et == "itransformer":
        return iTransformerEncoder(
            d=cfg.emb_dim, layers=cfg.layers, heads=cfg.heads,
            ffn_ratio=cfg.ffn_ratio, dropout=cfg.dropout,
            seq_len=cfg.itransformer_seq_len,
        )
    if et == "gcn":
        return GCNEncoder(
            d=cfg.emb_dim, gcn_layers=cfg.gcn_layers,
            dropout=cfg.dropout, learn_adj=cfg.gcn_learn_adj,
            num_nodes=4
        )
    if et == "mamba":
        return MambaEncoder(
            in_ch=4, d=cfg.emb_dim, patch_size=cfg.patch_size,
            layers=cfg.layers, dropout=cfg.dropout
        )
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")
