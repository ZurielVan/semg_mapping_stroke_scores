from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adjacency(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = a.size(0)
    i = torch.eye(n, device=a.device, dtype=a.dtype)
    a_tilde = a + i
    deg = a_tilde.sum(dim=1)
    d_inv_sqrt = torch.pow(deg + eps, -0.5)
    return d_inv_sqrt.unsqueeze(1) * a_tilde * d_inv_sqrt.unsqueeze(0)


class ChebGraphConv(nn.Module):
    """
    Chebyshev graph convolution on batched node features.
    x:   (B, N, Fin)
    adj: (N, N)
    out: (B, N, Fout)
    """

    def __init__(self, fin: int, fout: int, k_order: int):
        super().__init__()
        if k_order < 1:
            raise ValueError("k_order must be >= 1")
        self.k_order = int(k_order)
        self.weight = nn.Parameter(torch.empty(self.k_order, fin, fout))
        self.bias = nn.Parameter(torch.zeros(fout))
        nn.init.xavier_uniform_(self.weight)

    def _matmul_adj(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,bmf->bnf", adj, x)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        t0 = x
        supports = [t0]

        if self.k_order >= 2:
            t1 = self._matmul_adj(adj, t0)
            supports.append(t1)
            for _ in range(2, self.k_order):
                t2 = 2.0 * self._matmul_adj(adj, t1) - t0
                supports.append(t2)
                t0, t1 = t1, t2

        out = 0.0
        for k, tk in enumerate(supports):
            out = out + torch.einsum("bnf,fo->bno", tk, self.weight[k])
        out = out + self.bias
        return out


class STGCNLayer(nn.Module):
    """
    Official-style graph block:
    ChebConv(seq_len -> nb_chev_filter) + ReLU + Linear(nb_chev_filter -> seq_len).
    """

    def __init__(self, seq_len: int, nb_chev_filter: int, k_order: int):
        super().__init__()
        self.graph_conv = ChebGraphConv(seq_len, nb_chev_filter, k_order)
        self.linear = nn.Linear(nb_chev_filter, seq_len)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.graph_conv(x, adj)
        out = torch.relu(out)
        out = self.linear(out)
        return out


class DataEmbeddingInverted(nn.Module):
    """
    Same inverted projection spirit as official MGCN code:
    per-node time vector (len_input) -> d_model.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x))


class BiMambaEncoderLayer(nn.Module):
    """
    Bidirectional Mamba encoder layer (official MGCN style).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        d_state: int = 32,
        d_conv: int = 2,
        expand: int = 1,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "MGCN encoder requires `mamba-ssm`. Install it to use encoder_type='mgcn'."
            ) from e

        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x = self.mamba_fwd(x) + self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class MGCNEncoderOfficial(nn.Module):
    """
    MGCN-style encoder adapted to this project.
    Input:  (B, C=4, T)
    Output: (B, d)
    """

    def __init__(
        self,
        in_ch: int = 4,
        d: int = 128,
        seq_len: int = 256,
        k_order: int = 2,
        nb_chev_filter: int = 64,
        nb_time_filter: int = 64,
        d_model: int = 512,
        mamba_layers: int = 1,
        dropout: float = 0.1,
        learn_adj: bool = True,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.seq_len = int(seq_len)
        self.learn_adj = bool(learn_adj)

        self.stgcn_layer = STGCNLayer(
            seq_len=self.seq_len, nb_chev_filter=nb_chev_filter, k_order=k_order
        )
        self.enc_embedding = DataEmbeddingInverted(
            c_in=self.seq_len, d_model=d_model, dropout=dropout
        )
        self.encoder = nn.ModuleList(
            [
                BiMambaEncoderLayer(
                    d_model=d_model,
                    d_ff=4 * d_model,
                    dropout=dropout,
                    d_state=32,
                    d_conv=2,
                    expand=1,
                )
                for _ in range(mamba_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.project_time = nn.Linear(d_model, self.seq_len, bias=True)
        self.residual_conv = nn.Conv2d(
            in_channels=1, out_channels=nb_time_filter, kernel_size=(1, 1), stride=(1, 1)
        )
        self.project_feat_up = nn.Linear(1, nb_time_filter, bias=True)
        self.project_feat_down = nn.Linear(nb_time_filter, 1, bias=True)
        self.fuse_norm = nn.LayerNorm(nb_time_filter)
        self.out_proj = nn.Linear(self.seq_len, d)

        if self.learn_adj:
            self.adj_param = nn.Parameter(torch.eye(self.in_ch))
        else:
            self.register_buffer("adj_param", torch.ones(self.in_ch, self.in_ch))

    def _adj(self) -> torch.Tensor:
        # Keep the graph non-negative and normalized.
        a = torch.relu(self.adj_param)
        return _normalize_adjacency(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,C,T), got {tuple(x.shape)}")
        bsz, c, t = x.shape
        del bsz
        if c != self.in_ch:
            raise ValueError(f"Expected C={self.in_ch}, got C={c}")
        if t != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got T={t}")

        adj = self._adj()                            # (C,C)
        gcn_out = self.stgcn_layer(adj, x)           # (B,C,T)
        enc_out = self.enc_embedding(gcn_out)        # (B,C,d_model)
        for blk in self.encoder:
            enc_out = blk(enc_out)
        enc_out = self.encoder_norm(enc_out)

        mamba_out = self.project_time(enc_out)       # (B,C,T)
        mamba_out = self.project_feat_up(mamba_out.unsqueeze(-1))   # (B,C,T,Ft)

        x_res = self.residual_conv(x.unsqueeze(1))   # (B,Ft,C,T)
        x_res = x_res.permute(0, 2, 3, 1)            # (B,C,T,Ft)

        fused = self.fuse_norm(F.relu(x_res + mamba_out))
        fused = self.project_feat_down(fused).squeeze(-1)  # (B,C,T)

        pooled = fused.mean(dim=1)                   # (B,T)
        emb = self.out_proj(pooled)                  # (B,d)
        return emb
