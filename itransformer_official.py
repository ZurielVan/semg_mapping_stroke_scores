from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbeddingInverted(nn.Module):
    """
    Official iTransformer inverted embedding:
    input (B, L, N) -> tokens (B, N, d_model)
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, N, L)
        if x_mark is not None:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], dim=1)
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = False,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ):
        del tau, delta  # kept for interface compatibility
        _, _, _, e = queries.shape
        scale = self.scale or (1.0 / sqrt(e))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                raise ValueError("attn_mask is required when mask_flag=True")
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ):
        bsz, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        heads = self.n_heads

        queries = self.query_projection(queries).view(bsz, q_len, heads, -1)
        keys = self.key_projection(keys).view(bsz, k_len, heads, -1)
        values = self.value_projection(values).view(bsz, k_len, heads, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.reshape(bsz, q_len, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[nn.Module], norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class OfficialITransformerEncoder(nn.Module):
    """
    Official iTransformer encoder backbone adapted for this project.
    Input:  (B, C, T)
    Output: (B, d_model)
    """

    def __init__(
        self,
        in_ch: int = 4,
        seq_len: int = 256,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        use_norm: bool = False,
        factor: int = 5,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.seq_len = int(seq_len)
        self.use_norm = bool(use_norm)
        d_ff = int(d_model * ffn_ratio)

        self.enc_embedding = DataEmbeddingInverted(
            c_in=self.seq_len, d_model=d_model, dropout=dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B,C,T), got {tuple(x.shape)}")
        bsz, channels, _ = x.shape
        del bsz
        if channels != self.in_ch:
            raise ValueError(f"Expected C={self.in_ch}, got C={channels}")
        x_enc = x.transpose(1, 2)  # (B, T, C)
        if x_enc.size(1) != self.seq_len:
            raise ValueError(
                f"Expected T={self.seq_len} for official iTransformer embedding, got T={x_enc.size(1)}"
            )

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark=None)   # (B, C, d)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # (B, C, d)
        emb = enc_out.mean(dim=1)  # pool tokens -> (B, d)
        return emb
