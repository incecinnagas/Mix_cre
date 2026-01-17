# -*- coding: utf-8 -*-
"""Mix_cre Model definitions for Transformer+SSM hybrid architecture"""

from typing import Iterable, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65536) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float, stochastic_depth: float, max_len: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu"
        ) for _ in range(n_layers)])
        self.stochastic_depth = stochastic_depth
        self.drop_paths = nn.ModuleList([DropPath(stochastic_depth * (i + 1) / n_layers) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)  # [B, L, D]
        x = self.pos_enc(x)
        for layer, dp in zip(self.layers, self.drop_paths):
            residual = x
            x = layer(x)
            x = residual + dp(x)
        x = self.norm(x)
        return x  # [B, L, D]

    def backbone_layers(self) -> Iterable[nn.Module]:
        return list(self.layers)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 7, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.proj = nn.Linear(d_model, d_model * 2)
        self.out = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        h = self.norm(x)
        h1 = self.dwconv(h.transpose(1, 2)).transpose(1, 2)  # [B, L, D]
        gate, val = self.proj(h1).chunk(2, dim=-1)
        h2 = torch.sigmoid(gate) * F.gelu(val)
        h3 = self.out(h2)
        return x + self.alpha * h3


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn_vec = nn.Parameter(torch.randn(d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        h = torch.tanh(self.proj(x))  # [B, L, D]
        scores = torch.matmul(h, self.attn_vec) / math.sqrt(h.size(-1))  # [B, L]
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled


class Evo2MixModel(nn.Module):
    def __init__(
        self,
        num_species: int,
        d_model: int = 1024,
        n_layers: int = 8,
        n_heads: int = 8,
        tail: str = "mamba",
        tail_layers: int = 3,
        pool: str = "attn",
        max_len: int = 65536,
        dropout: float = 0.1,
        stochastic_depth: float = 0.1,
    ) -> None:
        super().__init__()
        vocab_size = 5
        
        # Transformer backbone
        self.backbone = TransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            max_len=max_len,
        )

        # SSM tail (Mamba blocks after Transformer)
        self.tail_type = tail
        if tail == "mamba":
            self.tail = nn.ModuleList([MambaBlock(d_model) for _ in range(tail_layers)])
        elif tail == "none":
            self.tail = nn.ModuleList([])
        else:
            raise ValueError("Unknown tail type: {}".format(tail))

        # Pooling
        if pool == "attn":
            self.pool = AttentionPooling(d_model)
        elif pool == "mean":
            self.pool = nn.Identity()
        else:
            raise ValueError("pool must be 'attn' or 'mean'")
        self.pool_type = pool

        # Prediction heads
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_species),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, L]
        x = self.backbone(tokens)  # [B, L, D]
        for block in self.tail:
            x = block(x)  # [B, L, D]
        if isinstance(self.pool, nn.Identity):
            pooled = x.mean(dim=1)
        else:
            pooled = self.pool(x)
        y_hat = self.reg_head(pooled).squeeze(-1)
        logits = self.cls_head(pooled)
        return y_hat, logits

    def backbone_layers(self) -> Iterable[nn.Module]:
        return self.backbone.backbone_layers()