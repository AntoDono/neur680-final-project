"""TabularTransformer: feature-as-token transformer for tabular brain data.

Each input feature is treated as one token (scalar → projected to d_model),
with a learned per-feature identity embedding added so the model knows which
biomarker each token represents.  All feature token outputs are pooled via a
learned attention query, and the resulting representation is classified by a
small MLP.

Because the identity embeddings are indexed by position (not value), the model
is not permutation-equivariant, but feature_cols ordering is kept fixed.
"""

import math

import torch
import torch.nn as nn

from src import config


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(1, d_model)

        # Per-feature identity embeddings — tells the model which biomarker
        # each token represents regardless of its scalar value
        self.feature_embed = nn.Embedding(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,        # pre-norm: easier to train from scratch
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling query — small random init avoids symmetry-breaking issues
        self.attn_pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self._scale = math.sqrt(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        feat_idx = torch.arange(F, device=x.device)

        h = self.feature_proj(x.unsqueeze(-1))      # (B, F, d_model)
        h = h + self.feature_embed(feat_idx)         # (B, F, d_model)
        h = self.transformer(h)                      # (B, F, d_model)

        # Attention pooling
        scores  = torch.einsum("d,bfd->bf", self.attn_pool_query, h) / self._scale
        weights = torch.softmax(scores, dim=-1)      # (B, F)
        pooled  = torch.einsum("bf,bfd->bd", weights, h)  # (B, d_model)

        return self.head(pooled).squeeze(-1)          # (B,)
