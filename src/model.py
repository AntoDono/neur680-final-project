"""TabularTransformer: feature-as-token transformer for tabular brain data.

Each input feature is treated as one token (scalar → projected to d_model).
A learnable CLS token is prepended; its final hidden state is used for
binary classification (PD vs HC).

Because there is no positional embedding, the model is permutation-equivariant
and can be applied to any number of features — useful for loading a checkpoint
trained on PPMI (ASEG-only, ~64 features) and fine-tuning on lab data
(ASEG + thickness, ~137 features) without architectural changes.
"""

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        x   = self.feature_proj(x.unsqueeze(-1))       # (B, F, d_model)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (B, F+1, d_model)
        x   = self.transformer(x)
        return self.head(x[:, 0, :]).squeeze(-1)        # (B,)
