from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ResMLP(nn.Module):
    """
    Residual MLP block with Pre-LayerNorm.

    Structure: x + MLP(LayerNorm(x))
    LayerNorm is applied on hidden_dim which is typically large (e.g., 512),
    ensuring stable statistics.
    """
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class ProprioProjector(nn.Module):
    def __init__(
        self,
        dim_proprio: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int = 8,
        num_res_blocks: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        # Input projection (no LayerNorm on low-dim proprio input)
        self.input_proj = nn.Linear(dim_proprio, hidden_dim)
        self.input_act = nn.GELU()

        # ResMLP blocks (LayerNorm inside, on hidden_dim which is high-dim)
        self.res_blocks = nn.ModuleList([
            ResMLP(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # Output projection with LayerNorm (on hidden_dim)
        self.output_ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_tokens * output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim_proprio] proprio features
        Returns:
            [B, num_tokens * output_dim] projected features
        """
        # Project to hidden dim
        x = self.input_proj(x)  # [B, dim_proprio] -> [B, hidden_dim]
        x = self.input_act(x)

        # Apply ResMLP blocks
        for block in self.res_blocks:
            x = block(x)

        # Final LayerNorm and output projection
        x = self.output_ln(x)
        x = self.output_proj(x)  # [B, hidden_dim] -> [B, num_tokens * output_dim]
        return x


class ActionEmbedder(nn.Module):
    """Project action chunk [T, D] -> [num_tokens, output_dim] for VLM prefix IC context."""

    def __init__(
        self,
        dim_action: int,
        num_actions: int,
        hidden_dim: int,
        output_dim: int,
        num_tokens: int = 8,
        num_res_blocks: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        input_dim = dim_action * num_actions
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU()

        self.res_blocks = nn.ModuleList([
            ResMLP(hidden_dim, mlp_ratio=mlp_ratio)
            for _ in range(num_res_blocks)
        ])

        self.output_ln = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_tokens * output_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: [*, T, D] action chunks (last two dims are flattened)
        Returns:
            [*, num_tokens, output_dim]
        """
        leading = action.shape[:-2]
        x = action.reshape(*leading, -1)  # [*, T*D]
        x = self.input_proj(x)
        x = self.input_act(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_ln(x)
        x = self.output_proj(x)  # [*, num_tokens * output_dim]
        return x.reshape(*leading, self.num_tokens, self.output_dim)


def init_action_embedder(
    action_space, config, output_dim: int,
) -> Optional[nn.Module]:
    """Create ActionEmbedder if IC is enabled (num_ic_frames > 0)."""
    num_ic = getattr(config, "num_ic_frames", 0)
    if num_ic <= 0:
        return None

    dim_action = action_space.dim_action
    num_actions = config.num_actions
    num_tokens = getattr(config, "num_action_tokens", 8)
    hidden_dim = getattr(config, "action_embedder_hidden_dim", 512)
    num_res_blocks = getattr(config, "action_embedder_num_res_blocks", 2)

    embedder = ActionEmbedder(
        dim_action=dim_action,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
        num_res_blocks=num_res_blocks,
    )

    logging.info(
        f"ActionEmbedder (ResMLP): {dim_action}x{num_actions} -> {hidden_dim} "
        f"(x{num_res_blocks} blocks) -> {num_tokens} x {output_dim}"
    )
    return embedder


def init_proprio_projector(
    action_space, config, projection_dim: int, default_num_tokens: int = 8, default_hidden: int = 512
) -> Tuple[Optional[nn.Module], int]:
    if not getattr(config, "proprio_to_vlm", False):
        return None, 0

    dim_proprio = getattr(action_space, "dim_proprio", action_space.dim_action)
    num_tokens = getattr(config, "num_proprio_tokens", default_num_tokens)
    hidden_dim = getattr(config, "proprio_projector_hidden_dim", default_hidden)
    num_res_blocks = getattr(config, "proprio_projector_num_res_blocks", 2)
    mlp_ratio = getattr(config, "proprio_projector_mlp_ratio", 4.0)
    dropout = getattr(config, "proprio_projector_dropout", 0.0)

    projector = ProprioProjector(
        dim_proprio=dim_proprio,
        hidden_dim=hidden_dim,
        output_dim=projection_dim,
        num_tokens=num_tokens,
        num_res_blocks=num_res_blocks,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    logging.info(
        f"Proprio projector (ResMLP): {dim_proprio} -> {hidden_dim} "
        f"(x{num_res_blocks} blocks, mlp_ratio={mlp_ratio}) -> {num_tokens} x {projection_dim}"
    )
    return projector, num_tokens
