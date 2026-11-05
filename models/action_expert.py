"""
Action Expert for Mixture-of-Transformers (MoT) action head.

The Action Expert is a lightweight transformer that shares attention with the VLM
backbone (Qwen3-VL) but uses its own FFN.
  - Expert hidden_size (e.g., 1024) can differ from VLM hidden_size (2048)
  - QKV projections bridge the gap: both map to the same attention space
    (same num_heads, num_kv_heads, head_dim)
  - Each expert has independent FFN operating at its own hidden_size

"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Sinusoidal timestep embedding. t: [B], returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ActionExpertLayer(nn.Module):
    """
    Single action expert layer, paired 1:1 with a VLM decoder layer for MoT.

    Mirrors Qwen3VLTextDecoderLayer structure (pre-norm, QKV + QK-norm, SwiGLU FFN)
    but with its own hidden_size. Shared attention requires matching
    num_heads, num_kv_heads, and head_dim with the VLM.

    QKV projections bridge dimensions:
      q_proj: Linear(hidden_size, num_heads * head_dim)      e.g. 1024 -> 2048
      k_proj: Linear(hidden_size, num_kv_heads * head_dim)   e.g. 1024 -> 1024
      o_proj: Linear(num_heads * head_dim, hidden_size)       e.g. 2048 -> 1024
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.input_layernorm = nn.RMSNorm(hidden_size)
        # QKV projections (project from expert hidden to shared attention space)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        # QK-Norm is needed for Qwen3-VL, but Gemma/PaliGemma attention has no QK norm.
        self.q_norm = nn.RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if use_qk_norm else nn.Identity()

        # Output projection (project from attention space back to expert hidden)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Pre-FFN norm
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

        # SwiGLU FFN (matching Qwen3 MLP structure)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def compute_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 1: Norm + QKV projection + QK-norm.

        Returns Q, K, V in shape [B, L, num_heads/num_kv_heads, head_dim] for concatenation
        with VLM QKV in the MoT forward loop.
        """
        B, L, _ = hidden_states.shape
        normed = self.input_layernorm(hidden_states)

        q = self.q_proj(normed).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(normed).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(normed).view(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def post_attention_ffn(self, hidden_states: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Phase 2: o_proj + residual + norm + SwiGLU FFN + residual.

        Args:
            hidden_states: [B, L, hidden_size] pre-attention hidden states (for residual)
            attn_output: [B, L, num_heads * head_dim] raw attention output (before o_proj)
        """
        # Attention residual
        hidden_states = hidden_states + self.o_proj(attn_output)

        # FFN
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))
        hidden_states = residual + hidden_states

        return hidden_states


class ActionExpert(nn.Module):
    """
    Complete Action Expert module for MoT.

    Components:
      - action_in_proj: projects noisy actions [B, T, dim_action] -> [B, T, hidden_size]
      - time_mlp: timestep embedding -> [B, hidden_size]
      - layers: N ActionExpertLayers (paired with VLM layers in MoT forward)
      - norm + action_out_proj: final prediction [B, T, hidden_size] -> [B, T, dim_action]
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 4096,
        dim_action: int = 20,
        dim_time: int = 32,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.num_layers = num_layers

        # Suffix embedding: noisy action -> hidden_size
        self.action_in_proj = nn.Linear(dim_action, hidden_size)

        # Timestep MLP: sinusoidal_emb(t) -> hidden_size
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_time, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            ActionExpertLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        ])

        # Final norm + output projection
        self.norm = nn.RMSNorm(hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, dim_action)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init output projection so initial prediction is zero (stable start for flow matching)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    def embed_suffix(self, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Build suffix embeddings from noisy actions + timestep.

        Args:
            noisy_action: [B, T_action, dim_action] noisy action sequence
            t: [B] timestep values in [0, 1)

        Returns:
            [B, T_action, hidden_size] suffix embeddings ready for MoT forward
        """
        dtype = self.action_in_proj.weight.dtype
        noisy_action = noisy_action.to(dtype=dtype)
        t = t.to(dtype=dtype)
        B, T = noisy_action.shape[:2]

        action_emb = self.action_in_proj(noisy_action)  # [B, T, hidden_size]

        time_emb = timestep_embedding(t, self.dim_time)  # [B, dim_time]
        time_emb = self.time_mlp(time_emb)  # [B, hidden_size]
        time_emb = time_emb.unsqueeze(1).expand(B, T, self.hidden_size)  # [B, T, hidden_size]

        return action_emb + time_emb

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Final norm + projection: [B, T, hidden_size] -> [B, T, dim_action]."""
        return self.action_out_proj(self.norm(hidden_states))
