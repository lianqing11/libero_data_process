"""
Mixture-of-Transformers (MoT) forward logic for VLM + Action Expert.

Since prefix tokens never attend to suffix tokens, the VLM prefix forward is
completely independent. We reuse Qwen3-VL's native language_model.forward()
with use_cache=True to get per-layer KV, then run the ActionExpert suffix
using the cached VLM KV.

Training:  language_model(prefix, use_cache=True) -> kv_cache -> suffix forward
Inference: same split, caller wraps in @torch.no_grad()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_expert import ActionExpert

# Global cache for IC suffix-to-suffix mask (constant for given K, T, device)
_S2S_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


class MixtureOfTransformers:
    """
    Shared-attention MoT that interleaves VLM (Qwen3-VL) and ActionExpert layers.

    Plain Python class (NOT nn.Module) -- has no learnable parameters of its own.
    VLM and ActionExpert are owned by the parent XVLA model; this class only
    holds references and orchestrates the forward pass.
    """

    def __init__(self, vlm: nn.Module, action_expert: ActionExpert, backbone_type: Optional[str] = None):
        self.vlm = vlm
        self.backbone_type = backbone_type or getattr(vlm.config, "model_type", "")
        lang_model = vlm.model.language_model
        self.vlm_language_model = lang_model
        self.rotary_emb = lang_model.rotary_emb
        self.action_expert = action_expert
        self.num_layers = len(lang_model.layers)

        assert self.num_layers == action_expert.num_layers, (
            f"VLM has {self.num_layers} layers but ActionExpert has "
            f"{action_expert.num_layers}. Set expert_num_layers={self.num_layers} in config."
        )

        text_config = vlm.config.text_config
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        suffix_embeds: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        suffix_mask: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
    ) -> torch.Tensor:
        """
        MoT forward (training).

        Phase 1: native VLM forward on prefix -> cached KV (gradients preserved).
        Phase 2: ActionExpert suffix forward using cached VLM KV.
        """
        kv_cache = self.run_prefix(
            prefix_embeds,
            prefix_pad_mask,
            position_ids,
            prefix_attention_mask=prefix_attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return self.forward_suffix_with_cache(
            suffix_embeds, kv_cache, prefix_pad_mask, suffix_mask,
        )

    def run_prefix(
        self,
        prefix_embeds: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        *,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run the VLM prefix and return per-layer KV cache."""
        attention_mask = prefix_attention_mask if prefix_attention_mask is not None else prefix_pad_mask
        forward_kwargs = {
            "inputs_embeds": prefix_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "return_dict": True,
        }
        if self.backbone_type == "paligemma":
            outputs = self.vlm.model(**forward_kwargs)
        else:
            forward_kwargs["visual_pos_masks"] = visual_pos_masks
            forward_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
            outputs = self.vlm_language_model(**forward_kwargs)
        return [outputs.past_key_values[i] for i in range(self.num_layers)]

    def forward_suffix_with_cache(
        self,
        suffix_embeds: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_mask: Optional[torch.Tensor] = None,
        suffix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run ActionExpert suffix forward reusing cached VLM KV.

        Args:
            suffix_embeds: [B, L_suf, expert_hidden_size]
            kv_cache: list of (K_vlm, V_vlm) per layer from run_vlm_prefix
            prefix_pad_mask: optional [B, L_pre] with 1=valid, 0=pad
            suffix_mask: optional [B, L_suf, L_pre + L_suf] block-causal mask for IC
        """
        kv_dtype = kv_cache[0][0].dtype
        suffix_embeds = suffix_embeds.to(dtype=kv_dtype)

        B, L_suf, _ = suffix_embeds.shape
        L_pre = kv_cache[0][0].shape[2]
        device = suffix_embeds.device

        if prefix_pad_mask is None:
            prefix_pad_mask = torch.ones(B, L_pre, device=device, dtype=torch.long)
        max_pos = prefix_pad_mask.sum(dim=1)
        suffix_offsets = torch.arange(L_suf, device=device)
        if self.backbone_type == "paligemma":
            suffix_pos = max_pos.unsqueeze(1) + suffix_offsets.unsqueeze(0) + 1
            rotary_position_ids = suffix_pos
        else:
            suffix_pos = max_pos.unsqueeze(1) + suffix_offsets.unsqueeze(0)
            rotary_position_ids = suffix_pos.unsqueeze(0).expand(3, B, L_suf)

        dummy = suffix_embeds.new_empty(B, L_suf, 1)
        cos, sin = self.rotary_emb(dummy, rotary_position_ids)
        cos = cos.to(dtype=suffix_embeds.dtype)
        sin = sin.to(dtype=suffix_embeds.dtype)

        if suffix_mask is not None:
            mask = suffix_mask.unsqueeze(1).to(dtype=suffix_embeds.dtype)
        else:
            suffix_causal = torch.zeros(L_suf, L_suf, device=device, dtype=suffix_embeds.dtype)
            suffix_to_prefix = torch.zeros(B, L_suf, L_pre, device=device, dtype=suffix_embeds.dtype)
            suffix_to_prefix.masked_fill_(~prefix_pad_mask[:, None, :].bool(), float("-inf"))
            mask = torch.cat(
                [suffix_to_prefix, suffix_causal.unsqueeze(0).expand(B, -1, -1)], dim=2
            ).unsqueeze(1)

        # Pre-allocate KV buffers for inference (no grad -> slice-assign is safe).
        # During training, use torch.cat to preserve gradient flow to VLM KV cache.
        use_buf = not torch.is_grad_enabled()
        if use_buf:
            L_total = L_pre + L_suf
            k_buf = kv_cache[0][0].new_empty(B, self.num_kv_heads, L_total, self.head_dim)
            v_buf = kv_cache[0][0].new_empty(B, self.num_kv_heads, L_total, self.head_dim)

        hidden_act = suffix_embeds

        for i in range(self.num_layers):
            expert_layer = self.action_expert.layers[i]
            cached_k_vlm, cached_v_vlm = kv_cache[i]

            q_act, k_act, v_act = expert_layer.compute_qkv(hidden_act)
            q_act = q_act.transpose(1, 2)
            k_act = k_act.transpose(1, 2)
            v_act = v_act.transpose(1, 2)

            q_act, k_act = _apply_rotary_pos_emb(q_act, k_act, cos, sin)

            if use_buf:
                k_buf[:, :, :L_pre, :] = cached_k_vlm
                k_buf[:, :, L_pre:, :] = k_act
                v_buf[:, :, :L_pre, :] = cached_v_vlm
                v_buf[:, :, L_pre:, :] = v_act
                k_full, v_full = k_buf, v_buf
            else:
                k_full = torch.cat([cached_k_vlm, k_act], dim=2)
                v_full = torch.cat([cached_v_vlm, v_act], dim=2)

            attn_out = F.scaled_dot_product_attention(
                q_act, k_full, v_full, attn_mask=mask, scale=self.scaling,
                enable_gqa=True,
            )

            attn_act = attn_out.transpose(1, 2).reshape(B, L_suf, self.num_heads * self.head_dim)
            hidden_act = expert_layer.post_attention_ffn(hidden_act, attn_act)

        return self.action_expert.decode(hidden_act)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings. cos/sin: [B, L, head_dim], Q/K: [B, heads, L, head_dim]."""
    cos = cos.unsqueeze(1)  # [B, 1, L, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_ic_suffix_mask(
    K: int,
    T: int,
    L_pre: int,
    frame_obs_ends: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build block-causal suffix mask for IC training (vectorized).

    Each suffix token in frame i attends to:
      - prefix tokens [0, frame_obs_ends[i]) for its sample
      - all suffix tokens belonging to frame i

    The suffix-to-suffix block (constant for given K, T) is cached globally.

    Returns [B, K*T, L_pre + K*T] float mask with 0=attend, -inf=block.
    """
    B = frame_obs_ends.shape[0]
    device = frame_obs_ends.device

    cache_key = (K, T, device)
    if cache_key not in _S2S_CACHE:
        frame_idx = torch.arange(K, device=device).repeat_interleave(T)
        s2s = frame_idx[None, :] == frame_idx[:, None]  # [K*T, K*T]
        _S2S_CACHE[cache_key] = (frame_idx, s2s)
    frame_idx, s2s = _S2S_CACHE[cache_key]

    pos_range = torch.arange(L_pre, device=device)
    ends = frame_obs_ends[:, frame_idx]  # [B, K*T]
    prefix_attend = (pos_range[None, None, :] < ends[:, :, None]) & pad_mask[:, None, :].bool()

    full_attend = torch.cat([prefix_attend, s2s.unsqueeze(0).expand(B, -1, -1)], dim=2)
    mask = torch.where(full_attend, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device))
    return mask
