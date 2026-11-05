"""
XVLA: Vision-Language-Action Model (Qwen3-VL + MoT).

VLM backbone: Qwen3-VL (decoder-only, 2B params)
Action head: Mixture-of-Transformers (MoT) — shared attention between VLM and ActionExpert.
"""

from __future__ import annotations

import logging
import math
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from transformers import PreTrainedModel
from PIL import Image
import json_numpy
import cv2

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import torch.nn as nn
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .hf_local import resolve_pretrained_load
from .proprio_projector import init_proprio_projector, init_action_embedder


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype_str, torch.bfloat16)


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      - Qwen3-VL backbone
      - MoT (Mixture-of-Transformers) action head with shared attention
      - Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, processor=None, _skip_vlm_weights=False, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self._processor = processor
        self.vlm_backbone_type = getattr(config, "vlm_backbone_type", "qwen3_vl")
        self.vlm_process_all_views = getattr(config, "vlm_process_all_views", False)

        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action

        # Initialize VLM backbone
        projection_dim = self._init_vlm(config, skip_weights=_skip_vlm_weights)

        # MoT action head
        from .action_expert import ActionExpert
        from .mot_forward import MixtureOfTransformers
        vlm_text_config = self.vlm.config.text_config
        use_qk_norm = getattr(config, "expert_use_qk_norm", None)
        if use_qk_norm is None:
            use_qk_norm = self.vlm_backbone_type != "paligemma"
        self.action_expert = ActionExpert(
            hidden_size=config.expert_hidden_size,
            num_layers=config.expert_num_layers,
            num_heads=vlm_text_config.num_attention_heads,
            num_kv_heads=vlm_text_config.num_key_value_heads,
            head_dim=getattr(vlm_text_config, "head_dim",
                             vlm_text_config.hidden_size // vlm_text_config.num_attention_heads),
            intermediate_size=config.expert_intermediate_size,
            dim_action=dim_action,
            dim_time=config.dim_time,
            use_qk_norm=use_qk_norm,
        )
        self.mot = MixtureOfTransformers(self.vlm, self.action_expert, backbone_type=self.vlm_backbone_type)
        self.action_embedder = init_action_embedder(self.action_space, config, projection_dim)
        logging.info(f"MoT action head: expert_hidden={config.expert_hidden_size}, "
                     f"layers={config.expert_num_layers}, intermediate={config.expert_intermediate_size}")

        # Learnable frame position embedding for IC: additive signal on all tokens per frame
        num_ic = getattr(config, 'num_ic_frames', 0)
        if num_ic > 0 and getattr(config, 'use_frame_position_embed', False):
            self.frame_position_embed = nn.Embedding(num_ic, projection_dim)
            nn.init.normal_(self.frame_position_embed.weight, std=0.02)
            logging.info(f"Frame position embedding: {num_ic} frames x {projection_dim} dim")
        else:
            self.frame_position_embed = None

        self.freeze_vision_encoder: bool = False

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ======================= Backbone Initialization =======================
    def _init_vlm(self, config: XVLAConfig, skip_weights: bool = False) -> int:
        backbone = getattr(config, "vlm_backbone_type", "qwen3_vl")
        if backbone == "qwen3_vl":
            return self._init_qwen3vl(config, skip_weights=skip_weights)
        if backbone == "paligemma":
            return self._init_paligemma(config, skip_weights=skip_weights)
        raise ValueError(f"Unsupported vlm_backbone_type: {backbone}")

    def _init_qwen3vl(self, config: XVLAConfig, skip_weights: bool = False) -> int:
        from transformers import Qwen3VLForConditionalGeneration

        pretrained_path = config.qwen3_pretrained
        resolved_path, load_kwargs = resolve_pretrained_load(
            pretrained_path,
            required_files=("config.json", "tokenizer_config.json", "preprocessor_config.json"),
            trust_remote_code=True,
        )
        torch_dtype = _get_torch_dtype(config.qwen3_torch_dtype)
        use_flash_attn = config.qwen3_use_flash_attn
        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

        if skip_weights:
            from transformers import AutoConfig
            logging.info(f"Creating Qwen3-VL skeleton from {resolved_path} (weights from checkpoint)")
            vlm_config = AutoConfig.from_pretrained(resolved_path, **load_kwargs)
            vlm_config._attn_implementation = attn_impl
            vlm_config.torch_dtype = torch_dtype
            self.vlm = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        else:
            logging.info(f"Loading Qwen3-VL from {resolved_path}")
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                resolved_path,
                dtype=torch_dtype,
                attn_implementation=attn_impl,
                **load_kwargs,
            )

        if self._processor is not None:
            self.qwen3_tokenizer = self._processor.tokenizer
        else:
            from transformers import AutoTokenizer
            self.qwen3_tokenizer = AutoTokenizer.from_pretrained(resolved_path, **load_kwargs)
            logging.info("Loaded tokenizer directly (will be updated when processor is set)")

        self.proprio_to_vlm = config.proprio_to_vlm
        projection_dim = self.vlm.config.text_config.hidden_size

        if self._processor is not None:
            self._cache_special_token_ids()
        else:
            self.qwen3_image_pad_id = None
            self.qwen3_proprio_pad_id = None
            self.qwen3_frame_start_id = None
            self.qwen3_frame_end_id = None

        if self.proprio_to_vlm:
            self.proprio_projector, self.num_proprio_tokens = init_proprio_projector(
                self.action_space, config, projection_dim, default_num_tokens=8, default_hidden=512
            )

        embed_tokens = self.vlm.model.language_model.embed_tokens
        current_vocab_size = embed_tokens.weight.shape[0]
        target_vocab_size = len(self.qwen3_tokenizer)

        if target_vocab_size > current_vocab_size:
            is_meta = embed_tokens.weight.device.type == "meta"
            if is_meta:
                self.vlm.resize_token_embeddings(target_vocab_size, mean_resizing=False)
                logging.info(f"Resized embeddings for checkpoint loading (special tokens)")
            else:
                self.vlm.resize_token_embeddings(target_vocab_size)
                logging.info(f"Resized embeddings from {current_vocab_size} to {target_vocab_size} for special tokens")
                init_pairs = self._processor.get_tokens_for_embedding_init()
                with torch.no_grad():
                    for token_name, ref_name in init_pairs:
                        token_id = self.qwen3_tokenizer.convert_tokens_to_ids(token_name)
                        ref_id = self.qwen3_tokenizer.convert_tokens_to_ids(ref_name)
                        if token_id >= 0 and ref_id >= 0:
                            embed_tokens.weight[token_id] = (
                                embed_tokens.weight[ref_id].clone() +
                                torch.randn_like(embed_tokens.weight[ref_id]) * 0.02
                            )
                            logging.info(f"Initialized {token_name} from {ref_name}")

        return projection_dim

    def _init_paligemma(self, config: XVLAConfig, skip_weights: bool = False) -> int:
        from transformers import PaliGemmaForConditionalGeneration

        pretrained_path = config.paligemma_pretrained
        resolved_path, load_kwargs = resolve_pretrained_load(
            pretrained_path,
            required_files=("config.json", "tokenizer_config.json", "preprocessor_config.json"),
            trust_remote_code=True,
        )
        torch_dtype = _get_torch_dtype(config.paligemma_torch_dtype)
        attn_impl = "flash_attention_2" if config.paligemma_use_flash_attn else "sdpa"

        if skip_weights:
            from transformers import AutoConfig
            logging.info(f"Creating PaliGemma skeleton from {resolved_path} (weights from checkpoint)")
            vlm_config = AutoConfig.from_pretrained(resolved_path, **load_kwargs)
            vlm_config._attn_implementation = attn_impl
            vlm_config.torch_dtype = torch_dtype
            self.vlm = PaliGemmaForConditionalGeneration._from_config(vlm_config)
        else:
            logging.info(f"Loading PaliGemma from {resolved_path}")
            self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                resolved_path,
                dtype=torch_dtype,
                attn_implementation=attn_impl,
                **load_kwargs,
            )

        if self._processor is not None:
            self.qwen3_tokenizer = self._processor.tokenizer
        else:
            from transformers import AutoTokenizer
            self.qwen3_tokenizer = AutoTokenizer.from_pretrained(resolved_path, **load_kwargs)
            logging.info("Loaded tokenizer directly (will be updated when processor is set)")

        self.proprio_to_vlm = config.proprio_to_vlm
        projection_dim = self.vlm.config.text_config.hidden_size

        if self._processor is not None:
            self._cache_special_token_ids()
        else:
            self.qwen3_image_pad_id = None
            self.qwen3_proprio_pad_id = None
            self.qwen3_frame_start_id = None
            self.qwen3_frame_end_id = None

        if self.proprio_to_vlm:
            self.proprio_projector, self.num_proprio_tokens = init_proprio_projector(
                self.action_space, config, projection_dim, default_num_tokens=8, default_hidden=512
            )

        embed_tokens = self.vlm.model.language_model.embed_tokens
        current_vocab_size = embed_tokens.weight.shape[0]
        target_vocab_size = len(self.qwen3_tokenizer)

        if target_vocab_size > current_vocab_size:
            is_meta = embed_tokens.weight.device.type == "meta"
            if is_meta:
                self.vlm.resize_token_embeddings(target_vocab_size, mean_resizing=False)
                logging.info("Resized PaliGemma embeddings for checkpoint loading (special tokens)")
            else:
                self.vlm.resize_token_embeddings(target_vocab_size)
                embed_tokens = self.vlm.model.language_model.embed_tokens
                logging.info(f"Resized PaliGemma embeddings from {current_vocab_size} to {target_vocab_size}")
                self._init_added_token_embeddings(embed_tokens)
        elif self._processor is not None and embed_tokens.weight.device.type != "meta":
            self._init_added_token_embeddings(embed_tokens)

        return projection_dim

    def _init_added_token_embeddings(self, embed_tokens) -> None:
        init_pairs = self._processor.get_tokens_for_embedding_init()
        with torch.no_grad():
            for token_name, ref_name in init_pairs:
                token_id = self.qwen3_tokenizer.convert_tokens_to_ids(token_name)
                ref_id = self.qwen3_tokenizer.convert_tokens_to_ids(ref_name)
                if token_id >= 0 and ref_id >= 0:
                    embed_tokens.weight[token_id] = (
                        embed_tokens.weight[ref_id].clone() +
                        torch.randn_like(embed_tokens.weight[ref_id]) * 0.02
                    )
                    logging.info(f"Initialized {token_name} from {ref_name}")

    def _cache_special_token_ids(self) -> None:
        """Cache special token IDs from tokenizer for efficient lookup."""
        if self.vlm_backbone_type == "paligemma":
            self.qwen3_image_pad_id = getattr(self._processor, "image_token_id", None)
        else:
            self.qwen3_image_pad_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.qwen3_action_pad_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|action_pad|>")
        self.qwen3_frame_start_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|frame_start|>")
        self.qwen3_frame_end_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|frame_end|>")
        if self.proprio_to_vlm:
            self.qwen3_proprio_pad_id = self.qwen3_tokenizer.convert_tokens_to_ids("<|proprio_pad|>")
            logging.info(f"Qwen3-VL special tokens: image_pad_id={self.qwen3_image_pad_id}, "
                        f"proprio_pad_id={self.qwen3_proprio_pad_id}")
        else:
            self.qwen3_proprio_pad_id = None

    def set_processor(self, processor) -> None:
        """Update processor after model init (needed when loading from checkpoint)."""
        self._processor = processor
        if processor is None:
            return
        self.qwen3_tokenizer = processor.tokenizer
        self._cache_special_token_ids()
        if hasattr(processor, 'action_tokenizer') and processor.action_tokenizer is not None:
            self.action_tokenizer = processor.action_tokenizer
        logging.info("Updated model with processor (tokenizer and special token IDs)")
        processor.num_proprio_tokens = getattr(self, 'num_proprio_tokens', 1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load XVLA, building Qwen3-VL processor first to ensure correct vocab size."""
        from .configuration_xvla import XVLAConfig
        from .processing_xvla import build_xvla_processor

        config = XVLAConfig.from_pretrained(pretrained_model_name_or_path)
        backbone_type = getattr(config, "vlm_backbone_type", "qwen3_vl")
        proprio_to_vlm = getattr(config, "proprio_to_vlm", False)
        vlm_path = (
            getattr(config, "paligemma_pretrained", "google/paligemma-3b-pt-224")
            if backbone_type == "paligemma"
            else getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
        )
        num_views = getattr(config, "num_views", 2)
        vlm_process_all_views = getattr(config, "vlm_process_all_views", False)

        processor = build_xvla_processor(
            pretrained_name_or_path=vlm_path,
            backbone_type=backbone_type,
            num_views=num_views,
            proprio_to_vlm=proprio_to_vlm,
            vlm_process_all_views=vlm_process_all_views,
        )
        logging.info(f"Built {backbone_type} processor for checkpoint loading (vocab_size={len(processor.tokenizer)})")
        kwargs['processor'] = processor

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_processor(processor)
        return model

    # ======================= Shared Prefix Helpers =======================

    def _language_model(self):
        return self.vlm.model.language_model

    def _effective_image_mask(self, image_mask: torch.Tensor) -> torch.Tensor:
        if image_mask.dim() < 2 or image_mask.shape[-1] != getattr(self.config, "num_views", 2):
            return image_mask
        if self.vlm_process_all_views:
            return image_mask
        mask = torch.zeros_like(image_mask)
        mask[..., 0] = image_mask[..., 0]
        return mask

    def _get_text_embeds_and_mask(self, input_ids):
        """Embed tokens and build attention mask. Returns (inputs_embeds, attention_mask, dtype, hidden_dim)."""
        lang_model = self._language_model()
        embed_ids = input_ids
        if self.vlm_backbone_type == "paligemma":
            image_token_id = self.vlm.config.image_token_id
            if image_token_id >= lang_model.embed_tokens.weight.shape[0]:
                embed_ids = input_ids.masked_fill(input_ids == image_token_id, 0)
        inputs_embeds = lang_model.embed_tokens(embed_ids)
        dtype = next(lang_model.parameters()).dtype
        pad_id = self.qwen3_tokenizer.pad_token_id
        mask = torch.ones_like(input_ids, dtype=torch.long) if pad_id is None else (input_ids != pad_id).long()
        return inputs_embeds, mask, dtype, inputs_embeds.shape[-1]

    def _encode_valid_images(self, pixel_values, image_mask, image_grid_thw):
        """Flatten views, filter by mask, encode. Returns (embeds_list, deepstack, valid_grid_thw, flat_mask)."""
        shape = pixel_values.shape
        if self.vlm_backbone_type == "paligemma":
            flat = pixel_values.reshape(-1, *shape[-3:])
        else:
            flat = pixel_values.reshape(-1, *shape[-2:])
        image_mask = self._effective_image_mask(image_mask)
        flat_mask = image_mask.reshape(-1).bool()
        valid = flat[flat_mask]
        vg = image_grid_thw.reshape(-1, 3)[flat_mask] if image_grid_thw is not None else None
        qwen_valid = valid
        if self.vlm_backbone_type != "paligemma" and vg is not None:
            pieces = []
            for patches, grid in zip(valid, vg):
                num_patches = int(grid.prod().item())
                if num_patches > patches.shape[0]:
                    raise ValueError(
                        f"image_grid_thw expects {num_patches} patches, "
                        f"but encoded image only has {patches.shape[0]}"
                    )
                pieces.append(patches[:num_patches])
            qwen_valid = torch.cat(pieces, dim=0) if pieces else valid.reshape(0, valid.shape[-1])
        if self.freeze_vision_encoder:
            with torch.no_grad():
                if self.vlm_backbone_type == "paligemma":
                    image_features = self.vlm.get_image_features(
                        valid.to(dtype=next(self.vlm.model.vision_tower.parameters()).dtype)
                    )
                    embeds_list, deepstack = list(image_features), None
                else:
                    embeds_list, deepstack = self.vlm.get_image_features(
                        qwen_valid.reshape(-1, qwen_valid.shape[-1]), vg,
                    )
            embeds_list = [e.detach() for e in embeds_list]
            if deepstack is not None:
                deepstack = [d.detach() for d in deepstack]
        else:
            if self.vlm_backbone_type == "paligemma":
                image_features = self.vlm.get_image_features(
                    valid.to(dtype=next(self.vlm.model.vision_tower.parameters()).dtype)
                )
                embeds_list, deepstack = list(image_features), None
            else:
                embeds_list, deepstack = self.vlm.get_image_features(
                    qwen_valid.reshape(-1, qwen_valid.shape[-1]), vg,
                )
        return embeds_list, deepstack, vg, flat_mask

    def _scatter_into_embeds(self, inputs_embeds, input_ids, token_id, flat_features, dtype):
        """Scatter flat_features into positions where input_ids == token_id."""
        mask = (input_ids == token_id)
        return inputs_embeds.masked_scatter(
            mask.unsqueeze(-1).expand_as(inputs_embeds),
            flat_features.to(dtype=dtype),
        ), mask

    def _finish_prefix(self, inputs_embeds, input_ids, grid_thw, attention_mask, dtype):
        """Cast to dtype, compute M-RoPE positions. Returns (prefix_embeds, position_ids)."""
        prefix_embeds = inputs_embeds.to(dtype=dtype)
        if self.vlm_backbone_type == "paligemma":
            position_ids = torch.cumsum(attention_mask, dim=1) * attention_mask
            return prefix_embeds, position_ids.long()
        if grid_thw is not None:
            grid_thw = grid_thw.to(device=input_ids.device, dtype=torch.long)
        position_ids, _ = self.vlm.model.get_rope_index(input_ids, grid_thw, None, attention_mask)
        return prefix_embeds, position_ids

    def _prefix_forward_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return attention_mask

    # ======================= Non-IC Prefix =======================

    def _prepare_prefix_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
        proprio: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Prepare VLM prefix for single-frame path."""
        inputs_embeds, attention_mask, dtype, hidden_dim = self._get_text_embeds_and_mask(input_ids)

        embeds_list, deepstack_all, valid_grid_thw, _ = self._encode_valid_images(
            pixel_values, image_mask,
            image_grid_thw if image_grid_thw is not None and image_grid_thw.dim() == 3 else None,
        )

        prefix_feats_flat = torch.cat(embeds_list, dim=0)
        inputs_embeds, image_pad_mask = self._scatter_into_embeds(
            inputs_embeds, input_ids, self.vlm.config.image_token_id, prefix_feats_flat, dtype,
        )
        if proprio is not None and self.proprio_to_vlm:
            proj_device = next(self.proprio_projector.parameters()).device
            projected = self.proprio_projector(proprio.to(device=proj_device, dtype=dtype))
            inputs_embeds, _ = self._scatter_into_embeds(
                inputs_embeds, input_ids, self.qwen3_proprio_pad_id,
                projected.view(-1, hidden_dim), dtype,
            )

        prefix_embeds, position_ids = self._finish_prefix(
            inputs_embeds, input_ids, valid_grid_thw, attention_mask, dtype,
        )
        return {
            'prefix_embeds': prefix_embeds,
            'pad_mask': attention_mask,
            'prefix_attention_mask': self._prefix_forward_mask(attention_mask, dtype),
            'position_ids': position_ids,
            'visual_pos_masks': image_pad_mask,
            'deepstack': deepstack_all,
        }

    # ======================= IC Prefix =======================

    def _prepare_ic_prefix_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
        proprios: torch.Tensor,
        ic_actions: Optional[torch.Tensor],
        frame_obs_ends: torch.Tensor,
        next_pixel_values: Optional[torch.FloatTensor] = None,
        next_image_grid_thw: Optional[torch.LongTensor] = None,
        next_proprios: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare K-frame IC prefix: encode images, scatter image/proprio/action_emb
        into pre-expanded placeholders, compute M-RoPE positions.
        """
        B, K, V = pixel_values.shape[:3]
        image_tail = pixel_values.shape[3:]

        effective_image_mask = self._effective_image_mask(image_mask)

        if next_pixel_values is not None:
            main_per_frame = pixel_values.view(B, K, V, *image_tail)
            next_per_frame = next_pixel_values.view(B, K - 1, V, *image_tail)
            main_mask_per = effective_image_mask[:, None, :].expand(B, K, V)
            next_mask_per = effective_image_mask[:, None, :].expand(B, K - 1, V)

            pv_parts, mask_parts, grid_parts = [], [], []
            main_grid = image_grid_thw.view(B, K, V, 3) if image_grid_thw is not None else None
            next_grid = next_image_grid_thw.view(B, K - 1, V, 3) if next_image_grid_thw is not None else None

            for k in range(K):
                pv_parts.append(main_per_frame[:, k])
                mask_parts.append(main_mask_per[:, k])
                if main_grid is not None:
                    grid_parts.append(main_grid[:, k])
                if k < K - 1:
                    pv_parts.append(next_per_frame[:, k])
                    mask_parts.append(next_mask_per[:, k])
                    if next_grid is not None:
                        grid_parts.append(next_grid[:, k])

            # Stack along dim=1 to preserve sample-major ordering for masked_scatter.
            # torch.cat([p.reshape(B*V,...), ...], dim=0) would give frame-major order,
            # which misaligns with masked_scatter's sample-first traversal when B>1.
            F = len(pv_parts)  # 2K-1 interleaved frame groups
            flat_pv = torch.stack(pv_parts, dim=1).reshape(B * F * V, *image_tail)
            flat_img_mask = torch.stack(mask_parts, dim=1).reshape(B * F * V)
            combined_grid = torch.stack(grid_parts, dim=1).reshape(
                -1, 3
            ) if grid_parts else None
        else:
            flat_img_mask = effective_image_mask[:, None, :].expand(B, K, V).reshape(B * K * V)
            flat_pv = pixel_values.reshape(B * K * V, *image_tail)
            combined_grid = image_grid_thw.view(B * K * V, 3) if image_grid_thw is not None else None

        inputs_embeds, attention_mask, dtype, hidden_dim = self._get_text_embeds_and_mask(input_ids)

        embeds_list, deepstack_all, valid_grid_thw, _ = self._encode_valid_images(
            flat_pv, flat_img_mask, combined_grid,
        )

        inputs_embeds, image_pad_mask = self._scatter_into_embeds(
            inputs_embeds, input_ids, self.vlm.config.image_token_id,
            torch.cat(embeds_list, dim=0), dtype,
        )

        if proprios is not None and self.proprio_to_vlm:
            proj_device = next(self.proprio_projector.parameters()).device
            if next_proprios is not None:
                main_p = proprios.view(B, K, -1)
                next_p = next_proprios.view(B, K - 1, -1)
                parts = []
                for k in range(K):
                    parts.append(main_p[:, k])
                    if k < K - 1:
                        parts.append(next_p[:, k])
                # Stack dim=1 for sample-major order (matches masked_scatter traversal)
                all_proprios = torch.stack(parts, dim=1).reshape(B * len(parts), -1)
            else:
                all_proprios = proprios.reshape(B * K, -1)
            projected = self.proprio_projector(all_proprios.to(device=proj_device, dtype=dtype))
            inputs_embeds, _ = self._scatter_into_embeds(
                inputs_embeds, input_ids, self.qwen3_proprio_pad_id,
                projected.view(-1, hidden_dim), dtype,
            )

        if ic_actions is not None and self.action_embedder is not None:
            emb_device = next(self.action_embedder.parameters()).device
            action_emb = self.action_embedder(ic_actions.to(device=emb_device, dtype=dtype))
            inputs_embeds, _ = self._scatter_into_embeds(
                inputs_embeds, input_ids, self.qwen3_action_pad_id,
                action_emb.reshape(-1, hidden_dim), dtype,
            )

        # Add learnable frame position embedding to all tokens within each frame
        if self.frame_position_embed is not None:
            for b in range(B):
                starts = (input_ids[b] == self.qwen3_frame_start_id).nonzero(as_tuple=True)[0]
                ends = (input_ids[b] == self.qwen3_frame_end_id).nonzero(as_tuple=True)[0]
                n_frames = min(len(starts), len(ends), K)
                for k in range(n_frames):
                    frame_id = n_frames - 1 - k  # reverse: query=0, nearest demo=1, ...
                    s = starts[k].item()
                    e = ends[k].item() + 1  # inclusive of <|frame_end|>
                    inputs_embeds[b, s:e] = inputs_embeds[b, s:e] + \
                        self.frame_position_embed.weight[frame_id].to(dtype=dtype)

        prefix_embeds, position_ids = self._finish_prefix(
            inputs_embeds, input_ids, valid_grid_thw, attention_mask, dtype,
        )
        return {
            'prefix_embeds': prefix_embeds,
            'pad_mask': attention_mask,
            'prefix_attention_mask': self._prefix_forward_mask(attention_mask, dtype),
            'position_ids': position_ids,
            'frame_obs_ends': frame_obs_ends,
            'visual_pos_masks': image_pad_mask,
            'deepstack': deepstack_all,
        }

    # ================================= training =================================
    @staticmethod
    def _sample_timesteps(B: int, device: torch.device) -> torch.Tensor:
        """Stratified timestep sampling for flow matching: [B] in [0, 1)."""
        return (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - 1e-5)

    @staticmethod
    def _add_flow_noise(action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between noise and action: noise*t + action*(1-t)."""
        ndim = action.ndim - 1
        shape = (-1,) + (1,) * ndim
        return torch.randn_like(action) * t.view(*shape) + action * (1 - t).view(*shape)

    def _mot_forward_and_loss(
        self,
        p: Dict[str, torch.Tensor],
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        gt_action: torch.Tensor,
        suffix_mask: Optional[torch.Tensor] = None,
        frame_weights: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Shared MoT training: embed suffix -> mot.forward -> loss.

        If ``frame_weights`` is provided (IC re-weighting enabled), the flat
        [B, K*T, D] pred/gt tensors are split into K frame slices and their
        per-frame loss dicts are combined as a weighted mean (Σ w_k·L_k / Σ w_k).
        """
        suffix_embeds = self.action_expert.embed_suffix(noisy_action, t)
        pred = self.mot.forward(
            p['prefix_embeds'], suffix_embeds, p['pad_mask'], p['position_ids'],
            suffix_mask,
            prefix_attention_mask=p.get('prefix_attention_mask'),
            visual_pos_masks=p.get('visual_pos_masks'),
            deepstack_visual_embeds=p.get('deepstack'),
        )
        gt = gt_action.to(dtype=pred.dtype)
        if frame_weights is None or K is None:
            return self.action_space.compute_loss(pred, gt)

        B, KT, D = pred.shape
        assert KT == K * (KT // K), f"pred second dim {KT} not divisible by K={K}"
        T = KT // K
        pred_kt = pred.reshape(B, K, T, D)
        gt_kt = gt.reshape(B, K, T, D)
        w = frame_weights.to(device=pred.device, dtype=pred.dtype)
        w_sum = w.sum().clamp_min(1e-8)
        agg: Dict[str, torch.Tensor] = {}
        for k in range(K):
            losses_k = self.action_space.compute_loss(pred_kt[:, k], gt_kt[:, k])
            coef = w[k] / w_sum
            for key, val in losses_k.items():
                agg[key] = agg.get(key, 0) + coef * val
        return agg

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward. Routes to IC path when action.ndim == 4."""
        if action.ndim == 4:
            return self.forward_diffusion_mot_ic(
                input_ids, image_input, image_mask,
                proprio, action, image_grid_thw, **kwargs
            )
        return self.forward_diffusion_mot(
            input_ids, image_input, image_mask,
            proprio, action, image_grid_thw, **kwargs
        )

    def forward_diffusion_mot(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """MoT training forward: VLM + ActionExpert share attention at every layer."""
        B = input_ids.shape[0]
        t = self._sample_timesteps(B, input_ids.device)
        action_noisy = self._add_flow_noise(action, t)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        p = self._prepare_prefix_embeds(
            input_ids, image_input, image_mask, image_grid_thw, proprio_m
        )
        return self._mot_forward_and_loss(p, action_noisy_m, t, action)

    def forward_diffusion_mot_ic(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        IC training: K frames, all-frame suffix with block-causal mask.

        Shapes: image_input [B,K,V,N,C], proprio [B,K,D], action [B,K,T,D],
                image_grid_thw [B,K*V,3], frame_obs_ends [B,K] in kwargs.
        """
        from .mot_forward import build_ic_suffix_mask

        B, K, T, D = action.shape
        t = self._sample_timesteps(B, input_ids.device)
        noisy_all = self._add_flow_noise(action, t)

        # Batched preprocess: K noisy frames plus optional (K-1) GT IC actions.
        main_proprio = proprio.reshape(B * K, -1)
        main_action = noisy_all.reshape(B * K, T, D)
        if K > 1:
            ctx_proprio = proprio[:, :-1].reshape(B * (K - 1), proprio.shape[-1])
            ctx_action = action[:, :-1].reshape(B * (K - 1), T, D)
            all_proprio = torch.cat([main_proprio, ctx_proprio], dim=0)
            all_action = torch.cat([main_action, ctx_action], dim=0)
        else:
            all_proprio = main_proprio
            all_action = main_action
        all_proprio_m, all_action_m = self.action_space.preprocess(all_proprio, all_action)
        proprio_m = all_proprio_m[:B * K].reshape(B, K, -1)
        noisy_all_m = all_action_m[:B * K].reshape(B, K, T, D)
        ic_actions_m = all_action_m[B * K:].reshape(B, K - 1, T, D) if K > 1 else None

        frame_obs_ends = kwargs.get("frame_obs_ends")
        if frame_obs_ends is None:
            raise ValueError("forward_diffusion_mot_ic requires 'frame_obs_ends' in kwargs")

        next_image_input = kwargs.get("next_image_input")
        next_image_grid_thw = kwargs.get("next_image_grid_thw")
        next_proprios = None
        if next_image_input is not None and self.proprio_to_vlm:
            # Prefer the dataset-provided next_proprio (aligned with chunk-end frame
            # used for next_image). Fall back to proprio[:, 1:K] for older datasets.
            next_proprio_raw = kwargs.get("next_proprio")
            if next_proprio_raw is None:
                next_proprio_raw = proprio[:, 1:K]
            next_proprios_raw = next_proprio_raw.reshape(B * (K - 1), -1)
            next_proprios_m, _ = self.action_space.preprocess(
                next_proprios_raw,
                torch.zeros(B * (K - 1), T, D, device=proprio.device, dtype=proprio.dtype),
            )
            next_proprios = next_proprios_m.reshape(B, K - 1, -1)

        p = self._prepare_ic_prefix_embeds(
            input_ids, image_input, image_mask, image_grid_thw,
            proprio_m, ic_actions_m, frame_obs_ends,
            next_pixel_values=next_image_input,
            next_image_grid_thw=next_image_grid_thw,
            next_proprios=next_proprios,
        )

        suffix_mask = build_ic_suffix_mask(
            K, T, p['prefix_embeds'].shape[1], p['frame_obs_ends'], p['pad_mask'],
        )
        noisy_flat = noisy_all_m.reshape(B, K * T, D)
        gt_flat = action.reshape(B, K * T, D)

        frame_weights = None
        if getattr(self.config, 'ic_frame_loss_reweight', False) and K > 1:
            w_min = float(self.config.ic_frame_loss_weight_min)
            w_max = float(self.config.ic_frame_loss_weight_max)
            frame_weights = torch.linspace(
                w_min, w_max, K, device=action.device, dtype=torch.float32,
            )

        return self._mot_forward_and_loss(
            p, noisy_flat, t, gt_flat, suffix_mask,
            frame_weights=frame_weights, K=K,
        )

    # ================================= inference =================================
    def _build_ic_inference_suffix_mask(
        self,
        frame_obs_ends: torch.Tensor,
        pad_mask: torch.Tensor,
        L_pre: int,
    ) -> torch.Tensor:
        """Build suffix mask for IC inference that matches training attention pattern.

        During training, ``build_ic_suffix_mask`` restricts the query frame's action
        tokens to attend only to prefix positions **before** ``frame_obs_ends[K-1]``.
        Without this mask, inference lets the suffix attend to the full prefix (including
        ``<|frame_end|>`` and ``<|im_end|>``), creating a train/eval mismatch.

        Returns [B, T, L_pre + T] mask with 0 = attend, -inf = block.
        """
        B = frame_obs_ends.shape[0]
        T = self.num_actions
        device = frame_obs_ends.device

        query_end = frame_obs_ends[:, -1]  # [B]
        pos_range = torch.arange(L_pre, device=device)

        # Suffix-to-prefix: attend to positions < query_end (matching training)
        prefix_attend = (pos_range[None, None, :] < query_end[:, None, None])
        prefix_attend = prefix_attend & pad_mask[:, None, :].bool()
        prefix_attend = prefix_attend.expand(B, T, L_pre)

        # Suffix-to-suffix: full attention among T action tokens (same as training)
        suffix_attend = torch.ones(B, T, T, device=device, dtype=torch.bool)

        full_attend = torch.cat([prefix_attend, suffix_attend], dim=2)
        return torch.where(
            full_attend,
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )

    def _cache_and_denoise(
        self,
        p: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Shared VLM KV cache + flow-matching denoising loop for MoT inference."""
        B = proprio.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        kv_cache = self.mot.run_prefix(
            p['prefix_embeds'],
            p['pad_mask'],
            p['position_ids'],
            prefix_attention_mask=p.get('prefix_attention_mask'),
            visual_pos_masks=p.get('visual_pos_masks'),
            deepstack_visual_embeds=p.get('deepstack'),
        )

        # Build IC suffix mask to match training attention pattern
        suffix_mask = None
        frame_obs_ends = p.get('frame_obs_ends')
        if frame_obs_ends is not None:
            L_pre = p['prefix_embeds'].shape[1]
            suffix_mask = self._build_ic_inference_suffix_mask(
                frame_obs_ends, p['pad_mask'], L_pre,
            )

        x1 = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        action = torch.zeros_like(x1)
        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device, dtype=dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            _, x_t_m = self.action_space.preprocess(proprio, x_t)
            suffix_embeds = self.action_expert.embed_suffix(x_t_m, t)
            action = self.mot.forward_suffix_with_cache(
                suffix_embeds, kv_cache, p['pad_mask'], suffix_mask,
            )
        action = self.action_space.postprocess(action)
        return action

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor = None,
        proprio: torch.Tensor = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        steps: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions. Routes to IC path when ic_actions is provided."""
        ic_actions = kwargs.pop("ic_actions", None)
        if ic_actions is not None:
            frame_obs_ends = kwargs.pop("frame_obs_ends", None)
            return self.generate_actions_diffusion_mot_ic(
                input_ids, image_input, image_mask,
                proprio, ic_actions, frame_obs_ends,
                image_grid_thw, steps, **kwargs,
            )
        return self.generate_actions_diffusion_mot(
            input_ids, image_input, image_mask,
            proprio, image_grid_thw, steps,
        )

    @torch.no_grad()
    def generate_actions_diffusion_mot(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        steps: int = 10,
    ) -> torch.Tensor:
        """Generate actions via MoT iterative denoising with KV caching."""
        self.eval()
        proprio_m, _ = self.action_space.preprocess(
            proprio,
            torch.zeros(proprio.shape[0], self.num_actions, self.action_space.dim_action,
                        device=proprio.device, dtype=proprio.dtype),
        )
        p = self._prepare_prefix_embeds(
            input_ids, image_input, image_mask, image_grid_thw, proprio_m
        )
        return self._cache_and_denoise(p, proprio, steps)

    @torch.no_grad()
    def generate_actions_diffusion_mot_ic(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        ic_actions: torch.Tensor,
        frame_obs_ends: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        steps: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions with IC demonstrations via MoT + KV caching."""
        self.eval()
        B, K = proprio.shape[:2]
        T = self.num_actions
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        dummy_action = torch.zeros(B * K, T, D, device=device, dtype=dtype)
        proprio_m, _ = self.action_space.preprocess(
            proprio.reshape(B * K, -1), dummy_action,
        )
        proprio_m = proprio_m.reshape(B, K, -1)

        ic_actions_m = None
        if K > 1:
            _, ic_actions_m = self.action_space.preprocess(
                proprio[:, :-1].reshape(B * (K - 1), proprio.shape[-1]),
                ic_actions.reshape(B * (K - 1), T, D),
            )
            ic_actions_m = ic_actions_m.reshape(B, K - 1, T, D)

        next_image_input = kwargs.get("next_image_input")
        next_image_grid_thw = kwargs.get("next_image_grid_thw")
        next_proprios = None
        if next_image_input is not None and self.proprio_to_vlm:
            next_proprio_raw = kwargs.get("next_proprio")
            if next_proprio_raw is None:
                next_proprio_raw = proprio[:, 1:K]
            next_proprios_raw = next_proprio_raw.reshape(B * (K - 1), -1)
            next_proprios_m, _ = self.action_space.preprocess(
                next_proprios_raw,
                torch.zeros(B * (K - 1), T, D, device=device, dtype=dtype),
            )
            next_proprios = next_proprios_m.reshape(B, K - 1, -1)

        p = self._prepare_ic_prefix_embeds(
            input_ids, image_input, image_mask, image_grid_thw,
            proprio_m, ic_actions_m, frame_obs_ends,
            next_pixel_values=next_image_input,
            next_image_grid_thw=next_image_grid_thw,
            next_proprios=next_proprios,
        )

        return self._cache_and_denoise(p, proprio[:, -1], steps)

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """Minimal FastAPI app for XVLA inference."""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload: continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))
                domain_id = torch.tensor([int(payload["domain_id"])], dtype=torch.long)

                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs.update({
                    "proprio": to_model(proprio.unsqueeze(0)),
                    "domain_id": domain_id.to(device),
                })

                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)


# ------------------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------------------
def prepare_batch(
    batch: Dict[str, Any],
    processor,
    device: torch.device,
    num_proprio_tokens: int = 1,
    num_action_tokens: int = 8,
    include_next_obs: bool = False,
    use_frame_sep: bool = False,
) -> Dict[str, torch.Tensor]:
    """Prepare training batch. Routes to IC encoding when num_frames >= 1."""
    batch = dict(batch)
    instructions = batch.pop("language_instruction", [])
    image_mask = batch.get("image_mask", None)
    image_grid_thw = batch.get("image_grid_thw", None)
    num_frames = batch.get("num_frames", None)
    next_image_grid_thw = batch.get("next_image_grid_thw", None)

    K = int(num_frames[0] if isinstance(num_frames, torch.Tensor) else num_frames) if num_frames is not None else 0

    has_next_obs = include_next_obs and K > 1 and "next_image_input" in batch

    if K >= 1:
        lang = processor.encode_ic_sequence(
            instructions,
            num_frames=K,
            image_grid_thw=image_grid_thw,
            image_mask=image_mask,
            num_proprio_tokens=num_proprio_tokens,
            num_action_tokens=num_action_tokens,
            include_next_obs=has_next_obs,
            next_image_grid_thw=next_image_grid_thw,
            use_frame_sep=use_frame_sep,
        )
    else:
        lang = processor.encode_language(
            instructions,
            image_mask=image_mask,
            image_grid_thw=image_grid_thw,
            num_proprio_tokens=num_proprio_tokens,
        )
    batch.update(lang)
    tensor_inputs = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_inputs[key] = value.to(device=device, non_blocking=True)
        else:
            tensor_inputs[key] = torch.as_tensor(value).to(device=device)
    return tensor_inputs


def build_vla_optimizer(
    model: XVLA,
    *,
    base_lr: float,
    weight_decay: float,
    betas: tuple,
    lr_coef_soft: float,
) -> AdamW:
    """Build optimizer with per-group learning rates (frozen VLM, trained heads)."""
    vlm_params = list(model.vlm.parameters())
    proprio_params = (
        list(model.proprio_projector.parameters())
        if getattr(model, 'proprio_projector', None) is not None else []
    )

    def _group(name, params, lr):
        return {"name": name, "params": params, "lr": lr, "weight_decay": weight_decay}

    expert_params = list(model.action_expert.parameters())
    exclude = set(map(id, vlm_params + expert_params + proprio_params))
    other_params = [p for p in model.parameters() if id(p) not in exclude]

    groups = [_group("vlm", vlm_params, 0.0),
              _group("action_expert", expert_params, base_lr)]
    if other_params:
        groups.append(_group("other", other_params, base_lr))
    if proprio_params:
        groups.append(_group("proprio_projector", proprio_params, base_lr))

    return AdamW(groups, betas=betas)


def update_vla_learning_rates(
    optim: torch.optim.Optimizer,
    step: int,
    *,
    learning_rate: float,
    learning_coef: float,
    freeze_steps: int,
    warmup_steps: int,
    total_iters: int,
    use_cosine_decay: bool,
    min_lr_ratio: float,
    model: Optional[XVLA] = None,
) -> None:
    """Update per-group learning rates with optional warmup + cosine decay."""
    base_lrs = {"vlm": learning_rate * learning_coef, "action_expert": learning_rate,
                 "other": learning_rate, "proprio_projector": learning_rate}
    frozen_groups = {"vlm"}

    for g in optim.param_groups:
        name = g.get("name")
        if name not in base_lrs:
            continue
        if step < freeze_steps:
            g["lr"] = 0.0 if name in frozen_groups else base_lrs[name]
        else:
            base = base_lrs[name]
            g["lr"] = _linear_warmup_cosine(step, freeze_steps, warmup_steps, total_iters, base, min_lr_ratio) if use_cosine_decay else base


def _linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Compute LR with linear warmup and cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)
