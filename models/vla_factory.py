# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------
"""
VLA Model Factory.

Provides unified loading for different VLA architectures:
  - XVLA (Qwen3-VL backbone + MoT action head)
  - OpenVLA family (openvla, openvla-oft, vla-adapter)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch


@dataclass
class VLAComponents:
    model: Any
    processor: Any
    prepare_batch_fn: Callable[[Dict[str, Any], Any, Any, torch.device], Dict[str, torch.Tensor]]
    build_optimizer_fn: Callable[[Any], torch.optim.Optimizer]
    update_lr_fn: Callable[[torch.optim.Optimizer, int], None]


# ---------------------------------------------------------------------------
# Helper: Load config from file or pretrained path
# ---------------------------------------------------------------------------
def _load_config(config_cls, config_source: str | None, default_config=None):
    """Load config from JSON file, pretrained path, or return default."""
    if config_source is None:
        return default_config if default_config is not None else config_cls()
    if os.path.isfile(config_source):
        return config_cls.from_json_file(config_source)
    return config_cls.from_pretrained(config_source)


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------
def _load_pretrained_weights_into(model, ckpt_path: str):
    """Load compatible weights from checkpoint into model (strict=False).

    Handles single-file and sharded safetensors. Skips shape mismatches.
    """
    from safetensors.torch import load_file
    import json as _json

    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_path, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            wmap = _json.load(f)["weight_map"]
        state_dict = {}
        for shard in sorted(set(wmap.values())):
            state_dict.update(load_file(os.path.join(ckpt_path, shard)))
    elif os.path.isfile(single_path):
        state_dict = load_file(single_path)
    else:
        logging.warning(f"No safetensors in {ckpt_path}, skipping weight loading")
        return

    model_state = model.state_dict()
    loaded = 0
    for key, param in state_dict.items():
        if key in model_state and param.shape == model_state[key].shape:
            model_state[key] = param
            loaded += 1
    model.load_state_dict(model_state, strict=False)

    new_keys = [k for k in model_state if k not in state_dict]
    logging.info(
        f"Loaded {loaded}/{len(model_state)} weights from {ckpt_path} | "
        f"{len(new_keys)} new (randomly init)"
    )
    if new_keys:
        prefixes = sorted(set(".".join(k.split(".")[:2]) for k in new_keys))
        logging.info(f"  New params: {prefixes}")


# ---------------------------------------------------------------------------
# XVLA Loader (Qwen3-VL backbone + MoT action head)
# ---------------------------------------------------------------------------
def _load_xvla(load_path: str | None, args) -> VLAComponents:
    """Load XVLA model (Qwen3-VL + MoT)."""
    from .configuration_xvla import XVLAConfig
    from .modeling_xvla import (
        XVLA,
        build_vla_optimizer,
        prepare_batch as xvla_prepare_batch,
        update_vla_learning_rates,
    )
    from .processing_xvla import build_xvla_processor

    resuming = args.resume_from_checkpoint is not None
    loading_checkpoint = resuming or (
        load_path is not None and args.config_path is None and args.vlm_pretrained is None
    )

    if loading_checkpoint:
        ckpt_path = args.resume_from_checkpoint or load_path
        model = XVLA.from_pretrained(ckpt_path)
        # Processor was built and set by from_pretrained
        processor = model._processor
    else:
        config_source = args.config_path or load_path
        if config_source is None:
            raise ValueError("Training XVLA from scratch requires --config_path pointing to an XVLA config.")
        config = _load_config(XVLAConfig, config_source)

        backbone_type = getattr(config, "vlm_backbone_type", "qwen3_vl")
        proprio_to_vlm = getattr(config, "proprio_to_vlm", False)
        default_vlm = (
            getattr(config, "paligemma_pretrained", "google/paligemma-3b-pt-224")
            if backbone_type == "paligemma"
            else getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
        )
        vlm_path = args.vlm_pretrained or default_vlm
        if backbone_type == "paligemma":
            config.paligemma_pretrained = vlm_path
        else:
            config.qwen3_pretrained = vlm_path
        num_views = getattr(config, "num_views", 2)
        vlm_process_all_views = getattr(config, "vlm_process_all_views", False)

        # Create processor FIRST so model can use it for embedding initialization
        processor = build_xvla_processor(
            pretrained_name_or_path=vlm_path,
            backbone_type=backbone_type,
            num_views=num_views,
            proprio_to_vlm=proprio_to_vlm,
            vlm_process_all_views=vlm_process_all_views,
        )
        model = XVLA(config, processor=processor)

        # --models CKPT --config_path CONFIG: build from config, load weights from checkpoint
        if load_path is not None and args.config_path is not None:
            _load_pretrained_weights_into(model, load_path)

    def build_optimizer_fn(model_instance):
        return build_vla_optimizer(
            model_instance,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
            lr_coef_soft=args.learning_coef,
        )

    def update_lr_fn(optim: torch.optim.Optimizer, step: int, model_instance=None):
        return update_vla_learning_rates(
            optim, step,
            learning_rate=args.learning_rate,
            learning_coef=args.learning_coef,
            freeze_steps=args.freeze_steps,
            warmup_steps=args.warmup_steps,
            total_iters=args.iters,
            use_cosine_decay=args.use_cosine_decay,
            min_lr_ratio=args.min_lr_ratio,
            model=model_instance,
        )

    def prepare_fn(batch, _model_instance, processor_instance, device):
        if processor_instance is None:
            raise ValueError("XVLA training requires a tokenizer/processor for language encoding.")
        npt = getattr(_model_instance, 'num_proprio_tokens', 1)
        nat = getattr(_model_instance.config, 'num_action_tokens', 8)
        inc_next = getattr(_model_instance.config, 'include_next_obs', False)
        frame_sep = getattr(_model_instance.config, 'use_frame_sep', False)
        return xvla_prepare_batch(
            batch, processor_instance, device,
            num_proprio_tokens=npt,
            num_action_tokens=nat,
            include_next_obs=inc_next,
            use_frame_sep=frame_sep,
        )

    return VLAComponents(
        model=model,
        processor=processor,
        prepare_batch_fn=prepare_fn,
        build_optimizer_fn=build_optimizer_fn,
        update_lr_fn=update_lr_fn,
    )


# ---------------------------------------------------------------------------
# OpenVLA Family Loader (openvla, openvla-oft, vla-adapter)
# ---------------------------------------------------------------------------
def _load_openvla_family(load_path: str | None, args, variant: str) -> VLAComponents:
    from .openvla.configuration_openvla import (
        OpenVLAAdapterConfig,
        OpenVLAConfig,
        OpenVLAOFTConfig,
    )
    from .openvla.modeling_openvla import (
        OpenVLA,
        OpenVLAAdapter,
        OpenVLAOFT,
        build_vla_optimizer,
        prepare_batch as openvla_prepare_batch,
        prepare_batch_adapter,
        prepare_batch_oft,
        update_vla_learning_rates,
    )

    # Variant-specific mappings
    VARIANT_MAP = {
        "openvla": (OpenVLA, OpenVLAConfig, openvla_prepare_batch, False),
        "openvla-oft": (OpenVLAOFT, OpenVLAOFTConfig, prepare_batch_oft, True),
        "vla-adapter": (OpenVLAAdapter, OpenVLAAdapterConfig, prepare_batch_adapter, True),
    }

    model_cls, config_cls, prepare_batch_fn, supports_action_chunk = VARIANT_MAP[variant]
    loading_checkpoint = args.resume_from_checkpoint is not None or load_path is not None

    if loading_checkpoint:
        model = model_cls.from_pretrained(load_path)
        processor = getattr(model, "processor", None)
    else:
        config = _load_config(config_cls, args.config_path or load_path)

        vlm_path = args.vlm_pretrained or args.models or getattr(config, "vlm_model_name", None)
        if vlm_path:
            config.vlm_model_name = vlm_path
            config.processor_name = getattr(config, "processor_name", None) or vlm_path
            config.tokenizer_name = getattr(config, "tokenizer_name", None) or vlm_path

        if supports_action_chunk and args.dataset_num_actions is not None:
            config.num_action_chunk = args.dataset_num_actions

        model = model_cls(config)
        processor = getattr(model, "processor", None)

    def build_optimizer_fn(model_instance):
        return build_vla_optimizer(
            model_instance,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=tuple(args.betas),
        )

    def update_lr_fn(optim: torch.optim.Optimizer, step: int):
        return update_vla_learning_rates(
            optim, step,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            total_iters=args.iters,
            use_cosine_decay=args.use_cosine_decay,
            min_lr_ratio=args.min_lr_ratio,
        )

    # Capture processor in closure for prepare_fn
    _processor = processor

    def prepare_fn(batch, model_instance, _processor_instance, device):
        return prepare_batch_fn(batch, model_instance, _processor, device)

    return VLAComponents(
        model=model,
        processor=processor,
        prepare_batch_fn=prepare_fn,
        build_optimizer_fn=build_optimizer_fn,
        update_lr_fn=update_lr_fn,
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
# Normalize architecture aliases to canonical names
_ARCH_ALIASES = {
    "xvla": "xvla",
    "openvla": "openvla",
    "openvla-oft": "openvla-oft",
    "openvla_oft": "openvla-oft",
    "openvlaoft": "openvla-oft",
    "vla-adapter": "vla-adapter",
    "vla_adapter": "vla-adapter",
    "vlaadapter": "vla-adapter",
}


def load_vla_components(model_arch: str, load_path: str, args) -> VLAComponents:
    """
    Load VLA model components based on architecture type.

    Supported architectures:
        - xvla: XVLA (Qwen3-VL + MoT)
        - openvla: OpenVLA base model
        - openvla-oft: OpenVLA with OFT
        - vla-adapter: VLA adapter model
    """
    arch = _ARCH_ALIASES.get(model_arch.lower())

    if arch is None:
        raise ValueError(f"Unknown model_arch '{model_arch}'. Supported: {list(_ARCH_ALIASES.keys())}")

    if arch == "xvla":
        return _load_xvla(load_path, args)

    # OpenVLA family (openvla, openvla-oft, vla-adapter)
    return _load_openvla_family(load_path, args, arch)
