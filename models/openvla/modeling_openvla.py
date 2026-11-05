from __future__ import annotations

import logging
import math
import traceback
from typing import Any, Dict, Iterable, List, Optional

import cv2
import json_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

from .configuration_openvla import OpenVLAAdapterConfig, OpenVLAConfig, OpenVLAOFTConfig
from .tokenization_openvla import OpenVLAActionTokenizer


class MLPResNetBlock(nn.Module):
    """Single pre-norm MLP block with residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x


class MLPResNet(nn.Module):
    """Lightweight MLP with residual blocks."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(MLPResNetBlock(hidden_dim) for _ in range(num_blocks))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """
    MLP head that regresses continuous actions from action-token hidden states.

    Expects action tokens to be grouped as ``[chunk, action_dim]`` when reshaped.
    """

    def __init__(self, *, hidden_size: int, action_dim: int, num_action_chunk: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_chunk = num_action_chunk
        hidden_dim = hidden_dim or hidden_size
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=hidden_size * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(action_hidden_states)

    def predict_action(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_hidden_states: [B, chunk * action_dim, hidden]
        Returns:
            Predicted continuous actions [B, chunk, action_dim]
        """
        bsz, seq_len, hidden = action_hidden_states.shape
        expected_len = self.num_action_chunk * self.action_dim
        if seq_len != expected_len:
            raise ValueError(
                f"Expected {expected_len} action tokens (chunk={self.num_action_chunk}, "
                f"action_dim={self.action_dim}), got {seq_len}."
            )
        reshaped = action_hidden_states.reshape(bsz, self.num_action_chunk, self.action_dim * hidden)
        return self.model(reshaped)


class AdapterRegressionHead(nn.Module):
    """
    Attention-enhanced regression head inspired by VLA-Adapter.

    Uses learnable adapter tokens and multi-head attention over action hidden states before
    a small MLP to predict continuous actions.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        num_action_chunk: int,
        num_adapter_tokens: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        use_pro_version: bool = False,
        proprio_dim: Optional[int] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_chunk = num_action_chunk
        self.hidden_size = hidden_size
        hidden_dim = hidden_dim or hidden_size
        self.num_heads = max(1, num_heads)
        if hidden_size % self.num_heads != 0:
            self.num_heads = 1
        self.adapter_tokens = nn.Parameter(torch.zeros(num_adapter_tokens, hidden_size))
        nn.init.normal_(self.adapter_tokens, mean=0.0, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        num_blocks = 4 if use_pro_version else 2
        self.mlp = MLPResNet(
            num_blocks=num_blocks,
            input_dim=hidden_size * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        self.proprio_proj = None
        if proprio_dim is not None and proprio_dim > 0:
            self.proprio_proj = nn.Linear(proprio_dim, hidden_size)

    def predict_action(self, action_hidden_states: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            action_hidden_states: [B, chunk*action_dim, hidden]
            proprio: Optional proprio features [B, P]
        Returns:
            [B, chunk, action_dim]
        """
        bsz, seq_len, hidden = action_hidden_states.shape
        expected_len = self.num_action_chunk * self.action_dim
        if seq_len != expected_len:
            raise ValueError(
                f"Expected {expected_len} action tokens (chunk={self.num_action_chunk}, "
                f"action_dim={self.action_dim}), got {seq_len}."
            )
        adapter = self.adapter_tokens.unsqueeze(0).expand(bsz, -1, -1)
        k = torch.cat([action_hidden_states, adapter], dim=1)
        v = k
        if proprio is not None:
            if self.proprio_proj is None:
                self.proprio_proj = nn.Linear(proprio.shape[-1], self.hidden_size).to(
                    device=action_hidden_states.device, dtype=action_hidden_states.dtype
                )
            p = self.proprio_proj(proprio).unsqueeze(1)
            k = torch.cat([k, p], dim=1)
            v = torch.cat([v, p], dim=1)

        attn_out, _ = self.attn(action_hidden_states, k, v)
        fused = attn_out.reshape(bsz, self.num_action_chunk, self.action_dim * hidden)
        return self.mlp(fused)


class OpenVLA(PreTrainedModel):
    """
    OpenVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • A multimodal VLM backbone (e.g., Qwen2.5-VL-3B).
      • A discretising tokenizer that maps continuous actions to textual tokens.
      • Utility helpers for training (teacher-forcing on action tokens) and inference.
    """

    config_class = OpenVLAConfig
    base_model_prefix = "openvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: OpenVLAConfig, *model_args, **model_kwargs):
        super().__init__(config)

        self.prompt_template: str = config.prompt_template
        self.predict_stop_token: bool = config.predict_stop_token
        self.action_dim: int = config.action_dim

        skip_vlm_init = bool(model_kwargs.pop("_skip_vlm_init", False))

        # Allow callers to inject pre-instantiated components to avoid re-loading heavy checkpoints.
        vlm = model_kwargs.pop("vlm", None)
        tokenizer = model_kwargs.pop("tokenizer", None)
        processor = model_kwargs.pop("processor", None)

        # Detect if the VLM is a Qwen-VL model (requires AutoModelForVision2Seq)
        model_name_lower = config.vlm_model_name.lower()

        if vlm is None:
            if skip_vlm_init:
                vlm_config = AutoConfig.from_pretrained(
                    config.vlm_model_name,
                    trust_remote_code=config.trust_remote_code,
                )
                vlm = AutoModelForVision2Seq.from_config(vlm_config)
            else:
                vlm = AutoModelForVision2Seq.from_pretrained(
                    config.vlm_model_name,
                    trust_remote_code=config.trust_remote_code,
                    **(config.vlm_model_kwargs or {}),
                )
        self.vlm = vlm

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name,
                trust_remote_code=config.trust_remote_code,
            )
        self.tokenizer = tokenizer

        self.action_tokenizer = OpenVLAActionTokenizer(
            tokenizer=tokenizer,
            bins=config.action_bins,
            min_action=config.action_min,
            max_action=config.action_max,
        )

        if processor is not None:
            self.processor = processor
        elif config.auto_processor:
            self.processor = AutoProcessor.from_pretrained(
                config.processor_name,
                trust_remote_code=config.trust_remote_code,
            )
        else:
            self.processor = None

        # Optional FastAPI endpoint holder (lazily created)
        self.app: FastAPI | None = None

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_token_ids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Teacher-forced training step.

        If ``actions`` or ``action_token_ids`` are provided, the method appends the discretised action tokens
        to the input sequence and ensures that the loss is only computed on those final tokens in the output.
        """
        input_ids, attention_mask, labels = self._append_action_tokens_for_training(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            action_token_ids=action_token_ids,
            action_mask=action_mask,
        )

        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _append_action_tokens_for_training(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
        actions: Optional[torch.Tensor],
        action_token_ids: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        if action_token_ids is None and actions is None:
            return input_ids, attention_mask, labels
        if action_token_ids is not None and actions is not None:
            raise ValueError("Provide either `actions` or `action_token_ids`, not both.")

        if action_token_ids is None and actions is not None:
            action_token_ids = torch.as_tensor(
                self.action_tokenizer.to_token_ids(actions),
                dtype=torch.long,
                device=input_ids.device,
            )

        if action_token_ids.ndim == 1:
            action_token_ids = action_token_ids.unsqueeze(0)
        if action_mask is not None and action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)

        if self.predict_stop_token and self.tokenizer.eos_token_id is not None:
            eos_tensor = torch.full(
                (action_token_ids.size(0), 1),
                fill_value=self.tokenizer.eos_token_id,
                dtype=action_token_ids.dtype,
                device=action_token_ids.device,
            )
            action_token_ids = torch.cat([action_token_ids, eos_tensor], dim=-1)
            if action_mask is not None:
                eos_mask = torch.ones(
                    (action_mask.size(0), 1),
                    dtype=action_mask.dtype,
                    device=action_mask.device,
                )
                action_mask = torch.cat([action_mask, eos_mask], dim=-1)

        batch_size, action_len = action_token_ids.shape[0], action_token_ids.shape[-1]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if labels is None:
            labels = torch.full_like(input_ids, fill_value=-100)

        device = input_ids.device
        new_input_ids = torch.cat([input_ids, action_token_ids.to(device=device)], dim=-1)
        new_attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((batch_size, action_len))],
            dim=-1,
        )
        new_labels = torch.cat(
            [labels, torch.full((batch_size, action_len), fill_value=-100, device=device, dtype=labels.dtype)],
            dim=-1,
        )
        new_labels[:, -action_len:] = action_token_ids.to(device=device)
        if action_mask is not None:
            mask = action_mask.to(device=device).bool()
            new_labels[:, -action_len:][~mask] = -100

        return new_input_ids, new_attention_mask, new_labels

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def build_prompt(self, instruction: str) -> str:
        instruction_clean = instruction.strip()
        return self.prompt_template.format(instruction=instruction_clean.lower())

    def _prepare_processor_inputs(
        self,
        processor,
        images: Optional[Iterable[Any]],
        prompt: str,
    ) -> Dict[str, Any]:
        if processor is None:
            raise ValueError("Processor is required when raw images are provided.")

        def _coerce_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
            """
            Align processor outputs to the Qwen-style forward signature.
            Supports legacy processors that emit Florence-style keys.
            """
            if outputs is None:
                return {}
            if "pixel_values" in outputs:
                return outputs

            out = dict(outputs)
            if "image_input" in out:
                img = out.pop("image_input")
                if isinstance(img, torch.Tensor):
                    if img.ndim == 5:  # [B, V, C, H, W] -> keep first view
                        img = img[:, 0]
                else:
                    img = torch.as_tensor(img)
                    if img.ndim == 5:
                        img = img[:, 0]
                out["pixel_values"] = img
                out.pop("image_mask", None)
            return out

        import inspect

        def _supports_kwargs(fn, kwargs: Dict[str, Any]) -> bool:
            sig = inspect.signature(fn)
            params = sig.parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return all(k in sig.parameters for k in kwargs.keys())

        call_kwargs = [
            {"images": images, "language_instruction": prompt},
            {"images": images, "text": prompt},
            {"image": images, "text": prompt},
        ]
        for kwargs in call_kwargs:
            if _supports_kwargs(processor.__call__, kwargs):
                return _coerce_outputs(processor(return_tensors="pt", **kwargs))

        return _coerce_outputs(processor(images=images, language_instruction=prompt))

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.vlm.parameters()).device
        dtype = next(self.vlm.parameters()).dtype
        out: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                cast_dtype = dtype if value.is_floating_point() else value.dtype
                out[key] = value.to(device=device, dtype=cast_dtype)
            elif isinstance(value, (list, tuple)):
                tensor = torch.as_tensor(value)
                cast_dtype = dtype if tensor.is_floating_point() else tensor.dtype
                out[key] = tensor.to(device=device, dtype=cast_dtype)
            else:
                out[key] = value
        return out

    def _select_unnorm_key(self, unnorm_key: Optional[str]) -> Optional[str]:
        stats = self.config.action_stats or {}
        if not stats:
            return None
        if unnorm_key is None:
            if len(stats) == 1:
                return next(iter(stats.keys()))
            raise ValueError(
                "Multiple datasets detected. Please provide `unnorm_key` to choose the de-normalization statistics."
            )
        if unnorm_key not in stats:
            raise KeyError(f"`unnorm_key={unnorm_key}` not found. Options: {list(stats.keys())}")
        return unnorm_key

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = self._select_unnorm_key(unnorm_key)
        if key is None:
            return None
        return self.config.action_stats[key]["action"]

    def get_action_dim(self) -> int:
        return self.action_dim

    @torch.inference_mode()
    def predict_action(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        inputs: Optional[Dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        unnorm_key: Optional[str] = None,
        return_token_ids: bool = False,
        **generate_kwargs: Any,
    ) -> np.ndarray | Dict[str, Any]:
        """
        Predict a continuous action for the provided instruction and images.
        """
        self.eval()
        processor = processor or self.processor

        if inputs is None:
            prompt = self.build_prompt(instruction)
            if processor is not None:
                inputs = self._prepare_processor_inputs(processor, images, prompt)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                inputs = dict(tokens)
        else:
            inputs = dict(inputs)
            if "input_ids" not in inputs:
                prompt = self.build_prompt(instruction)
                tokenized = self.tokenizer(prompt, return_tensors="pt")
                inputs["input_ids"] = tokenized["input_ids"]
                if "attention_mask" not in inputs and "attention_mask" in tokenized:
                    inputs["attention_mask"] = tokenized["attention_mask"]

        model_inputs = self._move_inputs_to_device(inputs)
        expected_tokens = self.get_action_dim() + (1 if self.predict_stop_token else 0)
        max_new_tokens = max_new_tokens or expected_tokens
        if max_new_tokens < expected_tokens:
            raise ValueError(f"`max_new_tokens` ({max_new_tokens}) must be >= {expected_tokens}.")

        generate_kwargs.setdefault("max_new_tokens", max_new_tokens)
        output_ids = self.vlm.generate(**model_inputs, **generate_kwargs)

        action_token_ids = output_ids[:, -expected_tokens:]
        if self.predict_stop_token:
            action_token_ids = action_token_ids[:, :-1]

        action_token_ids_np = action_token_ids[0].detach().cpu().numpy()
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)

        stats = self.get_action_stats(unnorm_key)
        if stats is not None:
            q01 = np.asarray(stats["q01"])
            q99 = np.asarray(stats["q99"])
            mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)))
            unnormed = 0.5 * (normalized_actions + 1.0) * (q99 - q01) + q01
            normalized_actions = np.where(mask, unnormed, normalized_actions)

        if return_token_ids:
            return {
                "actions": normalized_actions,
                "token_ids": action_token_ids_np,
            }
        return normalized_actions

    @torch.inference_mode()
    def generate_actions(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        **generate_kwargs: Any,
    ) -> np.ndarray:
        """
        Convenience wrapper mirroring the XVLA API surface.
        """
        return self.predict_action(
            instruction=instruction,
            images=images,
            processor=processor,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # FastAPI service
    # ------------------------------------------------------------------
    def _decode_images_from_payload(self, payload: Dict[str, Any]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for key in ("image0", "image1", "image2"):
            if key not in payload:
                continue
            value = json_numpy.loads(payload[key])
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    value = cv2.imdecode(value, cv2.IMREAD_COLOR)
                images.append(Image.fromarray(value))
            elif isinstance(value, (list, tuple)):
                images.append(Image.fromarray(np.array(value)))
            elif isinstance(value, str):
                images.append(Image.open(value))
        return images

    def _build_app(self, processor=None):
        if self.app is not None:
            return
        processor = processor or self.processor
        if processor is None:
            raise ValueError("A processor must be provided to serve OpenVLA.")

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                instruction = payload["language_instruction"]
                images = self._decode_images_from_payload(payload)
                generation_kwargs = payload.get("generation_kwargs", {})
                if "max_new_tokens" in payload:
                    generation_kwargs.setdefault("max_new_tokens", payload["max_new_tokens"])
                unnorm_key = payload.get("unnorm_key")
                actions = self.predict_action(
                    instruction=instruction,
                    images=images,
                    processor=processor,
                    unnorm_key=unnorm_key,
                    **generation_kwargs,
                )
                return JSONResponse({"action": actions.tolist()})
            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor=None, host: str = "0.0.0.0", port: int = 8000):
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)


class OpenVLAOFT(OpenVLA):
    """
    OpenVLA-OFT: action-chunk regression variant trained with an L1 loss.

    Uses placeholder action tokens to reserve decoding slots and regresses continuous actions
    directly from the corresponding hidden states.
    """

    config_class = OpenVLAOFTConfig
    base_model_prefix = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.num_actions: int = config.num_action_chunk

        hidden_size = (
            getattr(self.vlm.config, "hidden_size", None)
            or getattr(getattr(self.vlm.config, "text_config", None), "hidden_size", None)
            or getattr(self.vlm.config, "text_hidden_size", None)
        )
        if hidden_size is None:
            param = next(self.vlm.parameters(), None)
            if param is None:
                raise ValueError("Unable to infer hidden size for OpenVLAOFT.")
            hidden_size = int(param.shape[-1])

        placeholder = config.action_placeholder_token_id
        if placeholder is None:
            placeholder = self.tokenizer.pad_token_id
        if placeholder is None:
            placeholder = self.tokenizer.eos_token_id
        if placeholder is None:
            placeholder = 0
        self.placeholder_token_id: int = int(placeholder)

        self.action_head = L1RegressionActionHead(
            hidden_size=hidden_size,
            hidden_dim=config.action_head_hidden_dim or hidden_size,
            action_dim=self.action_dim,
            num_action_chunk=self.num_actions,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Trim/pad actions to the configured chunk length."""
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[..., : self.action_dim]
        bsz, seq, _ = actions.shape
        if seq < self.num_actions:
            pad = actions.new_zeros((bsz, self.num_actions - seq, self.action_dim))
            actions = torch.cat([actions, pad], dim=1)
        elif seq > self.num_actions:
            actions = actions[:, : self.num_actions]
        return actions

    def _append_action_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        action_len: int,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        bsz = input_ids.size(0)
        device = input_ids.device
        placeholders = torch.full(
            (bsz, action_len), fill_value=self.placeholder_token_id, device=device, dtype=torch.long
        )
        new_input_ids = torch.cat([input_ids, placeholders], dim=-1)
        new_attention = torch.cat(
            [attention_mask, attention_mask.new_ones((bsz, action_len))],
            dim=-1,
        )
        return new_input_ids, new_attention

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if actions is None:
            raise ValueError("OpenVLAOFT training requires `actions`.")

        actions = self._normalize_actions(actions)
        bsz, chunk, _ = actions.shape
        action_len = chunk * self.action_dim
        input_ids, attention_mask = self._append_action_placeholders(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_len=action_len,
        )

        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden = outputs.hidden_states[-1]
        action_hidden = last_hidden[:, -action_len:, :]
        pred_actions = self.action_head.predict_action(action_hidden)

        target_actions = actions.to(dtype=pred_actions.dtype)
        if action_mask is not None:
            mask = action_mask[:, :chunk].unsqueeze(-1).to(dtype=pred_actions.dtype, device=pred_actions.device)
            l1 = F.l1_loss(pred_actions, target_actions, reduction="none")
            loss = (l1 * mask).sum() / (mask.sum() * self.action_dim + 1e-8)
        else:
            loss = F.l1_loss(pred_actions, target_actions)

        return {"loss": loss, "pred_actions": pred_actions}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        inputs: Optional[Dict[str, Any]] = None,
        unnorm_key: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.eval()
        processor = processor or self.processor

        if inputs is None:
            prompt = self.build_prompt(instruction)
            if processor is not None:
                inputs = self._prepare_processor_inputs(processor, images, prompt)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                inputs = dict(tokens)
        else:
            inputs = dict(inputs)

        model_inputs = self._move_inputs_to_device(inputs)

        if "input_ids" not in model_inputs:
            raise ValueError("`input_ids` are required for OpenVLAOFT inference.")

        bsz = model_inputs["input_ids"].size(0)
        action_len = self.num_actions * self.action_dim
        model_inputs["input_ids"], model_inputs["attention_mask"] = self._append_action_placeholders(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            action_len=action_len,
        )

        model_inputs.setdefault("output_hidden_states", True)
        outputs = self.vlm(**model_inputs)
        hidden = outputs.hidden_states[-1]
        action_hidden = hidden[:, -action_len:, :]
        pred_actions = self.action_head.predict_action(action_hidden)
        actions_np = pred_actions[0].detach().cpu().numpy()

        stats = self.get_action_stats(unnorm_key)
        if stats is not None:
            q01 = np.asarray(stats["q01"])
            q99 = np.asarray(stats["q99"])
            mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)))
            unnormed = 0.5 * (actions_np + 1.0) * (q99 - q01) + q01
            actions_np = np.where(mask, unnormed, actions_np)

        return actions_np

    @torch.inference_mode()
    def generate_actions(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.predict_action(instruction=instruction, images=images, processor=processor, **kwargs)

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        kwargs.setdefault("_skip_vlm_init", True)
        return super().from_pretrained(*model_args, **kwargs)


class OpenVLAAdapter(OpenVLA):
    """
    VLA-Adapter style model: predicts action chunks via an adapter regression head.
    """

    config_class = OpenVLAAdapterConfig
    base_model_prefix = "vla_adapter"

    def __init__(self, config: OpenVLAAdapterConfig, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.num_actions: int = config.num_action_chunk
        hidden_size = (
            getattr(self.vlm.config, "hidden_size", None)
            or getattr(getattr(self.vlm.config, "text_config", None), "hidden_size", None)
            or getattr(self.vlm.config, "text_hidden_size", None)
        )
        if hidden_size is None:
            param = next(self.vlm.parameters(), None)
            if param is None:
                raise ValueError("Unable to infer hidden size for OpenVLAAdapter.")
            hidden_size = int(param.shape[-1])

        placeholder = config.action_placeholder_token_id
        if placeholder is None:
            placeholder = self.tokenizer.pad_token_id
        if placeholder is None:
            placeholder = self.tokenizer.eos_token_id
        if placeholder is None:
            placeholder = 0
        self.placeholder_token_id: int = int(placeholder)

        self.action_head = AdapterRegressionHead(
            hidden_size=hidden_size,
            hidden_dim=config.action_head_hidden_dim or hidden_size,
            action_dim=self.action_dim,
            num_action_chunk=self.num_actions,
            num_adapter_tokens=config.num_adapter_tokens,
            num_heads=config.adapter_num_heads,
            use_pro_version=config.use_pro_version,
            proprio_dim=None,
        )

    # Helpers
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[..., : self.action_dim]
        bsz, seq, _ = actions.shape
        if seq < self.num_actions:
            pad = actions.new_zeros((bsz, self.num_actions - seq, self.action_dim))
            actions = torch.cat([actions, pad], dim=1)
        elif seq > self.num_actions:
            actions = actions[:, : self.num_actions]
        return actions

    def _append_action_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        action_len: int,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        bsz = input_ids.size(0)
        device = input_ids.device
        placeholders = torch.full(
            (bsz, action_len), fill_value=self.placeholder_token_id, device=device, dtype=torch.long
        )
        new_input_ids = torch.cat([input_ids, placeholders], dim=-1)
        new_attention = torch.cat(
            [attention_mask, attention_mask.new_ones((bsz, action_len))],
            dim=-1,
        )
        return new_input_ids, new_attention

    # Training forward
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if actions is None:
            raise ValueError("OpenVLAAdapter training requires `actions`.")

        actions = self._normalize_actions(actions)
        bsz, chunk, _ = actions.shape
        action_len = chunk * self.action_dim
        input_ids, attention_mask = self._append_action_placeholders(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_len=action_len,
        )

        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden = outputs.hidden_states[-1]
        action_hidden = last_hidden[:, -action_len:, :]
        pred_actions = self.action_head.predict_action(action_hidden, proprio=proprio)

        target_actions = actions.to(dtype=pred_actions.dtype)
        if action_mask is not None:
            mask = action_mask[:, :chunk].unsqueeze(-1).to(dtype=pred_actions.dtype, device=pred_actions.device)
            l1 = F.l1_loss(pred_actions, target_actions, reduction="none")
            loss = (l1 * mask).sum() / (mask.sum() * self.action_dim + 1e-8)
        else:
            loss = F.l1_loss(pred_actions, target_actions)

        return {"loss": loss, "pred_actions": pred_actions}

    # Inference
    @torch.inference_mode()
    def predict_action(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        inputs: Optional[Dict[str, Any]] = None,
        unnorm_key: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.eval()
        processor = processor or self.processor

        if inputs is None:
            prompt = self.build_prompt(instruction)
            if processor is not None:
                inputs = self._prepare_processor_inputs(processor, images, prompt)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                inputs = dict(tokens)
        else:
            inputs = dict(inputs)

        model_inputs = self._move_inputs_to_device(inputs)
        if "input_ids" not in model_inputs:
            raise ValueError("`input_ids` are required for OpenVLAAdapter inference.")

        action_len = self.num_actions * self.action_dim
        model_inputs["input_ids"], model_inputs["attention_mask"] = self._append_action_placeholders(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            action_len=action_len,
        )
        model_inputs.setdefault("output_hidden_states", True)

        outputs = self.vlm(**model_inputs)
        hidden = outputs.hidden_states[-1]
        action_hidden = hidden[:, -action_len:, :]
        pred_actions = self.action_head.predict_action(action_hidden, proprio=None)

        actions_np = pred_actions[0].detach().cpu().numpy()
        stats = self.get_action_stats(unnorm_key)
        if stats is not None:
            q01 = np.asarray(stats["q01"])
            q99 = np.asarray(stats["q99"])
            mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)))
            unnormed = 0.5 * (actions_np + 1.0) * (q99 - q01) + q01
            actions_np = np.where(mask, unnormed, actions_np)

        return actions_np

    @torch.inference_mode()
    def generate_actions(
        self,
        instruction: str,
        images: Optional[Iterable[Any]] = None,
        processor=None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.predict_action(instruction=instruction, images=images, processor=processor, **kwargs)

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        kwargs.setdefault("_skip_vlm_init", True)
        return super().from_pretrained(*model_args, **kwargs)


# ------------------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------------------
def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor image to PIL Image."""
    if tensor.ndim == 4:
        tensor = tensor[0]  # Remove batch dim
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    arr = tensor.cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def prepare_batch(
    batch: Dict[str, Any],
    model: OpenVLA,
    processor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build model-ready batch using the configured processor.
    Defaults to the Qwen chat-template path when available, but falls back to a
    plain (images, text) call for processors like Prismatic/Prism-Qwen.
    """
    if processor is None:
        raise ValueError("OpenVLA training requires a processor for vision/text encoding.")

    batch = dict(batch)
    instructions = batch.get("language_instruction", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    prompts = [model.build_prompt(instr) for instr in instructions]

    # Extract images and convert to PIL format
    images_tensor = batch["image_input"]
    if isinstance(images_tensor, torch.Tensor) and images_tensor.ndim == 5:
        # [B, V, C, H, W] -> take first view for each sample
        images_tensor = images_tensor[:, 0]  # [B, C, H, W]

    # Convert tensors to PIL images for the processor
    pil_images = [_tensor_to_pil(img) for img in images_tensor]

    processor_outputs = None

    # Try the Qwen-style chat template path when available
    if hasattr(processor, "apply_chat_template") and callable(getattr(processor, "apply_chat_template", None)):
        try:
            conversations = []
            for prompt, pil_img in zip(prompts, pil_images):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                conversations.append(messages)

            texts = [
                processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

            processor_outputs = processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback
            logging.warning("OpenVLA: chat-template preprocessing failed; falling back to direct processor call (%s)", exc)
            processor_outputs = None

    # Fallback: direct processor call that works for Prismatic/Prism-Qwen processors
    if processor_outputs is None:
        import inspect

        def _supports_kwargs(fn, kwargs: Dict[str, Any]) -> bool:
            sig = inspect.signature(fn)
            params = sig.parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return all(k in sig.parameters for k in kwargs.keys())

        processor_kwargs = [
            {"images": pil_images, "text": prompts},
            {"image": pil_images, "text": prompts},
            {"images": pil_images, "language_instruction": prompts},
        ]
        for kwargs in processor_kwargs:
            if _supports_kwargs(processor.__call__, kwargs):
                processor_outputs = processor(
                    **kwargs,
                    return_tensors="pt",
                    padding=True,
                )
                break
        if processor_outputs is None:
            processor_outputs = processor(
                pil_images,
                prompts,
                return_tensors="pt",
                padding=True,
            )

    dtype = next(model.parameters()).dtype
    action_dim = model.get_action_dim()
    action_src = batch["action"]
    if action_src.size(-1) < action_dim:
        raise ValueError(f"Dataset action dim {action_src.size(-1)} < model action dim {action_dim}")

    inputs: Dict[str, torch.Tensor] = {}
    for k, v in processor_outputs.items():
        if isinstance(v, torch.Tensor):
            cast_dtype = dtype if v.is_floating_point() else v.dtype
            inputs[k] = v.to(device=device, dtype=cast_dtype, non_blocking=True)
    inputs["actions"] = action_src[:, 0, :action_dim].to(device=device, dtype=dtype, non_blocking=True)

    return inputs


def prepare_batch_oft(
    batch: Dict[str, Any],
    model: OpenVLAOFT,
    processor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build model-ready batch for OpenVLA-OFT with action chunks.
    """
    if processor is None:
        raise ValueError("OpenVLA-OFT training requires a processor for vision/text encoding.")

    batch = dict(batch)
    instructions = batch.get("language_instruction", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    prompts = [model.build_prompt(instr) for instr in instructions]

    images_tensor = batch["image_input"]
    if isinstance(images_tensor, torch.Tensor) and images_tensor.ndim == 5:
        images_tensor = images_tensor[:, 0]

    pil_images = [_tensor_to_pil(img) for img in images_tensor]

    processor_outputs = None
    if hasattr(processor, "apply_chat_template") and callable(getattr(processor, "apply_chat_template", None)):
        try:
            conversations = []
            for prompt, pil_img in zip(prompts, pil_images):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                conversations.append(messages)

            texts = [
                processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

            processor_outputs = processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback
            logging.warning("OpenVLA-OFT: chat-template preprocessing failed; falling back to direct processor call (%s)", exc)
            processor_outputs = None

    if processor_outputs is None:
        import inspect

        def _supports_kwargs(fn, kwargs: Dict[str, Any]) -> bool:
            sig = inspect.signature(fn)
            params = sig.parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return all(k in sig.parameters for k in kwargs.keys())

        processor_kwargs = [
            {"images": pil_images, "text": prompts},
            {"image": pil_images, "text": prompts},
            {"images": pil_images, "language_instruction": prompts},
        ]
        for kwargs in processor_kwargs:
            if _supports_kwargs(processor.__call__, kwargs):
                processor_outputs = processor(
                    **kwargs,
                    return_tensors="pt",
                    padding=True,
                )
                break
        if processor_outputs is None:
            processor_outputs = processor(
                pil_images,
                prompts,
                return_tensors="pt",
                padding=True,
            )

    dtype = next(model.parameters()).dtype
    action_dim = model.get_action_dim()
    action_src = batch["action"]
    if action_src.size(-1) < action_dim:
        raise ValueError(f"Dataset action dim {action_src.size(-1)} < model action dim {action_dim}")

    actions = action_src[:, : model.num_actions, :action_dim]

    inputs: Dict[str, torch.Tensor] = {}
    for k, v in processor_outputs.items():
        if isinstance(v, torch.Tensor):
            cast_dtype = dtype if v.is_floating_point() else v.dtype
            inputs[k] = v.to(device=device, dtype=cast_dtype, non_blocking=True)

    inputs["actions"] = actions.to(device=device, dtype=dtype, non_blocking=True)
    if "action_mask" in batch:
        inputs["action_mask"] = batch["action_mask"][:, : model.num_actions].to(
            device=device, dtype=torch.long, non_blocking=True
        )

    return inputs


def prepare_batch_adapter(
    batch: Dict[str, Any],
    model: OpenVLAAdapter,
    processor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build model-ready batch for VLA-Adapter regression.
    """
    if processor is None:
        raise ValueError("OpenVLAAdapter training requires a processor for vision/text encoding.")

    batch = dict(batch)
    instructions = batch.get("language_instruction", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    prompts = [model.build_prompt(instr) for instr in instructions]

    images_tensor = batch["image_input"]
    if isinstance(images_tensor, torch.Tensor) and images_tensor.ndim == 5:
        images_tensor = images_tensor[:, 0]

    pil_images = [_tensor_to_pil(img) for img in images_tensor]

    processor_outputs = None
    if hasattr(processor, "apply_chat_template") and callable(getattr(processor, "apply_chat_template", None)):
        try:
            conversations = []
            for prompt, pil_img in zip(prompts, pil_images):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                conversations.append(messages)

            texts = [
                processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

            processor_outputs = processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback
            logging.warning("OpenVLA-Adapter: chat-template preprocessing failed; falling back to direct processor call (%s)", exc)
            processor_outputs = None

    if processor_outputs is None:
        import inspect

        def _supports_kwargs(fn, kwargs: Dict[str, Any]) -> bool:
            sig = inspect.signature(fn)
            params = sig.parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return all(k in sig.parameters for k in kwargs.keys())

        processor_kwargs = [
            {"images": pil_images, "text": prompts},
            {"image": pil_images, "text": prompts},
            {"images": pil_images, "language_instruction": prompts},
        ]
        for kwargs in processor_kwargs:
            if _supports_kwargs(processor.__call__, kwargs):
                processor_outputs = processor(
                    **kwargs,
                    return_tensors="pt",
                    padding=True,
                )
                break
        if processor_outputs is None:
            processor_outputs = processor(
                pil_images,
                prompts,
                return_tensors="pt",
                padding=True,
            )

    dtype = next(model.parameters()).dtype
    action_dim = model.get_action_dim()
    action_src = batch["action"]
    if action_src.size(-1) < action_dim:
        raise ValueError(f"Dataset action dim {action_src.size(-1)} < model action dim {action_dim}")

    actions = action_src[:, : model.num_actions, :action_dim]

    inputs: Dict[str, torch.Tensor] = {}
    for k, v in processor_outputs.items():
        if isinstance(v, torch.Tensor):
            cast_dtype = dtype if v.is_floating_point() else v.dtype
            inputs[k] = v.to(device=device, dtype=cast_dtype, non_blocking=True)

    inputs["actions"] = actions.to(device=device, dtype=dtype, non_blocking=True)
    if "proprio" in batch:
        inputs["proprio"] = batch["proprio"].to(device=device, dtype=dtype, non_blocking=True)
    if "action_mask" in batch:
        inputs["action_mask"] = batch["action_mask"][:, : model.num_actions].to(
            device=device, dtype=torch.long, non_blocking=True
        )

    return inputs


def build_vla_optimizer(
    model: OpenVLA,
    *,
    base_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> AdamW:
    return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=betas)


def update_vla_learning_rates(
    optim: torch.optim.Optimizer,
    step: int,
    *,
    learning_rate: float,
    warmup_steps: int,
    total_iters: int,
    use_cosine_decay: bool,
    min_lr_ratio: float,
) -> None:
    if use_cosine_decay or warmup_steps > 0:
        lr = _linear_warmup_cosine(step, 0, warmup_steps, total_iters, learning_rate, min_lr_ratio)
    else:
        lr = learning_rate
    for group in optim.param_groups:
        group["lr"] = lr


def _linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)

    # ------------------------------------------------------------------
    # HF plumbing
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        kwargs.setdefault("_skip_vlm_init", True)
        return super().from_pretrained(*model_args, **kwargs)
