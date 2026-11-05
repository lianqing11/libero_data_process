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

from __future__ import annotations

from transformers import ProcessorMixin, AutoProcessor
from typing import List, Union, Dict, Any, Optional, Tuple
import torch
import numpy as np

from .hf_local import resolve_pretrained_load


class XVLAProcessor(ProcessorMixin):
    """
    XVLAProcessor: Unified multimodal processor for XVLA models (Qwen3-VL backbone).

    Handles:
      - Multi-view image inputs (e.g., from multiple cameras).
      - Batch processing for multiple samples.
      - Joint tokenization and image tensor preparation.
    """

    num_views: int = 2
    language_max_length: int = 50
    vlm_process_all_views: bool = False

    # Hugging Face ProcessorMixin-required metadata
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = (
        "AutoImageProcessor",
        "SiglipImageProcessor",
        "SiglipImageProcessorFast",
        "Qwen2VLImageProcessor",
        "Qwen2VLImageProcessorFast",
    )
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast", "GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        backbone_type: str = "qwen3_vl",
        vlm_process_all_views: bool = False,
        proprio_to_vlm: bool = False,
        language_max_length: int = 50,
        image_token: str = "<image>",
        image_token_id: Optional[int] = None,
        image_seq_length: Optional[int] = None,
    ):
        super().__init__(image_processor, tokenizer)
        self.backbone_type = backbone_type
        self.vlm_process_all_views = vlm_process_all_views
        self.proprio_to_vlm = proprio_to_vlm
        self.language_max_length = language_max_length
        self.num_proprio_tokens = 1  # updated by model.set_processor()
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length

    def get_tokens_for_embedding_init(self) -> List[Tuple[str, str]]:
        """
        Returns special tokens that need embedding initialization with reference tokens.

        These tokens should be initialized from the reference token's embedding + small noise.
        Used by the model to properly initialize newly added special tokens.
        """
        init_pairs = []

        if self.backbone_type == "paligemma":
            bos = self.tokenizer.bos_token or self.tokenizer.eos_token
            image_ref = self.image_token if self.image_token_id is not None else bos
            init_pairs.extend([
                ("<|frame_start|>", bos),
                ("<|frame_end|>", bos),
                ("<|frame_sep|>", bos),
                ("<|action_start|>", bos),
                ("<|action_pad|>", image_ref),
                ("<|action_end|>", bos),
            ])
            if self.proprio_to_vlm:
                init_pairs.extend([
                    ("<|proprio_start|>", bos),
                    ("<|proprio_pad|>", image_ref),
                    ("<|proprio_end|>", bos),
                ])
            return init_pairs

        if self.proprio_to_vlm:
            init_pairs.extend([
                ("<|proprio_start|>", "<|vision_start|>"),
                ("<|proprio_pad|>", "<|image_pad|>"),
                ("<|proprio_end|>", "<|vision_end|>"),
            ])
        init_pairs.append(("<|frame_sep|>", "<|frame_end|>"))

        return init_pairs

    @classmethod
    def from_pretrained_vlm(
        cls,
        pretrained_name_or_path: str,
        backbone_type: str = "qwen3_vl",
        num_views: int = 2,
        language_max_length: int = None,
        proprio_to_vlm: bool = False,
        vlm_process_all_views: bool = False,
        **kwargs,
    ) -> "XVLAProcessor":
        """
        Create processor from a pretrained Qwen3-VL model.
        """
        import logging

        resolved_path, load_kwargs = resolve_pretrained_load(
            pretrained_name_or_path,
            required_files=("tokenizer_config.json", "preprocessor_config.json"),
            **kwargs,
        )
        vlm_processor = AutoProcessor.from_pretrained(resolved_path, **load_kwargs)

        # Collect all special tokens to add
        # IC frame/action tokens are always added (needed for checkpoint compat)
        special_tokens_to_add = [
            "<|frame_start|>", "<|frame_end|>", "<|frame_sep|>",
            "<|action_start|>", "<|action_pad|>", "<|action_end|>",
        ]
        if proprio_to_vlm:
            special_tokens_to_add.extend(["<|proprio_start|>", "<|proprio_pad|>", "<|proprio_end|>"])

        if special_tokens_to_add:
            special_tokens = {"additional_special_tokens": special_tokens_to_add}
            num_added = vlm_processor.tokenizer.add_special_tokens(
                special_tokens, replace_additional_special_tokens=False,
            )
            if num_added > 0:
                logging.info(f"Added {num_added} special tokens to processor")

        if language_max_length is None:
            language_max_length = 50

        image_token = getattr(vlm_processor, "image_token", "<image>")
        image_token_id = getattr(vlm_processor, "image_token_id", None)
        image_seq_length = getattr(vlm_processor, "image_seq_length", None)

        processor = cls(
            image_processor=vlm_processor.image_processor,
            tokenizer=vlm_processor.tokenizer,
            backbone_type=backbone_type,
            vlm_process_all_views=vlm_process_all_views,
            proprio_to_vlm=proprio_to_vlm,
            language_max_length=language_max_length,
            image_token=image_token,
            image_token_id=image_token_id,
            image_seq_length=image_seq_length,
        )
        processor.num_views = num_views
        return processor

    # ================== LANGUAGE ENCODING ==================
    def encode_language(
        self,
        language_instruction: Union[str, List[str]],
        image_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        num_proprio_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize language instructions for Qwen3-VL.

        Returns {"input_ids": [B, L], "attention_mask": [B, L]}
        """
        if num_proprio_tokens is None:
            num_proprio_tokens = self.num_proprio_tokens
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        if self.backbone_type == "paligemma":
            return self._encode_language_paligemma(
                language_instruction, image_mask=image_mask, num_proprio_tokens=num_proprio_tokens,
            )

        # View name templates
        view_names = ["main view: ", "left wrist view: ", "right wrist view: "] + [
            f"view {i}: " for i in range(3, self.num_views)
        ]

        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for Qwen3-VL encode_language")
        merge_length = self.image_processor.merge_size ** 2
        image_pad = "<|image_pad|>"
        proprio_pad = "<|proprio_pad|>"

        formatted_instructions = []
        for idx, inst in enumerate(language_instruction):
            sample_mask = None
            if image_mask is not None:
                sample_mask = (image_mask[idx] if image_mask.dim() > 1 else image_mask).tolist()

            if self.vlm_process_all_views:
                valid_views = [i for i in range(self.num_views) if sample_mask is None or sample_mask[i]]
            else:
                valid_views = [0]

            parts = []
            for v in valid_views:
                n = int(image_grid_thw[idx, v].prod().item()) // merge_length
                tag = f"{view_names[v]}" if self.vlm_process_all_views else ""
                parts.append(f"{tag}<|vision_start|>{image_pad * n}<|vision_end|>")
            image_placeholders = "".join(parts)

            if self.proprio_to_vlm:
                proprio_ph = f"<|proprio_start|>{proprio_pad * num_proprio_tokens}<|proprio_end|>"
                formatted = f"<|im_start|>user\n{image_placeholders}{inst}{proprio_ph}<|im_end|>"
            else:
                formatted = f"<|im_start|>user\n{image_placeholders}{inst}<|im_end|>"

            formatted_instructions.append(formatted)

        inputs = self.tokenizer(
            formatted_instructions,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    def _encode_language_paligemma(
        self,
        language_instruction: List[str],
        image_mask: Optional[torch.Tensor] = None,
        num_proprio_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.image_seq_length is None:
            raise ValueError("PaliGemma image_seq_length is required for encode_language")
        image_token = self.image_token
        proprio_pad = "<|proprio_pad|>"
        bos = self.tokenizer.bos_token or ""

        formatted_instructions = []
        for idx, inst in enumerate(language_instruction):
            sample_mask = None
            if image_mask is not None:
                sample_mask = (image_mask[idx] if image_mask.dim() > 1 else image_mask).tolist()
            valid_views = [i for i in range(self.num_views) if sample_mask is None or sample_mask[i]]
            if not self.vlm_process_all_views:
                valid_views = valid_views[:1]
            image_placeholders = image_token * self.image_seq_length * len(valid_views)
            if self.proprio_to_vlm:
                proprio_ph = f"<|proprio_start|>{proprio_pad * num_proprio_tokens}<|proprio_end|>"
                formatted = f"{image_placeholders}{bos}{inst}{proprio_ph}\n"
            else:
                formatted = f"{image_placeholders}{bos}{inst}\n"
            formatted_instructions.append(formatted)

        inputs = self.tokenizer(
            formatted_instructions,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    # ================== IC SEQUENCE ENCODING ==================
    def encode_ic_sequence(
        self,
        language_instruction: Union[str, List[str]],
        num_frames: int,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        num_proprio_tokens: Optional[int] = None,
        num_action_tokens: int = 8,
        include_next_obs: bool = False,
        next_image_grid_thw: Optional[torch.Tensor] = None,
        use_frame_sep: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Build multi-frame IC input_ids for K frames.

        Template per IC frame (i<K-1), without next_obs:
          <|frame_start|>[views][proprio]<|action_start|>[action_pads]<|action_end|><|frame_end|>
        Template per IC frame (i<K-1), with next_obs:
          <|frame_start|>[views][proprio]<|action_start|>[action_pads]<|action_end|>[next_views][next_proprio]<|frame_end|>
        Query frame (i==K-1):
          <|frame_start|>[views][proprio]<|frame_end|>

        Returns dict with 'input_ids' [B, L], 'attention_mask' [B, L], 'frame_obs_ends' [B, K].
        """
        if num_proprio_tokens is None:
            num_proprio_tokens = self.num_proprio_tokens
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        if self.backbone_type == "paligemma":
            return self._encode_ic_sequence_paligemma(
                language_instruction,
                num_frames,
                image_mask=image_mask,
                num_proprio_tokens=num_proprio_tokens,
                num_action_tokens=num_action_tokens,
                include_next_obs=include_next_obs,
                use_frame_sep=use_frame_sep,
            )

        K = num_frames
        merge_length = self.image_processor.merge_size ** 2
        image_pad = "<|image_pad|>"
        proprio_pad = "<|proprio_pad|>"
        action_pad = "<|action_pad|>"

        view_names = ["main view: ", "left wrist view: ", "right wrist view: "] + [
            f"view {i}: " for i in range(3, self.num_views)
        ]

        formatted = []
        for idx, inst in enumerate(language_instruction):
            sample_mask = None
            if image_mask is not None:
                sample_mask = (image_mask[idx] if image_mask.dim() > 1 else image_mask).tolist()
            valid_views = [i for i in range(self.num_views) if sample_mask is None or sample_mask[i]]
            if not self.vlm_process_all_views:
                valid_views = valid_views[:1]

            instruction_part = f"<|im_start|>user\n{inst}"
            frame_parts = []
            for k in range(K):
                view_phs = []
                for v in valid_views:
                    n = int(image_grid_thw[idx, k * self.num_views + v].prod().item()) // merge_length
                    tag = f"{view_names[v]}" if self.vlm_process_all_views else ""
                    view_phs.append(f"{tag}<|vision_start|>{image_pad * n}<|vision_end|>")
                obs_str = "".join(view_phs)
                proprio_str = f"<|proprio_start|>{proprio_pad * num_proprio_tokens}<|proprio_end|>" if self.proprio_to_vlm else ""

                if k < K - 1:
                    action_str = f"<|action_start|>{action_pad * num_action_tokens}<|action_end|>"
                    next_obs_str = ""
                    if include_next_obs and next_image_grid_thw is not None:
                        next_view_phs = []
                        for v in valid_views:
                            nn_ = int(next_image_grid_thw[idx, k * self.num_views + v].prod().item()) // merge_length
                            tag = f"next {view_names[v]}" if self.vlm_process_all_views else ""
                            next_view_phs.append(f"{tag}<|vision_start|>{image_pad * nn_}<|vision_end|>")
                        next_obs_str = "".join(next_view_phs)
                        if self.proprio_to_vlm:
                            next_obs_str += f"<|proprio_start|>{proprio_pad * num_proprio_tokens}<|proprio_end|>"
                    frame_parts.append(f"<|frame_start|>{obs_str}{proprio_str}{action_str}{next_obs_str}<|frame_end|>")
                else:
                    frame_parts.append(f"<|frame_start|>{obs_str}{proprio_str}<|frame_end|>")

            joiner = "<|frame_sep|>" if use_frame_sep else ""
            full_text = instruction_part + joiner.join(frame_parts) + "<|im_end|>"
            formatted.append(full_text)

        inputs = self.tokenizer(formatted, return_tensors="pt", padding="longest", truncation=False)
        input_ids = inputs["input_ids"]

        # Compute frame_obs_ends [B, K]: position of last obs token before action or end per frame
        frame_start_id = self.tokenizer.convert_tokens_to_ids("<|frame_start|>")
        action_start_id = self.tokenizer.convert_tokens_to_ids("<|action_start|>")
        frame_end_id = self.tokenizer.convert_tokens_to_ids("<|frame_end|>")

        B, L = input_ids.shape
        frame_obs_ends = torch.zeros(B, K, dtype=torch.long)
        for b in range(B):
            frame_idx = 0
            for pos in range(L):
                tid = input_ids[b, pos].item()
                if tid == action_start_id and frame_idx < K:
                    frame_obs_ends[b, frame_idx] = pos
                    frame_idx += 1
                elif tid == frame_end_id and frame_idx == K - 1:
                    frame_obs_ends[b, frame_idx] = pos
                    frame_idx += 1

        return {"input_ids": input_ids, "attention_mask": inputs["attention_mask"], "frame_obs_ends": frame_obs_ends}

    def _encode_ic_sequence_paligemma(
        self,
        language_instruction: List[str],
        num_frames: int,
        image_mask: Optional[torch.Tensor] = None,
        num_proprio_tokens: Optional[int] = None,
        num_action_tokens: int = 8,
        include_next_obs: bool = False,
        use_frame_sep: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.image_seq_length is None:
            raise ValueError("PaliGemma image_seq_length is required for encode_ic_sequence")
        K = num_frames
        image_token = self.image_token
        proprio_pad = "<|proprio_pad|>"
        action_pad = "<|action_pad|>"
        bos = self.tokenizer.bos_token or ""

        formatted = []
        for idx, inst in enumerate(language_instruction):
            sample_mask = None
            if image_mask is not None:
                sample_mask = (image_mask[idx] if image_mask.dim() > 1 else image_mask).tolist()
            valid_views = [i for i in range(self.num_views) if sample_mask is None or sample_mask[i]]
            if not self.vlm_process_all_views:
                valid_views = valid_views[:1]
            view_ph = image_token * self.image_seq_length * len(valid_views)
            proprio_str = (
                f"<|proprio_start|>{proprio_pad * num_proprio_tokens}<|proprio_end|>"
                if self.proprio_to_vlm else ""
            )

            frame_parts = []
            for k in range(K):
                obs_str = view_ph + proprio_str
                if k < K - 1:
                    action_str = f"<|action_start|>{action_pad * num_action_tokens}<|action_end|>"
                    next_obs_str = view_ph + proprio_str if include_next_obs else ""
                    frame_parts.append(f"<|frame_start|>{obs_str}{action_str}{next_obs_str}<|frame_end|>")
                else:
                    frame_parts.append(f"<|frame_start|>{obs_str}<|frame_end|>")

            joiner = "<|frame_sep|>" if use_frame_sep else ""
            formatted.append(f"{bos}{inst}{joiner.join(frame_parts)}\n")

        inputs = self.tokenizer(formatted, return_tensors="pt", padding="longest", truncation=False)
        input_ids = inputs["input_ids"]
        frame_start_id = self.tokenizer.convert_tokens_to_ids("<|frame_start|>")
        action_start_id = self.tokenizer.convert_tokens_to_ids("<|action_start|>")
        frame_end_id = self.tokenizer.convert_tokens_to_ids("<|frame_end|>")

        B, L = input_ids.shape
        frame_obs_ends = torch.zeros(B, K, dtype=torch.long)
        for b in range(B):
            frame_idx = 0
            for pos in range(L):
                tid = input_ids[b, pos].item()
                if tid == action_start_id and frame_idx < K:
                    frame_obs_ends[b, frame_idx] = pos
                    frame_idx += 1
                elif tid == frame_end_id and frame_idx == K - 1:
                    frame_obs_ends[b, frame_idx] = pos
                    frame_idx += 1

        return {"input_ids": input_ids, "attention_mask": inputs["attention_mask"], "frame_obs_ends": frame_obs_ends}

    # ================== IMAGE ENCODING ==================
    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Normalize to batch form
        if not isinstance(images[0], (list, tuple)):
            images = [images]  # convert single sample to batch of size 1

        if self.backbone_type == "paligemma":
            return self._encode_image_paligemma(images, **kwargs)
        return self._encode_image_qwen3(images, **kwargs)

    def _encode_image_paligemma(
        self,
        images: List[List],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        from PIL import Image as PILImage

        batch_imgs, batch_masks = [], []
        for sample_imgs in images:
            sample_processed = []
            for img in sample_imgs:
                if not isinstance(img, PILImage.Image):
                    if isinstance(img, np.ndarray):
                        img = PILImage.fromarray(img)
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")
                processed = self.image_processor([img], return_tensors="pt", **kwargs)
                sample_processed.append(processed["pixel_values"].squeeze(0))

            sample_processed = sample_processed[:self.num_views]
            V_exist = len(sample_processed)
            if sample_processed:
                stacked = torch.stack(sample_processed, dim=0)
                if V_exist < self.num_views:
                    pad_shape = (self.num_views - V_exist, *stacked.shape[1:])
                    stacked = torch.cat([stacked, torch.zeros(pad_shape, dtype=stacked.dtype)], dim=0)
            else:
                size = getattr(self.image_processor, "size", {}) or {}
                height = int(size.get("height", size.get("shortest_edge", 224)))
                width = int(size.get("width", height))
                stacked = torch.zeros(self.num_views, 3, height, width)
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True
            batch_imgs.append(stacked)
            batch_masks.append(image_mask)

        return {
            "image_input": torch.stack(batch_imgs, dim=0),
            "image_mask": torch.stack(batch_masks, dim=0),
        }

    def _encode_image_qwen3(
        self,
        images: List[List],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Qwen3-VL style image encoding using NATIVE processor.

        Returns pre-embedded patches [num_patches, embed_dim] from Qwen3-VL's
        pre-trained Conv3d patch embedding, preserving full model capacity.
        """
        from PIL import Image as PILImage

        batch_imgs, batch_masks = [], []
        all_grid_thw = []

        for sample_imgs in images:
            sample_processed = []
            sample_grid_thw = []

            for img in sample_imgs:
                if not isinstance(img, PILImage.Image):
                    if isinstance(img, np.ndarray):
                        img = PILImage.fromarray(img)
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")

                processed = self.image_processor([img], return_tensors="pt", **kwargs)
                pixel_values = processed["pixel_values"]

                if "image_grid_thw" in processed:
                    sample_grid_thw.append(processed["image_grid_thw"])
                else:
                    import logging, math
                    logging.warning("image_grid_thw not in processor output, computing fallback")
                    num_patches = pixel_values.shape[0]
                    grid_size = int(math.sqrt(num_patches))
                    sample_grid_thw.append(torch.tensor([[1, grid_size, grid_size]]))

                sample_processed.append(pixel_values)

            V_exist = len(sample_processed)

            if sample_processed:
                max_patches = max(p.shape[0] for p in sample_processed)
                embed_dim = sample_processed[0].shape[1]

                padded_views = []
                for patches in sample_processed:
                    num_patches = patches.shape[0]
                    if num_patches < max_patches:
                        pad = torch.zeros(max_patches - num_patches, embed_dim,
                                         device=patches.device, dtype=patches.dtype)
                        patches = torch.cat([patches, pad], dim=0)
                    padded_views.append(patches)

                stacked = torch.stack(padded_views, dim=0)

                if V_exist < self.num_views:
                    pad_shape = (self.num_views - V_exist, max_patches, embed_dim)
                    pad_views = torch.zeros(pad_shape, device=stacked.device, dtype=stacked.dtype)
                    stacked = torch.cat([stacked, pad_views], dim=0)

                    if sample_grid_thw:
                        pad_grid = torch.zeros(self.num_views - V_exist, 3, dtype=torch.long)
                        sample_grid_thw.append(pad_grid)
            else:
                stacked = torch.zeros(self.num_views, 256, 1536)
                sample_grid_thw = [torch.tensor([[1, 16, 16]])] * self.num_views

            batch_imgs.append(stacked)

            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True
            batch_masks.append(image_mask)

            if sample_grid_thw:
                all_grid_thw.append(torch.cat(sample_grid_thw, dim=0))

        image_input = torch.stack(batch_imgs, dim=0)
        image_mask = torch.stack(batch_masks, dim=0)

        result = {"image_input": image_input, "image_mask": image_mask}

        if all_grid_thw:
            result["image_grid_thw"] = torch.stack(all_grid_thw, dim=0)

        return result

    # ================== COMBINED CALL ==================
    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Combine image and text encoding into a unified multimodal input.
        """
        outputs: Dict[str, Any] = {}

        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        if language_instruction is not None:
            image_mask = outputs.get("image_mask", None)
            image_grid_thw = outputs.get("image_grid_thw", None)
            outputs.update(self.encode_language(
                language_instruction, image_mask=image_mask, image_grid_thw=image_grid_thw,
            ))

        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs


# =============================================================================
# Factory function for creating processor
# =============================================================================

def build_xvla_processor(
    pretrained_name_or_path: Optional[str] = None,
    backbone_type: str = "qwen3_vl",
    num_views: int = 2,
    language_max_length: int = None,
    proprio_to_vlm: bool = False,
    vlm_process_all_views: bool = False,
    **kwargs,
) -> XVLAProcessor:
    """
    Factory function to build XVLA processor for the selected VLM backbone.
    """
    if pretrained_name_or_path is None:
        pretrained_name_or_path = (
            "google/paligemma-3b-pt-224"
            if backbone_type == "paligemma" else "Qwen/Qwen3-VL-2B-Instruct"
        )

    return XVLAProcessor.from_pretrained_vlm(
        pretrained_name_or_path=pretrained_name_or_path,
        backbone_type=backbone_type,
        num_views=num_views,
        language_max_length=language_max_length,
        proprio_to_vlm=proprio_to_vlm,
        vlm_process_all_views=vlm_process_all_views,
        **kwargs,
    )
