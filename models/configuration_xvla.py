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
XVLA Configuration (Qwen3-VL backbone + MoT action head).

Usage:
    config = XVLAConfig(
        qwen3_pretrained="Qwen/Qwen3-VL-2B-Instruct",
    )
"""

from typing import Optional, List
from transformers.configuration_utils import PretrainedConfig


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the XVLA model (Qwen3-VL backbone + MoT action head).
    """

    model_type = "xvla"

    def __init__(
        self,
        # === VLM backbone ===
        vlm_backbone_type: str = "qwen3_vl",
        qwen3_pretrained: str = "Qwen/Qwen3-VL-2B-Instruct",
        qwen3_torch_dtype: str = "bfloat16",
        qwen3_use_flash_attn: bool = True,
        paligemma_pretrained: str = "google/paligemma-3b-pt-224",
        paligemma_torch_dtype: str = "bfloat16",
        paligemma_use_flash_attn: bool = False,

        # === Multi-view settings ===
        num_views: int = 2,

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,
        dim_time: int = 32,
        # === Proprio projector (for feeding proprio to VLM) ===
        proprio_to_vlm: bool = False,
        num_proprio_tokens: int = 8,
        proprio_projector_hidden_dim: int = 512,
        proprio_projector_num_res_blocks: int = 2,
        proprio_projector_mlp_ratio: float = 4.0,
        proprio_projector_dropout: float = 0.0,
        # === MoT action head settings ===
        expert_hidden_size: int = 1024,
        expert_num_layers: int = 28,
        expert_intermediate_size: int = 4096,
        expert_use_qk_norm: Optional[bool] = None,
        # === IC (in-context) settings ===
        num_ic_frames: int = 0,
        ic_sampling_mode: str = "chain",
        include_next_obs: bool = False,
        ic_gap_range: Optional[List[int]] = None,
        num_action_tokens: int = 8,
        action_embedder_hidden_dim: int = 512,
        action_embedder_num_res_blocks: int = 2,
        # === Frame position / separator ===
        use_frame_sep: bool = False,
        use_frame_position_embed: bool = False,
        # === IC frame loss re-weighting ===
        ic_frame_loss_reweight: bool = False,
        ic_frame_loss_weight_min: float = 0.1,
        ic_frame_loss_weight_max: float = 1.0,
        **kwargs,
    ):
        # VLM backbone configuration
        self.vlm_backbone_type = vlm_backbone_type
        self.qwen3_pretrained = qwen3_pretrained
        self.qwen3_torch_dtype = qwen3_torch_dtype
        self.qwen3_use_flash_attn = qwen3_use_flash_attn
        self.paligemma_pretrained = paligemma_pretrained
        self.paligemma_torch_dtype = paligemma_torch_dtype
        self.paligemma_use_flash_attn = paligemma_use_flash_attn

        # Multi-view settings
        self.num_views = num_views

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        self.dim_time = dim_time
        self.proprio_to_vlm = proprio_to_vlm
        self.num_proprio_tokens = num_proprio_tokens
        self.proprio_projector_hidden_dim = proprio_projector_hidden_dim
        self.proprio_projector_num_res_blocks = proprio_projector_num_res_blocks
        self.proprio_projector_mlp_ratio = proprio_projector_mlp_ratio
        self.proprio_projector_dropout = proprio_projector_dropout

        # MoT action head settings
        self.expert_hidden_size = expert_hidden_size
        self.expert_num_layers = expert_num_layers
        self.expert_intermediate_size = expert_intermediate_size
        self.expert_use_qk_norm = expert_use_qk_norm

        # IC settings
        self.num_ic_frames = num_ic_frames
        self.ic_sampling_mode = ic_sampling_mode
        self.include_next_obs = include_next_obs
        self.ic_gap_range = ic_gap_range if ic_gap_range is not None else [0, 0]
        self.num_action_tokens = num_action_tokens
        self.action_embedder_hidden_dim = action_embedder_hidden_dim
        self.action_embedder_num_res_blocks = action_embedder_num_res_blocks

        # Frame position / separator
        self.use_frame_sep = use_frame_sep
        self.use_frame_position_embed = use_frame_position_embed

        # IC frame loss re-weighting
        self.ic_frame_loss_reweight = ic_frame_loss_reweight
        self.ic_frame_loss_weight_min = ic_frame_loss_weight_min
        self.ic_frame_loss_weight_max = ic_frame_loss_weight_max

        super().__init__(**kwargs)
