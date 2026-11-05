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
X-VLA Models Package

Qwen3-VL backbone + MoT (Mixture of Transformers) action head.

Usage:
    from models import XVLA, XVLAConfig

    config = XVLAConfig.from_pretrained("configs/libero/mot_chunk10.json")
    model = XVLA(config)
"""

# XVLA model and config
from .configuration_xvla import XVLAConfig
from .modeling_xvla import (
    XVLA,
    prepare_batch,
    build_vla_optimizer,
    update_vla_learning_rates,
)

# Processor
from .processing_xvla import XVLAProcessor, build_xvla_processor

# MoT action head
from .action_expert import ActionExpert
from .mot_forward import MixtureOfTransformers

# Action space
from .action_hub import build_action_space

__all__ = [
    # Model
    "XVLA",
    "XVLAConfig",
    # Processor
    "XVLAProcessor",
    "build_xvla_processor",
    # Training helpers
    "prepare_batch",
    "build_vla_optimizer",
    "update_vla_learning_rates",
    # Components
    "ActionExpert",
    "MixtureOfTransformers",
    "build_action_space",
]
