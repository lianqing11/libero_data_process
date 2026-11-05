from __future__ import annotations

from typing import Any, Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class OpenVLAConfig(PretrainedConfig):
    """
    Lightweight configuration for the OpenVLA wrapper.

    The config stores the identifier for the base VLM (e.g., ``Qwen/Qwen2.5-VL-3B-Instruct``),
    discretisation hyper-parameters for the action tokenizer, and metadata that is required
    to convert between the normalized OpenVLA action representation and the original action
    units from the training dataset.
    """

    model_type = "openvla"

    def __init__(
        self,
        vlm_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        processor_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        action_dim: int = 7,
        action_bins: int = 256,
        action_min: float = -1.0,
        action_max: float = 1.0,
        predict_stop_token: bool = True,
        prompt_template: str = "What action should the robot take to {instruction}?",
        action_stats: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        trust_remote_code: bool = True,
        auto_processor: bool = True,
        vlm_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            vlm_model_name: Hugging Face identifier for the base VLM checkpoint.
            processor_name: Identifier passed to ``AutoProcessor``. Defaults to ``vlm_model_name``.
            tokenizer_name: Identifier passed to ``AutoTokenizer``. Defaults to ``vlm_model_name``.
            action_dim: Number of continuous action dimensions that OpenVLA should predict.
            action_bins: Number of discretisation bins used by the action tokenizer.
            action_min: Minimum (normalized) action value prior to discretisation.
            action_max: Maximum (normalized) action value prior to discretisation.
            predict_stop_token: Whether the datasets include an end-of-action token target.
            prompt_template: Template used to build the action prediction prompt.
            action_stats: Optional per-dataset statistics for de-normalizing actions.
            trust_remote_code: Forwarded to HF ``Auto*`` loaders for models that require it.
            auto_processor: Attempt to automatically instantiate an ``AutoProcessor``.
            vlm_model_kwargs: Additional kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``.
        """
        super().__init__(**kwargs)

        self.vlm_model_name = vlm_model_name
        self.processor_name = processor_name or vlm_model_name
        self.tokenizer_name = tokenizer_name or vlm_model_name
        self.action_dim = int(action_dim)
        self.action_bins = int(action_bins)
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.predict_stop_token = bool(predict_stop_token)
        self.prompt_template = prompt_template
        self.action_stats = action_stats or {}
        self.trust_remote_code = bool(trust_remote_code)
        self.auto_processor = bool(auto_processor)
        self.vlm_model_kwargs = vlm_model_kwargs or {}

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output.update(
            {
                "vlm_model_name": self.vlm_model_name,
                "processor_name": self.processor_name,
                "tokenizer_name": self.tokenizer_name,
                "action_dim": self.action_dim,
                "action_bins": self.action_bins,
                "action_min": self.action_min,
                "action_max": self.action_max,
                "predict_stop_token": self.predict_stop_token,
                "prompt_template": self.prompt_template,
                "action_stats": self.action_stats,
                "trust_remote_code": self.trust_remote_code,
                "auto_processor": self.auto_processor,
                "vlm_model_kwargs": self.vlm_model_kwargs,
            }
        )
        return output


class OpenVLAOFTConfig(OpenVLAConfig):
    """
    Configuration for OpenVLA-OFT (OpenVLA with action-chunk regression).

    Extends ``OpenVLAConfig`` with settings for action chunking and the regression head.
    """

    model_type = "openvla-oft"

    def __init__(
        self,
        num_action_chunk: int = 8,
        action_head_hidden_dim: Optional[int] = None,
        action_placeholder_token_id: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            num_action_chunk: Number of future actions to predict per forward pass.
            action_head_hidden_dim: Hidden width for the regression MLP head (defaults to VLM hidden size).
            action_placeholder_token_id: Token id used to reserve slots for action tokens in the sequence.
        """
        super().__init__(**kwargs)
        self.num_action_chunk = int(num_action_chunk)
        self.action_head_hidden_dim = (
            int(action_head_hidden_dim) if action_head_hidden_dim is not None else None
        )
        self.action_placeholder_token_id = (
            int(action_placeholder_token_id) if action_placeholder_token_id is not None else None
        )

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output.update(
            {
                "num_action_chunk": self.num_action_chunk,
                "action_head_hidden_dim": self.action_head_hidden_dim,
                "action_placeholder_token_id": self.action_placeholder_token_id,
            }
        )
        return output


class OpenVLAAdapterConfig(OpenVLAConfig):
    """
    Configuration for VLA-Adapter style regression with adapter tokens and cross-attention.
    """

    model_type = "vla-adapter"

    def __init__(
        self,
        num_action_chunk: int = 8,
        num_adapter_tokens: int = 16,
        adapter_num_heads: int = 8,
        adapter_hidden_dim: Optional[int] = None,
        action_head_hidden_dim: Optional[int] = None,
        action_placeholder_token_id: Optional[int] = None,
        use_pro_version: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            num_action_chunk: Number of actions predicted per forward pass.
            num_adapter_tokens: Learnable adapter tokens mixed with action queries.
            adapter_num_heads: Attention heads used inside the adapter head.
            adapter_hidden_dim: Internal hidden size for the adapter attention projections (defaults to model hidden).
            action_head_hidden_dim: Hidden width for the regression MLP (defaults to model hidden).
            action_placeholder_token_id: Token id used to reserve slots for adapter/action tokens.
            use_pro_version: Enable deeper residual blocks inspired by VLA-Adapter Pro.
        """
        super().__init__(**kwargs)
        self.num_action_chunk = int(num_action_chunk)
        self.num_adapter_tokens = int(num_adapter_tokens)
        self.adapter_num_heads = int(adapter_num_heads)
        self.adapter_hidden_dim = adapter_hidden_dim
        self.action_head_hidden_dim = action_head_hidden_dim
        self.action_placeholder_token_id = (
            int(action_placeholder_token_id) if action_placeholder_token_id is not None else None
        )
        self.use_pro_version = bool(use_pro_version)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output.update(
            {
                "num_action_chunk": self.num_action_chunk,
                "num_adapter_tokens": self.num_adapter_tokens,
                "adapter_num_heads": self.adapter_num_heads,
                "adapter_hidden_dim": self.adapter_hidden_dim,
                "action_head_hidden_dim": self.action_head_hidden_dim,
                "action_placeholder_token_id": self.action_placeholder_token_id,
                "use_pro_version": self.use_pro_version,
            }
        )
        return output
