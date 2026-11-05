#!/usr/bin/env python3
"""
Run LIBERO evaluation directly against a locally loaded VLA model (no HTTP server).

Supports XVLA as well as OpenVLA, OpenVLA-OFT, and VLA-Adapter checkpoints. This
script can be used as a standalone evaluator or imported from training to trigger
periodic evaluations.
"""
import argparse
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Set these before importing transformers/huggingface_hub: they cache offline
# flags at import time. Evaluation only consumes pre-downloaded local assets.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import will be done dynamically based on --sim argument
try:
    from peft import PeftModel
except ImportError:  # Optional dependency
    PeftModel = None

logger = logging.getLogger("evaluate")

# Runtime perturbation types (applied at evaluation time, not baked into dataset)
RUNTIME_PERTURBATIONS_ALL = [
    "camera_star", "camera_plusplus", "camera_plusplus_star", "action", "action_star",
]
# Default runtime perturbations for libero_plus (excludes camera_star, camera_plusplus_star & action_star)
RUNTIME_PERTURBATIONS_DEFAULT = ["camera_plusplus", "action"]
# Base suites to evaluate with runtime perturbations
RUNTIME_PERTURBATION_BASE_SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]


def _resolve_checkpoint_output_dir(output_dir: str, model_path: str) -> Path:
    model_dir = Path(model_path).expanduser()
    run_dir_name = f"{model_dir.parent.name}_{model_dir.name}"
    base_dir = Path(output_dir).expanduser()
    if base_dir.name == run_dir_name:
        return base_dir
    return base_dir / run_dir_name


def _import_libero_client(sim_type: str):
    """
    Dynamically import the appropriate libero client module based on sim_type.

    Args:
        sim_type: Either 'libero' or 'libero_plus'

    Returns:
        Module containing LiberoAbsActionProcessor, _flip_agentview, and eval_libero
    """
    if sim_type == "libero":
        from evaluation.libero import libero_client as client_module
    elif sim_type == "libero_plus":
        from evaluation.libero import libero_plus_client as client_module
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}. Must be 'libero' or 'libero_plus'")

    return client_module


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _get_suite_episode_counts(
    client_module: Any,
    task_suites: Iterable[str],
    num_episodes: int,
    max_tasks: Optional[int],
) -> Dict[str, int]:
    """Return actual evaluated episode counts per suite."""
    counts: Dict[str, int] = {}
    for suite_name in task_suites:
        suite_keys = client_module.LIBERO_DATASETS[suite_name]
        total_tasks = sum(len(client_module.benchmark_dict[key]().tasks) for key in suite_keys)
        if max_tasks is not None and max_tasks > 0:
            total_tasks = min(total_tasks, max_tasks)
        counts[suite_name] = total_tasks * num_episodes
    return counts


class EvalRunResult(dict):
    """
    Success rates per suite plus average steps (including max horizon for failures).
    Behaves like a dict keyed by suite name for backward compatibility.
    """

    def __init__(self, success_rates: Dict[str, float], avg_steps: Dict[str, float]):
        super().__init__(success_rates)
        self.avg_steps = avg_steps

    def __repr__(self) -> str:
        return f"EvalRunResult(success_rate={dict(self)}, avg_steps={self.avg_steps})"




def _normalize_arch(name: str) -> str:
    name = name.lower()
    if name in {"openvla_oft", "openvlaoft"}:
        return "openvla-oft"
    if name in {"vla_adapter", "vlaadapter"}:
        return "vla-adapter"
    return name


class LocalVLAAgent:
    """
    Thin policy wrapper that calls an in-memory VLA model (XVLA or OpenVLA variants)
    to produce actions. Mirrors the HTTP client behaviour used by the FastAPI server.

    IC (in-context) mode: num_ic_frames=1 uses a query-only IC prompt. When
    num_ic_frames > 1, each model-call step is recorded into episode_history and
    as many available history entries as possible are used as IC context.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        denoising_steps: int = 10,
        domain_id: int = 3,
        model_arch: str = "xvla",
        unnorm_key: Optional[str] = None,
        gripper_close_threshold: float = 0.5,
        client_module: Any = None,
        action_mode: str = "ee6d",
        action_chunk_size: Optional[int] = None,
        gripper_curve_max_points: int = 200,
        num_ic_frames: int = 0,
        include_next_obs: bool = False,
        clone_query_as_demo: bool = False,
        ic_demo_mode: str = "normal",
        zero_ic_actions: bool = False,
        ic_stride: int = 1,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.denoising_steps = denoising_steps
        self.domain_id = torch.tensor([domain_id], device=self.device, dtype=torch.long)
        self.model_arch = _normalize_arch(model_arch)
        self.unnorm_key = unnorm_key
        self.gripper_close_threshold = gripper_close_threshold

        self.action_chunk_size = action_chunk_size
        if action_chunk_size is not None:
            logger.info(f"Action chunk size limited to {action_chunk_size}")

        self.use_qwen_template = True

        if client_module is None:
            from evaluation.libero import libero_client as client_module
        self.client_module = client_module
        self.rot_processor = client_module.LiberoAbsActionProcessor()
        self._flip_agentview = client_module._flip_agentview

        self.dtype = next(model.parameters()).dtype
        self.gripper_curve_max_points = gripper_curve_max_points

        # IC settings
        self.num_ic_frames = num_ic_frames
        self.include_next_obs = include_next_obs
        self.clone_query_as_demo = clone_query_as_demo
        if clone_query_as_demo:
            logger.info("IC demo will be cloned from query (diagnostic mode)")
        self.ic_demo_mode = ic_demo_mode
        self.zero_ic_actions = zero_ic_actions
        self.ic_stride = max(1, ic_stride)
        if self.ic_stride > 1:
            logger.info("IC stride: select every %d-th history entry as demo", self.ic_stride)
        if ic_demo_mode != "normal":
            logger.info("IC demo mode: %s", ic_demo_mode)
        if zero_ic_actions:
            logger.info("IC actions zeroed out (diagnostic mode)")

        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: deque[List[float]] = deque()
        self.gripper_trace: List[float] = []
        # Episode history for IC: each entry = {pil_images, proprio, action_chunk}
        self.episode_history: List[Dict] = []

    def _to_device(self, t) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.is_floating_point():
            return t.to(device=self.device, dtype=self.dtype)
        return t.to(device=self.device)

    @staticmethod
    def _to_pil_rgb(image: np.ndarray, *, view_name: str = "image") -> Image.Image:
        arr = np.asarray(image)
        if arr.ndim == 2:
            if min(arr.shape) <= 4:
                raise ValueError(f"{view_name} has invalid image shape {arr.shape}; expected HxWx3.")
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3:
            if min(arr.shape[:2]) <= 4:
                raise ValueError(f"{view_name} has invalid image shape {arr.shape}; expected HxWx3.")
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] != 3:
                raise ValueError(f"{view_name} has invalid image shape {arr.shape}; expected HxWx3.")
        else:
            raise ValueError(f"{view_name} has invalid image shape {arr.shape}; expected HxWx3.")
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
                if arr.size and float(np.nanmax(arr)) <= 1.0 and float(np.nanmin(arr)) >= 0.0:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(arr))

    @staticmethod
    def _pad_patch_dim(tensor: torch.Tensor, target_patches: int) -> torch.Tensor:
        patches = tensor.shape[-2]
        if patches == target_patches:
            return tensor
        if patches > target_patches:
            raise ValueError(f"Cannot pad patch dimension from {patches} down to {target_patches}.")
        pad_shape = (*tensor.shape[:-2], target_patches - patches, tensor.shape[-1])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=-2)

    @classmethod
    def _stack_patch_tensors(cls, tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        target_patches = max(t.shape[-2] for t in tensors)
        return torch.stack([cls._pad_patch_dim(t, target_patches) for t in tensors], dim=dim)

    def _extract_images(self, obs: Dict) -> Tuple[List[Image.Image], List[bool]]:
        """
        Extract images from observation and return with validity mask.

        Returns:
            images: List of PIL Images (None for missing views)
            valid_mask: List of bools indicating which views are valid
        """
        main_view = self._flip_agentview(obs["agentview_image"])
        wrist_view = obs.get("robot0_eye_in_hand_image")  # May be None

        images = []
        valid_mask = []

        # Main view (always expected)
        if main_view is not None:
            images.append(self._to_pil_rgb(main_view, view_name="agentview_image"))
            valid_mask.append(True)
        else:
            # Should not happen, but handle gracefully
            images.append(Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)))
            valid_mask.append(False)

        # Wrist view (optional)
        if wrist_view is not None:
            images.append(self._to_pil_rgb(wrist_view, view_name="robot0_eye_in_hand_image"))
            valid_mask.append(True)
        else:
            # No wrist view - add placeholder
            images.append(Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)))
            valid_mask.append(False)

        return images, valid_mask

    # ------------------------------------------------------------------
    # IC demo construction from current-episode history
    # ------------------------------------------------------------------

    def _select_ic_entries(self, K_ic: int) -> List[Dict]:
        """Select up to K_ic demo entries from episode_history based on ic_demo_mode.

        If fewer than K_ic entries are available, return only what exists
        (no left-padding / repetition).

        When ``ic_stride > 1``, entries are sampled every ``ic_stride`` steps
        from the history.  For example with stride=3 and mode="normal", the
        selected indices (counting from the end) are -1, -4, -7, …
        """
        mode = self.ic_demo_mode
        stride = self.ic_stride

        if mode == "fixed_first":
            # Always repeat the first entry K_ic times (constant IC context)
            return [self.episode_history[0]] * K_ic

        hist = self.episode_history

        if mode == "early":
            # earliest entries with stride
            indices = list(range(0, len(hist), stride))[:K_ic]
            return [hist[i] for i in indices]

        # mode == "normal": most recent entries with stride
        # Walk backwards from the latest entry: -1, -1-stride, -1-2*stride, …
        indices = []
        for k in range(K_ic):
            idx = len(hist) - 1 - k * stride
            if idx < 0:
                break
            indices.append(idx)
        # Reverse so that chronological order is preserved (oldest first)
        indices.reverse()
        return [hist[i] for i in indices]

    def _build_ic_demos_from_history(
        self, current_pil_images: Optional[List] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build IC demo dict from the current episode's history buffer.

        Supports multiple demo selection modes via self.ic_demo_mode:
          - "normal": most recent K-1 entries (default)
          - "fixed_first": always repeat the first entry K-1 times
          - "early": earliest K-1 entries from history

        Returns None if history is completely empty.
        """
        K_ic = self.num_ic_frames - 1  # demo frames needed
        if K_ic <= 0 or len(self.episode_history) < 1:
            return None

        entries = self._select_ic_entries(K_ic)

        all_images, all_proprios, all_actions, all_grids = [], [], [], []
        all_next_images, all_next_grids, all_next_proprios = [], [], []

        for i, entry in enumerate(entries):
            img_out = self.processor.encode_image(entry["pil_images"])
            all_images.append(img_out["image_input"].squeeze(0))
            if "image_grid_thw" in img_out:
                all_grids.append(img_out["image_grid_thw"].squeeze(0))

            all_proprios.append(torch.from_numpy(entry["proprio"].copy()).float())

            act = entry["action_chunk"]  # [T, D]
            if act.ndim == 1:
                act = act[None, :]
            T, D = act.shape
            if D < 20:
                pad = np.zeros((T, 20 - D), dtype=act.dtype)
                act = np.concatenate([act, pad], axis=-1)
            all_actions.append(torch.from_numpy(act[:, :20].copy()).float())

            if self.include_next_obs:
                next_pil_images = entry.get("next_pil_images")
                next_proprio = entry.get("next_proprio")
                if next_pil_images is None:
                    if entry is self.episode_history[-1] and current_pil_images is not None:
                        next_pil_images = current_pil_images
                        next_proprio = self.proprio
                    else:
                        raise ValueError("IC history entry is missing endpoint next observation.")
                next_img_out = self.processor.encode_image(next_pil_images)
                all_next_images.append(next_img_out["image_input"].squeeze(0))
                if "image_grid_thw" in next_img_out:
                    all_next_grids.append(next_img_out["image_grid_thw"].squeeze(0))
                if next_proprio is None:
                    next_proprio = entry["proprio"]
                all_next_proprios.append(torch.from_numpy(np.asarray(next_proprio).copy()).float())

        result = {
            "images": self._stack_patch_tensors(all_images, dim=0),
            "proprios": torch.stack(all_proprios, dim=0),
            "actions": torch.stack(all_actions, dim=0),
        }
        if all_grids:
            result["image_grid_thw"] = torch.cat(all_grids, dim=0)
        if all_next_images:
            result["next_images"] = self._stack_patch_tensors(all_next_images, dim=0)
        if all_next_grids:
            result["next_image_grid_thw"] = torch.cat(all_next_grids, dim=0)
        if all_next_proprios:
            result["next_proprios"] = torch.stack(all_next_proprios, dim=0)
        return result

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_xvla_inputs(self, obs: Dict, goal: str) -> Dict[str, torch.Tensor]:
        """Prepare inputs for XVLA model inference. Routes to IC path when demos exist."""
        images, valid_mask = self._extract_images(obs)
        closed_loop = np.concatenate(
            [obs["robo_pos"], obs["robo_ori"], np.array([0.0])], axis=-1
        )
        obs_proprio = np.concatenate([closed_loop, np.zeros_like(closed_loop)], axis=-1)
        self.proprio = obs_proprio

        if self.num_ic_frames > 0:
            ic_demos = self._build_ic_demos_from_history(current_pil_images=images) if self.num_ic_frames > 1 else None
            return self._prepare_xvla_inputs_ic(images, valid_mask, obs_proprio, goal, ic_demos)

        img_outputs = self.processor.encode_image(images)
        explicit_mask = torch.tensor(valid_mask, dtype=torch.bool)
        lang_outputs = self.processor.encode_language(
            goal, image_mask=explicit_mask,
            image_grid_thw=img_outputs.get("image_grid_thw"),
        )
        processed = {**img_outputs, **lang_outputs}
        processed["image_mask"] = explicit_mask.unsqueeze(0)

        inputs = {k: self._to_device(v) for k, v in processed.items()}
        inputs.update(
            {
                "proprio": torch.as_tensor(self.proprio, device=self.device, dtype=self.dtype).unsqueeze(0),
                "domain_id": self.domain_id,
            }
        )
        return inputs

    def _prepare_xvla_inputs_ic(
        self,
        query_images: List[Image.Image],
        valid_mask: List[bool],
        query_proprio: np.ndarray,
        goal: str,
        demos: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Build IC inputs: stack demo frames + query frame, encode IC sequence."""
        K_ic = 0 if demos is None else demos['images'].shape[0]
        K = K_ic + 1  # total frames = IC demos + 1 query
        has_next_obs = demos is not None and self.include_next_obs and 'next_images' in demos

        # Encode query images
        query_img_out = self.processor.encode_image(query_images)
        query_pv = query_img_out['image_input']           # [1, V, N, C]
        query_grid = query_img_out.get('image_grid_thw')  # [1, V, 3] or None
        next_images = demos.get('next_images') if has_next_obs else None

        # [Diagnostic] Clone query as demo: replace demo images/proprio with query's
        if self.clone_query_as_demo and demos is not None:
            demos = dict(demos)  # shallow copy
            demos['images'] = query_pv.squeeze(0).unsqueeze(0).expand(K_ic, -1, -1, -1)
            if query_grid is not None:
                demos['image_grid_thw'] = query_grid.squeeze(0).unsqueeze(0).expand(K_ic, -1, -1).reshape(K_ic * query_grid.shape[1], 3)
            demos['proprios'] = torch.as_tensor(query_proprio, dtype=torch.float32).unsqueeze(0).expand(K_ic, -1)
            if self.zero_ic_actions:
                demos['actions'] = torch.zeros_like(demos['actions'])

        # Stack: [1, K, V, N, C]
        query_pv_exp = query_pv.unsqueeze(1)
        if demos is not None:
            target_patches = max(demos['images'].shape[-2], query_pv_exp.shape[-2])
            if next_images is not None:
                target_patches = max(target_patches, next_images.shape[-2])
            demo_images = self._pad_patch_dim(demos['images'], target_patches)
            query_pv_exp = self._pad_patch_dim(query_pv_exp, target_patches)
            if next_images is not None:
                next_images = self._pad_patch_dim(next_images, target_patches)
        all_pv = (
            query_pv_exp if demos is None
            else torch.cat([demo_images.unsqueeze(0), query_pv_exp], dim=1)
        )

        # Grid THW: [1, K*V, 3] — only present for Qwen-style backbones
        all_grid = None
        if demos is not None and 'image_grid_thw' in demos:
            demo_grid = demos['image_grid_thw'].unsqueeze(0)
            if query_grid is not None:
                all_grid = torch.cat([demo_grid, query_grid.view(1, -1, 3)], dim=1)
            else:
                all_grid = demo_grid
        elif query_grid is not None:
            all_grid = query_grid.view(1, -1, 3)

        # Proprios: [1, K, D]
        query_prop = torch.as_tensor(query_proprio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        all_prop = (
            query_prop if demos is None
            else torch.cat([demos['proprios'].unsqueeze(0), query_prop], dim=1)
        )

        # IC actions: [1, K-1, T, D]
        if demos is None:
            num_actions = getattr(self.model, 'num_actions', getattr(self.model.config, 'num_actions', 10))
            dim_action = getattr(self.model.action_space, 'dim_action', 20)
            ic_actions = torch.empty(1, 0, num_actions, dim_action, dtype=torch.float32)
        else:
            ic_actions = demos['actions'].unsqueeze(0)

        explicit_mask = torch.tensor(valid_mask, dtype=torch.bool)
        npt = getattr(self.model, 'num_proprio_tokens', 1)
        nat = getattr(self.model.config, 'num_action_tokens', 8)

        # Next-obs for encode_ic_sequence
        next_image_grid_thw = None
        if has_next_obs and 'next_image_grid_thw' in demos:
            next_image_grid_thw = demos['next_image_grid_thw'].unsqueeze(0)

        use_frame_sep = getattr(self.model.config, 'use_frame_sep', False)
        lang_outputs = self.processor.encode_ic_sequence(
            goal,
            num_frames=K,
            image_grid_thw=all_grid,
            image_mask=explicit_mask,
            num_proprio_tokens=npt,
            num_action_tokens=nat,
            include_next_obs=has_next_obs,
            next_image_grid_thw=next_image_grid_thw,
            use_frame_sep=use_frame_sep,
        )

        inputs = {k: self._to_device(v) for k, v in lang_outputs.items()}
        inputs["image_input"] = self._to_device(all_pv)
        inputs["image_mask"] = self._to_device(explicit_mask.unsqueeze(0))
        if all_grid is not None:
            inputs["image_grid_thw"] = self._to_device(all_grid)
        inputs["proprio"] = self._to_device(all_prop)
        inputs["ic_actions"] = self._to_device(ic_actions)
        inputs["domain_id"] = self.domain_id

        if has_next_obs:
            inputs["next_image_input"] = self._to_device(next_images.unsqueeze(0))
            if next_image_grid_thw is not None:
                inputs["next_image_grid_thw"] = self._to_device(next_image_grid_thw)
            if 'next_proprios' in demos:
                inputs["next_proprio"] = self._to_device(demos['next_proprios'].unsqueeze(0))

        return inputs

    def _convert_actions(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Unsupported action shape {actions.shape}, expected [T, D] or [D].")

        dim = actions.shape[1]
        if dim >= 9:
            target_eef = actions[:, :3]
            target_axis = self.rot_processor.Rotate6D_to_AxisAngle(actions[:, 3:9])
            target_grip = actions[:, 9:10] if dim > 9 else actions[:, -1:]
        elif dim >= 7:
            target_eef = actions[:, :3]
            target_axis = actions[:, 3:6]
            target_grip = actions[:, 6:7]
        else:
            raise ValueError(f"Expected at least 7 action dims, got {dim}.")

        return np.concatenate([target_eef, target_axis, target_grip], axis=-1)

    def _queue_plan(self, raw_action: np.ndarray) -> None:
        actions = np.asarray(raw_action)
        if actions.ndim == 1:
            actions = actions[None, :]

        # Limit action chunk size if specified (for debugging cumulative errors)
        if self.action_chunk_size is not None and self.action_chunk_size > 0:
            original_len = actions.shape[0]
            actions = actions[:self.action_chunk_size]
            if original_len > self.action_chunk_size:
                logger.debug(f"Action chunk truncated: {original_len} -> {self.action_chunk_size}")

        if self.model_arch == "xvla":
            # Update proprio from last predicted action
            # proprio is 20-dim: [left_arm(10), right_arm(10)]
            # actions[-1] is 10-dim (single arm) or 20-dim (dual arm)
            if self.proprio is None:
                # Initialize as 20-dim with zeros
                self.proprio = np.zeros(20, dtype=np.float32)

            action_dim = actions.shape[1]
            if action_dim >= 20:
                # Dual arm: copy all 20 dims
                self.proprio[:20] = actions[-1, :20].copy()
            else:
                # Single arm: copy to left arm, right arm stays 0
                self.proprio[:action_dim] = actions[-1, :action_dim].copy()
                self.proprio[10:] = 0.0  # Right arm padded with 0

        final_action = self._convert_actions(actions)
        if final_action.size > 0:
            self.gripper_trace.extend(final_action[:, -1].tolist())
        for row in final_action.tolist():
            self.action_plan.append(row)

    def _discretize_gripper(self, raw_value: float) -> float:
        return 1.0 if raw_value > self.gripper_close_threshold else -1.0

    def write_gripper_curve(self, out_dir: Path, tag: str) -> None:

        if not self.gripper_trace:
            logger.info("No gripper trace collected for %s", tag)
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        curve_path = out_dir / f"gripper_curve_{tag}.csv"
        with open(curve_path, "w", encoding="utf-8") as f:
            f.write("idx,gripper\n")
            for i, v in enumerate(self.gripper_trace):
                f.write(f"{i},{v}\n")
        max_points = max(1, int(self.gripper_curve_max_points))
        preview = self.gripper_trace[:max_points]
        logger.info(
            "Gripper curve %s saved to %s (len=%d, min=%.3f, max=%.3f)",
            tag,
            curve_path,
            len(self.gripper_trace),
            float(np.min(self.gripper_trace)),
            float(np.max(self.gripper_trace)),
        )
        logger.info("Gripper curve %s first_%d: %s", tag, len(preview), preview)

    def _run_xvla_inference(self, obs: Dict, goal: str) -> np.ndarray:
        """Run XVLA model inference and return raw action chunk [T, D]."""
        inputs = self._prepare_xvla_inputs(obs, goal)
        inputs.pop("attention_mask", None)
        with torch.inference_mode():
            action = (
                self.model.generate_actions(steps=self.denoising_steps, **inputs)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )
        return action

    def _record_ic_history(self, obs: Dict, action_chunk: np.ndarray) -> None:
        """Record an executed action chunk into IC episode history."""
        if self.num_ic_frames <= 1:
            return
        pil_images, _ = self._extract_images(obs)
        # Store only the portion that will actually be executed (respecting action_chunk_size)
        chunk = np.asarray(action_chunk).copy()
        if self.action_chunk_size is not None and self.action_chunk_size > 0:
            chunk = chunk[:self.action_chunk_size]
        self.episode_history.append({
            "pil_images": pil_images,
            "proprio": self.proprio.copy() if self.proprio is not None else np.zeros(20, dtype=np.float32),
            "action_chunk": chunk,
            "next_pil_images": None,
            "next_proprio": None,
        })

    def _fill_latest_history_next_obs(self, obs: Dict) -> None:
        if self.num_ic_frames <= 1 or not self.episode_history:
            return
        latest = self.episode_history[-1]
        if latest.get("next_pil_images") is None:
            latest["next_pil_images"], _ = self._extract_images(obs)
            # Capture proprio at the chunk-end obs to match next_image_input.
            closed_loop = np.concatenate(
                [obs["robo_pos"], obs["robo_ori"], np.array([0.0])], axis=-1
            )
            latest["next_proprio"] = np.concatenate(
                [closed_loop, np.zeros_like(closed_loop)], axis=-1
            )

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            if self.model_arch == "xvla":
                self._fill_latest_history_next_obs(obs)
                action = self._run_xvla_inference(obs, goal)
                # --- IC diagnostic logging ---
                if self.num_ic_frames > 1 and len(self.episode_history) <= 3:
                    ic_tag = f"IC(history={len(self.episode_history)})"
                    act = np.asarray(action)
                    grip_idx = min(9, act.shape[1] - 1)
                    prop_grip_idx = min(9, len(self.proprio) - 1)
                    logger.info(
                        "[IC-diag] inference #%d (%s): action shape=%s, "
                        "xyz_first=[%.4f,%.4f,%.4f], xyz_last=[%.4f,%.4f,%.4f], "
                        "grip_first=%.6f, grip_last=%.6f, "
                        "proprio=[%.4f,%.4f,%.4f,...,grip=%.4f]",
                        len(self.episode_history), ic_tag, act.shape,
                        act[0, 0], act[0, 1], act[0, 2],
                        act[-1, 0], act[-1, 1], act[-1, 2],
                        act[0, grip_idx], act[-1, grip_idx],
                        self.proprio[0], self.proprio[1], self.proprio[2], self.proprio[prop_grip_idx],
                    )
                    if self.episode_history:
                        prev = self.episode_history[-1]
                        prev_act = prev["action_chunk"]
                        prev_grip_idx = min(9, prev_act.shape[1] - 1)
                        prev_prop_grip_idx = min(9, len(prev["proprio"]) - 1)
                        logger.info(
                            "[IC-diag] IC demo action: xyz_first=[%.4f,%.4f,%.4f], "
                            "xyz_last=[%.4f,%.4f,%.4f], grip_first=%.6f, grip_last=%.6f, "
                            "demo_proprio=[%.4f,%.4f,%.4f,...,grip=%.4f]",
                            prev_act[0, 0], prev_act[0, 1], prev_act[0, 2],
                            prev_act[-1, 0], prev_act[-1, 1], prev_act[-1, 2],
                            prev_act[0, prev_grip_idx], prev_act[-1, prev_grip_idx],
                            prev["proprio"][0], prev["proprio"][1], prev["proprio"][2], prev["proprio"][prev_prop_grip_idx],
                        )
                self._record_ic_history(obs, action)
            else:
                images, _ = self._extract_images(obs)
                with torch.inference_mode():
                    action = self.model.generate_actions(
                        instruction=goal,
                        images=images,
                        processor=self.processor,
                        unnorm_key=self.unnorm_key,
                    )
            self._queue_plan(np.asarray(action))

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        action_predict[-1] = self._discretize_gripper(action_predict[-1])
        return action_predict



def load_model_and_processor(
    model_path: str,
    processor_path: Optional[str],
    lora_path: Optional[str],
    device: torch.device,
    model_arch: str,
) -> tuple[torch.nn.Module, Any]:
    def resolve_processor_path(path: Optional[str]) -> str:
        if path and Path(path).exists():
            return path
        if path and path != model_path and not Path(path).exists():
            logger.warning("Processor path %s not found, falling back to model path %s", path, model_path)
        return model_path

    arch = _normalize_arch(model_arch)
    processor_resolved = resolve_processor_path(processor_path)

    if arch == "xvla":
        from models.modeling_xvla import XVLA
        logger.info(f"Loading XVLA model with device_map={device} for faster loading...")
        model = XVLA.from_pretrained(
            model_path,
            device_map=device,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )

        num_views = getattr(model.config, "num_views", 2)
        backbone_type = getattr(model.config, "vlm_backbone_type", "qwen3_vl")
        processor = model._processor
        if processor_resolved != model_path:
            from models.processing_xvla import build_xvla_processor

            processor = build_xvla_processor(
                pretrained_name_or_path=processor_resolved,
                backbone_type=backbone_type,
                num_views=num_views,
                proprio_to_vlm=getattr(model.config, "proprio_to_vlm", False),
                vlm_process_all_views=getattr(model.config, "vlm_process_all_views", False),
            )
            model.set_processor(processor)
            logger.info("Loaded XVLA processor from %s", processor_resolved)
        logger.info(f"Loaded XVLA model with processor (num_views={num_views})")

        uses_flash_attn = (
            getattr(model.config, 'paligemma_use_flash_attn', False)
            if backbone_type == "paligemma"
            else getattr(model.config, 'qwen3_use_flash_attn', False)
        )

        if uses_flash_attn or backbone_type == "paligemma":
            logger.info("Using bfloat16 for %s evaluation", backbone_type)
        else:
            model = model.to(torch.float32)
            logger.info("Using float32 for evaluation")

        if lora_path:
            if PeftModel is None:
                raise ImportError("peft is required for loading LoRA weights.")
            model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        return model, processor

    from models.openvla.modeling_openvla import OpenVLA, OpenVLAAdapter, OpenVLAOFT
    model_cls_map = {
        "openvla": OpenVLA,
        "openvla-oft": OpenVLAOFT,
        "vla-adapter": OpenVLAAdapter,
    }
    if arch not in model_cls_map:
        raise ValueError(f"Unsupported model_arch {arch}.")

    model_cls = model_cls_map[arch]
    model = model_cls.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    try:
        processor = AutoProcessor.from_pretrained(processor_resolved, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Falling back to model.processor because AutoProcessor load failed: %s", exc)
        processor = getattr(model, "processor", None)
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if getattr(model, "processor", None) is None:
        try:
            model.processor = processor
        except Exception:
            pass
    if lora_path:
        logger.warning("Ignoring lora_path for %s models; LoRA loading only supported for XVLA.", arch)
    model.eval()
    return model, processor


def run_libero_eval(
    model: torch.nn.Module,
    processor: Any,
    save_path: Path,
    task_suites: Iterable[str],
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = "abs",
    denoising_steps: int = 10,
    domain_id: int = 3,
    device: Optional[torch.device] = None,
    save_video: bool = True,
    model_arch: str = "xvla",
    openvla_unnorm_key: Optional[str] = None,
    gripper_close_threshold: float = 0.5,
    sim_type: str = "libero",
    max_tasks: Optional[int] = None,
    action_mode: str = "ee6d",
    action_chunk_size: Optional[int] = None,
    gripper_curve_max_points: int = 200,
    num_ic_frames: int = 0,
    include_next_obs: bool = False,
    clone_query_as_demo: bool = False,
    ic_demo_mode: str = "normal",
    zero_ic_actions: bool = False,
    ic_stride: int = 1,
    perturbation: Optional[str] = None,
    perturbation_level: int = 2,
) -> tuple[EvalRunResult, Optional[Dict[str, float]], Dict[str, List[Dict]]]:
    client_module = _import_libero_client(sim_type)

    device = device or next(model.parameters()).device
    agent = LocalVLAAgent(
        model=model,
        processor=processor,
        device=device,
        denoising_steps=denoising_steps,
        domain_id=domain_id,
        model_arch=model_arch,
        unnorm_key=openvla_unnorm_key,
        gripper_close_threshold=gripper_close_threshold,
        client_module=client_module,
        action_mode=action_mode,
        action_chunk_size=action_chunk_size,
        gripper_curve_max_points=gripper_curve_max_points,
        num_ic_frames=num_ic_frames,
        include_next_obs=include_next_obs,
        clone_query_as_demo=clone_query_as_demo,
        ic_demo_mode=ic_demo_mode,
        zero_ic_actions=zero_ic_actions,
        ic_stride=ic_stride,
    )

    if num_ic_frames > 1:
        logger.info(
            "IC mode: num_ic_frames=%d, include_next_obs=%s. "
            "History grows up to %d demo frames; partial history is used as available.",
            num_ic_frames, include_next_obs, num_ic_frames - 1,
        )

    save_path.mkdir(parents=True, exist_ok=True)

    success_rates, avg_steps, raw_per_suite = client_module.eval_libero(
        agent=agent,
        save_path=save_path,
        num_episodes=num_episodes,
        init_seed=init_seed,
        act_type=act_type,
        task_suites=task_suites,
        save_video=save_video,
        max_tasks=max_tasks,
        perturbation=perturbation,
        perturbation_level=perturbation_level,
    )

    return EvalRunResult(success_rates, avg_steps), None, raw_per_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Direct LIBERO evaluation (no server needed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument(
        "--sim",
        type=str,
        default="libero",
        choices=["libero", "libero_plus"],
        help="Simulation type: 'libero' for standard LIBERO evaluation, 'libero_plus' for LIBERO-Plus evaluation with fine-grained perturbation types.",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="xvla",
        choices=["xvla", "openvla", "openvla-oft", "openvla_oft", "openvlaoft", "vla-adapter", "vla_adapter", "vlaadapter"],
        help="Which model family to evaluate.",
    )
    parser.add_argument("--processor_path", type=str, default=None, help="Optional processor path (defaults to model).")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA weights to merge for evaluation.")
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=None,
        help="LIBERO suites to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs", help="Directory for eval logs/videos.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task.")
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate per suite. Use for quick testing (e.g., --max_tasks 20). Default: all tasks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"], help="Action type for env.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference.")
    parser.add_argument("--denoising_steps", type=int, default=10, help="Diffusion denoising steps.")
    parser.add_argument("--domain_id", type=int, default=3, help="Domain id used by the model.")
    parser.add_argument(
        "--openvla_unnorm_key",
        type=str,
        default=None,
        help="De-normalization key for OpenVLA action stats when multiple datasets are present.",
    )
    parser.add_argument(
        "--gripper_close_threshold",
        type=float,
        default=0.5,
        help="Threshold for mapping gripper logits to open/close commands.",
    )
    parser.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        default=False,
        help="Save evaluation rollouts as videos (default: off).",
    )
    parser.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Disable saving videos to speed up evaluation.",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="ee6d",
        help="Action mode for evaluation.",
    )
    # Action chunk size control (for debugging cumulative errors)
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=None,
        help="Only execute first N actions from each action chunk (default: full model chunk). "
             "Use 1 to test single-step prediction, 5 for short horizon, etc. "
             "This is useful for debugging cumulative errors.",
    )
    parser.add_argument(
        "--gripper_curve_max_points",
        type=int,
        default=200,
        help="Max number of gripper values to print per episode (all values are saved to CSV).",
    )
    # IC arguments
    parser.add_argument(
        "--num_ic_frames",
        type=int,
        default=None,
        help="Total IC frames (K). Omit to auto-detect from model config. 0 = baseline prompt. 1 = IC query-only prompt. K>1 means K-1 demo frames + 1 query frame.",
    )
    parser.add_argument(
        "--clone_query_as_demo",
        action="store_true",
        help="[Diagnostic] Replace IC demo images/proprio with query's to test if old obs causes jitter.",
    )
    parser.add_argument(
        "--ic_demo_mode",
        type=str,
        default="normal",
        choices=["normal", "fixed_first", "early"],
        help="[Diagnostic] IC demo selection mode. "
             "normal: most recent K-1 frames (default). "
             "fixed_first: always repeat first frame K-1 times (constant context). "
             "early: earliest K-1 frames from episode history.",
    )
    parser.add_argument(
        "--zero_ic_actions",
        action="store_true",
        help="[Diagnostic] Zero out IC demo actions. Combine with --clone_query_as_demo to test if action embeddings matter.",
    )
    parser.add_argument(
        "--ic_stride",
        type=int,
        default=1,
        help="Stride for IC demo selection from history. "
             "1 = every history entry is eligible (default). "
             "k = select every k-th entry (e.g., stride=3 picks frames 0,3,6,… in 'early' mode "
             "or …,-7,-4,-1 in 'normal' mode).",
    )
    # Runtime perturbation arguments
    parser.add_argument(
        "--perturbation",
        type=str,
        default=None,
        choices=["camera_star", "camera_plusplus", "camera_plusplus_star", "action", "action_star"],
        help="Runtime perturbation type to apply during evaluation. "
             "camera_star: larger agentview camera perturbation. "
             "camera_plusplus: agentview perturbation + wrist view image flip. "
             "action: systematic calibration shift on actions.",
    )
    parser.add_argument(
        "--perturbation_level",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Severity level for action perturbation (1=mild +-5mm/+-1deg, "
             "2=moderate +-10mm/+-3deg, 3=severe +-20mm/+-6deg). Default: 2.",
    )
    parser.add_argument(
        "--runtime_perturbations",
        nargs='+',
        default=None,
        help="Runtime perturbation types to evaluate on original base suites. "
             "Available: camera_star, camera_plusplus, camera_plusplus_star, action, action_star. "
             "Use 'all' for all 5 types, 'none' to skip. "
             "Default for --sim libero_plus: camera_plusplus, action "
             "(camera_star, camera_plusplus_star and action_star are excluded by default). "
             "Default for --sim libero: none.",
    )
    parser.add_argument("--libero_root", required=True, help="LIBERO or LIBERO-Plus checkout (contains libero/libero).")
    return parser.parse_args()


def _main(args) -> None:
    logging.basicConfig(level=logging.INFO)

    if args.task_suites is None:
        if args.sim == "libero":
            args.task_suites = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
        elif args.sim == "libero_plus":
            args.task_suites = ["libero_plus_camera", "libero_plus_robot", "libero_plus_language", "libero_plus_light", "libero_plus_background", "libero_plus_noise", "libero_plus_layout"]
        else:
            raise ValueError(f"Unknown sim type: {args.sim}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    arch = _normalize_arch(args.model_arch)
    logger.info("Loading %s model on device %s", arch, device)
    model, processor = load_model_and_processor(
        args.model_path, args.processor_path, args.lora_path, device, arch,
    )

    # Auto-read IC settings from model config when not specified by user
    if args.num_ic_frames is None and hasattr(model, 'config'):
        cfg_ic = getattr(model.config, 'num_ic_frames', 0)
        args.num_ic_frames = int(cfg_ic)
        logger.info("Auto-detected num_ic_frames=%d from model config", args.num_ic_frames)
    elif args.num_ic_frames is None:
        args.num_ic_frames = 0

    include_next_obs = getattr(model.config, 'include_next_obs', False) if hasattr(model, 'config') else False

    out_dir = _resolve_checkpoint_output_dir(args.output_dir, args.model_path)
    logger.info(
        "Starting LIBERO evaluation | sim=%s | arch=%s | suites=%s | episodes=%s | "
        "num_ic_frames=%d | include_next_obs=%s | save_dir=%s",
        args.sim, arch, args.task_suites, args.episodes,
        args.num_ic_frames, include_next_obs, out_dir,
    )
    if args.action_chunk_size is not None:
        logger.info(f"Action chunk size: {args.action_chunk_size}")

    if args.perturbation:
        logger.info("Runtime perturbation: %s (level=%d)", args.perturbation, args.perturbation_level)

    # Resolve runtime perturbations
    if args.runtime_perturbations is None:
        runtime_perts = RUNTIME_PERTURBATIONS_DEFAULT if args.sim == "libero_plus" else []
    elif args.runtime_perturbations == ["none"]:
        runtime_perts = []
    elif "all" in args.runtime_perturbations:
        runtime_perts = RUNTIME_PERTURBATIONS_ALL
    else:
        runtime_perts = args.runtime_perturbations

    if runtime_perts:
        logger.info("Runtime perturbation passes: %s", runtime_perts)

    results, recon_stats, dataset_raw = run_libero_eval(
        model=model,
        processor=processor,
        save_path=out_dir,
        task_suites=args.task_suites,
        num_episodes=args.episodes,
        init_seed=args.seed,
        act_type=args.act_type,
        denoising_steps=args.denoising_steps,
        domain_id=args.domain_id,
        device=device,
        save_video=args.save_video,
        model_arch=arch,
        openvla_unnorm_key=args.openvla_unnorm_key,
        gripper_close_threshold=args.gripper_close_threshold,
        sim_type=args.sim,
        max_tasks=args.max_tasks,
        action_mode=args.action_mode,
        action_chunk_size=args.action_chunk_size,
        gripper_curve_max_points=args.gripper_curve_max_points,
        num_ic_frames=args.num_ic_frames,
        include_next_obs=include_next_obs,
        clone_query_as_demo=args.clone_query_as_demo,
        ic_demo_mode=args.ic_demo_mode,
        zero_ic_actions=args.zero_ic_actions,
        ic_stride=args.ic_stride,
        perturbation=args.perturbation,
        perturbation_level=args.perturbation_level,
    )

    # ---- Runtime perturbation evaluation passes ----
    runtime_results = {}  # pert_type -> EvalRunResult
    runtime_raw: Dict[str, Dict[str, List[Dict]]] = {}
    if runtime_perts:
        for pert_type in runtime_perts:
            pert_level = args.perturbation_level
            logger.info("Running runtime perturbation: %s (level=%d)", pert_type, pert_level)
            pert_out_dir = out_dir / f"runtime_{pert_type}"
            pert_result, _, pert_raw = run_libero_eval(
                model=model,
                processor=processor,
                save_path=pert_out_dir,
                task_suites=RUNTIME_PERTURBATION_BASE_SUITES,
                num_episodes=args.episodes,
                init_seed=args.seed,
                act_type=args.act_type,
                denoising_steps=args.denoising_steps,
                domain_id=args.domain_id,
                device=device,
                save_video=args.save_video,
                model_arch=arch,
                openvla_unnorm_key=args.openvla_unnorm_key,
                gripper_close_threshold=args.gripper_close_threshold,
                sim_type=args.sim,
                max_tasks=args.max_tasks,
                action_mode=args.action_mode,
                action_chunk_size=args.action_chunk_size,
                gripper_curve_max_points=args.gripper_curve_max_points,
                num_ic_frames=args.num_ic_frames,
                include_next_obs=include_next_obs,
                clone_query_as_demo=args.clone_query_as_demo,
                ic_demo_mode=args.ic_demo_mode,
                zero_ic_actions=args.zero_ic_actions,
                ic_stride=args.ic_stride,
                perturbation=pert_type,
                perturbation_level=pert_level,
            )
            runtime_results[pert_type] = pert_result
            runtime_raw[pert_type] = pert_raw

    summary_client = _import_libero_client(args.sim)
    for raw, names in [(dataset_raw, args.task_suites)] + [
            (raw, RUNTIME_PERTURBATION_BASE_SUITES) for raw in runtime_raw.values()]:
        expected = _get_suite_episode_counts(summary_client, names, args.episodes, args.max_tasks)
        for name, count in expected.items():
            if len(raw.get(name, [])) != count:
                raise RuntimeError(f"Incomplete evaluation for {name}: expected {count} episodes")

    from evaluation.run_support import write_final_summary
    write_final_summary(out_dir, dataset_raw, runtime_raw)

    # ---- Print results summary ----
    summary_client = _import_libero_client(args.sim)
    suite_episode_counts = _get_suite_episode_counts(summary_client, args.task_suites, args.episodes, args.max_tasks)
    sim_label = "LIBERO-Plus" if args.sim == "libero_plus" else "LIBERO"
    print("\n" + "=" * 80)
    print(f"  {sim_label} Evaluation Results Summary")
    print("=" * 80)

    # Dataset perturbation suites
    print("  [Dataset Perturbation Suites]")
    for suite_name in args.task_suites:
        success_rate = results.get(suite_name, 0.0)
        total_suite_episodes = suite_episode_counts.get(suite_name, 0)
        successes = int(round(success_rate * total_suite_episodes))
        avg_steps = results.avg_steps.get(suite_name, 0.0)
        print(
            f"  {suite_name:40s}: {success_rate:6.2%} ({successes}/{total_suite_episodes}) | "
            f"avg steps: {avg_steps:7.2f}"
        )
    print("-" * 80)
    overall_avg = _mean(results.get(suite_name, 0.0) for suite_name in args.task_suites)
    overall_avg_steps = _mean(results.avg_steps.get(suite_name, 0.0) for suite_name in args.task_suites)
    print(
        f"  {'Dataset Overall':40s}: {overall_avg:6.2%} | "
        f"avg steps: {overall_avg_steps:7.2f} | mean over {len(args.task_suites)} suites"
    )

    # Runtime perturbation results
    if runtime_results:
        print("\n  [Runtime Perturbation Passes]")
        for pert_type, pert_res in runtime_results.items():
            pert_suite_names = list(pert_res.keys())
            pert_avg = _mean(pert_res.get(suite_name, 0.0) for suite_name in pert_suite_names)
            pert_avg_steps = _mean(pert_res.avg_steps.get(suite_name, 0.0) for suite_name in pert_suite_names)
            print(
                f"  {pert_type:40s}: {pert_avg:6.2%} | "
                f"avg steps: {pert_avg_steps:7.2f} | mean over {len(pert_suite_names)} suites"
            )
        print("-" * 80)
        runtime_overall = _mean(_mean(r.values()) for r in runtime_results.values() if r)
        runtime_overall_steps = _mean(
            _mean(r.avg_steps.get(suite_name, 0.0) for suite_name in r.keys())
            for r in runtime_results.values() if r
        )
        print(
            f"  {'Runtime Overall':40s}: {runtime_overall:6.2%} | "
            f"avg steps: {runtime_overall_steps:7.2f} | mean over {len(runtime_results)} perturbation passes"
        )

    print("=" * 80)

    logger.info("Evaluation finished: success=%s avg_steps=%s", dict(results), results.avg_steps)
    if runtime_results:
        for pert_type, pert_res in runtime_results.items():
            logger.info("Runtime %s: success=%s avg_steps=%s", pert_type, dict(pert_res), pert_res.avg_steps)


def main() -> None:
    import random
    from evaluation.run_support import execute_recorded_run, validate_options
    args = parse_args()
    validate_options(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    execute_recorded_run(args, lambda: _main(args),
                         _resolve_checkpoint_output_dir(args.output_dir, args.model_path))


if __name__ == "__main__":
    main()
