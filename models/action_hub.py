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
from typing import Iterable, Tuple, Dict, Type
import torch
import torch.nn as nn
import numpy as np

# Lazy import scipy to avoid CUDA/BLAS conflicts
def _get_scipy_rotation():
    from scipy.spatial.transform import Rotation
    return Rotation
# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections (e.g. apply sigmoid).
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------------------
    # Core supervised loss
    # ---------------------------------------------------------------------
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    # ---------------------------------------------------------------------
    # Space-level hooks
    # ---------------------------------------------------------------------
    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


def _rotate6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d (CONCATENATED format) to rotation matrix.

    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    This matches libero_client.Mat_to_Rotate6D format.
    """
    a1 = rot6d[..., :3]   # first column: [col1[0], col1[1], col1[2]]
    a2 = rot6d[..., 3:6]  # second column: [col2[0], col2[1], col2[2]]

    # Gram-Schmidt orthogonalization
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_rotate6d(mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to rotate6d (CONCATENATED format).

    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    col1 = mat[..., :, 0]
    col2 = mat[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


def _matrix_to_axis_angle(mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle.

    Handles degenerate matrices (e.g., from all-zero rotate6d) as identity.

    Args:
        mat: Rotation matrix of shape (..., 3, 3)

    Returns:
        Axis-angle of shape (..., 3), where ||aa|| = angle
    """
    batch_shape = mat.shape[:-2]
    mat = mat.reshape(-1, 3, 3)

    # Check for degenerate matrices - valid rotation has trace in [-1, 3]
    trace = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    is_valid = (trace >= -1.0 - 1e-6) & (trace <= 3.0 + 1e-6)

    # Matrix -> Quaternion (w, x, y, z)
    s = torch.sqrt((trace + 1.0).clamp(min=1e-8)) * 2
    w = 0.25 * s
    x = (mat[:, 2, 1] - mat[:, 1, 2]) / s.clamp(min=1e-8)
    y = (mat[:, 0, 2] - mat[:, 2, 0]) / s.clamp(min=1e-8)
    z = (mat[:, 1, 0] - mat[:, 0, 1]) / s.clamp(min=1e-8)

    # Normalize quaternion
    quat = torch.stack([w, x, y, z], dim=-1)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Ensure w >= 0 (canonical form)
    quat = torch.where(quat[:, 0:1] < 0, -quat, quat)

    # Quaternion -> Axis-angle
    xyz = quat[:, 1:4]
    sin_half = xyz.norm(dim=-1).clamp(min=1e-8)
    angle = 2 * torch.atan2(sin_half, quat[:, 0])
    axis = xyz / sin_half.unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)

    # Zero axis-angle for invalid/identity matrices
    aa = torch.where(is_valid.unsqueeze(-1) & (sin_half.unsqueeze(-1) > 1e-6), aa, torch.zeros_like(aa))

    return aa.reshape(*batch_shape, 3)


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix using Rodrigues formula.

    Args:
        aa: Axis-angle of shape (..., 3), where ||aa|| = angle and aa/||aa|| = axis

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # theta = ||aa||
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis = aa / theta  # normalized axis, shape (..., 3)

    theta = theta.squeeze(-1)  # (...)

    # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    # where K is the skew-symmetric matrix of axis
    cos_t = torch.cos(theta)  # (...)
    sin_t = torch.sin(theta)  # (...)
    one_minus_cos = 1 - cos_t  # (...)

    # Extract axis components
    kx, ky, kz = axis[..., 0], axis[..., 1], axis[..., 2]

    # Build rotation matrix elements
    # R = I + sin(theta)*K + (1-cos(theta))*K^2
    # Expanded form:
    r00 = cos_t + kx * kx * one_minus_cos
    r01 = kx * ky * one_minus_cos - kz * sin_t
    r02 = kx * kz * one_minus_cos + ky * sin_t
    r10 = ky * kx * one_minus_cos + kz * sin_t
    r11 = cos_t + ky * ky * one_minus_cos
    r12 = ky * kz * one_minus_cos - kx * sin_t
    r20 = kz * kx * one_minus_cos - ky * sin_t
    r21 = kz * ky * one_minus_cos + kx * sin_t
    r22 = cos_t + kz * kz * one_minus_cos

    # Stack into matrix
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


# =============================================================================
# Rotation Conversion Helpers
# =============================================================================
def _broadcast_proprio(R_proprio: torch.Tensor, R_other: torch.Tensor) -> torch.Tensor:
    """Broadcast proprio matrix if action has extra time dimension."""
    if R_proprio.ndim == R_other.ndim - 1:
        R_proprio = R_proprio.unsqueeze(-3)
    return R_proprio


def convert_rotate6d_from_abs_to_rel(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """R_rel = R_proprio^T @ R_abs (rot6d -> rot6d). Zero input -> zero output."""
    R_proprio = _rotate6d_to_matrix(proprio)
    R_abs = _rotate6d_to_matrix(action)
    R_proprio = _broadcast_proprio(R_proprio, R_abs)
    result = _matrix_to_rotate6d(torch.matmul(R_proprio.transpose(-2, -1), R_abs))

    return result

def convert_rotate6d_from_rel_to_abs(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """R_abs = R_proprio @ R_rel (rot6d -> rot6d). Zero input -> zero output."""
    R_proprio = _rotate6d_to_matrix(proprio)
    R_rel = _rotate6d_to_matrix(action)
    R_proprio = _broadcast_proprio(R_proprio, R_rel)
    result = _matrix_to_rotate6d(torch.matmul(R_proprio, R_rel))

    return result

def convert_abs_rot6d_to_rel_axisangle(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """R_rel = R_proprio^T @ R_abs -> axis_angle (rot6d -> aa). Zero input -> zero output."""
    R_proprio = _rotate6d_to_matrix(proprio)
    R_abs = _rotate6d_to_matrix(action)
    R_proprio = _broadcast_proprio(R_proprio, R_abs)
    result = _matrix_to_axis_angle(torch.matmul(R_proprio.transpose(-2, -1), R_abs))

    return result

def convert_rel_axisangle_to_abs_rot6d(proprio: torch.Tensor, rel_aa: torch.Tensor) -> torch.Tensor:
    """R_abs = R_proprio @ aa_to_matrix(rel_aa) (aa -> rot6d). Zero input -> zero output."""
    R_proprio = _rotate6d_to_matrix(proprio)
    R_rel = _axis_angle_to_matrix(rel_aa)
    R_proprio = _broadcast_proprio(R_proprio, R_rel)
    result = _matrix_to_rotate6d(torch.matmul(R_proprio, R_rel))

    return result

# =============================================================================
# Full Action Conversion (20D dual-arm layout)
# Layout: [pos1(3), rot1(6), grip1(1), pos2(3), rot2(6), grip2(1)]
# =============================================================================

# Index constants for 20D layout
_P1, _R1, _G1 = slice(0, 3), slice(3, 9), slice(9, 10)
_P2, _R2, _G2 = slice(10, 13), slice(13, 19), slice(19, 20)

# Index constants for 14D layout (axis-angle)
_P1_14, _AA1_14, _G1_14 = slice(0, 3), slice(3, 6), slice(6, 7)
_P2_14, _AA2_14, _G2_14 = slice(7, 10), slice(10, 13), slice(13, 14)


def convert_action_abs_to_rel_rot6d(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """20D abs -> 20D rel (rotation as rot6d)."""
    if proprio.ndim == action.ndim - 1:
        proprio = proprio.unsqueeze(-2)
    return torch.cat([
        action[..., _P1] - proprio[..., _P1],
        convert_rotate6d_from_abs_to_rel(proprio[..., _R1], action[..., _R1]),
        action[..., _G1],
        action[..., _P2] - proprio[..., _P2],
        convert_rotate6d_from_abs_to_rel(proprio[..., _R2], action[..., _R2]),
        action[..., _G2],
    ], dim=-1)


def convert_action_rel_rot6d_to_abs(proprio: torch.Tensor, rel_action: torch.Tensor) -> torch.Tensor:
    """20D rel -> 20D abs (rotation as rot6d)."""
    if proprio.ndim == rel_action.ndim - 1:
        proprio = proprio.unsqueeze(-2)
    return torch.cat([
        rel_action[..., _P1] + proprio[..., _P1],
        convert_rotate6d_from_rel_to_abs(proprio[..., _R1], rel_action[..., _R1]),
        rel_action[..., _G1],
        rel_action[..., _P2] + proprio[..., _P2],
        convert_rotate6d_from_rel_to_abs(proprio[..., _R2], rel_action[..., _R2]),
        rel_action[..., _G2],
    ], dim=-1)


def convert_action_abs_to_rel_axisangle(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """20D abs (rot6d) -> 14D rel (axis-angle)."""
    if proprio.ndim == action.ndim - 1:
        proprio = proprio.unsqueeze(-2)
    return torch.cat([
        action[..., _P1] - proprio[..., _P1],
        convert_abs_rot6d_to_rel_axisangle(proprio[..., _R1], action[..., _R1]),
        action[..., _G1],
        action[..., _P2] - proprio[..., _P2],
        convert_abs_rot6d_to_rel_axisangle(proprio[..., _R2], action[..., _R2]),
        action[..., _G2],
    ], dim=-1)


def convert_action_rel_axisangle_to_abs(proprio: torch.Tensor, rel_action: torch.Tensor) -> torch.Tensor:
    """14D rel (axis-angle) -> 20D abs (rot6d)."""
    if proprio.ndim == rel_action.ndim - 1:
        proprio = proprio.unsqueeze(-2)
    return torch.cat([
        rel_action[..., _P1_14] + proprio[..., _P1],
        convert_rel_axisangle_to_abs_rot6d(proprio[..., _R1], rel_action[..., _AA1_14]),
        rel_action[..., _G1_14],
        rel_action[..., _P2_14] + proprio[..., _P2],
        convert_rel_axisangle_to_abs_rot6d(proprio[..., _R2], rel_action[..., _AA2_14]),
        rel_action[..., _G2_14],
    ], dim=-1)


def _euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (xyz convention) to rotation matrix.

    Args:
        euler: Tensor of shape (..., 3) with Euler angles in radians (rx, ry, rz)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    # Rotation matrix for xyz (extrinsic) = Rz @ Ry @ Rx
    # Row 0
    r00 = cos_y * cos_z
    r01 = sin_x * sin_y * cos_z - cos_x * sin_z
    r02 = cos_x * sin_y * cos_z + sin_x * sin_z
    # Row 1
    r10 = cos_y * sin_z
    r11 = sin_x * sin_y * sin_z + cos_x * cos_z
    r12 = cos_x * sin_y * sin_z - sin_x * cos_z
    # Row 2
    r20 = -sin_y
    r21 = sin_x * cos_y
    r22 = cos_x * cos_y

    # Stack into matrix [..., 3, 3]
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def convert_euler_to_rotate6d(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to rotate6d representation.

    Args:
        euler: Tensor of shape (..., 3) with Euler angles in radians (xyz convention)

    Returns:
        Tensor of shape (..., 6) with rotate6d representation
    """
    pos1 = euler[..., 0:3]
    rot6d1 = euler[..., 3:6]
    grip1 = euler[..., 6:7]
    pos2 = euler[..., 7:10]
    rot6d2 = euler[..., 10:13]
    grip2 = euler[..., 13:14]
    mat1 = _euler_to_matrix(rot6d1)
    mat1 = _matrix_to_rotate6d(mat1)
    mat2 = _euler_to_matrix(rot6d2)
    mat2 = _matrix_to_rotate6d(mat2)
    return torch.cat([pos1, mat1, grip1, pos2, mat2, grip2], dim=-1)



def convert_rotate6d_to_euler(abs_traj: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d to Euler angles at dims [3:9] and optionally [13:19].

    Input layout (20D dual-arm):  [pos1(3), rot6d1(6), grip1(1), pos2(3), rot6d2(6), grip2(1)]
    Output layout (14D dual-arm): [pos1(3), euler1(3), grip1(1), pos2(3), euler2(3), grip2(1)]

    Input layout (10D single-arm):  [pos(3), rot6d(6), grip(1)]
    Output layout (7D single-arm):  [pos(3), euler(3), grip(1)]
    """
    if abs_traj.ndim == 1:
        abs_traj = abs_traj.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    dim = abs_traj.shape[-1]

    # Single arm case: dims [3:9] contain rotate6d
    assert dim == 20, f"Expected dim=20 (dual-arm), got {dim}"

    pos1 = abs_traj[..., :3]
    euler1 = _rotate6d_to_euler_torch(abs_traj[..., 3:9])
    grip1 = abs_traj[..., 9:10]
    pos2 = abs_traj[..., 10:13]
    euler2 = _rotate6d_to_euler_torch(abs_traj[..., 13:19])
    grip2 = abs_traj[..., 19:20]

    result = torch.cat([pos1, euler1, grip1, pos2, euler2, grip2], dim=-1)

    if squeeze_output:
        result = result.squeeze(0)

    return result

def convert_euler_from_abs_to_rel(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert absolute Euler angles to relative.

    Simple subtraction for Euler angles (approximation valid for small angles).

    Args:
        proprio: Base Euler angles, shape (3,) or (..., 3)
        action: Absolute Euler angles, shape (..., 3)

    Returns:
        Relative Euler angles, same shape as action
    """
    return action - proprio


def convert_euler_from_rel_to_abs(proprio: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Convert relative Euler angles to absolute.

    Simple addition for Euler angles (approximation valid for small angles).

    Args:
        proprio: Base Euler angles, shape (3,) or (..., 3)
        action: Relative Euler angles, shape (..., 3)

    Returns:
        Absolute Euler angles, same shape as action
    """
    return action + proprio



def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    """Convert rotate6d (CONCATENATED format) to euler xyz angles.

    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., :3]   # first column
    a2 = v6[..., 3:6]  # second column
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    R = _get_scipy_rotation()
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    """Convert rotate6d (CONCATENATED format) to quaternion.

    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]
    """
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1, a2 = v6[..., :3], v6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    R = _get_scipy_rotation()
    return R.from_matrix(rot_mats).as_quat(scalar_first = scalar_first)


def _rotate6d_to_euler_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rotate6d (CONCATENATED format) to Euler angles (xyz convention).

    CONCATENATED format: [col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]

    Args:
        rot6d: Tensor of shape (..., 6) representing the first two columns of rotation matrix.

    Returns:
        Tensor of shape (..., 3) with Euler angles in radians.
    """
    original_shape = rot6d.shape[:-1]
    rot6d_flat = rot6d.reshape(-1, 6)

    # Check for degenerate cases (all zeros or very small norm)
    norm = torch.linalg.norm(rot6d_flat, dim=-1)
    valid_mask = norm > 1e-6

    # Initialize output with zeros (for degenerate cases)
    euler_flat = torch.zeros(rot6d_flat.shape[0], 3, dtype=rot6d.dtype, device=rot6d.device)

    if valid_mask.any():
        valid_rot6d = rot6d_flat[valid_mask]

        # Extract the two column vectors (CONCATENATED format)
        a1 = valid_rot6d[..., :3]   # first column
        a2 = valid_rot6d[..., 3:6]  # second column

        # Gram-Schmidt orthogonalization with clamping to avoid division by zero
        norm_a1 = torch.linalg.norm(a1, dim=-1, keepdim=True).clamp(min=1e-8)
        b1 = a1 / norm_a1
        proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = a2 - proj
        norm_b2 = torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = b2 / norm_b2
        b3 = torch.cross(b1, b2, dim=-1)

        # Stack to form rotation matrix [..., 3, 3]
        rot_mat = torch.stack([b1, b2, b3], dim=-1)

        # Convert rotation matrix to Euler angles (xyz convention)
        rot_mat_np = rot_mat.detach().cpu().numpy()

        # Handle any remaining NaN/Inf in rotation matrices
        valid_matrices = np.isfinite(rot_mat_np).all(axis=(1, 2))
        euler_valid = np.zeros((rot_mat_np.shape[0], 3), dtype=np.float64)

        if valid_matrices.any():
            R = _get_scipy_rotation()
            euler_valid[valid_matrices] = R.from_matrix(rot_mat_np[valid_matrices]).as_euler('xyz')

        euler_flat[valid_mask] = torch.from_numpy(euler_valid).to(dtype=rot6d.dtype, device=rot6d.device)

    return euler_flat.reshape(*original_shape, 3)


# =============================================================================
# Action Space Implementations
# =============================================================================


@register_action("ee6d")
class EE6DActionSpace(BaseActionSpace):
    """End-effector layout with xyz, 6D rotation, and gripper channels."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 1.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0

    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape, "pred/target shapes must match"
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        # Gripper BCE
        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        # XYZ position
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE

        # Rotation 6D
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action

@register_action("joint")
class JointActionSpace(BaseActionSpace):
    """Joint-space layout with joints + gripper only."""

    dim_action = 14
    gripper_idx = (6, 13)
    GRIPPER_SCALE = 0.1
    JOINTS_SCALE = 1.0

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        g_losses = [self.bce(pred[:, :, gi], target[:, :, gi]) for gi in self.gripper_idx]
        gripper_loss = sum(g_losses) / len(self.gripper_idx) * self.GRIPPER_SCALE

        joints_idx = tuple(i for i in range(D) if i not in set(self.gripper_idx))
        joints_loss = self.mse(pred[:, :, joints_idx], target[:, :, joints_idx]) * self.JOINTS_SCALE

        return {
            "joints_loss": joints_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """Zero-out gripper channels in proprio/action."""
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., self.gripper_idx] = 0.0
        action_m[..., self.gripper_idx] = 0.0
        return proprio_m, action_m

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid to gripper logits."""
        if action.size(-1) > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


@register_action("agibot_ee6d")
class AGIBOTEE6DActionSpace(BaseActionSpace):
    """AGI-bot variant of EE6DActionSpace using MSE for all components."""

    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 10.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, T, D = pred.shape
        _ensure_indices_valid(D, self.gripper_idx, "gripper_idx")

        gripper_loss = self.mse(pred[:, :, self.gripper_idx], target[:, :, self.gripper_idx]) * self.GRIPPER_SCALE
        pos_loss = (
            self.mse(pred[:, :, self.POS_IDX_1], target[:, :, self.POS_IDX_1]) +
            self.mse(pred[:, :, self.POS_IDX_2], target[:, :, self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[:, :, self.ROT_IDX_1], target[:, :, self.ROT_IDX_1]) +
            self.mse(pred[:, :, self.ROT_IDX_2], target[:, :, self.ROT_IDX_2])
        ) * self.ROT_SCALE

        return {
            "position_loss": pos_loss,
            "rotate6D_loss": rot_loss,
            "gripper_loss": gripper_loss,
        }

    def preprocess(self, proprio, action, mode="train"):
        """No preprocessing applied in AGIBOT variant."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """AGIBOT does not postprocess."""
        return action


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "EE6DActionSpace",
    "JointActionSpace",
    "AGIBOTEE6DActionSpace",
    "ACTION_REGISTRY",
    # Rotate6D conversions
    "convert_rotate6d_from_abs_to_rel",
    "convert_rotate6d_from_rel_to_abs",
    # Axis-angle conversions
    "convert_abs_rot6d_to_rel_axisangle",
    "convert_rel_axisangle_to_abs_rot6d",
    # Full action conversions (20D <-> 14D axis-angle)
    "convert_action_abs_to_rel_axisangle",
    "convert_action_rel_axisangle_to_abs",
    # Full action conversions (20D <-> 20D rot6d)
    "convert_action_abs_to_rel_rot6d",
    "convert_action_rel_rot6d_to_abs",
    # Euler conversions
    "convert_euler_from_rel_to_abs",
    "convert_euler_from_abs_to_rel",
    "convert_euler_to_rotate6d",
]




if __name__ == "__main__":
    # =========================================================================
    # Test 1: rotate6d abs<->rel conversion
    # =========================================================================
    print("=" * 60)
    print("Test 1: rotate6d abs<->rel conversion")
    print("=" * 60)
    proprio_rot = torch.randn(1, 1, 6)
    action_rot = torch.randn(1, 2, 6)
    rotation_matrix = _rotate6d_to_matrix(action_rot)

    relative_rot = convert_rotate6d_from_abs_to_rel(proprio_rot, action_rot)
    recovered_rot = convert_rotate6d_from_rel_to_abs(proprio_rot, relative_rot)
    rotation_matrix_new = _rotate6d_to_matrix(recovered_rot)

    test1_pass = torch.allclose(rotation_matrix, rotation_matrix_new, atol=1e-5)
    print(f"rotate6d abs->rel->abs reversible: {test1_pass}")

    # =========================================================================
    # Test 2: axis-angle abs<->rel conversion (rotation only)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 2: axis-angle abs<->rel conversion (rotation only)")
    print("=" * 60)
    proprio_rot6d = torch.randn(1, 1, 6)
    action_rot6d = torch.randn(1, 2, 6)
    original_matrix = _rotate6d_to_matrix(action_rot6d)

    # abs rot6d -> rel axis-angle
    rel_aa = convert_abs_rot6d_to_rel_axisangle(proprio_rot6d, action_rot6d)
    print(f"Relative axis-angle shape: {rel_aa.shape}")  # Should be (1, 2, 3)

    # rel axis-angle -> abs rot6d
    recovered_rot6d = convert_rel_axisangle_to_abs_rot6d(proprio_rot6d, rel_aa)
    recovered_matrix = _rotate6d_to_matrix(recovered_rot6d)

    test2_pass = torch.allclose(original_matrix, recovered_matrix, atol=1e-5)
    print(f"axis-angle abs->rel->abs reversible: {test2_pass}")

    # =========================================================================
    # Test 3: full action conversion (20D abs rot6d <-> 14D rel axis-angle)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 3: full action conversion (20D abs rot6d <-> 14D rel axis-angle)")
    print("=" * 60)
    # Create random 20D proprio and action
    proprio_20d = torch.randn(2, 20)  # (Batch, 20)
    action_20d = torch.randn(2, 4, 20)  # (Batch, Time, 20)

    # Save original rotation matrices for verification
    orig_rot1 = _rotate6d_to_matrix(action_20d[..., 3:9])
    orig_rot2 = _rotate6d_to_matrix(action_20d[..., 13:19])

    # 20D abs -> 14D rel
    rel_action_14d = convert_action_abs_to_rel_axisangle(proprio_20d, action_20d)
    print(f"Relative action shape: {rel_action_14d.shape}")  # Should be (2, 4, 14)

    # 14D rel -> 20D abs
    recovered_action_20d = convert_action_rel_axisangle_to_abs(proprio_20d, rel_action_14d)
    print(f"Recovered action shape: {recovered_action_20d.shape}")  # Should be (2, 4, 20)

    # Verify position
    pos1_match = torch.allclose(action_20d[..., 0:3], recovered_action_20d[..., 0:3], atol=1e-5)
    pos2_match = torch.allclose(action_20d[..., 10:13], recovered_action_20d[..., 10:13], atol=1e-5)
    print(f"Position 1 match: {pos1_match}")
    print(f"Position 2 match: {pos2_match}")

    # Verify rotation (compare rotation matrices)
    recov_rot1 = _rotate6d_to_matrix(recovered_action_20d[..., 3:9])
    recov_rot2 = _rotate6d_to_matrix(recovered_action_20d[..., 13:19])
    rot1_match = torch.allclose(orig_rot1, recov_rot1, atol=1e-5)
    rot2_match = torch.allclose(orig_rot2, recov_rot2, atol=1e-5)
    print(f"Rotation 1 match: {rot1_match}")
    print(f"Rotation 2 match: {rot2_match}")

    # Verify gripper
    grip1_match = torch.allclose(action_20d[..., 9:10], recovered_action_20d[..., 9:10], atol=1e-5)
    grip2_match = torch.allclose(action_20d[..., 19:20], recovered_action_20d[..., 19:20], atol=1e-5)
    print(f"Gripper 1 match: {grip1_match}")
    print(f"Gripper 2 match: {grip2_match}")

    test3_pass = pos1_match and pos2_match and rot1_match and rot2_match and grip1_match and grip2_match
    print(f"\nFull action 20D->14D->20D reversible: {test3_pass}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (rotate6d abs<->rel):      {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (axis-angle abs<->rel):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (full action 20D<->14D):   {'PASS' if test3_pass else 'FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\nAll tests: {'PASS' if all_pass else 'FAIL'}")
