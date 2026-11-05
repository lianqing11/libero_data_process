"""
Runtime perturbation module for LIBERO-Plus evaluation.

Three perturbation types applied at evaluation time:
  - camera_star:     Larger agentview camera perturbation
  - camera_plusplus:  Agentview perturbation + wrist view image flip
  - action:          Systematic calibration shift on actions
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Optional

try:
    import robosuite.utils.transform_utils as T
except ImportError:
    T = None


# ---------------------------------------------------------------------------
# Camera transform helpers
# (Adapted from libero_kitchen_tabletop_manipulation.py)
# ---------------------------------------------------------------------------

def scale_distance_from_pivot(original_quat=None, original_pos=None, scale_factor=1.3):
    """Scale position distance from pivot (0, 0, 0.8), keep quaternion unchanged."""
    result = {}
    pivot_point = np.array([0, 0, 0.8])

    if original_quat is not None:
        result['new_quat'] = original_quat

    if original_pos is not None:
        original_pos = np.array(original_pos)
        vec_to_point = original_pos - pivot_point
        scaled_vec = vec_to_point * scale_factor
        new_pos = pivot_point + scaled_vec
        result['new_pos'] = new_pos.tolist() if isinstance(new_pos, np.ndarray) else new_pos

    return result


def rotate_around_y(original_quat=None, original_pos=None, degrees=0):
    """Rotate quaternion/position around a Y-parallel axis through (0, 0, 0.8)."""
    result = {}
    axis = np.array([0, 1, 0])
    axis_point = np.array([0, 0, 0.8])
    custom_rotation = Rotation.from_rotvec(np.radians(-degrees) * axis)

    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = custom_rotation * original_rot
        nq = combined_rot.as_quat()
        result['new_quat'] = [float(nq[3]), float(nq[0]), float(nq[1]), float(nq[2])]

    if original_pos is not None:
        translated_pos = np.array(original_pos) - axis_point
        rotated_pos = custom_rotation.apply(translated_pos)
        final_pos = rotated_pos + axis_point
        result['new_pos'] = final_pos.tolist() if isinstance(final_pos, np.ndarray) else final_pos

    return result


def rotate_around_z(original_quat=None, original_pos=None, degrees=0):
    """Rotate quaternion/position around the world Z axis."""
    result = {}
    z_rotation = Rotation.from_euler('z', degrees, degrees=True)

    if original_quat is not None:
        original_rot = Rotation.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        combined_rot = z_rotation * original_rot
        nq = combined_rot.as_quat()
        result['new_quat'] = [float(nq[3]), float(nq[0]), float(nq[1]), float(nq[2])]

    if original_pos is not None:
        rotated_pos = z_rotation.apply(np.array(original_pos))
        result['new_pos'] = rotated_pos.tolist() if isinstance(rotated_pos, np.ndarray) else rotated_pos

    return result


def _apply_camera_transforms(pos, quat, horizontal, vertical, scale,
                             endpoint_rot, endpoint_vert):
    """Apply the full agentview camera transform chain and return new (pos, quat)."""
    if int(vertical) != 0:
        r = rotate_around_y(original_quat=quat, original_pos=pos, degrees=int(vertical))
        pos, quat = r['new_pos'], r['new_quat']
        r = rotate_around_z(original_quat=quat, original_pos=pos, degrees=int(horizontal))
        pos, quat = r['new_pos'], r['new_quat']
    elif int(horizontal) != 0:
        r = rotate_around_z(original_quat=quat, original_pos=pos, degrees=int(horizontal))
        pos, quat = r['new_pos'], r['new_quat']

    if float(scale) != 1.0:
        r = scale_distance_from_pivot(original_quat=quat, original_pos=pos, scale_factor=float(scale))
        pos, quat = r['new_pos'], r['new_quat']

    if int(endpoint_rot) != 0:
        r = rotate_around_z(original_quat=quat, degrees=int(endpoint_rot))
        quat = r['new_quat']

    if int(endpoint_vert) != 0:
        r = rotate_around_y(original_quat=quat, degrees=int(endpoint_vert))
        quat = r['new_quat']

    return pos, quat


def _modify_agentview(env, pos, quat):
    """Write agentview camera pos/quat to the MuJoCo sim model."""
    sim = env.env.sim if hasattr(env, 'env') else env.sim
    cam_id = sim.model.camera_name2id("agentview")
    sim.model.cam_pos[cam_id] = np.array(pos, dtype=np.float64)
    sim.model.cam_quat[cam_id] = np.array(quat, dtype=np.float64)


def _read_agentview(env):
    """Read current agentview pos/quat from the MuJoCo sim model."""
    sim = env.env.sim if hasattr(env, 'env') else env.sim
    cam_id = sim.model.camera_name2id("agentview")
    pos = sim.model.cam_pos[cam_id].copy().tolist()
    quat = sim.model.cam_quat[cam_id].copy().tolist()
    return pos, quat


# ---------------------------------------------------------------------------
# 1. CameraStarPerturbation  (larger agentview perturbation)
# ---------------------------------------------------------------------------

# Predefined configs for visualization (deterministic, easy to inspect)
CAMERA_STAR_VIS_CONFIGS = [
    {"horizontal": 0,   "vertical": 0,  "scale": 1.0, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "baseline"},
    {"horizontal": 85,  "vertical": 0,  "scale": 1.0, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "H+85"},
    {"horizontal": -85, "vertical": 0,  "scale": 1.0, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "H-85"},
    {"horizontal": 0,   "vertical": 28, "scale": 1.0, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "V+28"},
    {"horizontal": 0,   "vertical": 0,  "scale": 2.4, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "S2.4"},
    {"horizontal": 70,  "vertical": 20, "scale": 1.8, "endpoint_rot": 0,  "endpoint_vert": 0,  "label": "H70_V20_S1.8"},
    {"horizontal": 0,   "vertical": 0,  "scale": 1.0, "endpoint_rot": 18, "endpoint_vert": 18, "label": "ER18_EV18"},
]


class CameraStarPerturbation:
    """Extended agentview camera perturbation with larger parameter ranges."""

    # Extended ranges (compared to standard camera perturbation)
    H_RANGE = (-90, 90)
    V_RANGE = (0, 30)
    S_RANGE = (1.0, 2.5)
    ER_RANGE = (-20, 20)
    EV_RANGE = (-20, 20)

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def sample_config(self) -> Dict:
        """Sample a random camera_star config."""
        h = self.rng.uniform(*self.H_RANGE)
        v = self.rng.uniform(*self.V_RANGE)
        s = self.rng.uniform(*self.S_RANGE)
        er = self.rng.uniform(*self.ER_RANGE)
        ev = self.rng.uniform(*self.EV_RANGE)
        return {
            "horizontal": round(h, 1),
            "vertical": round(v, 1),
            "scale": round(s, 2),
            "endpoint_rot": round(er, 1),
            "endpoint_vert": round(ev, 1),
        }

    def apply(self, env, config: Optional[Dict] = None) -> Dict:
        """Apply camera_star perturbation to the environment.

        Must be called after env.reset() / set_init_state().
        """
        if config is None:
            config = self.sample_config()

        pos, quat = _read_agentview(env)
        new_pos, new_quat = _apply_camera_transforms(
            pos, quat,
            config["horizontal"], config["vertical"], config["scale"],
            config["endpoint_rot"], config["endpoint_vert"],
        )
        _modify_agentview(env, new_pos, new_quat)
        return config


# ---------------------------------------------------------------------------
# 2. CameraPlusPlusPerturbation  (agentview + wrist flip)
# ---------------------------------------------------------------------------

WRIST_FLIP_MODES = ["none", "hflip", "vflip", "rotate180"]

# Wrist camera default FOV and perturbation range
WRIST_DEFAULT_FOVY = 75.0
WRIST_FOVY_RANGE = (60.0, 90.0)  # ±20% from default — simulates different lens

# Predefined configs for visualization
CAMERA_PP_VIS_CONFIGS = [
    {"agentview": {"horizontal": 40,  "vertical": 0,  "scale": 1.3, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "none",      "wrist_fovy": 75,  "label": "wrist_none_fov75"},
    {"agentview": {"horizontal": 40,  "vertical": 0,  "scale": 1.3, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "hflip",     "wrist_fovy": 60,  "label": "wrist_hflip_fov60"},
    {"agentview": {"horizontal": 40,  "vertical": 0,  "scale": 1.3, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "vflip",     "wrist_fovy": 90,  "label": "wrist_vflip_fov90"},
    {"agentview": {"horizontal": 40,  "vertical": 0,  "scale": 1.3, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "rotate180", "wrist_fovy": 65,  "label": "wrist_rot180_fov65"},
    {"agentview": {"horizontal": -50, "vertical": 10, "scale": 1.5, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "hflip",     "wrist_fovy": 85,  "label": "wrist_hflip_fov85_agH-50"},
]


def _modify_wrist_fovy(env, fovy: float):
    """Set wrist camera FOV. Works at runtime without breaking body attachment."""
    sim = env.env.sim if hasattr(env, 'env') else env.sim
    cam_id = sim.model.camera_name2id("robot0_eye_in_hand")
    sim.model.cam_fovy[cam_id] = float(fovy)


class CameraPlusPlusPerturbation:
    """Agentview perturbation (standard camera range) + wrist view flip + FOV scale."""

    # Standard camera ranges (same as existing LIBERO-Plus camera perturbation)
    H_RANGE = (-75, 75)
    V_RANGE = (0, 15)
    S_RANGE = (1.0, 2.0)
    ER_RANGE = (-10, 10)
    EV_RANGE = (-10, 10)

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def sample_config(self) -> Dict:
        """Sample agentview config + wrist flip mode + wrist FOV."""
        h = self.rng.uniform(*self.H_RANGE)
        v = self.rng.uniform(*self.V_RANGE)
        s = self.rng.uniform(*self.S_RANGE)
        er = self.rng.uniform(*self.ER_RANGE)
        ev = self.rng.uniform(*self.EV_RANGE)
        flip = self.rng.choice(WRIST_FLIP_MODES)
        fovy = self.rng.uniform(*WRIST_FOVY_RANGE)
        return {
            "agentview": {
                "horizontal": round(h, 1),
                "vertical": round(v, 1),
                "scale": round(s, 2),
                "endpoint_rot": round(er, 1),
                "endpoint_vert": round(ev, 1),
            },
            "wrist_flip": flip,
            "wrist_fovy": round(fovy, 1),
        }

    def apply_agentview(self, env, config: Dict) -> None:
        """Modify agentview camera via sim.model."""
        av = config["agentview"]
        pos, quat = _read_agentview(env)
        new_pos, new_quat = _apply_camera_transforms(
            pos, quat,
            av["horizontal"], av["vertical"], av["scale"],
            av["endpoint_rot"], av["endpoint_vert"],
        )
        _modify_agentview(env, new_pos, new_quat)

    @staticmethod
    def apply_wrist_fovy(env, config: Dict) -> None:
        """Modify wrist camera FOV via sim.model. Call once after env reset."""
        fovy = config.get("wrist_fovy", WRIST_DEFAULT_FOVY)
        if fovy != WRIST_DEFAULT_FOVY:
            _modify_wrist_fovy(env, fovy)

    @staticmethod
    def apply_wrist(obs: Dict, config: Dict) -> Dict:
        """Apply image-level flip to wrist view in observation dict.

        Modifies obs in-place and returns it.
        """
        mode = config.get("wrist_flip", "none")
        key = "robot0_eye_in_hand_image"
        if key not in obs or mode == "none":
            return obs

        img = obs[key]
        if mode == "hflip":
            img = np.flip(img, axis=1).copy()
        elif mode == "vflip":
            img = np.flip(img, axis=0).copy()
        elif mode == "rotate180":
            img = np.flip(np.flip(img, axis=0), axis=1).copy()
        obs[key] = img
        return obs


# ---------------------------------------------------------------------------
# 2b. CameraPlusPlusStarPerturbation  (star = wider ranges)
# ---------------------------------------------------------------------------

CAMERA_PP_STAR_VIS_CONFIGS = [
    {"agentview": {"horizontal": 85,  "vertical": 25, "scale": 2.0, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "hflip",     "wrist_fovy": 55,  "label": "star_H85_V25_S2_hflip_fov55"},
    {"agentview": {"horizontal": -80, "vertical": 0,  "scale": 2.4, "endpoint_rot": 0, "endpoint_vert": 0},
     "wrist_flip": "vflip",     "wrist_fovy": 95,  "label": "star_H-80_S2.4_vflip_fov95"},
    {"agentview": {"horizontal": 60,  "vertical": 28, "scale": 1.5, "endpoint_rot": 15, "endpoint_vert": 15},
     "wrist_flip": "rotate180", "wrist_fovy": 50,  "label": "star_combined_rot180_fov50"},
    {"agentview": {"horizontal": 0,   "vertical": 0,  "scale": 2.5, "endpoint_rot": 18, "endpoint_vert": 18},
     "wrist_flip": "hflip",     "wrist_fovy": 100, "label": "star_S2.5_ER18_hflip_fov100"},
]


class CameraPlusPlusStarPerturbation(CameraPlusPlusPerturbation):
    """Star variant of camera_plusplus with wider parameter ranges."""

    H_RANGE = (-90, 90)
    V_RANGE = (0, 30)
    S_RANGE = (1.0, 2.5)
    ER_RANGE = (-20, 20)
    EV_RANGE = (-20, 20)

    def sample_config(self) -> Dict:
        """Sample with wider wrist FOV range (50-100 vs 60-90)."""
        h = self.rng.uniform(*self.H_RANGE)
        v = self.rng.uniform(*self.V_RANGE)
        s = self.rng.uniform(*self.S_RANGE)
        er = self.rng.uniform(*self.ER_RANGE)
        ev = self.rng.uniform(*self.EV_RANGE)
        flip = self.rng.choice(WRIST_FLIP_MODES)
        fovy = self.rng.uniform(50.0, 100.0)
        return {
            "agentview": {
                "horizontal": round(h, 1),
                "vertical": round(v, 1),
                "scale": round(s, 2),
                "endpoint_rot": round(er, 1),
                "endpoint_vert": round(ev, 1),
            },
            "wrist_flip": flip,
            "wrist_fovy": round(fovy, 1),
        }


# ---------------------------------------------------------------------------
# 3. ActionPerturbation  (calibration shift)
# ---------------------------------------------------------------------------

ACTION_SEVERITY = {
    1: {"pos_range": 0.005, "rot_range_deg": 1.0},   # +-5mm,  +-1 deg
    2: {"pos_range": 0.010, "rot_range_deg": 3.0},   # +-10mm, +-3 deg
    3: {"pos_range": 0.020, "rot_range_deg": 6.0},   # +-20mm, +-6 deg
}

# Predefined configs for visualization
ACTION_VIS_CONFIGS = [
    {"level": 1, "label": "action_level1 (+-5mm, +-1deg)"},
    {"level": 2, "label": "action_level2 (+-10mm, +-3deg)"},
    {"level": 3, "label": "action_level3 (+-20mm, +-6deg)"},
]


class ActionPerturbation:
    """Systematic calibration shift applied to every action in an episode."""

    def __init__(self, seed: int, level: int = 2):
        self.rng = np.random.RandomState(seed)
        self.level = level
        assert level in ACTION_SEVERITY, f"level must be 1, 2, or 3, got {level}"

    def sample_bias(self) -> Dict:
        """Sample a fixed calibration bias (called once per episode)."""
        cfg = ACTION_SEVERITY[self.level]
        pos_bias = self.rng.uniform(-cfg["pos_range"], cfg["pos_range"], size=3)
        rot_range_rad = np.radians(cfg["rot_range_deg"])
        # Sample rotation bias as axis-angle: random axis, random angle
        rot_axis = self.rng.randn(3)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-8)
        rot_angle = self.rng.uniform(-rot_range_rad, rot_range_rad)
        rot_bias_aa = rot_axis * rot_angle
        return {"pos_bias": pos_bias, "rot_bias_aa": rot_bias_aa}

    @staticmethod
    def apply(action: np.ndarray, bias: Dict) -> np.ndarray:
        """Apply calibration shift to a 7D absolute action [pos3, aa3, grip1].

        Position: additive offset.
        Rotation: quaternion composition (bias * target).
        Gripper: unchanged.
        """
        perturbed = action.copy()
        # Position shift
        perturbed[:3] = perturbed[:3] + bias["pos_bias"]
        # Rotation shift via quaternion composition
        target_quat = T.axisangle2quat(action[3:6])
        bias_quat = T.axisangle2quat(bias["rot_bias_aa"])
        composed_quat = T.quat_multiply(bias_quat, target_quat)
        perturbed[3:6] = T.quat2axisangle(composed_quat)
        return perturbed


# ---------------------------------------------------------------------------
# 3b. ActionStarPerturbation  (star = larger bias)
# ---------------------------------------------------------------------------

ACTION_STAR_SEVERITY = {
    1: {"pos_range": 0.015, "rot_range_deg": 5.0},    # +-15mm, +-5 deg
    2: {"pos_range": 0.030, "rot_range_deg": 10.0},   # +-30mm, +-10 deg
    3: {"pos_range": 0.050, "rot_range_deg": 15.0},   # +-50mm, +-15 deg
}

ACTION_STAR_VIS_CONFIGS = [
    {"level": 1, "label": "action_star_lv1 (+-15mm, +-5deg)"},
    {"level": 2, "label": "action_star_lv2 (+-30mm, +-10deg)"},
    {"level": 3, "label": "action_star_lv3 (+-50mm, +-15deg)"},
]


class ActionStarPerturbation(ActionPerturbation):
    """Star variant of action perturbation with larger calibration bias."""

    def __init__(self, seed: int, level: int = 2):
        self.rng = np.random.RandomState(seed)
        self.level = level
        assert level in ACTION_STAR_SEVERITY, f"level must be 1, 2, or 3, got {level}"

    def sample_bias(self) -> Dict:
        cfg = ACTION_STAR_SEVERITY[self.level]
        pos_bias = self.rng.uniform(-cfg["pos_range"], cfg["pos_range"], size=3)
        rot_range_rad = np.radians(cfg["rot_range_deg"])
        rot_axis = self.rng.randn(3)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-8)
        rot_angle = self.rng.uniform(-rot_range_rad, rot_range_rad)
        rot_bias_aa = rot_axis * rot_angle
        return {"pos_bias": pos_bias, "rot_bias_aa": rot_bias_aa}


# ---------------------------------------------------------------------------
# Convenience: create perturbation from name
# ---------------------------------------------------------------------------

def create_perturbation(name: str, seed: int, level: int = 2):
    """Factory to create a perturbation object from its name."""
    if name == "camera_star":
        return CameraStarPerturbation(seed)
    elif name == "camera_plusplus":
        return CameraPlusPlusPerturbation(seed)
    elif name == "camera_plusplus_star":
        return CameraPlusPlusStarPerturbation(seed)
    elif name == "action":
        return ActionPerturbation(seed, level=level)
    elif name == "action_star":
        return ActionStarPerturbation(seed, level=level)
    else:
        raise ValueError(f"Unknown perturbation: {name}")
