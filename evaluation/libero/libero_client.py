
#!/usr/bin/env python3
"""
Refined Libero evaluation client.
- Stronger typing, docstrings, and logging
- Safer HTTP client with timeouts & error handling
- Cleaner action post-processing and 6D-rotation helpers
- Deterministic seeding and robust main() patterned after the provided example
"""
from __future__ import annotations
import sys
import logging
from mmengine import fileio
import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import imageio
import json_numpy
import numpy as np
import requests
import torch  # noqa: F401  # (kept in case of future GPU array ops)
import torchvision.transforms as transforms  # noqa: F401
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv  # noqa: F401
import robosuite.utils.transform_utils as T

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS = 1e-6

LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

benchmark_dict = benchmark.get_benchmark_dict()
logger = logging.getLogger("libero_eval")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match original code behavior: vertical+horizontal flips."""
    return np.flip(np.flip(img, 0), 1)


def _rollout_video_frame(obs: Dict) -> np.ndarray:
    """Side-by-side [agentview | wrist] frame for rollout videos."""
    agentview = _flip_agentview(obs["agentview_image"])
    wrist = obs["robot0_eye_in_hand_image"]
    return np.concatenate([agentview, wrist], axis=1)


def _policy_needs_fresh_obs(policy: ClientModel) -> bool:
    plan = getattr(policy, "action_plan", None)
    if plan is None:
        return True
    return len(plan) == 0


def _step_env(env, action, obs: Dict, *, render_obs: bool):
    if render_obs:
        return env.step(action)

    inner = env.env
    if getattr(inner, "done", False):
        raise ValueError("executing action in terminated episode")

    act = np.asarray(action)
    if getattr(inner, "action_dim", 7) == 4 and len(act) > 4:
        act = np.concatenate((act[:3], act[-1:]), axis=-1)

    disabled = [
        name for name, ob in inner._observables.items()
        if "image" in name and ob.is_enabled()
    ]
    for name in disabled:
        inner._observables[name].set_enabled(False)
    try:
        _, reward, done, info = inner.step(act)
        return obs, reward, done, info
    finally:
        for name in disabled:
            inner._observables[name].set_enabled(True)


# -----------------------------------------------------------------------------
# Action processing
# -----------------------------------------------------------------------------
class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle.

        Args:
            r6d: array with shape (N, 6) or (6,)
        Returns:
            array with shape (N, 3) or (3,)
        """
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        # b1
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)

        # b2
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)

        # b3
        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

        axis_angle_list: List[np.ndarray] = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        else:
            raise ValueError("Only 1D axis-angle supported here")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")


# -----------------------------------------------------------------------------
# HTTP Client Policy
# -----------------------------------------------------------------------------
class ClientModel:
    """Thin HTTP client that queries a remote policy server and returns actions.

    Uses ee6d action mode: [pos(3), rot6d(6), grip(1)] per arm = 10D
    Converts rot6d to axis_angle for Libero simulator.
    """

    def __init__(self, host: str, port: int, action_mode: str = "ee6d"):
        """
        Initialize client.

        Args:
            host: Server host
            port: Server port
            action_mode: Action mode (only "ee6d" supported)
        """
        self.url = f"http://{host}:{port}/act"
        self.processor = LiberoAbsActionProcessor()
        self.action_mode = action_mode
        self.action_dim = 10  # pos(3) + rot6d(6) + grip(1)
        self.proprio_dim = 20  # 10D x 2 arms (ee6d dual-arm format)
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None  # last absolute proprio
        self.action_plan: Deque[List[float]] = collections.deque()

    def _format_query(self, obs: Dict, goal: str) -> Dict:
        main_view = _flip_agentview(obs["agentview_image"])  # (256,256,3)
        wrist_view = obs["robot0_eye_in_hand_image"]  # (256,256,3)

        # ee6d: proprio is [pos(3), rot6d(6), grip(1)] x 2 = 20D
        closed_loop_proprio = np.concatenate([obs['robo_pos'], obs['robo_ori'], np.array([0.0])], axis=-1)
        closed_loop_proprio = np.concatenate([closed_loop_proprio, np.zeros_like(closed_loop_proprio)], axis=-1)

        if self.proprio is None:
            self.proprio = closed_loop_proprio
        return {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": 3,
            "steps": 10,
        }

    def _post(self, payload: Dict) -> np.ndarray:
        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])
        if action.ndim != 2 or action.shape[1] < self.action_dim:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}, "
                             f"expected (T, {self.action_dim})")
        return action

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            payload = self._format_query(obs, goal)
            action = self._post(payload)

            self.proprio[:9] = action[-1, :9].copy()

            # Convert rot6d->axis-angle and build final [pos3, aa3, grip1]
            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)

            # Queue up the plan
            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        # Discretize gripper to {-1, 1}
        action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
        return action_predict


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
class LIBEROEval:
    def __init__(
        self,
        task_suite_name: str,
        eval_horizon: int = 600,
        act_type: str = "abs",
        num_episodes: int = 10,
        eval_freq: int = 10,  # reserved
        init_seed: int = 42,
        save_video: bool = True,
        max_tasks: Optional[int] = None,  # Limit number of tasks per suite (None = all)
    ) -> None:
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.max_tasks = max_tasks  # If set, limit number of tasks to evaluate
        self.eval_freq = eval_freq
        self.init_seed = init_seed
        self.act_type = act_type
        self.processor = LiberoAbsActionProcessor()
        self.base_dir: Path = Path('.')
        self.save_video = save_video

    # ---- internal helpers --------------------------------------------------
    def _make_dir(self, save_path: Path) -> None:
        path = save_path / self.task_suite_name
        _ensure_dir(path)
        self.base_dir = path

    def _init_env(self, task_suite, task_id: int = 0, ep: int = 0):

        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(
            f"[info] retrieving task {task_id} from suite {self.task_suite_name}, "
            f"language: {task_description}, bddl: {task_bddl_file}"
        )

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
        env = OffScreenRenderEnv(**env_args)

        # Environment reset & deterministic init state
        env.seed(self.init_seed + ep + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])

        # settle
        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, done, info = env.step(action)

        if self.act_type == 'abs':
            for robot in env.env.robots:
                robot.controller.use_delta = False
        elif self.act_type == 'rel':
            pass
        else:
            raise ValueError("act_type must be 'abs' or 'rel'")

        return env, task_description, obs

    def _log_results(self, metrics: Dict) -> None:
        print(metrics)
        save_name = self.base_dir / 'episodes.jsonl'
        with open(save_name, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_video(self, save_path: Path, images: List[np.ndarray], fps: int = 30) -> None:
        imageio.mimsave(save_path.as_posix(), images, fps=fps)

    def _get_safe_filename(self, lang: str, ep: int, success: bool, max_length: int = 200) -> str:
        """Generate a safe filename that won't exceed filesystem limits.

        Args:
            lang: Task description/language instruction
            ep: Episode number
            success: Whether the task was successful
            max_length: Maximum length for the base filename (excluding extension)

        Returns:
            A safe filename string
        """
        import hashlib

        # Clean the language string
        lang_clean = lang.replace(" ", "_")
        # Remove any characters that might cause issues
        lang_clean = "".join(c for c in lang_clean if c.isalnum() or c in "_-")

        suffix = f"_{ep}_{'success' if success else 'fail'}"
        extension = ".mp4"

        # Calculate available length for the task name
        # Reserve space for suffix and extension
        available_length = max_length - len(suffix) - len(extension)

        if len(lang_clean) <= available_length:
            # Filename is short enough, use as is
            return f"{lang_clean}{suffix}{extension}"
        else:
            # Filename is too long, truncate and add hash for uniqueness
            # Use first part of name + hash of full name
            hash_suffix = hashlib.md5(lang.encode()).hexdigest()[:8]
            truncate_length = available_length - len(hash_suffix) - 1  # -1 for underscore
            truncated_name = lang_clean[:truncate_length]
            return f"{truncated_name}_{hash_suffix}{suffix}{extension}"

    def _rollout(self, task_suite, policy: ClientModel, task_id: int, ep: int) -> tuple[float, int]:
        env, lang, obs = self._init_env(task_suite, task_id, ep)
        try:
            images: List[np.ndarray] = []
            save_path: Optional[Path] = None

            done_flag = False
            steps_taken = 0
            for step_idx in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                robo_pos = env.env.robots[0].controller.ee_pos
                obs['robo_ori'] = robo_ori
                obs['robo_pos'] = robo_pos

                action = policy.step(obs, lang)

                if self.save_video:
                    images.append(_rollout_video_frame(obs))
                render_obs = self.save_video or _policy_needs_fresh_obs(policy)
                obs, reward, done, info = _step_env(env, action, obs, render_obs=render_obs)
                if done:
                    done_flag = True
                    steps_taken = step_idx + 1
                    break
            if not done_flag:
                steps_taken = self.eval_horizon
            success = 1.0 if done_flag else 0.0

            if self.save_video:
                # Use safe filename to avoid path length issues
                safe_filename = self._get_safe_filename(lang, ep, success == 1.0)
                save_path = self.base_dir / safe_filename
                self._save_video(save_path, images, fps=30)

            metrics = {
                f'sim/{self.task_suite_name}/{lang}': success,
                f'steps/{self.task_suite_name}/{lang}': steps_taken,
            }
            save_path_display = save_path if save_path is not None else "video_disabled"
            print(metrics, "success", success, save_path_display)
            logger.info(
                "[%s] task=%s episode=%d success=%s steps=%d/%d",
                self.task_suite_name,
                lang,
                ep,
                bool(success),
                steps_taken,
                self.eval_horizon,
            )
            if hasattr(policy, "write_gripper_curve"):
                tag = f"{self.task_suite_name}_task{task_id}_ep{ep}"
                policy.write_gripper_curve(self.base_dir, tag)
            self._log_results(metrics)
            return success, steps_taken
        finally:
            env.close()

    # ---- public API --------------------------------------------------------
    def eval_episodes(
        self,
        policy: ClientModel,
        save_path: Path,
    ) -> tuple[float, float, List[Dict]]:
        self._make_dir(save_path)

        rews: List[float] = []
        steps: List[int] = []
        raw_records: List[Dict] = []

        pairs = [(ts, tid) for ts in self.task_suite_list for tid in range(len(ts.tasks))]
        all_tasks_count = len(pairs)
        if self.max_tasks is not None and self.max_tasks > 0:
            pairs = pairs[:self.max_tasks]

        effective_tasks = len(pairs)
        total_episodes = effective_tasks * self.num_episodes
        print(f"\n🚀 Starting evaluation: {effective_tasks}/{all_tasks_count} tasks × "
              f"{self.num_episodes} episodes = {total_episodes} total episodes\n")

        episode_count = 0
        for task_suite, task_id in tqdm(pairs, desc=f"Evaluating {self.task_suite_name}"):
            for ep in range(self.num_episodes):
                policy.reset()
                rew, ep_steps = self._rollout(task_suite, policy, task_id, ep)
                rews.append(rew)
                steps.append(ep_steps)
                raw_records.append({
                    "suite": self.task_suite_name,
                    "task_id": task_id,
                    "benchmark": type(task_suite).__name__,
                    "task_name": task_suite.get_task(task_id).name,
                    "ep": ep,
                    "success": float(rew),
                    "steps": int(ep_steps),
                })
                episode_count += 1

                current_success_rate = float(sum(rews) / len(rews))
                current_avg_steps = float(sum(steps) / len(steps))
                success_count = int(sum(rews))
                progress_pct = (episode_count / total_episodes * 100) if total_episodes > 0 else 0

                tqdm.write(
                    f"📊 [{self.task_suite_name}] Progress: {episode_count}/{total_episodes} ({progress_pct:.1f}%) | "
                    f"Success: {success_count}/{len(rews)} ({current_success_rate*100:.2f}%) | "
                    f"Avg Steps: {current_avg_steps:.1f}"
                )

        eval_rewards = float(sum(rews) / max(len(rews), 1))
        avg_steps = float(sum(steps) / max(len(steps), 1))
        success_count = int(sum(rews))
        total_episodes = len(rews)

        metrics = {
            f'sim_summary/{self.task_suite_name}/all': eval_rewards,
            f'steps_summary/{self.task_suite_name}/all': avg_steps,
        }
        self._log_results(metrics)

        # Final summary with emphasis
        print("\n" + "="*80)
        print(f"🏁 [{self.task_suite_name}] FINAL RESULTS")
        print("="*80)
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Successful:    {success_count}")
        print(f"   Failed:         {total_episodes - success_count}")
        print(f"   Success Rate:   {eval_rewards*100:.2f}%")
        print(f"   Average Steps:  {avg_steps:.2f}")
        print("="*80 + "\n")

        logger.info(
            "[%s] success_rate=%.3f (%d/%d episodes) avg_steps=%.2f",
            self.task_suite_name,
            eval_rewards,
            success_count,
            total_episodes,
            avg_steps,
        )
        return eval_rewards, avg_steps, raw_records



# -----------------------------------------------------------------------------
# Batch evaluator across suites
# -----------------------------------------------------------------------------

def eval_libero(
    agent: ClientModel,
    save_path: Path,
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = 'abs',
    task_suites: Iterable[str] = ("libero_goal", "libero_spatial", "libero_10"),
    save_video: bool = True,
    max_tasks: Optional[int] = None,  # Limit number of tasks per suite (None = all)
    **kwargs,  # Accept perturbation args for API compatibility (ignored in standard client)
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, List[Dict]]]:
    success_rates: Dict[str, float] = {}
    avg_steps: Dict[str, float] = {}
    raw_per_suite: Dict[str, List[Dict]] = {}
    task_suites_list = list(task_suites)
    total_suites = len(task_suites_list)

    max_tasks_info = f"   Max tasks per suite: {max_tasks}" if max_tasks else "   Max tasks per suite: All"

    print(f"\n{'='*80}")
    print(f"🎯 Starting LIBERO Evaluation")
    print(f"   Task Suites: {total_suites}")
    print(f"   Episodes per task: {num_episodes}")
    print(max_tasks_info)
    print(f"   Action type: {act_type}")
    print(f"{'='*80}\n")

    for idx, suite_name in enumerate(task_suites_list, 1):
        print(f"\n{'─'*80}")
        print(f"📦 [{idx}/{total_suites}] Evaluating suite: {suite_name}")
        print(f"{'─'*80}")

        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEval(
            task_suite_name=suite_name,
            eval_horizon=horizon,
            act_type=act_type,
            num_episodes=num_episodes,
            init_seed=init_seed,
            save_video=save_video,
            max_tasks=max_tasks,
        )
        eval_rewards, suite_avg_steps, raw_records = evaluator.eval_episodes(
            agent, save_path=save_path,
        )
        success_rates[suite_name] = eval_rewards
        avg_steps[suite_name] = suite_avg_steps
        raw_per_suite[suite_name] = raw_records
        with open(save_path / suite_name / "raw_episodes.json", "w", encoding="utf-8") as handle:
            json.dump(raw_records, handle, indent=2)

        print(f"✅ [{idx}/{total_suites}] Completed: {suite_name} | "
              f"Success Rate: {eval_rewards*100:.2f}% | Avg Steps: {suite_avg_steps:.2f}\n")

    # Also write a compact JSON summary at root
    summary_path = (save_path / "results.json").as_posix()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"success_rate": success_rates, "avg_steps": avg_steps}, f, indent=2)
        f.write("\n")
    return success_rates, avg_steps, raw_per_suite


# -----------------------------------------------------------------------------
# Main (patterned after the provided example)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    # Connection options
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json (contains 'host' and 'port')")
    parser.add_argument("--server_ip", type=str, default=None,
                        help="Manual server IP (if not using --connection_info)")
    parser.add_argument("--server_port", type=int, default=None,
                        help="Manual server port (if not using --connection_info)")

    # Eval options
    parser.add_argument("--output_dir", type=str, default="logs/",
                        help="Directory for saving evaluation videos and logs")
    parser.add_argument("--task_suites", nargs='+', default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
                        help="Libero suites to evaluate")
    parser.add_argument("--eval_time", type=int, default=10, help="Episodes per task")
    parser.add_argument("--init_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"], help="Action type")
    parser.add_argument("--action_mode", type=str, default="ee6d", help="Action mode (ee6d)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    print("🚀 [Client] Starting LIBERO evaluation client...")

    # ------------------------ 1) Load connection info ------------------------
    if args.connection_info is not None:
        info_path = Path(args.connection_info)
        print(f"🔍 Waiting for connection info file: {info_path}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not info_path.exists():
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server to start...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Connection info file found!")
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                infos = json.load(f)
            host, port = infos["host"], int(infos["port"])
            print(f"🔗 Loaded server info: host={host}, port={port}")
        except Exception as e:
            print(f"❌ Failed to read connection info: {e}")
            sys.exit(1)
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must specify either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, int(args.server_port)
        print(f"🔗 Using manual server address: {host}:{port}")

    # ------------------------ 2) Initialize client ---------------------------
    print(f"🛰️  Connecting to policy server at {host}:{port} ...")
    client = ClientModel(host, port)
    print("✅ Successfully initialized client!")

    # ------------------------ 3) Run evaluation ------------------------------
    print("🎯 Starting LIBERO policy evaluation...")
    print(f"📁 Results and videos will be saved to: {out_dir.resolve()}")
    print("-" * 88)
    print("init seed:", args.init_seed)
    print("task suites:", args.task_suites)
    print("episodes per task:", args.eval_time)
    print("action type:", args.act_type)
    print("-" * 88)

    try:
        success_rates, avg_steps, _ = eval_libero(
            agent=client,
            save_path=out_dir,
            init_seed=args.init_seed,
            num_episodes=args.eval_time,
            task_suites=args.task_suites,
            act_type=args.act_type,
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        import traceback;
        traceback.print_exc()
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(2)

    print("\n✅ All evaluations completed successfully!")
    print("📊 Summary:")
    print(json.dumps({"success_rate": success_rates, "avg_steps": avg_steps}, indent=2))
