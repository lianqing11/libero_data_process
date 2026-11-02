#!/usr/bin/env python3
"""LIBERO / LIBERO-Plus -> XVLA training HDF5 and metadata, in one file.

External dependencies: numpy, h5py, robosuite, LIBERO (LIBERO-Plus for variants),
and torch only for .pruned_init overrides. No imports from this repository.

Examples (run with a simulator-enabled Python environment):
  python prepare_libero_data.py generate --input /data/LIBERO/libero_spatial \
      --suite libero_spatial --modes base --output-root /data/libero_prepared
  python prepare_libero_data.py generate --input /data/LIBERO \
      --modes camera camera_plusplus action --variants-per-demo 5 \
      --libero-root /path/to/libero_plus --output-root /data/libero_prepared
  python prepare_libero_data.py generate --input /data/LIBERO \
      --modes language light background add noise --variants-per-demo 2 \
      --libero-root /path/to/libero_plus --output-root /data/libero_prepared
  python prepare_libero_data.py meta --root /data/libero_prepared

Schema: images [N+1,H,W,3], abs_action_6d [N+1,7] (xyz + axis-angle +
gripper), rewards/dones [N]. The training loader converts rotation to rot6d
and slices action chunks; this script deliberately saves complete trajectories.
Agentview is rotated 180 degrees; wrist is raw except explicit wrist flips.
Action perturbation is ONE delta step after settling, before recording/replay.
Its sampled numbers are controller inputs, not verified physical displacement.
Camera++ star retains standard discrete agentview configs and widens wrist FOV.
Base replays all actions and uses final done, matching the original converter;
Plus stops on done and checks env.check_success(), matching the Plus generator.

Outputs are never overwritten. Use a new directory for a new run. Failed
replays produce report entries but no training HDF5. Metadata validates schema;
it does not certify image/action temporal alignment or task success independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

MODES = ('base', 'language', 'light', 'background', 'add', 'noise', 'camera',
         'camera_plusplus', 'camera_plusplus_star', 'action', 'action_star')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10',
          'libero_90', 'libero_mix')
IMAGE_KEYS = ('observation/agentview_rgb', 'observation/eye_in_hand_rgb')
DEFAULT_OUTPUT = '/VLA-Data/scripts/lianqing/data/libero_prepared'


def load_dependencies(libero_root=None, simulation=False):
    global np, h5py, T, OffScreenRenderEnv, get_libero_path, BDDLUtils
    import numpy as np
    import h5py
    if simulation:
        if libero_root:
            sys.path.insert(0, str(Path(libero_root).resolve()))
        import robosuite.utils.transform_utils as T
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs import bddl_utils as BDDLUtils


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp',
                                     delete=False) as f:
        temp = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(temp, path)


def validate_hdf5(path):
    with h5py.File(path, 'r') as f:
        action = f['abs_action_6d']
        if action.ndim != 2 or action.shape[1] != 7 or len(action) < 2:
            raise ValueError(f'{path}: expected abs_action_6d [N+1,7], N>=1')
        if not np.isfinite(action[:]).all():
            raise ValueError(f'{path}: nonfinite action')
        for key in IMAGE_KEYS:
            image = f[key]
            if image.ndim != 4 or image.shape[0] != len(action) or image.shape[-1] != 3:
                raise ValueError(f'{path}: invalid {key} shape {image.shape}')
        for key in ('rewards', 'dones'):
            if f[key].shape != (len(action) - 1,):
                raise ValueError(f'{path}: invalid {key} length')
        if not str(f.attrs.get('instruction', '')).strip():
            raise ValueError(f'{path}: missing instruction')


def build_meta(root, output=None):
    root = Path(root).resolve()
    paths = sorted(root.rglob('*.hdf5'))
    if not paths:
        raise ValueError(f'No HDF5 files under {root}')
    for path in paths:
        validate_hdf5(path)
    output = Path(output) if output else root / 'meta.json'
    write_json(output, dict(observation_key=list(IMAGE_KEYS), dataset_name='libero',
                           language_instruction_key='instruction',
                           datalist=[str(p) for p in paths]))
    print(f'Metadata: {output} ({len(paths)} episodes)', flush=True)


def camera_configs():
    return [(h, v, s) for h in (0, 10, 20, 30, 40, 50, 60, 70,
                               285, 295, 305, 315, 325, 335, 345, 355)
            for v in (0, 15) for s in (1., 1.25, 1.5, 1.75, 2.)]


def encoded_bddl(path, camera=None, noise=0):
    if camera is None and noise == 0:
        return str(path)
    h, v, scale = camera or (0, 0, 1.)
    suffix = f'_view_{int(h)}_{int(v)}_{int(scale * 100)}_0_0_initstate_0'
    if noise:
        suffix += f'_noise_{noise}'
    return str(path.with_suffix('')) + suffix + '.bddl'


def resolve_task(source, data, suite_override):
    name = Path(str(data.attrs['bddl_file_name'])).name
    suite = suite_override
    if not suite:
        hints = str(source) + '/' + str(data.attrs['bddl_file_name'])
        suite = next((s for s in SUITES if s in hints), None)
    if not suite:
        raise ValueError(f'Cannot resolve suite for {source}; supply --suite')
    bddl = Path(get_libero_path('bddl_files')) / suite / name
    if not bddl.is_file():
        raise FileNotFoundError(bddl)
    return suite, bddl


def sample_variant(mode, original, suite, rng, level):
    """Sample a variant; categorical modes remain separate, with no overlays."""
    info = {'perturbation': mode}
    bddl, init_file, pre_action, wrist_flip, wrist_fov = original, None, None, 'none', None
    camera, noise = None, 0
    if mode in ('language', 'light', 'background', 'add'):
        kind = 'table' if mode == 'background' else mode
        patterns = [f'{original.stem}_{kind}_*.bddl']
        if kind == 'table':
            patterns.append(f'{original.stem}_tb_*.bddl')
        candidates = sorted({p for pat in patterns for p in original.parent.glob(pat)
                             if re.search(r'_(?:language|light|table|tb|add)_\d+$', p.stem)})
        if not candidates:
            raise ValueError(f'No {mode} BDDL variants for {original.stem}')
        bddl = candidates[int(rng.randint(len(candidates)))]
        info.update(variant_type=kind, bddl=str(bddl))
        if mode == 'add':
            candidate = Path(get_libero_path('init_states')) / 'libero_newobj' / suite / (bddl.stem + '.pruned_init')
            if candidate.exists():
                init_file = candidate
    if mode.startswith('camera'):
        configs = camera_configs()
        camera = configs[int(rng.randint(len(configs)))]
        info['camera_config'] = list(camera)
        if mode != 'camera':
            wrist_flip = str(rng.choice(['none', 'hflip', 'vflip', 'rotate180']))
            limits = (50., 100.) if mode.endswith('_star') else (60., 90.)
            wrist_fov = round(float(rng.uniform(*limits)), 1)
            info.update(wrist_flip=wrist_flip, wrist_fovy=wrist_fov)
    if mode == 'noise':
        noise = int(rng.choice([5, 10, 15]))
        info['noise_level'] = noise
    if mode in ('action', 'action_star'):
        ranges = ((.015, 5.), (.030, 10.), (.050, 15.)) if mode.endswith('_star') else ((.005, 1.), (.010, 3.), (.020, 6.))
        pos_range, rot_degrees = ranges[level - 1]
        pos = rng.uniform(-pos_range, pos_range, size=3)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis) + 1e-8
        rotation = axis * rng.uniform(-math.radians(rot_degrees), math.radians(rot_degrees))
        pre_action = np.concatenate([pos, rotation, [-1.]])
        info.update(level=level, pos_bias_m=pos.tolist(), rot_bias_aa=rotation.tolist(),
                    application='one_delta_step_after_settle',
                    units_note='controller delta inputs; physical displacement depends on scaling')
    return encoded_bddl(bddl, camera, noise), init_file, pre_action, wrist_flip, wrist_fov, info


def record_frame(env, obs, grip, wrist_flip):
    agent = np.flip(obs['agentview_image'], axis=(0, 1)).copy()
    wrist = obs['robot0_eye_in_hand_image']
    axes = {'none': (), 'hflip': (1,), 'vflip': (0,), 'rotate180': (0, 1)}[wrist_flip]
    wrist = np.flip(wrist, axis=axes).copy() if axes else wrist.copy()
    controller = env.env.robots[0].controller
    aa = T.quat2axisangle(T.mat2quat(controller.goal_ori))
    target = np.concatenate([controller.goal_pos, aa, [grip]])
    return agent, wrist, target


def replay(env, actions, mode, pre_action=None, wrist_flip='none'):
    # Original LIBERO uses openvla's dummy action (gripper=-1); Plus uses zeros.
    settle_action = np.zeros(7)
    if mode == 'base':
        settle_action[-1] = -1.
    for _ in range(10):
        obs, _, done, _ = env.step(settle_action.copy())
    if pre_action is not None:
        obs, _, done, _ = env.step(pre_action.tolist())
    frames = [record_frame(env, obs, -1., wrist_flip)]
    previous = None
    for action in actions:
        noop = np.linalg.norm(action[:-1]) < 1e-4
        if noop and (previous is None or action[-1] == previous[-1]):
            continue
        obs, _, done, _ = env.step(action.tolist())
        frames.append(record_frame(env, obs, float(action[-1]), wrist_flip))
        previous = action
        if done and mode != 'base':
            break
    success = bool(done) if mode == 'base' else bool(env.check_success())
    if not success or len(frames) < 2:
        return None
    return tuple(np.stack(items) for items in zip(*frames))


def save_episode(path, trajectory, instruction, task, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous data. Remove incomplete output on error.
    f = h5py.File(path, 'x')
    try:
        with f:
            agent, wrist, action = trajectory
            for key, value in zip((*IMAGE_KEYS, 'abs_action_6d'), (agent, wrist, action)):
                f.create_dataset(key, data=value)
            terminal = np.zeros(len(action) - 1, dtype=np.uint8)
            terminal[-1] = 1
            f.create_dataset('rewards', data=terminal)
            f.create_dataset('dones', data=terminal)
            f.attrs['instruction'] = instruction
            f.attrs['task_name'] = task
            f.attrs['perturbation_type'] = json.dumps(info)
        validate_hdf5(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def generate(args):
    source = Path(args.input).resolve()
    files = [source] if source.is_file() else sorted(source.rglob('*_demo.hdf5'))
    if not files:
        raise ValueError(f'No raw *_demo.hdf5 under {source}')
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Output directory is not empty: {root}; choose a new directory')
    root.mkdir(parents=True, exist_ok=True)
    report = {'arguments': vars(args), 'episodes': []}
    errors = 0
    try:
        for source_file in files:
            with h5py.File(source_file, 'r') as f:
                data = f['data']
                suite, original = resolve_task(source_file, data, args.suite)
                instruction = json.loads(data.attrs['problem_info'])['language_instruction']
                demos = sorted(data.keys(), key=lambda x: int(x.rsplit('_', 1)[-1]))
                for demo_id in demos[:args.max_demos]:
                    actions = data[demo_id]['actions'][:]
                    initial = data[demo_id]['states'][0]
                    for mode in args.modes:
                        count = 1 if mode == 'base' else args.variants_per_demo
                        for variant in range(count):
                            identity = f'{suite}/{original.stem}/{demo_id}/{mode}/{variant}/{args.seed}'
                            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'little')
                            rng = np.random.RandomState(seed)
                            out = root / mode / suite / f'{original.stem}_{demo_id}_v{variant}.hdf5'
                            entry = dict(source=str(source_file), demo=demo_id, mode=mode,
                                         seed=seed, output=str(out))
                            env = None
                            try:
                                bddl, init_file, pre, flip, fov, info = sample_variant(mode, original, suite, rng, args.action_level)
                                state = initial
                                if init_file:
                                    import torch
                                    states = torch.load(init_file, weights_only=False)
                                    idx = int(rng.randint(len(states))) if states.ndim > 1 else None
                                    state = states[idx] if idx is not None else states
                                    info.update(init_state_file=str(init_file), init_state_idx=idx)
                                text = BDDLUtils.get_problem_info(bddl)['language_instruction'] if mode == 'language' else instruction
                                env = OffScreenRenderEnv(bddl_file_name=bddl,
                                                         camera_heights=args.resolution,
                                                         camera_widths=args.resolution)
                                env.reset()
                                env.set_init_state(state)
                                if fov is not None:
                                    cam = env.sim.model.camera_name2id('robot0_eye_in_hand')
                                    env.sim.model.cam_fovy[cam] = fov
                                env.sim.forward()
                                traj = replay(env, actions, mode, pre, flip)
                                info.update(source=str(source_file), demo_id=demo_id, seed=seed)
                                if traj is not None:
                                    save_episode(out, traj, text, original.stem, info)
                                entry['status'] = 'saved' if traj is not None else 'replay_failed'
                            except Exception as exc:
                                errors += 1
                                entry.update(status='error', error=f'{type(exc).__name__}: {exc}')
                            finally:
                                if env is not None:
                                    env.close()
                            report['episodes'].append(entry)
                            print(json.dumps(entry), flush=True)
    finally:
        write_json(root / 'generation_report.json', report)
    if any(e['status'] == 'saved' for e in report['episodes']):
        build_meta(root)
    else:
        raise RuntimeError(f'No successful episodes; inspect {root / "generation_report.json"}')
    if errors:
        raise RuntimeError(f'{errors} generation errors; successful outputs retained, inspect report')


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='Replay raw LIBERO demonstrations with optional perturbations')
    gen.add_argument('--input', required=True, help='Raw task HDF5, suite directory, or dataset root')
    gen.add_argument('--suite', choices=SUITES, help='Override suite inference for a single suite')
    gen.add_argument('--libero-root', help='Directory containing the LIBERO / LIBERO-Plus Python package')
    gen.add_argument('--output-root', default=DEFAULT_OUTPUT, help='New/empty output directory')
    gen.add_argument('--modes', nargs='+', choices=MODES, default=['base'])
    gen.add_argument('--variants-per-demo', type=positive, default=5, help='Base always uses one variant')
    gen.add_argument('--max-demos', type=positive, help='Maximum demos per input HDF5')
    gen.add_argument('--action-level', type=int, choices=(1, 2, 3), default=2)
    gen.add_argument('--resolution', type=positive, default=224)
    gen.add_argument('--seed', type=int, default=42)
    meta = sub.add_parser('meta', help='Validate prepared HDF5 and build metadata without simulator imports')
    meta.add_argument('--root', required=True)
    meta.add_argument('--output', help='Defaults to <root>/meta.json')
    args = parser.parse_args()
    if args.command == 'generate':
        args.modes = list(dict.fromkeys(args.modes))
    load_dependencies(getattr(args, 'libero_root', None), args.command == 'generate')
    if args.command == 'generate':
        generate(args)
    else:
        build_meta(args.root, args.output)


if __name__ == '__main__':
    main()
