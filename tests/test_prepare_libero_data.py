"""Offline contract tests: no simulator package, network or GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'prepare', Path(__file__).resolve().parents[1] / 'scripts/prepare_libero_data.py')
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
prep.load_dependencies()
np = prep.np


class FakeEnv:
    def __init__(self, done_at=13, success=True):
        self.calls = []
        self.done_at = done_at
        self.success = success
        self.image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.controller = types.SimpleNamespace(goal_pos=np.zeros(3), goal_ori=np.eye(3))
        self.env = types.SimpleNamespace(robots=[types.SimpleNamespace(controller=self.controller)])

    def step(self, action):
        self.calls.append(np.array(action))
        self.controller.goal_pos = np.full(3, len(self.calls))
        return {'agentview_image': self.image, 'robot0_eye_in_hand_image': self.image}, 0, len(self.calls) >= self.done_at, {}

    def check_success(self):
        return self.success


class PreparationTests(unittest.TestCase):
    def setUp(self):
        transform = types.SimpleNamespace(mat2quat=lambda x: x, quat2axisangle=lambda x: np.zeros(3))
        self.transform_patch = patch.object(prep, 'T', transform, create=True)
        self.transform_patch.start()
        self.addCleanup(self.transform_patch.stop)
        self.actions = np.array([
            [0, 0, 0, 0, 0, 0, -1],  # Initial noop skipped.
            [.2, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],  # Same gripper noop skipped.
            [0, 0, 0, 0, 0, 0, 1],   # Gripper-only change retained.
            [.3, 0, 0, 0, 0, 0, 1],
        ])

    def test_base_settle_and_complete_replay(self):
        env = FakeEnv(done_at=11)
        traj = prep.replay(env, self.actions, 'base')
        self.assertEqual(len(env.calls), 13)  # Base continues after done.
        self.assertTrue(all(a[-1] == -1 for a in env.calls[:10]))
        self.assertEqual(traj[2].shape, (4, 7))
        np.testing.assert_array_equal(traj[2][:, -1], [-1, -1, 1, 1])

    def test_plus_single_shift_flips_and_stop(self):
        env = FakeEnv()
        shift = np.array([.01, .02, .03, 0, 0, 0, -1])
        traj = prep.replay(env, self.actions, 'action', shift, 'hflip')
        self.assertEqual(len(env.calls), 13)
        self.assertTrue(all(np.all(a == 0) for a in env.calls[:10]))
        np.testing.assert_array_equal(env.calls[10], shift)
        np.testing.assert_array_equal(traj[0][0], env.image[::-1, ::-1])
        np.testing.assert_array_equal(traj[1][0], env.image[:, ::-1])
        self.assertEqual(traj[2].shape, (3, 7))
        np.testing.assert_array_equal(traj[2][0, :3], [11, 11, 11])

    def test_unsuccessful_and_empty_replays_are_rejected(self):
        self.assertIsNone(prep.replay(FakeEnv(success=False), self.actions, 'camera'))
        self.assertIsNone(prep.replay(FakeEnv(), np.zeros((2, 7)), 'camera'))
        self.assertIsNone(prep.replay(FakeEnv(done_at=100), self.actions, 'base'))

    def test_variant_ranges_and_determinism(self):
        self.assertEqual(len(set(prep.camera_configs())), 160)
        for mode in ('base', 'camera', 'camera_plusplus', 'camera_plusplus_star', 'noise', 'action', 'action_star'):
            first = prep.sample_variant(mode, Path('/tmp/task.bddl'), 'libero_spatial', np.random.RandomState(42), 2)
            second = prep.sample_variant(mode, Path('/tmp/task.bddl'), 'libero_spatial', np.random.RandomState(42), 2)
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[-1], second[-1])
            if mode.startswith('action'):
                limit = .030 if mode.endswith('star') else .010
                self.assertTrue((np.abs(first[2][:3]) <= limit).all())
                self.assertEqual(first[2][-1], -1)
            if mode.startswith('camera_plusplus'):
                low, high = (50, 100) if mode.endswith('star') else (60, 90)
                self.assertTrue(low <= first[4] <= high)

    def test_save_validate_meta_and_exclusive_creation(self):
        traj = prep.replay(FakeEnv(), self.actions, 'base')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base/libero_spatial/demo.hdf5'
            prep.save_episode(path, traj, 'test instruction', 'task', {'seed': 42})
            prep.validate_hdf5(path)
            prep.build_meta(directory)
            meta = json.loads((Path(directory) / 'meta.json').read_text())
            self.assertEqual(meta['dataset_name'], 'libero')
            self.assertEqual(meta['datalist'], [str(path)])
            with prep.h5py.File(path, 'r') as f:
                np.testing.assert_array_equal(f['dones'][:], [0, 0, 1])
                self.assertEqual(json.loads(f.attrs['perturbation_type'])['seed'], 42)
            with self.assertRaises(FileExistsError):
                prep.save_episode(path, traj, 'changed', 'task', {})
            with prep.h5py.File(path, 'a') as f:
                f['abs_action_6d'][0, 0] = np.nan
            with self.assertRaises(ValueError):
                prep.build_meta(directory)

    def test_empty_metadata_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prep.build_meta(directory)


if __name__ == '__main__':
    unittest.main()
