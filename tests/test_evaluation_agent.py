"""Policy history and chunk tests with a tiny CPU model, no simulator/weights."""
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

HAS_MODEL_DEPS = all(importlib.util.find_spec(name) for name in ('torch', 'transformers'))


@unittest.skipUnless(HAS_MODEL_DEPS, 'optional evaluation dependencies not installed')
class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from evaluation.scripts.evaluate_libero import LocalVLAAgent
        cls.torch = torch
        cls.agent_type = LocalVLAAgent

    def make_agent(self, **kwargs):
        return self.agent_type(
            self.torch.nn.Linear(1, 1), None, self.torch.device('cpu'),
            client_module=SimpleNamespace(LiberoAbsActionProcessor=lambda: None,
                                          _flip_agentview=lambda x: x), **kwargs)

    def test_ic_stride_and_partial_history(self):
        agent = self.make_agent(num_ic_frames=4, ic_stride=2)
        agent.episode_history = list(range(6))
        self.assertEqual(agent._select_ic_entries(3), [1, 3, 5])
        agent.episode_history = [0, 1]
        self.assertEqual(agent._select_ic_entries(3), [1])
        agent.ic_demo_mode = 'early'
        self.assertEqual(agent._select_ic_entries(3), [0])

    def test_reset_prevents_cross_episode_context(self):
        agent = self.make_agent(num_ic_frames=3)
        agent.episode_history.append({'old_episode': True})
        agent.action_plan.append(np.zeros(7))
        agent.proprio = np.ones(20)
        agent.gripper_trace.append(1.0)
        agent.reset()
        self.assertEqual(agent.episode_history, [])
        self.assertEqual(len(agent.action_plan), 0)
        self.assertIsNone(agent.proprio)
        self.assertEqual(agent.gripper_trace, [])

    def test_chunk_execution_and_gripper_threshold(self):
        from unittest.mock import Mock
        agent = self.make_agent(action_chunk_size=2)
        actions = np.array([[.1, .2, .3, 0, 0, 0, .5],
                            [.4, .5, .6, 0, 0, 0, .6],
                            [.7, .8, .9, 0, 0, 0, .2]])
        agent._run_xvla_inference = Mock(return_value=actions)
        first = agent.step({}, 'task')
        second = agent.step({}, 'task')
        self.assertEqual(agent._run_xvla_inference.call_count, 1)
        np.testing.assert_allclose(first[:3], actions[0, :3])
        np.testing.assert_allclose(second[:3], actions[1, :3])
        self.assertEqual(first[-1], -1)
        self.assertEqual(second[-1], 1)
        agent.step({}, 'task')
        self.assertEqual(agent._run_xvla_inference.call_count, 2)


if __name__ == '__main__':
    unittest.main()
