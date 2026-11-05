"""Offline evaluation bookkeeping tests; no model or simulator required."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.run_support import (
    configure_libero, execute_recorded_run, summarize, validate_options,
    write_final_summary,
)


class EvaluationSupportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'config.json').write_text('{}')
        self.libero = self.root / 'LIBERO'
        package = self.libero / 'libero/libero'
        for name in ('bddl_files', 'init_files', 'assets'):
            (package / name).mkdir(parents=True)
        (package / '__init__.py').touch()
        self.args = SimpleNamespace(
            episodes=1, denoising_steps=10, max_tasks=1, action_chunk_size=None,
            num_ic_frames=None, ic_stride=1, runtime_perturbations=['none'],
            perturbation=None, sim='libero', act_type='abs',
            model_path=str(self.checkpoint), libero_root=str(self.libero),
        )
        self.environ = patch.dict(os.environ)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))

    def test_invalid_or_silently_ignored_combinations_rejected(self):
        validate_options(self.args)
        for update in ({'episodes': 0}, {'max_tasks': -1}, {'ic_stride': 0},
                       {'runtime_perturbations': ['action']},
                       {'runtime_perturbations': ['none', 'action']},
                       {'sim': 'libero_plus', 'runtime_perturbations': None, 'act_type': 'rel'},
                       {'sim': 'libero_plus', 'perturbation': 'action', 'runtime_perturbations': None}):
            values = vars(self.args) | update
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_options(SimpleNamespace(**values))

    def test_configuration_is_isolated_and_checks_assets(self):
        config = self.root / 'run/config'
        values = configure_libero(self.libero, config)
        self.assertEqual(os.environ['LIBERO_CONFIG_PATH'], str(config))
        self.assertEqual(json.loads((config / 'config_plus.yaml').read_text()), values)
        self.assertEqual(json.loads((config / 'config.yaml').read_text()), values)
        (self.libero / 'libero/libero/assets').rmdir()
        with self.assertRaisesRegex(ValueError, 'Missing assets'):
            configure_libero(self.libero, config)

    def test_completed_run_and_overwrite_guard(self):
        output = self.root / 'run'
        rows = {'spatial': [{'success': 1, 'steps': 3}]}
        execute_recorded_run(self.args, lambda: write_final_summary(output, rows, {}), output)
        record = json.loads((output / 'run_manifest.json').read_text())
        self.assertEqual(record['status'], 'complete')
        self.assertFalse(record['weights_hash_verified'])
        self.assertEqual(len(record['checkpoint_config_sha256']), 64)
        self.assertIn('evaluation/run_support.py', record['source_sha256'])
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            execute_recorded_run(self.args, lambda: None, output)

    def test_failed_run_records_error_and_has_no_final_summary(self):
        output = self.root / 'failed'
        def fail():
            raise RuntimeError('simulator failed')
        with self.assertRaisesRegex(RuntimeError, 'simulator failed'):
            execute_recorded_run(self.args, fail, output)
        record = json.loads((output / 'run_manifest.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertIn('simulator failed', record['error'])
        self.assertFalse((output / 'summary.json').exists())

    def test_macro_and_micro_use_different_denominators(self):
        result = summarize({'a': [{'success': 1, 'steps': 2}],
                            'b': [{'success': 0, 'steps': 800}] * 3})
        self.assertEqual(result['episodes'], 4)
        self.assertEqual(result['macro_success_rate'], 0.5)
        self.assertEqual(result['micro_success_rate'], 0.25)
        self.assertEqual(result['suites']['b']['avg_steps'], 800)
        with self.assertRaises(ValueError):
            summarize({'empty': []})
        with self.assertRaises(ValueError):
            summarize({'bad': [{'success': 2, 'steps': 1}]})


if __name__ == '__main__':
    unittest.main()
