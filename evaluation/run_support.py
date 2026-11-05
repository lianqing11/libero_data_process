"""Portable run setup and result accounting; imports only the standard library."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUNTIME_TYPES = {'camera_star', 'camera_plusplus', 'camera_plusplus_star', 'action', 'action_star'}


def validate_options(args):
    for name in ('episodes', 'denoising_steps'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('max_tasks', 'action_chunk_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f'{name} must be positive')
    if args.num_ic_frames is not None and args.num_ic_frames < 0:
        raise ValueError('num_ic_frames must be nonnegative')
    if args.ic_stride < 1:
        raise ValueError('ic_stride must be positive')
    values = args.runtime_perturbations or []
    if len(values) != len(set(values)):
        raise ValueError('Duplicate runtime perturbations are not allowed')
    suites = getattr(args, 'task_suites', None) or []
    if len(suites) != len(set(suites)):
        raise ValueError('Duplicate task suites are not allowed')
    if any(x not in RUNTIME_TYPES | {'all', 'none'} for x in values):
        raise ValueError(f'Unknown runtime perturbation: {values}')
    if ('none' in values or 'all' in values) and len(values) != 1:
        raise ValueError('Use all/none alone')
    has_runtime = bool(args.perturbation or (values and values != ['none'])
                       or (args.sim == 'libero_plus' and args.runtime_perturbations is None))
    if args.sim == 'libero' and has_runtime:
        raise ValueError('Runtime perturbations require --sim libero_plus; standard client ignores them')
    if has_runtime and args.act_type != 'abs':
        raise ValueError('Runtime evaluation requires --act_type abs')
    if args.perturbation and args.runtime_perturbations != ['none']:
        raise ValueError('For a single --perturbation pass, also specify --runtime_perturbations none')
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.libero_root = str(Path(args.libero_root).expanduser().resolve())
    if not Path(args.model_path).is_dir():
        raise ValueError('model_path must be an existing local checkpoint directory')


def configure_libero(root, config_dir):
    root = Path(root).expanduser().resolve()
    package = root / 'libero/libero'
    if not (package / '__init__.py').is_file():
        raise ValueError(f'Invalid LIBERO checkout: {root}')
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    values = {'benchmark_root': str(package), 'bddl_files': str(package / 'bddl_files'),
              'init_states': str(package / 'init_files'),
              'datasets': str(root / 'libero/datasets'), 'assets': str(package / 'assets')}
    for key in ('bddl_files', 'init_states', 'assets'):
        if not Path(values[key]).is_dir():
            raise ValueError(f'Missing {key}: {values[key]}')
    # JSON is valid YAML. Both upstream names are supported in an isolated dir.
    for name in ('config.yaml', 'config_plus.yaml'):
        (config_dir / name).write_text(json.dumps(values, indent=2) + '\n')
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    sys.path.insert(0, str(root))
    return values


def _git_revision(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def execute_recorded_run(args, callback, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for folder in ('evaluation', 'models'):
        for path in sorted((repo / folder).rglob('*.py')):
            source_hashes[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = Path(args.model_path)
    config = checkpoint / 'config.json'
    record = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'arguments': vars(args), 'code_commit': _git_revision(repo),
              'source_sha256': source_hashes,
              'libero_commit': _git_revision(args.libero_root),
              'checkpoint_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None,
              'checkpoint_files': [{'name': p.name, 'bytes': p.stat().st_size}
                                   for p in sorted(checkpoint.iterdir()) if p.is_file()],
              'weights_hash_verified': False}
    record_path = output / 'run_manifest.json'
    try:
        record['libero_paths'] = configure_libero(args.libero_root, output / '.libero_config')
        # Evaluation consumes prepared local assets and never starts HF transfers.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        callback()
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + '\n')


def summarize(raw):
    suites = {}
    for name, rows in raw.items():
        if not rows:
            raise ValueError(f'No completed episodes for {name}')
        successes = sum(float(r['success']) for r in rows)
        if any(float(r['success']) not in (0., 1.) or int(r['steps']) < 0 for r in rows):
            raise ValueError(f'Invalid episode metrics for {name}')
        suites[name] = {'episodes': len(rows), 'successes': int(successes),
                        'success_rate': successes / len(rows),
                        'avg_steps': sum(r['steps'] for r in rows) / len(rows)}
    total = sum(s['episodes'] for s in suites.values())
    return {'suites': suites, 'episodes': total,
            'macro_success_rate': sum(s['success_rate'] for s in suites.values()) / len(suites) if suites else None,
            'micro_success_rate': sum(s['successes'] for s in suites.values()) / total if total else None}


def write_final_summary(output, dataset_raw, runtime_raw):
    payload = {'dataset': summarize(dataset_raw),
               'runtime': {name: summarize(raw) for name, raw in runtime_raw.items()}}
    (Path(output) / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
