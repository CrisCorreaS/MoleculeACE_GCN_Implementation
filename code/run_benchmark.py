"""Train and evaluate one GCN_GRAPH dataset/fold task from a validated manifest.

Inputs are a manifest row, the corrected dataset CSV and its published YAML.
Outputs under results/<campaign>/<dataset>/fold_<k>/ are checkpoints, official
and best-validation predictions/metrics, task_record.json and status.json.
The completed full campaign contains 30 datasets and 150 such tasks.

Execution requires the historical Linux/CUDA environment inside a Slurm GPU
allocation. Use the same checkout for the executing script and --project-root;
input resolution and recorded source hashing are not interchangeable roots.
See docs/reproduction.md for preparing a new campaign and scheduling all tasks.

After generating reproduction_manifest.csv for a new campaign, one allocated
task can be run from that checkout's code directory:
    python run_benchmark.py --manifest ../jobs/reproduction_manifest.csv --campaign gcn_full_reproduction --task-id 0

The official state follows the upstream epoch-ceiling rule; secondary predictions
always use the best-validation state from the same training run. A completed
valid task is skipped. An unfinished retry removes its previous task artefacts
and temporary files before starting again; it does not resume optimiser state.
Completion is recorded only after the new outputs pass validation.
"""
import argparse
import datetime
import importlib.util
import inspect
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from MoleculeACE.benchmark.artifacts import (ACTIVITY_SCALE, PREDICTION_FIELDS, atomic_write_csv, atomic_write_json,
                                              canonical_json_sha256, fold_metrics, read_json, sha256_file)
from MoleculeACE.benchmark.campaign import (CampaignError, check_input_hashes, config_path, data_path,
                                             load_manifest, resolve_project_root, source_test_rows, task_dir,
                                             validate_task_outputs)
from MoleculeACE.benchmark.gpu_allocation import allocation_report

SLURM_VARIABLES = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_STEP_ID',
                   'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS', 'SLURM_JOB_ACCOUNT', 'SLURM_CPUS_PER_TASK',
                   'SLURM_MEM_PER_NODE', 'SLURM_JOB_NODELIST', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
                   'SLURM_GPUS_ON_NODE', 'SLURM_RESTART_COUNT', 'CUDA_VISIBLE_DEVICES')   # CUDA_VISIBLE_DEVICES is only read, never set
# After a node failure the dead node's lock can outlive it on NFS until its lease expires (typically ~90 s)
LOCK_WAIT_AFTER_REQUEUE_SECONDS = 600
PACKAGES = ('torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'rdkit-pypi', 'numpy', 'pandas',
            'scikit-learn', 'scipy', 'PyYAML', 'python-Levenshtein')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def package_versions():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover - Python < 3.8
        return {}
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def validate_training_source_root(root: Path) -> Path:
    """Require the entry point and scientific imports to belong to one checkout.

    Compare resolved paths, so equivalent relative paths and symlinks are valid.
    Loaded modules are checked using their actual files and package search paths;
    import specifications also cover lazy model/featurisation modules without
    importing Torch or requesting a GPU. Run this before creating task outputs.
    """
    root = Path(root).resolve()

    def require_path(actual, expected, label):
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise CampaignError(f'Training source-root mismatch for {label}: '
                                f'expected {expected}, found {actual}')

    require_path(__file__, root / 'code' / 'run_benchmark.py', 'run_benchmark')
    package = root / 'code' / 'MoleculeACE'
    sources = {}
    for source in sorted(package.rglob('*.py')):
        if '__pycache__' in source.parts:
            continue
        parts = source.relative_to(package.parent).with_suffix('').parts
        name = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
        sources[name] = source
    for name, source in sources.items():
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise CampaignError(f'Cannot verify training import {name}: {exc}') from exc
        require_path(None if spec is None else spec.origin, source, name + ' import origin')
    for name, module in list(sys.modules.items()):
        if name != 'MoleculeACE' and not name.startswith('MoleculeACE.'):
            continue
        if module is None or name not in sources:
            raise CampaignError(f'Unrecognised training module: {name}')
        require_path(getattr(module, '__file__', None), sources[name], name + ' loaded file')
        for location in getattr(module, '__path__', ()):
            require_path(location, sources[name].parent, name + ' package search path')
    # Direct imports can retain an old function even after its module is replaced.
    for name, value in list(globals().items()):
        if inspect.isfunction(value) and value.__module__.startswith('MoleculeACE.'):
            expected = sources.get(value.__module__)
            if expected is None:
                raise CampaignError(f'Unrecognised training function: {name}')
            require_path(inspect.getsourcefile(value), expected, name + ' function source')
    return root


def source_hashes(root: Path):
    """Hash Python sources beneath the selected root for campaign provenance.

    Parameters
    ----------
    root : pathlib.Path
        The executing training checkout's repository root.

    Returns
    -------
    dict
        Relative Python-source paths mapped to SHA-256 digests. Published data and
        configuration hashes are recorded separately in the manifest, not here.

    Notes
    -----
    This reads root/code. The training entry point validates the executing
    checkout and import paths before creating task outputs or recording hashes.
    """
    code = root / 'code'
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(code.rglob('*.py')) if '__pycache__' not in p.parts}


def git_state(root: Path):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'code', 'jobs'], cwd=str(root),
                                        text=True, stderr=subprocess.DEVNULL)
        return dict(commit=commit, dirty_code_or_jobs=bool(dirty.strip()))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty_code_or_jobs=None, note='git not available in this copy')


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    # Linux reports ru_maxrss in KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def restart_count():
    """ SLURM_RESTART_COUNT: how many times Slurm has requeued this job (absent or empty = first attempt) """
    value = os.environ.get('SLURM_RESTART_COUNT') or '0'
    if not value.isdigit():
        raise CampaignError(f"Invalid SLURM_RESTART_COUNT {value!r}")
    return int(value)


def acquire_lock(folder: Path, wait_seconds: float = 0):
    """ Advisory lock released by the kernel even after SIGKILL. Two copies of one task cannot run at once.
    wait_seconds > 0 (requeued jobs) polls until the lock of a dead attempt is released """
    handle = open(folder / '.lock', 'a+')
    try:
        import fcntl
    except ImportError:
        return handle  # Windows: only used by local functional tests, never for training
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise CampaignError(f"Task already running elsewhere (lock held): {folder}")
            time.sleep(10)


TASK_ARTIFACTS = ('predictions.csv', 'metrics.json', 'predictions_best_val.csv', 'metrics_best_val.json',
                  'best_model.pt', 'final_model.pt', 'task_record.json')


def clean_previous_attempt(folder: Path):
    """ Remove every artifact and temporary file (*.tmp, left by a write interrupted by a node failure) of an earlier,
    unfinished attempt, so that nothing of it can mix with the new one. Returns what was found, for the record """
    previous = None
    status_path = folder / 'status.json'
    if status_path.exists():
        try:
            old = read_json(status_path)
            previous = dict(status=old.get('status'),
                            slurm_job_id=old.get('slurm_job_id') or (old.get('slurm') or {}).get('SLURM_JOB_ID'),
                            restart_count=(old.get('attempt') or {}).get('restart_count'))
            # Old records only have job ID. New worker jobs can contain several independent manifest tasks/steps.
            for key in ('manifest_task_id', 'slurm_array_job_id', 'slurm_array_task_id', 'slurm_step_id'):
                if key in (old.get('attempt') or {}):
                    previous[key] = old['attempt'][key]
        except (OSError, ValueError) as exc:
            previous = dict(status='unreadable', error=repr(exc))
    removed, other_files = [], []
    for path in sorted(folder.iterdir()):
        if path.is_file() and (path.name in TASK_ARTIFACTS or path.name.endswith('.tmp')):
            path.unlink()
            removed.append(path.name)
        elif path.name not in ('.lock', 'status.json'):
            other_files.append(path.name)    # e.g. NFS .nfs* files; never read by the validator
    return dict(previous=previous, removed=removed, other_files=other_files)


def _prediction_rows(task, predictions, effective_hash):
    return [dict(id=f"{task['dataset']}:{csv_row}", csv_row=csv_row, smiles=smiles, y_true=y_true, y_pred=y_pred,
                 cliff_mol=cliff, fold=task['fold'], dataset=task['dataset'], campaign=task['campaign'],
                 config_sha256=task['config_sha256'], effective_config_sha256=effective_hash,
                 activity_scale=ACTIVITY_SCALE)
            for csv_row, smiles, y_true, y_pred, cliff in predictions]


def write_task_outputs(root, task, predictions, predictions_best_val, record):
    """ Write both prediction sets, their metrics and task_record.json, validate all, then mark the task completed.

    Official result (predictions.csv, metrics.json): the state upstream would have evaluated -- the best state after
    early stopping, the last-epoch state at the epoch limit (record['training']['official_state']).
    Secondary result (predictions_best_val.csv, metrics_best_val.json): the best validated state, always; flagged as
    not official. Both come from the same training run (no retraining).

    :param predictions / predictions_best_val: lists of (csv_row, smiles, y_true, y_pred, cliff_mol), test order
    :param record: dict with at least effective_config_sha256, training, timing, resources, provenance,
                   yaml_epochs_ignored (artifact hashes and metrics are added here)
    """
    folder = task_dir(root, task)
    official_state = record['training']['official_state']
    checkpoints = ['best_model.pt'] + (['final_model.pt'] if official_state == 'last_epoch' else [])
    for name in checkpoints:
        if not (folder / name).is_file():
            raise CampaignError(f"Missing task-local checkpoint {name}")
    outputs = {}
    for label, preds, prediction_file, metrics_file in (
            ('official', predictions, 'predictions.csv', 'metrics.json'),
            ('best_validation_secondary', predictions_best_val, 'predictions_best_val.csv', 'metrics_best_val.json')):
        rows = _prediction_rows(task, preds, record['effective_config_sha256'])
        metrics = fold_metrics([r['y_true'] for r in rows], [r['y_pred'] for r in rows],
                               [r['cliff_mol'] for r in rows])
        atomic_write_csv(folder / prediction_file, rows, PREDICTION_FIELDS)
        state = official_state if label == 'official' else 'best_validation'
        atomic_write_json(folder / metrics_file, dict(metrics, dataset=task['dataset'], fold=task['fold'],
                                                      campaign=task['campaign'], activity_scale=ACTIVITY_SCALE,
                                                      result=label, model_state=state))
        outputs[label] = metrics
    record = dict(record, task=task, metrics=outputs['official'],
                  metrics_best_val_secondary=outputs['best_validation_secondary'],
                  artifact_sha256={name: sha256_file(folder / name)
                                   for name in ['predictions.csv', 'metrics.json', 'predictions_best_val.csv',
                                                'metrics_best_val.json'] + checkpoints})
    atomic_write_json(folder / 'task_record.json', record)
    validate_task_outputs(root, task, require_completed=False)
    atomic_write_json(folder / 'status.json', dict(status='completed', task=task, finished=now(),
                                                   slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                                   attempt=record.get('attempt')))
    return outputs['official']


def train_task(root, task, folder):
    """Fit one internal fold and predict the unchanged external test partition.

    Parameters
    ----------
    root : pathlib.Path
        Selected campaign root; use the same checkout as the imported package.
    task : dict
        Validated manifest task with dataset, fold, seed and protocol fields.
    folder : pathlib.Path
        Existing task directory for best/final checkpoint writes.

    Returns
    -------
    tuple
        Official predictions, secondary best-validation predictions and a record
        of configuration, splits, training, timing and resources. Prediction tuples
        contain csv_row, smiles, y_true, y_pred and cliff_mol; y uses -log10(nM).

    Notes
    -----
    Only internal validation controls stopping. Published YAML epochs are discarded
    in favour of the retained 300-epoch model default for scientific stages.
    This function requires the pinned CUDA runtime and a valid Slurm GPU allocation.
    """
    root = validate_training_source_root(root)
    import numpy as np
    import torch
    from MoleculeACE.benchmark.const import Descriptors
    from MoleculeACE.benchmark.utils import Data, get_config, molecular_folds
    from MoleculeACE.models import GCN
    validate_training_source_root(root)

    allocation, problems = allocation_report(torch, check_process=False)
    if problems:
        raise RuntimeError('GPU allocation check failed before training: ' + '; '.join(problems))
    print('GPU allocation before training: ' + json.dumps(allocation))
    seed = task['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    timing = {}

    started = time.perf_counter()
    data = Data(str(data_path(task['dataset'])))
    test_rows = source_test_rows(task['dataset'])
    if [r[1] for r in test_rows] != data.smiles_test:
        raise CampaignError('Test-set order differs between Data and the source CSV')
    folds = molecular_folds(data.y_train, n_folds=task['n_folds'], seed=seed)
    split = folds[task['fold']]
    train_idx, val_idx = [int(i) for i in split['train_idx']], [int(i) for i in split['val_idx']]
    data.featurize_data(Descriptors.GRAPH)
    timing['load_and_featurize_seconds'] = time.perf_counter() - started

    # Upstream Experiments/benchmark.py: the YAML is loaded and 'epochs' popped; the rest goes to GCN(**hyper)
    published = get_config(str(config_path(task['dataset'])))
    hyperparameters = dict(published)
    yaml_epochs = hyperparameters.pop('epochs', None)
    model = GCN(**hyperparameters)
    # The model now holds a CUDA context: verify it before training any epoch.
    allocation, problems = allocation_report(torch)
    if problems:
        raise RuntimeError('GPU process allocation check failed: ' + '; '.join(problems))
    if task['stage'] != 'pilot' and model.epochs != task['max_epochs']:
        raise CampaignError(f"GCN constructor gives {model.epochs} epochs, manifest declares {task['max_epochs']}")
    signature = inspect.signature(GCN.__init__).parameters
    effective = {name: p.default for name, p in signature.items()
                 if p.default is not inspect.Parameter.empty}
    effective.update(hyperparameters)
    effective.update(epochs=task['max_epochs'], early_stopping_patience=task['patience'],
                     early_stopping_mode=task['early_stopping_mode'], train_batch_size=64, train_shuffle=False,
                     predict_batch_size=32, optimizer='Adam', loss='MSELoss', n_folds=task['n_folds'],
                     fold=task['fold'], split_seed=seed, model_init_seed='torch.manual_seed(42) in GCNmodel',
                     official_result='upstream state: best after early stopping, last weights at the epoch limit',
                     secondary_result='best validated state (predictions_best_val.csv), same training run')

    x_tr = [data.x_train[i] for i in train_idx]
    y_tr = [data.y_train[i] for i in train_idx]
    x_val = [data.x_train[i] for i in val_idx]
    y_val = [data.y_train[i] for i in val_idx]

    started = time.perf_counter()
    training = model.train(x_tr, y_tr, x_val, y_val, early_stopping_patience=task['patience'],
                           epochs=task['max_epochs'], early_stopping_mode=task['early_stopping_mode'],
                           checkpoint_path=str(folder / 'best_model.pt'),
                           final_checkpoint_path=str(folder / 'final_model.pt'), print_every_n=25)
    torch.cuda.synchronize()
    timing['training_seconds'] = time.perf_counter() - started
    timing['seconds_per_epoch'] = timing['training_seconds'] / training['epochs_run']

    def predict_test():
        values = [float(v) for v in model.predict(data.x_test).reshape(-1).tolist()]
        if len(values) != len(test_rows):
            raise CampaignError('Number of predictions differs from the test set')
        return values

    # The model ends training in its best validated state (secondary result, always written)
    started = time.perf_counter()
    y_best = predict_test()
    if training['official_state'] == 'last_epoch':
        # Upstream reached the epoch limit and evaluated the last weights: official result from final_model.pt
        if model.load_checkpoint(str(folder / 'final_model.pt')) != training['epochs_run']:
            raise CampaignError('final_model.pt is not the last-epoch state')
        y_official = predict_test()
        model.load_checkpoint(str(folder / 'best_model.pt'))
    else:
        y_official = y_best
    timing['prediction_seconds'] = time.perf_counter() - started

    device = torch.cuda.current_device()
    resources = dict(peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                     peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                     gpu_name=torch.cuda.get_device_name(device), nvidia_smi=allocation['nvidia_smi_allocated'],
                     gpu_allocation=allocation,
                     torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                     cudnn_deterministic=torch.backends.cudnn.deterministic,
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    predictions = [(csv_row, smiles, y, pred, cliff)
                   for (csv_row, smiles, y, cliff), pred in zip(test_rows, y_official)]
    predictions_best_val = [(csv_row, smiles, y, pred, cliff)
                            for (csv_row, smiles, y, cliff), pred in zip(test_rows, y_best)]
    record = dict(published_config=published, yaml_epochs_ignored=yaml_epochs, effective_config=effective,
                  effective_config_sha256=canonical_json_sha256(effective), seed=seed,
                  split=dict(n_train_total=len(data.y_train), n_fold_train=len(train_idx), n_val=len(val_idx),
                             n_test=len(test_rows), train_idx=train_idx, val_idx=val_idx),
                  training=training, timing=timing, resources=resources)
    return predictions, predictions_best_val, record


def run(root, manifest, campaign, task_id):
    """Validate a manifest task, lock its output directory and execute or skip it.

    Parameters
    ----------
    root : pathlib.Path
        Executing checkout and campaign-output root.
    manifest : pathlib.Path
        Manifest containing the requested task and its input hashes.
    campaign : str
        Campaign identity that must match the manifest.
    task_id : int
        Zero-based manifest row.

    Returns
    -------
    None
        Writes task evidence only for an unfinished attempt. Completed valid tasks
        are read and skipped; errors are reported through the existing status path.
    """
    root = validate_training_source_root(root)
    tasks = load_manifest(manifest)
    if tasks[0]['campaign'] != campaign:
        raise CampaignError(f"--campaign {campaign!r} does not match the manifest campaign {tasks[0]['campaign']!r}")
    if not 0 <= task_id < len(tasks):
        raise CampaignError(f"Task id {task_id} outside the manifest (0..{len(tasks) - 1})")
    task = tasks[task_id]
    check_input_hashes(task)
    if not os.environ.get('SLURM_JOB_ID'):
        raise CampaignError('Training runs only inside a Slurm allocation (never on the login node or a laptop)')

    folder = task_dir(root, task)
    folder.mkdir(parents=True, exist_ok=True)
    attempt = dict(restart_count=restart_count(), manifest_task_id=task_id,
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                   slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                   slurm_step_id=os.environ.get('SLURM_STEP_ID'), started=now())
    lock = acquire_lock(folder, LOCK_WAIT_AFTER_REQUEUE_SECONDS if attempt['restart_count'] else 0)
    try:
        status_path = folder / 'status.json'
        if status_path.exists() and read_json(status_path).get('status') == 'completed':
            # Never overwrite a completed task silently: it must still validate, otherwise stop and report
            validate_task_outputs(root, task)
            print(f"Already completed and valid, skipping: {folder} (restart_count={attempt['restart_count']})")
            return
        # A retry of an unfinished task starts clean: no artifact or temporary file of an earlier attempt survives
        attempt.update(clean_previous_attempt(folder))
        if attempt['restart_count'] or attempt['previous']:
            print(f"Retry of {folder}: {json.dumps(attempt)}")
        started = time.perf_counter()
        atomic_write_json(status_path, dict(status='running', task=task, started=now(), host=socket.gethostname(),
                                            pid=os.getpid(), slurm={k: os.environ.get(k) for k in SLURM_VARIABLES},
                                            attempt=attempt))
        predictions, predictions_best_val, record = train_task(root, task, folder)
        record['attempt'] = attempt
        record['timing']['total_seconds'] = time.perf_counter() - started
        record['resources']['peak_rss_bytes'] = peak_rss_bytes()
        record['provenance'] = dict(python=platform.python_version(), platform=platform.platform(),
                                    host=socket.gethostname(), versions=package_versions(),
                                    source_sha256=source_hashes(root), git=git_state(root),
                                    manifest=str(Path(manifest).resolve()), manifest_sha256=sha256_file(manifest),
                                    slurm={k: os.environ.get(k) for k in SLURM_VARIABLES}, finished=now())
        metrics = write_task_outputs(root, task, predictions, predictions_best_val, record)
        print(json.dumps(dict(task=f"{task['dataset']}/fold_{task['fold']}", **metrics,
                              official_state=record['training']['official_state'],
                              best_epoch=record['training']['best_epoch'],
                              epochs_run=record['training']['epochs_run'],
                              restart_count=attempt['restart_count'],
                              seconds=round(record['timing']['total_seconds'], 1))))
    except BaseException as exc:
        atomic_write_json(folder / 'status.json', dict(status='failed', task=task, error=repr(exc),
                                                       traceback=traceback.format_exc(), failed=now(),
                                                       slurm_job_id=os.environ.get('SLURM_JOB_ID'), attempt=attempt))
        raise
    finally:
        lock.close()


def main():
    """Parse the single-task CLI and report campaign errors with exit status two."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--campaign', required=True, help='Must equal the campaign column of the manifest')
    parser.add_argument('--task-id', required=True, type=int, help='Row of the manifest (independent of the array worker)')
    parser.add_argument('--project-root', default=None, help='Default: $PROJECT_ROOT, else derived from this file')
    args = parser.parse_args()
    root = resolve_project_root(args.project_root)
    try:
        run(root, args.manifest.resolve(), args.campaign, args.task_id)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
