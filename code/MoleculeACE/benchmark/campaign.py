"""Define campaign tasks, input paths and saved-evidence validation.

A task is one dataset/internal-fold model evaluated on the unchanged published
test set. The completed full campaign has 30 datasets and five folds per dataset.
Shared pilot/demo stage rules remain for historical compatibility and synthetic
tests; they do not require historical pilot/demo results. See docs/protocol.md.

Inputs are manifest rows, corrected CSV/YAML files and saved task artefacts.
Readers return validated task/metric records; write_manifest explicitly writes
a manifest. Validation reads checkpoint bytes for hashes without deserialising
models or rerunning inference. Module import does not run a campaign.

These helpers do not require Torch, but package initialisation imports NumPy,
pandas and PyYAML. CLI consumers include make_manifest.py, task_status.py and
aggregate_results.py. Data/config paths follow the imported package; the
project root selects the campaign-output tree.
"""

import csv
import math
import os
import re
from pathlib import Path

from MoleculeACE.benchmark.artifacts import (ACTIVITY_SCALE, PREDICTION_FIELDS, fold_metrics, read_json,
                                              read_predictions, sha256_file)
from MoleculeACE.benchmark.const import CONFIG_PATH, DATA_PATH, RANDOM_SEED, datasets
from MoleculeACE.models.training import EARLY_STOPPING_MODES, UPSTREAM_CONSECUTIVE, EarlyStopping

STRATEGY = 'GCN_GRAPH'
N_FOLDS = 5
PATIENCE = 10                       # Experiments/benchmark.py: cross_validate(..., early_stopping=10)
UPSTREAM_MAX_EPOCHS = 300           # GCN.__init__ default; the YAML 'epochs' key is popped upstream
UPSTREAM_EPOCHS_POLICY = 'upstream_gcn_default_300'
PILOT_EPOCHS_POLICY = 'pilot_cap'
STAGES = ('pilot', 'demo', 'full')
STAGE_N_DATASETS = {'pilot': 1, 'demo': 5, 'full': 30}
SUMMARY_DIR = '_summary'

MANIFEST_FIELDS = ['task_id', 'campaign', 'stage', 'scientific', 'strategy', 'dataset', 'fold', 'n_folds', 'seed',
                   'max_epochs', 'epochs_policy', 'patience', 'early_stopping_mode', 'deviation',
                   'data_sha256', 'config_sha256']
_INT_FIELDS = ('task_id', 'scientific', 'fold', 'n_folds', 'seed', 'max_epochs', 'patience')
_CAMPAIGN_FIELDS = ('campaign', 'stage', 'scientific', 'strategy', 'n_folds', 'seed', 'max_epochs', 'epochs_policy',
                    'patience', 'early_stopping_mode', 'deviation')
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')


class CampaignError(ValueError):
    """ Raised for any manifest, campaign or artifact inconsistency """


# ----------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------

def default_project_root() -> Path:
    # code/MoleculeACE/benchmark/campaign.py -> project root is three folders above 'benchmark'
    """Return the repository containing this imported campaign module."""
    return Path(__file__).resolve().parents[3]


def resolve_project_root(cli_value=None) -> Path:
    """Resolve and check a repository root for campaign outputs.

    Parameters
    ----------
    cli_value : str or path-like, optional
        Explicit root; otherwise use PROJECT_ROOT, then this module's repository.

    Returns
    -------
    pathlib.Path
        Absolute root containing code/MoleculeACE.

    Notes
    -----
    This helper permits an independent output root for post-hoc analysis.
    Input paths follow the imported package. Before creating training outputs,
    run_benchmark separately verifies that the entry point and scientific imports
    resolve to the selected checkout.
    """
    value = cli_value or os.environ.get('PROJECT_ROOT') or default_project_root()
    root = Path(value).resolve()
    if not (root / 'code' / 'MoleculeACE').is_dir():
        raise CampaignError(f"Not a project root (missing code/MoleculeACE): {root}")
    return root


def data_path(dataset: str) -> Path:
    """Return the imported package's corrected CSV path for a dataset identifier."""
    return Path(DATA_PATH) / f"{dataset}.csv"


def config_path(dataset: str) -> Path:
    """Return the imported package's published GCN_GRAPH YAML path."""
    return Path(CONFIG_PATH) / 'benchmark' / dataset / f"{STRATEGY}.yml"


def campaign_dir(root, campaign: str) -> Path:
    """Return results/campaign beneath the selected output root."""
    return Path(root) / 'results' / campaign


def task_dir(root, task) -> Path:
    """Return the dataset/fold directory for a task beneath its campaign root."""
    return campaign_dir(root, task['campaign']) / task['dataset'] / f"fold_{task['fold']}"


# ----------------------------------------------------------------------------------------------------------------
# Manifests
# ----------------------------------------------------------------------------------------------------------------

def build_manifest(campaign: str, stage: str, names, folds=None, max_epochs=None,
                   early_stopping_mode: str = UPSTREAM_CONSECUTIVE, seed: int = RANDOM_SEED):
    """ Build (and validate) the rows of a stage manifest. Task ids follow dataset order, then fold order """
    if stage not in STAGES:
        raise CampaignError(f"Unknown stage {stage!r}")
    pilot = stage == 'pilot'
    folds = list(range(N_FOLDS)) if folds is None else list(folds)
    max_epochs = UPSTREAM_MAX_EPOCHS if max_epochs is None else int(max_epochs)
    if pilot:
        deviation = (f"pilot_not_scientific:{len(folds)}_of_{N_FOLDS}_folds;"
                     f"max_epochs_{max_epochs}_instead_of_{UPSTREAM_MAX_EPOCHS}")
    elif stage == 'demo':
        deviation = f"dataset_subset:{len(names)}_of_{len(datasets)}"
    else:
        deviation = 'none'
    rows = []
    for dataset in names:
        for fold in folds:
            rows.append(dict(task_id=len(rows), campaign=campaign, stage=stage, scientific=0 if pilot else 1,
                             strategy=STRATEGY, dataset=dataset, fold=fold, n_folds=N_FOLDS, seed=seed,
                             max_epochs=max_epochs,
                             epochs_policy=PILOT_EPOCHS_POLICY if pilot else UPSTREAM_EPOCHS_POLICY,
                             patience=PATIENCE, early_stopping_mode=early_stopping_mode, deviation=deviation,
                             data_sha256=sha256_file(data_path(dataset)) if dataset in datasets else '',
                             config_sha256=sha256_file(config_path(dataset)) if dataset in datasets else ''))
    validate_manifest(rows)
    return rows


def validate_manifest(rows):
    """ Check schema, uniqueness, one campaign/protocol per manifest and the stage rules """
    if not rows:
        raise CampaignError('Empty manifest')
    first = rows[0]
    for i, row in enumerate(rows):
        if list(row.keys()) != MANIFEST_FIELDS:
            raise CampaignError(f"Row {i}: columns differ from {MANIFEST_FIELDS}")
        if row['task_id'] != i:
            raise CampaignError(f"Row {i}: task_id must equal the row index (Slurm array index)")
        for field in _CAMPAIGN_FIELDS:
            if row[field] != first[field]:
                raise CampaignError(f"Row {i}: mixed campaign protocol field {field!r}")
        if row['dataset'] not in datasets:
            raise CampaignError(f"Row {i}: unknown dataset {row['dataset']!r}")
        if not 0 <= row['fold'] < row['n_folds']:
            raise CampaignError(f"Row {i}: fold out of range")
        for field in ('data_sha256', 'config_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row[field]):
                raise CampaignError(f"Row {i}: invalid {field}")

    stage, campaign = first['stage'], first['campaign']
    if stage not in STAGES:
        raise CampaignError(f"Unknown stage {stage!r}")
    if not _SAFE_NAME.fullmatch(campaign) or not campaign.startswith(f"gcn_{stage}_"):
        raise CampaignError(f"Campaign must be a safe name starting with 'gcn_{stage}_', got {campaign!r}")
    if first['strategy'] != STRATEGY or first['n_folds'] != N_FOLDS or first['patience'] != PATIENCE:
        raise CampaignError('Strategy, number of folds and patience are fixed by the published protocol')
    if first['early_stopping_mode'] not in EARLY_STOPPING_MODES:
        raise CampaignError(f"Unknown early-stopping mode {first['early_stopping_mode']!r}")
    if not 0 <= first['seed'] < 2 ** 32:
        raise CampaignError('Seed out of range')

    keys = [(r['dataset'], r['fold']) for r in rows]
    if len(set(keys)) != len(keys):
        raise CampaignError('Duplicate (dataset, fold) tasks')
    names = list(dict.fromkeys(r['dataset'] for r in rows))
    if len(names) != STAGE_N_DATASETS[stage]:
        raise CampaignError(f"Stage {stage} needs {STAGE_N_DATASETS[stage]} datasets, got {len(names)}")

    if stage == 'pilot':
        if first['scientific'] != 0 or first['epochs_policy'] != PILOT_EPOCHS_POLICY:
            raise CampaignError('A pilot is never scientific and uses the pilot epoch cap')
        if not 1 <= first['max_epochs'] < UPSTREAM_MAX_EPOCHS:
            raise CampaignError('A pilot must cap epochs below the upstream 300')
        if not first['deviation'].startswith('pilot_not_scientific:'):
            raise CampaignError('Pilot limitations must be explicit in the deviation field')
    else:
        if first['scientific'] != 1 or first['epochs_policy'] != UPSTREAM_EPOCHS_POLICY \
                or first['max_epochs'] != UPSTREAM_MAX_EPOCHS:
            raise CampaignError('Scientific stages use the upstream epoch rule (max 300) without caps')
        for name in names:
            if sorted(r['fold'] for r in rows if r['dataset'] == name) != list(range(N_FOLDS)):
                raise CampaignError(f"Scientific stages need all {N_FOLDS} folds of {name}")
    return rows


def write_manifest(path, rows):
    from MoleculeACE.benchmark.artifacts import atomic_write_csv
    validate_manifest(rows)
    atomic_write_csv(path, rows, MANIFEST_FIELDS)


def load_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise CampaignError(f"Manifest not found: {path}")
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise CampaignError(f"Manifest columns differ from {MANIFEST_FIELDS}: {path}")
        rows = []
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise CampaignError(f"Ragged manifest row in {path}")
            row = {k: raw[k] for k in MANIFEST_FIELDS}
            try:
                for field in _INT_FIELDS:
                    row[field] = int(row[field])
            except ValueError as exc:
                raise CampaignError(f"Non-integer manifest value in {path}: {exc}") from exc
            rows.append(row)
    return validate_manifest(rows)


def check_input_hashes(task):
    """ The data and published configuration must be the files the manifest was generated from """
    if sha256_file(data_path(task['dataset'])) != task['data_sha256']:
        raise CampaignError(f"Data file changed since the manifest was made: {task['dataset']}")
    if sha256_file(config_path(task['dataset'])) != task['config_sha256']:
        raise CampaignError(f"Published configuration changed since the manifest was made: {task['dataset']}")


def source_test_rows(dataset: str):
    """Read published test rows in their original CSV order.

    Parameters
    ----------
    dataset : str
        Corrected benchmark dataset identifier.

    Returns
    -------
    list of tuple
        (csv_row, smiles, y, cliff_mol), where y = -log10(concentration in nM).
        CSV row indices are zero-based data-row indices, excluding the header.
    """
    with data_path(dataset).open('r', encoding='utf-8', newline='') as handle:
        return [(i, r['smiles'], float(r['y']), int(r['cliff_mol']))
                for i, r in enumerate(csv.DictReader(handle)) if r['split'] == 'test']


# ----------------------------------------------------------------------------------------------------------------
# Task outputs
# ----------------------------------------------------------------------------------------------------------------

def _artifact(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"Missing or empty artifact: {path}")
    return path


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


# Official result = state upstream would have evaluated; secondary = best validated state (flagged, same run)
OFFICIAL_FILES = ('predictions.csv', 'metrics.json')
SECONDARY_FILES = ('predictions_best_val.csv', 'metrics_best_val.json')


def _check_predictions(folder, task, predictions_name, metrics_name, effective_hash, where):
    """ Predictions aligned with the source test set; saved metrics equal the ones recomputed from them """
    rows = read_predictions(_artifact(folder, predictions_name))
    expected = source_test_rows(task['dataset'])
    if len(rows) != len(expected):
        raise CampaignError(f"{predictions_name} covers {len(rows)} of {len(expected)} test molecules: {where}")
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise CampaignError(f"Duplicate molecule ids in {predictions_name}: {where}")
    for row, (csv_row, smiles, y, cliff) in zip(rows, expected):
        if (row['id'] != f"{task['dataset']}:{csv_row}" or row['csv_row'] != csv_row or row['smiles'] != smiles
                or row['y_true'] != y or row['cliff_mol'] != cliff or row['fold'] != task['fold']
                or row['dataset'] != task['dataset'] or row['campaign'] != task['campaign']
                or row['config_sha256'] != task['config_sha256']
                or row['effective_config_sha256'] != effective_hash
                or row['activity_scale'] != ACTIVITY_SCALE):
            raise CampaignError(f"{predictions_name} row does not match the source test set/task ({row['id']}): "
                                f"{where}")
    recomputed = fold_metrics([r['y_true'] for r in rows], [r['y_pred'] for r in rows],
                              [r['cliff_mol'] for r in rows])
    saved = read_json(_artifact(folder, metrics_name))
    for key in ('n_test', 'n_cliff'):
        if saved.get(key) != recomputed[key]:
            raise CampaignError(f"Saved {key} in {metrics_name} differs from predictions: {where}")
    for key in ('rmse', 'rmse_cliff'):
        if not _close(saved.get(key), recomputed[key]):
            raise CampaignError(f"Saved {key} in {metrics_name} differs from the value recomputed from "
                                f"predictions: {where}")
    return rows, recomputed


def validate_task_outputs(root, task, require_completed: bool = True):
    """Check task identity, protocol history, artefact hashes and saved metrics.

    Parameters
    ----------
    root : path-like
        Repository root containing the task's results hierarchy.
    task : dict
        Validated manifest task.
    require_completed : bool
        Require a matching completed status before checking the other artefacts.

    Returns
    -------
    dict
        Official and secondary metrics, state selection, timing and provenance.

    Raises
    ------
    CampaignError
        For missing, malformed or inconsistent evidence.

    Notes
    -----
    Prediction rows must align with the published test set. Metrics are recalculated
    from saved predictions; checkpoint files are hashed but never deserialised.
    This is evidence validation, not a repeat of model inference or training.
    """
    folder = task_dir(root, task)
    where = f"{task['campaign']}/{task['dataset']}/fold_{task['fold']}"
    if not folder.is_dir():
        raise CampaignError(f"Missing task folder: {where}")
    try:
        if require_completed:
            status = read_json(_artifact(folder, 'status.json'))
            if status.get('status') != 'completed' or status.get('task') != task:
                raise CampaignError(f"Task not completed or status identity mismatch: {where}")
        record = read_json(_artifact(folder, 'task_record.json'))
        if record.get('task') != task:
            raise CampaignError(f"task_record.json belongs to another task: {where}")

        training = record['training']
        official_state = _check_training(training, task, where)
        hashes = record['artifact_sha256']
        names = list(OFFICIAL_FILES + SECONDARY_FILES) + ['best_model.pt']
        if official_state == 'last_epoch':
            names.append('final_model.pt')
        elif (folder / 'final_model.pt').exists() or 'final_model.pt' in hashes:
            raise CampaignError(f"final_model.pt present although the official state is the best one: {where}")
        if sorted(hashes) != sorted(names):
            raise CampaignError(f"Recorded artifacts differ from the expected set {sorted(names)}: {where}")
        for name in names:
            if sha256_file(_artifact(folder, name)) != hashes[name]:
                raise CampaignError(f"Hash mismatch (corrupted or replaced) for {name}: {where}")

        effective_hash = record['effective_config_sha256']
        official_rows, official = _check_predictions(folder, task, OFFICIAL_FILES[0], OFFICIAL_FILES[1],
                                                      effective_hash, where)
        best_rows, best = _check_predictions(folder, task, SECONDARY_FILES[0], SECONDARY_FILES[1],
                                             effective_hash, where)
        if official_state == 'best_validation' and \
                [r['y_pred'] for r in official_rows] != [r['y_pred'] for r in best_rows]:
            raise CampaignError(f"Official and best-validation predictions must be identical here: {where}")
        return dict(task=task, metrics=official, metrics_best_val=best, official_state=official_state,
                    best_epoch=training['best_epoch'], epochs_run=training['epochs_run'],
                    stop_reason=training['stop_reason'],
                    upstream_would_restore_best=training['upstream_would_restore_best'],
                    seconds=record['timing']['total_seconds'], resources=record['resources'],
                    provenance=record['provenance'], yaml_epochs_ignored=record['yaml_epochs_ignored'])
    except CampaignError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise CampaignError(f"Malformed or corrupted artifacts in {where}: {exc!r}") from exc


def _check_training(training, task, where):
    """ Replay the declared early-stopping rule on the saved history; return the official state it implies """
    history = training['history']
    epochs_run = training['epochs_run']
    if epochs_run != len(history) or not 1 <= epochs_run <= task['max_epochs'] \
            or training['epochs_max'] != task['max_epochs']:
        raise CampaignError(f"Epoch counts inconsistent with the manifest: {where}")
    stopper = EarlyStopping(task['patience'], task['early_stopping_mode'])
    stopped_at = None
    for entry in history:
        _, stop = stopper.update(entry['val_loss'], entry['epoch'])
        if stop:
            stopped_at = entry['epoch']
            break
    reason = 'early_stopping' if stopped_at is not None else 'epoch_limit'
    if stopped_at not in (None, epochs_run) or (stopped_at is None and epochs_run != task['max_epochs']):
        raise CampaignError(f"Training length does not follow the declared early-stopping rule: {where}")
    # Upstream checked the patience at the start of the next epoch: no restore if it ran out at the last epoch
    upstream_restores = stopped_at is not None and stopped_at < task['max_epochs']
    official = 'last_epoch' if (not upstream_restores and stopper.best_epoch != epochs_run) else 'best_validation'
    if training['stop_reason'] != reason or training['best_epoch'] != stopper.best_epoch \
            or training['early_stopping']['mode'] != task['early_stopping_mode'] \
            or training['restored_best'] is not True \
            or training['upstream_would_restore_best'] is not upstream_restores \
            or training['official_state'] != official:
        raise CampaignError(f"Best epoch / stop reason / mode / official state inconsistent with the history: "
                            f"{where}")
    return official
