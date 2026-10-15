"""Report saved campaign task states, optionally validating completed evidence.

Inputs are a manifest and the corresponding status/task records. Output is a
console listing and state counts; this script does not write project files.
NumPy, pandas and PyYAML are required by package import, but Torch is not.
From code/ in the complete submission:
    python task_status.py --manifest ../jobs/manifest_full.csv --campaign gcn_full_v1 --validate

A stored 'running' state is not proof that a process is still alive; on the
cluster compare it with scheduler accounting. Unreadable or malformed evidence
is listed as invalid without preventing later tasks from being inspected.
Exit codes are 0 for a readable listing, 1 if any task has invalid evidence, and
2 for a manifest/campaign error. A valid stored failed state is not malformed.
Unexpected programming errors propagate. See docs/reproduction.md for the
independent evidence validator.
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from MoleculeACE.benchmark.artifacts import read_json
from MoleculeACE.benchmark.campaign import (CampaignError, load_manifest, resolve_project_root, task_dir,
                                             validate_task_outputs)


class TaskEvidenceError(ValueError):
    """A task's saved JSON is unreadable or lacks the fields used in its listing."""


def _read_mapping(path, allow_missing=False):
    """Read a JSON object, converting only expected file/JSON errors to evidence errors."""
    try:
        record = read_json(path)
    except FileNotFoundError as exc:
        if allow_missing:
            return None
        raise TaskEvidenceError(f"{path.name}: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise TaskEvidenceError(f"{path.name}: {exc}") from exc
    if not isinstance(record, dict):
        raise TaskEvidenceError(f"{path.name}: expected a JSON object")
    return record


def _required(record, field, expected):
    """Return a required field of the expected JSON type; booleans are not numbers."""
    if field not in record:
        raise TaskEvidenceError(f"Missing field {field!r}")
    value = record[field]
    if not isinstance(value, expected) or isinstance(value, bool):
        raise TaskEvidenceError(f"Invalid type for field {field!r}")
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise TaskEvidenceError(f"Out-of-range field {field!r}") from exc
        if not finite:
            raise TaskEvidenceError(f"Non-finite field {field!r}")
    return value


def describe_task(root, task, validate=False):
    """Return one stored state and detail; raise TaskEvidenceError for invalid evidence."""
    folder = task_dir(root, task)
    path = folder / 'status.json'
    record = _read_mapping(path, allow_missing=True)
    if record is None:
        return 'pending', ''
    status = _required(record, 'status', str)
    if status not in {'pending', 'running', 'failed', 'completed'}:
        raise TaskEvidenceError(f"Unknown stored status {status!r}")
    detail = ''
    if status == 'failed':
        error = record.get('error', '')
        if not isinstance(error, str):
            raise TaskEvidenceError("Invalid type for field 'error'")
        detail = error[:160]
    if status == 'completed':
        record = _read_mapping(folder / 'task_record.json')
        metrics = _required(record, 'metrics', dict)
        training = _required(record, 'training', dict)
        timing = _required(record, 'timing', dict)
        rmse = _required(metrics, 'rmse', (int, float))
        cliff_value = _required(metrics, 'rmse_cliff', (int, float, type(None)))
        cliff = 'None' if cliff_value is None else f"{cliff_value:.3f}"
        best_epoch = _required(training, 'best_epoch', int)
        epochs_run = _required(training, 'epochs_run', int)
        stop_reason = _required(training, 'stop_reason', str)
        seconds = _required(timing, 'total_seconds', (int, float))
        detail = (f"rmse={rmse:.3f} rmse_cliff={cliff} best_epoch={best_epoch}/"
                  f"{epochs_run} ({stop_reason}) {seconds:.0f}s")
        if validate:
            try:
                validate_task_outputs(root, task)
            except CampaignError as exc:
                raise TaskEvidenceError(str(exc)) from exc
    return status, detail


def main():
    """Print every manifest task and return 1 if any saved evidence is invalid."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--project-root', default=None)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    try:
        root = resolve_project_root(args.project_root)
        tasks = load_manifest(args.manifest.resolve())
        if tasks[0]['campaign'] != args.campaign:
            raise CampaignError(f"--campaign does not match the manifest campaign {tasks[0]['campaign']!r}")
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts = Counter()
    for task in tasks:
        try:
            status, detail = describe_task(root, task, validate=args.validate)
        except TaskEvidenceError as exc:
            status, detail = 'invalid', str(exc)
        counts[status] += 1
        print(f"{task['task_id']:>4}  {task['dataset']:<16} fold_{task['fold']}  {status:<10} {detail}")
    print(', '.join(f"{k}={v}" for k, v in sorted(counts.items())), f"(total {len(tasks)})")
    return 1 if counts['invalid'] else 0


if __name__ == '__main__':
    sys.exit(main())
