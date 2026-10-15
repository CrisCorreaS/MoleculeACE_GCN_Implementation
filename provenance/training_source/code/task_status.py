"""Show the state of every task of a campaign (pending / running / completed / failed). Read-only; no torch needed.

Run from code/:
    python task_status.py --manifest ../jobs/manifest_pilot.csv --campaign gcn_pilot_v1 [--validate]

'running' only means that status.json says so: a task killed by Slurm (time limit, OOM) stays 'running'.
Cross-check with sacct. --validate also checks the artifacts of completed tasks, as the aggregator does.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from MoleculeACE.benchmark.artifacts import read_json
from MoleculeACE.benchmark.campaign import (CampaignError, load_manifest, resolve_project_root, task_dir,
                                             validate_task_outputs)


def main():
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
        sys.exit(2)

    counts = Counter()
    for task in tasks:
        path = task_dir(root, task) / 'status.json'
        detail = ''
        try:
            status = read_json(path).get('status', 'unknown') if path.exists() else 'pending'
        except ValueError:
            status = 'unreadable'
        if status == 'failed':
            detail = read_json(path).get('error', '')[:160]
        if status == 'completed':
            record = task_dir(root, task) / 'task_record.json'
            if record.exists():
                r = read_json(record)
                m, t = r['metrics'], r['training']
                cliff = 'None' if m['rmse_cliff'] is None else f"{m['rmse_cliff']:.3f}"
                detail = (f"rmse={m['rmse']:.3f} rmse_cliff={cliff} best_epoch={t['best_epoch']}/"
                          f"{t['epochs_run']} ({t['stop_reason']}) {r['timing']['total_seconds']:.0f}s")
            if args.validate:
                try:
                    validate_task_outputs(root, task)
                except CampaignError as exc:
                    status, detail = 'invalid', str(exc)
        counts[status] += 1
        print(f"{task['task_id']:>4}  {task['dataset']:<16} fold_{task['fold']}  {status:<10} {detail}")
    print(', '.join(f"{k}={v}" for k, v in sorted(counts.items())), f"(total {len(tasks)})")


if __name__ == '__main__':
    main()
