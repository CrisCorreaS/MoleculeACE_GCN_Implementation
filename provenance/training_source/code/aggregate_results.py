"""Validate one campaign against its manifest and aggregate RMSE / RMSE_cliff per dataset.

Run from code/:
    python aggregate_results.py --manifest ../jobs/manifest_demo.csv --campaign gcn_demo_v1

Fails (exit 2, listing every problem) if the manifest does not exist or belongs to another campaign, if a task is
missing, failed or unexpected, if a folder holds another task's artifacts, if an artifact is corrupted (hash),
if predictions are missing, incomplete, duplicated or misaligned with the source test set, or if a metric is
non-finite or differs from the value recomputed from predictions.

Official metrics follow the upstream rule (best state after early stopping, last-epoch weights when the epoch limit is
reached); columns *_best_val_secondary hold the best-validated-state metrics of the same run (secondary, flagged).
Per dataset, scores are the arithmetic mean over folds of per-fold RMSE (upstream protocol), not the RMSE of
averaged predictions. The SD over folds describes internal folds evaluated on the same test set; folds are not
independent seeds. Works identically for 1, 5 or 30 datasets. Outputs go to results/<campaign>/_summary/.
"""
import argparse
import datetime
import math
import statistics
import sys
from pathlib import Path

from MoleculeACE.benchmark.artifacts import atomic_write_csv, atomic_write_json, sha256_file
from MoleculeACE.benchmark.campaign import (SUMMARY_DIR, CampaignError, campaign_dir, check_input_hashes,
                                             load_manifest, resolve_project_root, validate_task_outputs)

PER_FOLD_FIELDS = ['campaign', 'stage', 'scientific', 'dataset', 'fold', 'n_test', 'n_cliff', 'rmse', 'rmse_cliff',
                   'delta_cliff_minus_all', 'official_state', 'rmse_best_val_secondary',
                   'rmse_cliff_best_val_secondary', 'best_epoch', 'epochs_run', 'stop_reason',
                   'upstream_would_restore_best',
                   'yaml_epochs_ignored', 'seconds', 'peak_vram_allocated_bytes', 'peak_rss_bytes']
PER_DATASET_FIELDS = ['campaign', 'stage', 'scientific', 'dataset', 'n_folds', 'n_test', 'n_cliff',
                      'rmse_mean', 'rmse_sd', 'rmse_cliff_mean', 'rmse_cliff_sd', 'delta_mean', 'delta_sd',
                      'folds_official_last_epoch', 'rmse_best_val_secondary_mean',
                      'rmse_cliff_best_val_secondary_mean', 'delta_best_val_secondary_mean',
                      'best_epoch_mean', 'epochs_run_mean', 'folds_at_epoch_limit', 'seconds_total',
                      'peak_vram_allocated_bytes_max', 'peak_rss_bytes_max']


def _mean_sd(values):
    if any(v is None for v in values):
        return None, None
    mean = statistics.mean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else None)


def collect(root, manifest, campaign):
    tasks = load_manifest(manifest)
    if tasks[0]['campaign'] != campaign:
        raise CampaignError(f"--campaign {campaign!r} does not match the manifest campaign {tasks[0]['campaign']!r}")
    folder = campaign_dir(root, campaign)
    if not folder.is_dir():
        raise CampaignError(f"No results for campaign {campaign}: {folder}")

    errors = []
    expected = {(t['dataset'], f"fold_{t['fold']}") for t in tasks}
    names = {t['dataset'] for t in tasks}
    for entry in sorted(folder.iterdir()):
        if entry.name.startswith('_') or not entry.is_dir():   # _summary/, _analysis/: our own outputs
            continue
        if entry.name not in names:
            errors.append(f"Unexpected dataset folder (not in the manifest): {entry}")
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (entry.name, sub.name) not in expected:
                errors.append(f"Unexpected task folder (not in the manifest): {sub}")

    results = []
    for task in tasks:
        try:
            check_input_hashes(task)
            results.append(validate_task_outputs(root, task))
        except CampaignError as exc:
            errors.append(str(exc))

    signatures = {(repr(sorted(r['provenance']['source_sha256'].items())),
                   repr(sorted(r['provenance']['versions'].items())), r['provenance']['python']) for r in results}
    if len(signatures) > 1:
        errors.append('Tasks of this campaign ran with different code, package versions or Python; '
                      'rerun them in a new campaign instead of mixing')
    if errors:
        raise CampaignError(f"{len(errors)} problem(s) in campaign {campaign}:\n  - " + '\n  - '.join(errors))
    return tasks, results


def summarise(tasks, results):
    per_fold = []
    for r in results:
        t, m = r['task'], r['metrics']
        for key in ('rmse', 'rmse_cliff'):
            if m[key] is not None and not math.isfinite(m[key]):
                raise CampaignError(f"Non-finite {key} in {t['dataset']}/fold_{t['fold']}")
        per_fold.append(dict(
            campaign=t['campaign'], stage=t['stage'], scientific=t['scientific'], dataset=t['dataset'],
            fold=t['fold'], n_test=m['n_test'], n_cliff=m['n_cliff'], rmse=m['rmse'], rmse_cliff=m['rmse_cliff'],
            delta_cliff_minus_all=None if m['rmse_cliff'] is None else m['rmse_cliff'] - m['rmse'],
            official_state=r['official_state'], rmse_best_val_secondary=r['metrics_best_val']['rmse'],
            rmse_cliff_best_val_secondary=r['metrics_best_val']['rmse_cliff'],
            best_epoch=r['best_epoch'], epochs_run=r['epochs_run'], stop_reason=r['stop_reason'],
            upstream_would_restore_best=r['upstream_would_restore_best'],
            yaml_epochs_ignored=r['yaml_epochs_ignored'], seconds=r['seconds'],
            peak_vram_allocated_bytes=r['resources'].get('peak_vram_allocated_bytes'),
            peak_rss_bytes=r['resources'].get('peak_rss_bytes')))

    per_dataset = []
    for name in dict.fromkeys(t['dataset'] for t in tasks):
        rows = [f for f in per_fold if f['dataset'] == name]
        rmse_mean, rmse_sd = _mean_sd([f['rmse'] for f in rows])
        cliff_mean, cliff_sd = _mean_sd([f['rmse_cliff'] for f in rows])
        delta_mean, delta_sd = _mean_sd([f['delta_cliff_minus_all'] for f in rows])
        best_mean = _mean_sd([f['rmse_best_val_secondary'] for f in rows])[0]
        best_cliff_mean = _mean_sd([f['rmse_cliff_best_val_secondary'] for f in rows])[0]
        vram = [f['peak_vram_allocated_bytes'] for f in rows if f['peak_vram_allocated_bytes'] is not None]
        rss = [f['peak_rss_bytes'] for f in rows if f['peak_rss_bytes'] is not None]
        per_dataset.append(dict(
            campaign=rows[0]['campaign'], stage=rows[0]['stage'], scientific=rows[0]['scientific'], dataset=name,
            n_folds=len(rows), n_test=rows[0]['n_test'], n_cliff=rows[0]['n_cliff'],
            rmse_mean=rmse_mean, rmse_sd=rmse_sd, rmse_cliff_mean=cliff_mean, rmse_cliff_sd=cliff_sd,
            delta_mean=delta_mean, delta_sd=delta_sd,
            best_epoch_mean=statistics.mean(f['best_epoch'] for f in rows),
            epochs_run_mean=statistics.mean(f['epochs_run'] for f in rows),
            folds_official_last_epoch=sum(f['official_state'] == 'last_epoch' for f in rows),
            rmse_best_val_secondary_mean=best_mean, rmse_cliff_best_val_secondary_mean=best_cliff_mean,
            delta_best_val_secondary_mean=(None if best_cliff_mean is None
                                           else _mean_sd([f['rmse_cliff_best_val_secondary']
                                                          - f['rmse_best_val_secondary'] for f in rows])[0]),
            folds_at_epoch_limit=sum(f['stop_reason'] == 'epoch_limit' for f in rows),
            seconds_total=math.fsum(f['seconds'] for f in rows),
            peak_vram_allocated_bytes_max=max(vram) if vram else None, peak_rss_bytes_max=max(rss) if rss else None))
    return per_fold, per_dataset


def aggregate(root, manifest, campaign):
    tasks, results = collect(root, manifest, campaign)
    per_fold, per_dataset = summarise(tasks, results)
    out = campaign_dir(root, campaign) / SUMMARY_DIR
    atomic_write_csv(out / 'per_fold.csv', per_fold, PER_FOLD_FIELDS)
    atomic_write_csv(out / 'per_dataset.csv', per_dataset, PER_DATASET_FIELDS)
    first = tasks[0]
    atomic_write_json(out / 'summary.json', dict(
        campaign=campaign, stage=first['stage'], scientific=bool(first['scientific']),
        deviation=first['deviation'], early_stopping_mode=first['early_stopping_mode'],
        max_epochs=first['max_epochs'], n_datasets=len(per_dataset), n_tasks=len(per_fold),
        manifest=str(Path(manifest).resolve()), manifest_sha256=sha256_file(manifest),
        per_fold_sha256=sha256_file(out / 'per_fold.csv'), per_dataset_sha256=sha256_file(out / 'per_dataset.csv'),
        aggregation='per dataset: arithmetic mean over folds of per-fold RMSE and RMSE_cliff on the same test set',
        official_result='upstream rule: best state after early stopping, last-epoch weights at the epoch limit',
        secondary_result='*_best_val_secondary: best validated state of the same run; NOT the official metric',
        uncertainty='SD over internal folds (ddof=1); folds share the test set and are not independent seeds',
        created=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return per_fold, per_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--project-root', default=None)
    args = parser.parse_args()
    try:
        root = resolve_project_root(args.project_root)
        per_fold, per_dataset = aggregate(root, args.manifest.resolve(), args.campaign)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    for row in per_dataset:
        cliff = 'None' if row['rmse_cliff_mean'] is None else f"{row['rmse_cliff_mean']:.3f}"
        print(f"{row['dataset']:<16} folds={row['n_folds']} RMSE={row['rmse_mean']:.3f} RMSE_cliff={cliff}")
    print(f"Validated {len(per_fold)} tasks of {args.campaign}: "
          f"{campaign_dir(resolve_project_root(args.project_root), args.campaign) / SUMMARY_DIR}")


if __name__ == '__main__':
    main()
