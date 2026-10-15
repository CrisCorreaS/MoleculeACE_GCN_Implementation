"""Generate the manifest of a stage (pilot, demo, full). One row = one Slurm array task = one (dataset, fold).

Run from code/ (no training; standard library only):
    python make_manifest.py --stage pilot --campaign gcn_pilot_v1 --datasets CHEMBL2835_Ki --folds 0 1 --max-epochs 20
    python make_manifest.py --stage demo  --campaign gcn_demo_v1  --datasets D1 D2 D3 D4 D5
    python make_manifest.py --stage full  --campaign gcn_full_v1

Default output: jobs/manifest_<stage>.csv. A manifest is never overwritten once its campaign has results: to
change anything (targets, early-stopping mode, code), create a new campaign name (e.g. gcn_demo_v2).
"""
import argparse
import sys
from pathlib import Path

from MoleculeACE.benchmark.campaign import (CampaignError, build_manifest, campaign_dir, load_manifest,
                                             resolve_project_root, write_manifest)
from MoleculeACE.benchmark.const import datasets
from MoleculeACE.models.training import EARLY_STOPPING_MODES, UPSTREAM_CONSECUTIVE

PILOT_FOLDS = [0, 1]
PILOT_MAX_EPOCHS = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['pilot', 'demo', 'full'])
    parser.add_argument('--campaign', required=True, help="Must start with 'gcn_<stage>_'")
    parser.add_argument('--datasets', nargs='+', help='pilot: 1 dataset; demo: 5; full: omit (all 30)')
    parser.add_argument('--folds', nargs='+', type=int, help=f'pilot only (default {PILOT_FOLDS})')
    parser.add_argument('--max-epochs', type=int, help=f'pilot only (default {PILOT_MAX_EPOCHS})')
    parser.add_argument('--early-stopping-mode', choices=EARLY_STOPPING_MODES, default=UPSTREAM_CONSECUTIVE)
    parser.add_argument('--out', type=Path, help='Default: <project>/jobs/manifest_<stage>.csv')
    parser.add_argument('--replace', action='store_true',
                        help='Replace an existing manifest whose campaign has no results yet')
    parser.add_argument('--project-root', default=None)
    args = parser.parse_args()
    try:
        root = resolve_project_root(args.project_root)
        pilot = args.stage == 'pilot'
        if not pilot and (args.folds is not None or args.max_epochs is not None):
            raise CampaignError('--folds and --max-epochs are pilot-only: scientific stages use the full protocol')
        names = args.datasets if args.datasets else (datasets if args.stage == 'full' else None)
        if names is None:
            raise CampaignError(f"--datasets is required for stage {args.stage}")
        rows = build_manifest(args.campaign, args.stage, names,
                              folds=(args.folds or PILOT_FOLDS) if pilot else None,
                              max_epochs=(args.max_epochs or PILOT_MAX_EPOCHS) if pilot else None,
                              early_stopping_mode=args.early_stopping_mode)
        out = (args.out or root / 'jobs' / f"manifest_{args.stage}.csv").resolve()
        if out.exists():
            old = load_manifest(out)[0]['campaign']
            if campaign_dir(root, old).exists():
                raise CampaignError(f"{out} belongs to campaign {old}, which already has results; "
                                    f"write the new manifest elsewhere with a new campaign name")
            if not args.replace:
                raise CampaignError(f"{out} exists (campaign {old}); pass --replace to overwrite it")
        write_manifest(out, rows)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"{len(rows)} tasks ({len(set(r['dataset'] for r in rows))} datasets) -> {out}")
    print(f"campaign={rows[0]['campaign']} stage={rows[0]['stage']} scientific={rows[0]['scientific']} "
          f"max_epochs={rows[0]['max_epochs']} early_stopping={rows[0]['early_stopping_mode']} "
          f"deviation={rows[0]['deviation']}")


if __name__ == '__main__':
    main()
