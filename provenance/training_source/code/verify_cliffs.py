"""Recompute the published cliff labels (cliff_mol) with the pinned RDKit and compare. Never modifies the CSVs.

Run from code/ on CIAI (needs rdkit and Levenshtein; CPU; the largest dataset takes minutes):
    python verify_cliffs.py --datasets CHEMBL2835_Ki CHEMBL4203_Ki     # or --all

Upstream definition (benchmark/data_prep.py in 7e6de0b): ActivityCliffs(smiles, bioactivity in nM) over the whole
dataset (train + test), similarity >= 0.9 in ECFP-Tanimoto OR generic-scaffold Tanimoto OR normalised Levenshtein,
and potency fold change > 10 (strict). The training/evaluation always uses the published labels.
Writes results/data_check/cliffs_<dataset>.json; exit code 2 if any label differs.
"""
import argparse
import csv
import sys
import time

from MoleculeACE.benchmark.artifacts import atomic_write_json, sha256_file
from MoleculeACE.benchmark.campaign import data_path, resolve_project_root
from MoleculeACE.benchmark.const import datasets


def check(dataset):
    from MoleculeACE.benchmark.cliffs import ActivityCliffs
    with data_path(dataset).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    started = time.perf_counter()
    cliffs = ActivityCliffs([r['smiles'] for r in rows], [float(r['exp_mean [nM]']) for r in rows])
    recomputed = cliffs.get_cliff_molecules(return_smiles=False, similarity=0.9, potency_fold=10)
    differ = [i for i, (r, c) in enumerate(zip(rows, recomputed)) if int(r['cliff_mol']) != int(c)]
    return dict(dataset=dataset, sha256=sha256_file(data_path(dataset)), n=len(rows),
                n_cliff_published=sum(int(r['cliff_mol']) for r in rows), n_cliff_recomputed=int(sum(recomputed)),
                n_differ=len(differ), differing_rows=differ[:200], seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', choices=datasets)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--project-root', default=None)
    args = parser.parse_args()
    if not args.all and not args.datasets:
        parser.error('give --datasets or --all')
    root = resolve_project_root(args.project_root)
    bad = 0
    for dataset in (datasets if args.all else args.datasets):
        result = check(dataset)
        atomic_write_json(root / 'results' / 'data_check' / f"cliffs_{dataset}.json", result)
        bad += result['n_differ'] > 0
        print(f"{dataset:<16} n={result['n']} published={result['n_cliff_published']} "
              f"recomputed={result['n_cliff_recomputed']} differ={result['n_differ']} ({result['seconds']:.0f}s)")
    if bad:
        print(f"{bad} dataset(s) with differing cliff labels (published labels are still the ones used)",
              file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
