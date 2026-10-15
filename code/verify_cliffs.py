"""Recompute and compare published activity-cliff labels on CPU.

Inputs are corrected package CSVs with SMILES, exp_mean [nM] and cliff_mol.
ActivityCliffs sees the complete train-plus-test dataset. Structural similarity
is at least 0.9 in ECFP/Tanimoto, generic-scaffold Tanimoto or normalised
SMILES/Levenshtein; the concentration ratio is strictly greater than 10.
The model's logarithmic responses are not passed to that calculation.

Use the pinned chemistry environment described in docs/reproduction.md:
RDKit, Levenshtein, NumPy and tqdm, plus pandas/PyYAML via package import.
Torch and PyG are not required. The script writes
results/data_check/cliffs_<dataset>.json but never replaces the CSV labels.
Use a separately prepared scratch copy to preserve existing reports; from code/:
    python verify_cliffs.py --all

--datasets selects named subsets. Exit status two means at least one label
differs. Input paths follow the imported package; --project-root selects the
report destination. See docs/protocol.md for the concentration-scale definition.
"""
import argparse
import csv
import sys
import time

from MoleculeACE.benchmark.artifacts import atomic_write_json, sha256_file
from MoleculeACE.benchmark.campaign import data_path, resolve_project_root
from MoleculeACE.benchmark.const import datasets


def check(dataset):
    """Recompute one dataset's cliff membership and compare row by row.

    Parameters
    ----------
    dataset : str
        Corrected benchmark identifier.

    Returns
    -------
    dict
        Input SHA-256, molecule and cliff counts, total mismatch count, first
        200 differing zero-based data-row indices and elapsed seconds. The function
        itself writes nothing and leaves the published labels unchanged.
    """
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
    """Select datasets and write cliff-comparison reports, failing on differences."""
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
