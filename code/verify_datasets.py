"""Check all corrected benchmark CSVs against their scales and corrected Table 1.

Inputs are the 30 package CSVs, metadata and references/corrected_table1.csv.
Checks cover columns, split/cliff values, duplicate SMILES strings, counts,
rounded cliff percentages, y = -log10(concentration in nM), and pKi/pEC50 = y + 9.
This does not recompute cliff labels or fit a model. Package import requires
NumPy, pandas and PyYAML; chemistry and Torch are not required.

The CLI writes results/data_check/datasets.csv and datasets.json. To preserve
the supplied reports, use a prepared scratch copy from docs/reproduction.md.
From its code directory:
    python verify_datasets.py --markdown

Exit status two reports unsuccessful checks. Retained a priori pilot/demo
selection describes the historical selection rule and never uses model results.
"""
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from MoleculeACE.benchmark.artifacts import atomic_write_csv, atomic_write_json, sha256_file
from MoleculeACE.benchmark.campaign import data_path, resolve_project_root
from MoleculeACE.benchmark.const import DATA_PATH, datasets

COLUMNS = ['smiles', 'exp_mean [nM]', 'y', 'cliff_mol', 'split', 'y [pEC50/pKi]']
FIELDS = ['dataset', 'target', 'abbreviation', 'type', 'target_class', 'n', 'n_train', 'n_test', 'n_cliff', 'pct_cliff',
          'n_test_cliff', 'pct_test_cliff', 'table1_n', 'table1_n_test', 'table1_pct_cliff', 'table1_pct_test_cliff',
          'y_mean', 'y_sd', 'y_min', 'y_max', 'matches_table1', 'problems', 'sha256']


def describe(dataset, table_row, metadata_row):
    """Read one corrected CSV and compare its counts and response scales.

    Parameters
    ----------
    dataset : str
        Dataset identifier resolved in the imported package.
    table_row, metadata_row : dict
        Corrected reference counts and optional target metadata.

    Returns
    -------
    dict
        Dataset characteristics, input hash, matches_table1 and concise problems.
        This reads input files without changing published rows or labels.
    """
    problems = []
    with data_path(dataset).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            problems.append(f"columns {reader.fieldnames}")
        rows = list(reader)
    ys = []
    for i, r in enumerate(rows):
        nm, y, p = float(r['exp_mean [nM]']), float(r['y']), float(r['y [pEC50/pKi]'])
        ys.append(y)
        if r['split'] not in ('train', 'test') or r['cliff_mol'] not in ('0', '1'):
            problems.append(f"row {i}: split/cliff value")
        if not (nm > 0 and math.isclose(y, -math.log10(nm), abs_tol=1e-9) and math.isclose(p, y + 9, abs_tol=1e-9)):
            problems.append(f"row {i}: label scale")
    if len({r['smiles'] for r in rows}) != len(rows):
        problems.append('duplicate SMILES')
    test = [r for r in rows if r['split'] == 'test']
    n_cliff = sum(r['cliff_mol'] == '1' for r in rows)
    n_test_cliff = sum(r['cliff_mol'] == '1' for r in test)
    out = dict(dataset=dataset, target=table_row['target'],
               abbreviation=metadata_row.get('Abbriviation', ''), type=dataset.split('_')[1],
               target_class=metadata_row.get('Receptor Class', ''), n=len(rows), n_train=len(rows) - len(test),
               n_test=len(test), n_cliff=n_cliff, pct_cliff=100 * n_cliff / len(rows), n_test_cliff=n_test_cliff,
               pct_test_cliff=100 * n_test_cliff / len(test), table1_n=int(table_row['n']),
               table1_n_test=int(table_row['n_test']), table1_pct_cliff=int(table_row['percent_cliff']),
               table1_pct_test_cliff=int(table_row['percent_test_cliff']), y_mean=statistics.mean(ys),
               y_sd=statistics.stdev(ys), y_min=min(ys), y_max=max(ys), sha256=sha256_file(data_path(dataset)))
    if (out['n'], out['n_test']) != (out['table1_n'], out['table1_n_test']):
        problems.append('n / n_test differ from Table 1')
    for mine, published in (('pct_cliff', 'table1_pct_cliff'), ('pct_test_cliff', 'table1_pct_test_cliff')):
        if abs(out[mine] - out[published]) > 0.5 + 1e-9:
            problems.append(f"{mine} {out[mine]:.2f} vs Table 1 {out[published]}")
    out['matches_table1'] = not problems
    out['problems'] = '; '.join(problems[:5])
    return out


def check_all(root):
    """Return input-check records for all 30 corrected benchmark datasets.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing references/corrected_table1.csv. Dataset files and
        metadata are read from the imported package's Data directory.

    Returns
    -------
    list of dict
        Dataset descriptions in corrected-reference target order; no files written.
    """
    with (root / 'references' / 'corrected_table1.csv').open(encoding='utf-8', newline='') as handle:
        table = {r['dataset']: r for r in csv.DictReader(handle)}
    with (Path(DATA_PATH) / 'metadata' / 'datasets.csv').open(encoding='utf-8', newline='') as handle:
        metadata = {r['Dataset']: r for r in csv.DictReader(handle)}
    if set(table) != set(datasets):
        raise ValueError('references/corrected_table1.csv does not list exactly the 30 benchmark datasets')
    return [describe(d, table[d], metadata.get(d, {})) for d in sorted(datasets, key=lambda d: table[d]['target'])]


def _tercile_bounds(values):
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=3, method='inclusive')
    return q[0], q[1]


def propose_targets(rows):
    """ A priori target proposal from data characteristics only (never from GCN results).

    Size groups are terciles of n. Demo (5): small/lowest %cliff, small/highest %cliff, large/lowest %cliff,
    large/highest %cliff, and the medium-size dataset whose %cliff is closest to the median of the 30.
    Pilot (1): the smallest dataset not in the demo.
    """
    small_max, large_min = _tercile_bounds([r['n'] for r in rows])
    small = [r for r in rows if r['n'] <= small_max]
    large = [r for r in rows if r['n'] >= large_min]
    medium = [r for r in rows if small_max < r['n'] < large_min]
    median_cliff = statistics.median(r['pct_cliff'] for r in rows)
    picks = [('pequeño, % cliffs mínimo', min(small, key=lambda r: r['pct_cliff'])),
             ('pequeño, % cliffs máximo', max(small, key=lambda r: r['pct_cliff'])),
             ('mediano, % cliffs más cercano a la mediana', min(medium, key=lambda r: abs(r['pct_cliff'] - median_cliff))),
             ('grande, % cliffs mínimo', min(large, key=lambda r: r['pct_cliff'])),
             ('grande, % cliffs máximo', max(large, key=lambda r: r['pct_cliff']))]
    chosen = {r['dataset'] for _, r in picks}
    pilot = min((r for r in rows if r['dataset'] not in chosen), key=lambda r: r['n'])
    return dict(small_max=small_max, large_min=large_min, median_cliff=median_cliff, demo=picks, pilot=pilot)


def main():
    """Check inputs and write the requested dataset reports and console table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', default=None)
    parser.add_argument('--markdown', action='store_true', help='Print a Markdown table for docs/')
    parser.add_argument('--propose', action='store_true', help='Print the a priori pilot/demo target proposal')
    args = parser.parse_args()
    root = resolve_project_root(args.project_root)
    rows = check_all(root)
    if args.propose:
        p = propose_targets(rows)
        print(f"Terciles de n: pequeño <= {p['small_max']:.0f}, grande >= {p['large_min']:.0f}; "
              f"mediana de % cliffs = {p['median_cliff']:.1f}")
        print('| Papel | Dataset | Target | Tipo | n (test) | % cliff (cliffs en test) |')
        print('|---|---|---|---|---|---|')
        for role, r in p['demo'] + [('piloto (el más pequeño fuera de la demo)', p['pilot'])]:
            print(f"| {role} | {r['dataset']} | {r['target']} | {r['type']} | {r['n']} ({r['n_test']}) "
                  f"| {r['pct_cliff']:.1f} ({r['n_test_cliff']}) |")
    out = root / 'results' / 'data_check'
    atomic_write_csv(out / 'datasets.csv', rows, FIELDS)
    atomic_write_json(out / 'datasets.json', dict(n_datasets=len(rows), n_rows=sum(r['n'] for r in rows),
                                                  all_match_table1=all(r['matches_table1'] for r in rows),
                                                  datasets=rows))
    if args.markdown:
        print('| Dataset | Target | Tipo | Clase | n (test) | % cliff (test) | Tabla 1 |')
        print('|---|---|---|---|---|---|---|')
        for r in rows:
            print(f"| {r['dataset']} | {r['target']} | {r['type']} | {r['target_class']} | {r['n']} ({r['n_test']}) "
                  f"| {r['pct_cliff']:.1f} ({r['pct_test_cliff']:.1f}) | {'coincide' if r['matches_table1'] else 'NO'} |")
    bad = [r for r in rows if not r['matches_table1']]
    for r in bad:
        print(f"MISMATCH {r['dataset']}: {r['problems']}", file=sys.stderr)
    print(f"{len(rows) - len(bad)}/{len(rows)} datasets match the corrected Table 1; "
          f"{sum(r['n'] for r in rows)} molecules. Written to {out}")
    if bad:
        sys.exit(2)


if __name__ == '__main__':
    main()
