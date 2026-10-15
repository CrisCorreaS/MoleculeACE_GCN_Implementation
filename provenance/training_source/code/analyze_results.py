"""Demo analysis of one validated campaign: tables and figure against the corrected Table 1 and published GCN values.

Run from code/ (light, CPU; after all tasks of the campaign are completed):
    python analyze_results.py --manifest ../jobs/manifest_demo.csv --campaign gcn_demo_v1

The campaign is re-validated first (same checks as aggregate_results.py). Outputs in results/<campaign>/_analysis/:
    datasets_vs_table1.csv   evaluated targets: n, n_test, %cliff of the CSVs vs the corrected Table 1 (2024)
    comparison.csv           RMSE / RMSE_cliff (mean and SD over folds), their difference, published GCN values
    figure_rmse_cliff.png/.pdf  (a) RMSE vs RMSE_cliff, (b) difference vs training molecules, (c) difference vs %cliff
    analysis.md              the same tables in Markdown, with the interpretation limits

With few targets (5 in the demo) every relationship is descriptive: no general conclusion, no p-values.
"""
import argparse
import csv
import datetime
import hashlib
import io
import math
import statistics
import sys
from pathlib import Path

from MoleculeACE.benchmark.artifacts import atomic_write_bytes, atomic_write_csv, atomic_write_json, read_json, \
    sha256_file
from MoleculeACE.benchmark.campaign import CampaignError, campaign_dir, resolve_project_root
from aggregate_results import collect, summarise
from verify_datasets import check_all

PUBLISHED_SHA256 = '6aa88c897f16bbb328793a2917289e3972b3fc8b50cc56be840f6714ea9baae5'   # upstream CRLF bytes
PUBLISHED_SOURCE_COMMIT = '45cf46d77f7681a7758d6447ffc3f239b349f406'
PUBLISHED_CORRECTION_DOI = '10.1021/acs.jcim.3c01576'
COMPARISON_FIELDS = ['dataset', 'target', 'abbreviation', 'type', 'n', 'n_train', 'n_test', 'pct_cliff', 'n_test_cliff',
                     'pct_test_cliff', 'n_folds', 'rmse_mean', 'rmse_sd', 'rmse_cliff_mean', 'rmse_cliff_sd',
                     'delta_mean', 'delta_sd', 'published_rmse', 'published_rmse_cliff', 'published_delta',
                     'rmse_minus_published', 'rmse_cliff_minus_published', 'published_provenance_verified',
                     'folds_official_last_epoch', 'rmse_best_val_secondary_mean',
                     'rmse_cliff_best_val_secondary_mean']
TABLE1_FIELDS = ['dataset', 'target', 'type', 'n', 'table1_n', 'n_test', 'table1_n_test', 'pct_cliff',
                 'table1_pct_cliff', 'pct_test_cliff', 'table1_pct_test_cliff', 'matches_table1']

# Reference palette (dataviz skill): slot 1 = ours (filled), slot 2 = published (hollow); ink for text
OURS, PUBLISHED = '#2a78d6', '#eb6834'
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def published_gcn(root):
    """ Published GCN_GRAPH results, accepting only LF/CRLF transport of the pinned upstream bytes. """
    folder = root / 'references' / 'published'
    path = folder / 'MoleculeACE_results.csv'
    provenance = read_json(folder / 'provenance.json')
    contents = path.read_bytes()
    # The pinned upstream blob uses CRLF; this checkout uses LF. Reconstruct only those
    # separators before hashing: do not strip whitespace, reserialise CSV cells, or replace
    # bare CR characters. Changed values, row order and trailing newlines must still fail.
    upstream_bytes = contents.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    source_url = (f"https://github.com/molML/MoleculeACE/blob/{PUBLISHED_SOURCE_COMMIT}/"
                  'MoleculeACE/Data/results/MoleculeACE_results.csv')
    verified = (hashlib.sha256(upstream_bytes).hexdigest() == PUBLISHED_SHA256 == provenance.get('sha256')
                and provenance.get('file') == path.name
                and provenance.get('source_commit') == PUBLISHED_SOURCE_COMMIT
                and provenance.get('source_url') == source_url
                and provenance.get('correction_doi') == PUBLISHED_CORRECTION_DOI
                and provenance.get('corrected_results_verified') is True)
    if not verified:
        return {}, False
    # Parse the same bytes that were verified, without a second file read.
    rows = {r['dataset']: r for r in csv.DictReader(io.StringIO(contents.decode('utf-8'), newline=''))
            if r['algorithm'] == 'GCN' and r['descriptor'] == 'GRAPH'}
    return {d: (float(r['rmse']), float(r['cliff_rmse'])) for d, r in rows.items()}, True


def build_tables(root, per_dataset):
    characteristics = {r['dataset']: r for r in check_all(root)}
    published, verified = published_gcn(root)
    table1, comparison = [], []
    for row in per_dataset:
        c = characteristics[row['dataset']]
        table1.append({k: c[k] for k in TABLE1_FIELDS})
        pub = published.get(row['dataset']) if verified else None
        comparison.append(dict(
            dataset=row['dataset'], target=c['target'], abbreviation=c['abbreviation'], type=c['type'], n=c['n'], n_train=c['n_train'],
            n_test=c['n_test'], pct_cliff=c['pct_cliff'], n_test_cliff=c['n_test_cliff'],
            pct_test_cliff=c['pct_test_cliff'], n_folds=row['n_folds'], rmse_mean=row['rmse_mean'],
            rmse_sd=row['rmse_sd'], rmse_cliff_mean=row['rmse_cliff_mean'], rmse_cliff_sd=row['rmse_cliff_sd'],
            delta_mean=row['delta_mean'], delta_sd=row['delta_sd'],
            published_rmse=pub[0] if pub else None, published_rmse_cliff=pub[1] if pub else None,
            published_delta=pub[1] - pub[0] if pub else None,
            rmse_minus_published=row['rmse_mean'] - pub[0] if pub else None,
            rmse_cliff_minus_published=(row['rmse_cliff_mean'] - pub[1]
                                        if pub and row['rmse_cliff_mean'] is not None else None),
            published_provenance_verified=verified, folds_official_last_epoch=row['folds_official_last_epoch'],
            rmse_best_val_secondary_mean=row['rmse_best_val_secondary_mean'],
            rmse_cliff_best_val_secondary_mean=row['rmse_cliff_best_val_secondary_mean']))
    return table1, comparison, verified


def _short(row):
    # Upstream abbreviation (metadata/datasets.csv), with the activity type when a target has two datasets
    return f"{row['abbreviation'] or row['dataset']} ({row['type']})"


def make_figure(comparison, title, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.titlecolor': INK, 'text.color': INK})
    rows = [r for r in comparison if r['rmse_cliff_mean'] is not None]
    label = len(rows) <= 10
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), facecolor=SURFACE)
    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    ax = axes[0]
    values = [v for r in rows for v in (r['rmse_mean'], r['rmse_cliff_mean'], r['published_rmse'],
                                        r['published_rmse_cliff']) if v is not None]
    lo, hi = min(values) - 0.1, max(values) + 0.1
    ax.plot([lo, hi], [lo, hi], color=INK_2, linestyle='--', linewidth=1, zorder=1)
    for offset in (0.5, -0.5):
        ax.plot([lo, hi], [lo + offset, hi + offset], color=GRID, linestyle='--', linewidth=1, zorder=1)
    if any(r['published_rmse'] is not None for r in rows):
        ax.scatter([r['published_rmse'] for r in rows if r['published_rmse'] is not None],
                   [r['published_rmse_cliff'] for r in rows if r['published_rmse'] is not None],
                   s=42, facecolors='none', edgecolors=PUBLISHED, linewidths=1.6, zorder=2,
                   label='GCN publicado (2024)')
    for r in rows:
        if r['published_rmse'] is not None:
            ax.plot([r['rmse_mean'], r['published_rmse']], [r['rmse_cliff_mean'], r['published_rmse_cliff']],
                    color=GRID, linewidth=1, zorder=1)
    ax.errorbar([r['rmse_mean'] for r in rows], [r['rmse_cliff_mean'] for r in rows],
                xerr=[r['rmse_sd'] or 0 for r in rows], yerr=[r['rmse_cliff_sd'] or 0 for r in rows],
                fmt='o', markersize=6, color=OURS, markeredgecolor=SURFACE, markeredgewidth=1.5,
                ecolor=OURS, elinewidth=1, capsize=0, zorder=3, label='Este trabajo (media ± DE entre folds)')
    if label:
        for r in rows:
            ax.annotate(_short(r), (r['rmse_mean'], r['rmse_cliff_mean']), xytext=(5, 4),
                        textcoords='offset points', fontsize=7, color=INK_2)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='RMSE (todo el test)', ylabel='RMSE$_{cliff}$',
           title='(a) RMSE frente a RMSE$_{cliff}$')
    ax.legend(loc='upper left', fontsize=7, frameon=False)

    for ax, key, xlabel, panel in ((axes[1], 'n_train', 'Moléculas de entrenamiento', '(b)'),
                                   (axes[2], 'pct_test_cliff', '% de cliffs en el test', '(c)')):
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.errorbar([r[key] for r in rows], [r['delta_mean'] for r in rows],
                    yerr=[r['delta_sd'] or 0 for r in rows], fmt='o', markersize=6, color=OURS,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, ecolor=OURS, elinewidth=1, capsize=0, zorder=3)
        pub = [r for r in rows if r['published_delta'] is not None]
        if pub:
            ax.scatter([r[key] for r in pub], [r['published_delta'] for r in pub], s=42, facecolors='none',
                       edgecolors=PUBLISHED, linewidths=1.6, zorder=2)
        if label:
            for r in rows:
                ax.annotate(_short(r), (r[key], r['delta_mean']), xytext=(5, 4),
                            textcoords='offset points', fontsize=7, color=INK_2)
        for r in pub:
            ax.plot([r[key], r[key]], [r['delta_mean'], r['published_delta']], color=GRID, linewidth=1, zorder=1)
        ax.set(xlabel=xlabel, ylabel='RMSE$_{cliff}$ − RMSE', title=f"{panel} Δ frente a {xlabel.lower()}")

    fig.suptitle(title, fontsize=10, color=INK)
    fig.text(0.5, 0.005, f"{len(rows)} targets: relación descriptiva, no una conclusión general. "
             'Barras: DE entre folds internos (mismo test; no son semillas independientes).',
             ha='center', fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ('png', 'pdf'):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=ext, dpi=200, facecolor=SURFACE)
        atomic_write_bytes(f"{out_base}.{ext}", buffer.getvalue())
    plt.close(fig)


def _fmt(value, digits=3):
    return '—' if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def markdown(campaign, stage, scientific, table1, comparison, verified):
    lines = [f"# Análisis de la campaña {campaign} ({stage})", '']
    if not scientific:
        lines += ['> **PILOTO: resultados NO científicos** (folds y épocas limitados). Solo validan el pipeline.', '']
    lines += ['## Targets evaluados frente a la Tabla 1 corregida (2024)', '',
              '| Dataset | Target | Tipo | n (Tabla 1) | n test (Tabla 1) | % cliff (Tabla 1) | % cliff test (Tabla 1) |',
              '|---|---|---|---|---|---|---|']
    for r in table1:
        lines.append(f"| {r['dataset']} | {r['target']} | {r['type']} | {r['n']} ({r['table1_n']}) | "
                     f"{r['n_test']} ({r['table1_n_test']}) | {r['pct_cliff']:.1f} ({r['table1_pct_cliff']}) | "
                     f"{r['pct_test_cliff']:.1f} ({r['table1_pct_test_cliff']}) |")
    lines += ['', '## RMSE y RMSE_cliff (media ± DE entre folds) frente al GCN publicado', '',
              '| Dataset | folds | RMSE | RMSE_cliff | Δ = RMSE_cliff − RMSE | RMSE publ. | RMSE_cliff publ. | Δ publ. |'
              ' RMSE − publ. | RMSE_cliff − publ. |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in comparison:
        pm = lambda m, s: f"{_fmt(m)} ± {_fmt(s)}" if m is not None else '—'
        lines.append(f"| {r['dataset']} | {r['n_folds']} | {pm(r['rmse_mean'], r['rmse_sd'])} | "
                     f"{pm(r['rmse_cliff_mean'], r['rmse_cliff_sd'])} | {pm(r['delta_mean'], r['delta_sd'])} | "
                     f"{_fmt(r['published_rmse'])} | {_fmt(r['published_rmse_cliff'])} | {_fmt(r['published_delta'])} | "
                     f"{_fmt(r['rmse_minus_published'])} | {_fmt(r['rmse_cliff_minus_published'])} |")
    lines += ['', '## Resultado secundario (NO oficial): mejor estado de validación', '',
              'Mismo entrenamiento, sin reentrenar. Solo difiere del oficial en los folds que llegaron al límite de '
              'épocas sin early stopping (el upstream evaluaba entonces los pesos de la última época).', '',
              '| Dataset | folds con estado oficial = última época | RMSE (mejor val.) | RMSE_cliff (mejor val.) |',
              '|---|---|---|---|']
    for r in comparison:
        lines.append(f"| {r['dataset']} | {r['folds_official_last_epoch']} | {_fmt(r['rmse_best_val_secondary_mean'])} "
                     f"| {_fmt(r['rmse_cliff_best_val_secondary_mean'])} |")
    lines += ['', 'Notas:',
              '- Métrica oficial = regla upstream: mejor estado tras early stopping; pesos de la última época si se '
              'alcanza el límite de épocas. La figura usa solo la métrica oficial.',
              '- Escala: y = −log10(nM); una diferencia de RMSE no cambia si se usa pKi/pEC50 = y + 9.',
              '- Media aritmética de los RMSE por fold (protocolo upstream), no RMSE de predicciones promediadas.',
              '- La DE describe folds internos evaluados en el mismo test; no son réplicas independientes.',
              f"- Valores publicados: `references/published/MoleculeACE_results.csv`, blob del commit upstream 45cf46d "
              f"(2023-09-29). Procedencia verificada por hash: **{'sí' if verified else 'NO (no se comparan)'}**. "
              'Son una única media de 5 folds por target, sin dispersión publicada; el modo de early stopping de '
              'esas ejecuciones no está registrado (ver docs/upstream_changes.md).',
              f"- Con {len(comparison)} targets, las relaciones con el tamaño y el % de cliffs son descriptivas "
              '(y ambos están correlacionados en este benchmark); no permiten conclusiones generales.', '']
    return '\n'.join(lines)


def analyze(root, manifest, campaign):
    tasks, results = collect(root, manifest, campaign)
    _, per_dataset = summarise(tasks, results)
    table1, comparison, verified = build_tables(root, per_dataset)
    first = tasks[0]
    out = campaign_dir(root, campaign) / '_analysis'
    atomic_write_csv(out / 'datasets_vs_table1.csv', table1, TABLE1_FIELDS)
    atomic_write_csv(out / 'comparison.csv', comparison, COMPARISON_FIELDS)
    title = f"GCN_GRAPH, campaña {campaign}" + ('' if first['scientific'] else ' — PILOTO, NO CIENTÍFICO')
    make_figure(comparison, title, out / 'figure_rmse_cliff')
    atomic_write_bytes(out / 'analysis.md', markdown(campaign, first['stage'], first['scientific'], table1,
                                                     comparison, verified).encode('utf-8'))
    atomic_write_json(out / 'analysis.json', dict(
        campaign=campaign, manifest_sha256=sha256_file(manifest), published_provenance_verified=verified,
        scientific=bool(first['scientific']), n_datasets=len(comparison),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--project-root', default=None)
    args = parser.parse_args()
    try:
        root = resolve_project_root(args.project_root)
        comparison = analyze(root, args.manifest.resolve(), args.campaign)
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Analysed {len(comparison)} datasets -> {campaign_dir(root, args.campaign) / '_analysis'}")


if __name__ == '__main__':
    main()
