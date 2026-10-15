"""Published reference integrity and demo analysis on a synthetic 5-target campaign."""
import csv
import hashlib
import importlib.util
import json
import shutil
import unittest

from MoleculeACE.benchmark.campaign import default_project_root
from helpers import CampaignTestCase

DEMO = ['CHEMBL4203_Ki', 'CHEMBL4616_EC50', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL244_Ki']


class ReferenceTestCase(CampaignTestCase):
    def setUp(self):
        super().setUp()
        shutil.copytree(default_project_root() / 'references' / 'published', self.root / 'references' / 'published')
        shutil.copy(default_project_root() / 'references' / 'corrected_table1.csv', self.root / 'references')
        self.reference_path = self.root / 'references' / 'published' / 'MoleculeACE_results.csv'
        self.provenance_path = self.reference_path.with_name('provenance.json')


class PublishedProvenance(ReferenceTestCase):
    def test_lf_and_crlf_preserve_verified_published_values(self):
        from analyze_results import published_gcn
        lf = self.reference_path.read_bytes().replace(b'\r\n', b'\n')
        expected = None
        for separator in (b'\n', b'\r\n'):
            with self.subTest(separator=separator):
                self.reference_path.write_bytes(lf.replace(b'\n', separator))
                rows, verified = published_gcn(self.root)
                self.assertTrue(verified)
                self.assertEqual(len(rows), 30)
                self.assertEqual(rows['CHEMBL214_Ki'], (1.006804061, 1.083560898))
                if expected is not None:
                    self.assertEqual(rows, expected)
                expected = rows

    def test_changed_gcn_or_non_gcn_values_are_rejected(self):
        from analyze_results import published_gcn
        original = self.reference_path.read_bytes()
        for old, new in ((b'GCN,GRAPH,0,1.006804061', b'GCN,GRAPH,0,9.006804061'),
                         (b'AFP,GRAPH,0,1.347354231', b'AFP,GRAPH,0,9.347354231')):
            with self.subTest(algorithm=old.split(b',')[0]):
                self.assertIn(old, original)
                self.reference_path.write_bytes(original.replace(old, new, 1))
                self.assertEqual(published_gcn(self.root), ({}, False))

    def test_normalisation_does_not_ignore_other_byte_changes(self):
        from analyze_results import published_gcn
        lf = self.reference_path.read_bytes().replace(b'\r\n', b'\n')
        for changed in (b' ' + lf, lf + b'\n', lf.replace(b'\n', b'\r', 1)):
            with self.subTest(change=changed[:70]):
                self.reference_path.write_bytes(changed)
                self.assertEqual(published_gcn(self.root), ({}, False))

    def test_provenance_identity_and_verification_flag_are_required(self):
        from analyze_results import published_gcn
        original = json.loads(self.provenance_path.read_text(encoding='utf-8'))
        for field, wrong in (('sha256', '0' * 64), ('source_commit', '0' * 40),
                             ('source_url', 'https://example.org/unrelated.csv'),
                             ('correction_doi', '10.1021/acs.jcim.2c01073'),
                             ('file', 'unrelated.csv'), ('corrected_results_verified', False)):
            with self.subTest(field=field):
                changed = dict(original, **{field: wrong})
                self.provenance_path.write_text(json.dumps(changed), encoding='utf-8')
                self.assertEqual(published_gcn(self.root), ({}, False))

    def test_replacing_metadata_hash_cannot_bless_changed_results(self):
        from analyze_results import published_gcn
        changed = self.reference_path.read_bytes().replace(
            b'GCN,GRAPH,0,1.006804061', b'GCN,GRAPH,0,9.006804061', 1)
        self.reference_path.write_bytes(changed)
        provenance = json.loads(self.provenance_path.read_text(encoding='utf-8'))
        provenance['sha256'] = hashlib.sha256(
            changed.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
        self.provenance_path.write_text(json.dumps(provenance), encoding='utf-8')
        self.assertEqual(published_gcn(self.root), ({}, False))

    def test_unverified_values_are_excluded_from_comparison(self):
        from analyze_results import build_tables
        self.reference_path.write_bytes(self.reference_path.read_bytes().replace(
            b'GCN,GRAPH,0,1.006804061', b'GCN,GRAPH,0,9.006804061', 1))
        summary = dict(dataset='CHEMBL214_Ki', n_folds=5, rmse_mean=1.0, rmse_sd=0.1,
                       rmse_cliff_mean=1.2, rmse_cliff_sd=0.1, delta_mean=0.2, delta_sd=0.1,
                       folds_official_last_epoch=0, rmse_best_val_secondary_mean=1.0,
                       rmse_cliff_best_val_secondary_mean=1.2)
        _, comparison, verified = build_tables(self.root, [summary])
        self.assertFalse(verified)
        self.assertFalse(comparison[0]['published_provenance_verified'])
        for field in ('published_rmse', 'published_rmse_cliff', 'published_delta',
                      'rmse_minus_published', 'rmse_cliff_minus_published'):
            self.assertIsNone(comparison[0][field], field)


@unittest.skipIf(importlib.util.find_spec('matplotlib') is None, 'matplotlib not installed')
class Analysis(ReferenceTestCase):
    def test_demo_tables_figure_and_published_comparison(self):
        from analyze_results import analyze
        manifest, _ = self.make_campaign('gcn_demo_t', 'demo', DEMO)
        comparison = analyze(self.root, manifest, 'gcn_demo_t')
        self.assertEqual([r['dataset'] for r in comparison], DEMO)
        self.assertTrue(all(r['published_provenance_verified'] for r in comparison))
        # CHEMBL214_Ki published GCN values (upstream 45cf46d)
        with (self.root / 'references' / 'published' / 'MoleculeACE_results.csv').open(newline='') as handle:
            ref = [r for r in csv.DictReader(handle) if r['dataset'] == 'CHEMBL214_Ki' and r['algorithm'] == 'GCN'][0]
        row = [r for r in comparison if r['dataset'] == 'CHEMBL214_Ki'][0]
        self.assertEqual(row['published_rmse'], float(ref['rmse']))
        self.assertAlmostEqual(row['rmse_minus_published'], row['rmse_mean'] - float(ref['rmse']))
        out = self.root / 'results' / 'gcn_demo_t' / '_analysis'
        for name in ('comparison.csv', 'datasets_vs_table1.csv', 'figure_rmse_cliff.png', 'figure_rmse_cliff.pdf',
                     'analysis.md'):
            self.assertGreater((out / name).stat().st_size, 0, name)
        # the analysis folder must not break a later re-validation of the campaign
        from aggregate_results import aggregate
        self.assertEqual(len(aggregate(self.root, manifest, 'gcn_demo_t')[0]), 25)

    def test_pilot_is_labelled_not_scientific(self):
        from analyze_results import analyze
        manifest, _ = self.make_campaign('gcn_pilot_t', 'pilot', ['CHEMBL2835_Ki'], folds=[0, 1], max_epochs=20)
        analyze(self.root, manifest, 'gcn_pilot_t')
        text = (self.root / 'results' / 'gcn_pilot_t' / '_analysis' / 'analysis.md').read_text(encoding='utf-8')
        self.assertIn('NO científicos', text)


if __name__ == '__main__':
    unittest.main()
