"""Test fold aggregation and rejection of invalid synthetic campaign evidence.

Inputs are real dataset identities plus generated predictions, manifests and
fake checkpoints under temporary roots. Cases cover five- and 30-dataset
aggregation, paired differences, stage flags and selected corruption failures;
no historical pilot/demo result folders are required. Package imports require
NumPy, pandas and PyYAML, but no Torch or model training.

From the repository root:
    python -B tools/run_scientific_tests.py --selection test_aggregation

Temporary fixtures are removed by the test helper. See docs/reproduction.md.
"""
import csv
import json
import shutil
import statistics

from MoleculeACE.benchmark.artifacts import atomic_write_json, read_json, sha256_file
from MoleculeACE.benchmark.campaign import CampaignError, SUMMARY_DIR, task_dir
from MoleculeACE.benchmark.const import datasets
from aggregate_results import aggregate
from helpers import CampaignTestCase, write_synthetic_task

FIVE = ['CHEMBL2034_Ki', 'CHEMBL4203_Ki', 'CHEMBL1871_Ki', 'CHEMBL218_EC50', 'CHEMBL239_EC50']


class Success(CampaignTestCase):
    def test_demo_five_targets(self):
        manifest, rows = self.make_campaign('gcn_demo_t', 'demo', FIVE)
        per_fold, per_dataset = aggregate(self.root, manifest, 'gcn_demo_t')
        self.assertEqual((len(per_fold), len(per_dataset)), (25, 5))
        first = [f for f in per_fold if f['dataset'] == FIVE[0]]
        self.assertAlmostEqual(per_dataset[0]['rmse_mean'], statistics.mean(f['rmse'] for f in first))
        self.assertAlmostEqual(per_dataset[0]['rmse_sd'], statistics.stdev(f['rmse'] for f in first))
        self.assertAlmostEqual(per_dataset[0]['delta_mean'],
                               statistics.mean(f['rmse_cliff'] - f['rmse'] for f in first))
        summary = read_json(self.root / 'results' / 'gcn_demo_t' / SUMMARY_DIR / 'summary.json')
        self.assertTrue(summary['scientific'])
        self.assertEqual(summary['n_tasks'], 25)

    def test_full_thirty_targets_same_code(self):
        manifest, _ = self.make_campaign('gcn_full_t', 'full', datasets)
        per_fold, per_dataset = aggregate(self.root, manifest, 'gcn_full_t')
        self.assertEqual((len(per_fold), len(per_dataset)), (150, 30))

    def test_pilot_is_flagged_not_scientific(self):
        manifest, _ = self.make_campaign('gcn_pilot_t', 'pilot', ['CHEMBL2034_Ki'], folds=[0, 1], max_epochs=20)
        per_fold, per_dataset = aggregate(self.root, manifest, 'gcn_pilot_t')
        self.assertEqual((per_dataset[0]['n_folds'], per_dataset[0]['scientific']), (2, 0))
        self.assertFalse(read_json(self.root / 'results' / 'gcn_pilot_t' / SUMMARY_DIR / 'summary.json')['scientific'])


class Failures(CampaignTestCase):
    @classmethod
    def setUpClass(cls):
        # Build the valid campaign once; each test tampers with its own copy
        template = CampaignTestCase('run')
        template.setUp()
        template.make_campaign('gcn_demo_t', 'demo', FIVE)
        cls.template = template

    @classmethod
    def tearDownClass(cls):
        cls.template.tearDown()

    def setUp(self):
        super().setUp()
        shutil.rmtree(self.root)
        shutil.copytree(self.template.root, self.root)
        self.manifest = self.root / 'jobs' / 'manifest_gcn_demo_t.csv'
        from MoleculeACE.benchmark.campaign import load_manifest
        self.rows = load_manifest(self.manifest)
        self.task = self.rows[7]
        self.folder = task_dir(self.root, self.task)

    def assertFails(self, fragment, manifest=None, campaign='gcn_demo_t'):
        with self.assertRaises(CampaignError) as ctx:
            aggregate(self.root, manifest or self.manifest, campaign)
        self.assertIn(fragment, str(ctx.exception))

    def test_nonexistent_manifest(self):
        self.assertFails('Manifest not found', manifest=self.root / 'jobs' / 'nope.csv')

    def test_other_campaign(self):
        self.assertFails('does not match the manifest campaign', campaign='gcn_demo_other')

    def test_missing_task(self):
        shutil.rmtree(self.folder)
        self.assertFails('Missing task folder')

    def test_failed_or_running_task(self):
        atomic_write_json(self.folder / 'status.json', dict(status='running', task=self.task))
        self.assertFails('not completed')

    def test_missing_predictions(self):
        (self.folder / 'predictions.csv').unlink()
        self.assertFails('Missing or empty artifact')

    def test_corrupted_predictions(self):
        path = self.folder / 'predictions.csv'
        path.write_bytes(path.read_bytes().replace(b',0,', b',1,', 1))
        self.assertFails('Hash mismatch')

    def test_corrupted_checkpoint(self):
        (self.folder / 'best_model.pt').write_bytes(b'other bytes')
        self.assertFails('Hash mismatch')

    def _rewrite_predictions(self, edit):
        """ Tamper with predictions and fix the recorded hash, so only the content checks can catch it """
        path = self.folder / 'predictions.csv'
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows = edit(rows)
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        record = read_json(self.folder / 'task_record.json')
        record['artifact_sha256']['predictions.csv'] = sha256_file(path)
        atomic_write_json(self.folder / 'task_record.json', record)

    def test_nonfinite_prediction(self):
        self._rewrite_predictions(lambda rows: [dict(r, y_pred='nan') if i == 0 else r for i, r in enumerate(rows)])
        self.assertFails('Malformed or corrupted')

    def test_incomplete_predictions(self):
        self._rewrite_predictions(lambda rows: rows[:-1])
        self.assertFails('predictions.csv covers')

    def test_duplicated_molecule(self):
        self._rewrite_predictions(lambda rows: rows[:-1] + rows[:1])
        self.assertFails('Duplicate molecule ids')

    def test_misaligned_labels(self):
        self._rewrite_predictions(lambda rows: [dict(r, y_true=str(float(r['y_true']) + 1)) if i == 3 else r
                                                for i, r in enumerate(rows)])
        self.assertFails('does not match the source test set')

    def test_saved_metric_differs_from_predictions(self):
        path = self.folder / 'metrics.json'
        metrics = read_json(path)
        metrics['rmse'] += 0.1
        atomic_write_json(path, metrics)
        record = read_json(self.folder / 'task_record.json')
        record['artifact_sha256']['metrics.json'] = sha256_file(path)
        atomic_write_json(self.folder / 'task_record.json', record)
        self.assertFails('differs from the value recomputed')

    def test_nonfinite_saved_metric(self):
        path = self.folder / 'metrics.json'
        path.write_text(path.read_text(encoding='utf-8').replace('"rmse": ', '"rmse": NaN, "x": ', 1),
                        encoding='utf-8')
        record = read_json(self.folder / 'task_record.json')
        record['artifact_sha256']['metrics.json'] = sha256_file(path)
        atomic_write_json(self.folder / 'task_record.json', record)
        self.assertFails('Malformed or corrupted')

    def test_duplicate_copied_from_another_fold(self):
        other = task_dir(self.root, self.rows[8])
        shutil.rmtree(other)
        shutil.copytree(self.folder, other)
        self.assertFails('identity mismatch')
        # even with a matching status.json, the record identity is checked
        atomic_write_json(other / 'status.json', dict(status='completed', task=self.rows[8]))
        self.assertFails('belongs to another task')

    def test_unexpected_folders(self):
        (self.root / 'results' / 'gcn_demo_t' / FIVE[0] / 'fold_7').mkdir()
        (self.root / 'results' / 'gcn_demo_t' / 'CHEMBL262_Ki').mkdir()
        self.assertFails('Unexpected task folder')

    def test_training_history_inconsistent_with_mode(self):
        record = read_json(self.folder / 'task_record.json')
        record['training']['best_epoch'] += 1
        atomic_write_json(self.folder / 'task_record.json', record)
        self.assertFails('inconsistent with the history')

    def test_mixed_code_versions(self):
        record = read_json(self.folder / 'task_record.json')
        record['provenance']['source_sha256'] = {'a.py': 'changed'}
        atomic_write_json(self.folder / 'task_record.json', record)
        self.assertFails('different code')

    def test_official_last_epoch_needs_final_checkpoint(self):
        from MoleculeACE.benchmark.campaign import build_manifest
        pilot = build_manifest('gcn_pilot_t', 'pilot', ['CHEMBL2835_Ki'], folds=[0], max_epochs=20)[0]
        write_synthetic_task(self.root, pilot, self.rng, n_improving=15)   # plateau: reaches the 20-epoch limit
        folder = task_dir(self.root, pilot)
        record = read_json(folder / 'task_record.json')
        self.assertEqual(record['training']['official_state'], 'last_epoch')
        self.assertNotEqual(read_json(folder / 'metrics.json')['rmse'], read_json(folder / 'metrics_best_val.json')['rmse'])
        from MoleculeACE.benchmark.campaign import validate_task_outputs
        validate_task_outputs(self.root, pilot)
        (folder / 'final_model.pt').unlink()
        with self.assertRaises(CampaignError):
            validate_task_outputs(self.root, pilot)

    def test_stale_final_checkpoint_when_official_is_best(self):
        (self.folder / 'final_model.pt').write_bytes(b'left over')
        self.assertFails('final_model.pt present')

    def test_official_and_best_predictions_must_match_when_official_is_best(self):
        path = self.folder / 'predictions_best_val.csv'
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['y_pred'] = str(float(rows[0]['y_pred']) + 0.5)
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        from MoleculeACE.benchmark.artifacts import fold_metrics
        metrics = read_json(self.folder / 'metrics_best_val.json')
        metrics.update(fold_metrics([float(r['y_true']) for r in rows], [float(r['y_pred']) for r in rows],
                                    [int(r['cliff_mol']) for r in rows]))
        atomic_write_json(self.folder / 'metrics_best_val.json', metrics)
        record = read_json(self.folder / 'task_record.json')
        for name in ('predictions_best_val.csv', 'metrics_best_val.json'):
            record['artifact_sha256'][name] = sha256_file(self.folder / name)
        atomic_write_json(self.folder / 'task_record.json', record)
        self.assertFails('must be identical')

    def test_all_problems_are_reported_together(self):
        shutil.rmtree(self.folder)
        (task_dir(self.root, self.rows[0]) / 'metrics.json').unlink()
        with self.assertRaises(CampaignError) as ctx:
            aggregate(self.root, self.manifest, 'gcn_demo_t')
        self.assertIn('2 problem(s)', str(ctx.exception))

    def test_rerun_of_one_task_keeps_campaign_valid(self):
        write_synthetic_task(self.root, self.task, self.rng)
        per_fold, _ = aggregate(self.root, self.manifest, 'gcn_demo_t')
        self.assertEqual(len(per_fold), 25)


if __name__ == '__main__':
    import unittest
    unittest.main()
