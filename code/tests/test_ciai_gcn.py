"""Retain historical GCN environment checks with an explicit scientific selection.

The submission runner selects Versions, Featurization, Folds and CliffLabels.
The first two require the pinned Torch/PyG stack; Folds requires scikit-learn;
CliffLabels requires RDKit and Levenshtein. Package imports also need NumPy,
pandas and PyYAML, and chemistry helpers use tqdm. These four classes do not
train models. Missing dependencies produce explicit skips.

From the repository root, in the appropriate existing environment:
    python -B tools/run_scientific_tests.py --selection test_ciai_gcn.Versions test_ciai_gcn.Featurization test_ciai_gcn.Folds test_ciai_gcn.CliffLabels

The retained GCNOnGPU class is excluded by that runner. It trains diagnostic
models and refers to historical diagnostic infrastructure that is not active
in the submission. Do not run this module wholesale to verify the delivery.
Importing it may inspect CUDA availability when Torch is already installed.
See docs/reproduction.md for the supported checks and their limitations.
"""
import importlib.util
import inspect
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

HAVE = {name: importlib.util.find_spec(name) is not None
        for name in ('torch', 'torch_geometric', 'rdkit', 'sklearn', 'Levenshtein', 'yaml')}
PENDING = 'PENDIENTE EN CIAI: requires {}'


def needs(*names):
    missing = [n for n in names if not HAVE[n]]
    return unittest.skipIf(bool(missing), PENDING.format(', '.join(missing)))


def needs_gpu_allocation():
    if not (HAVE['torch'] and HAVE['torch_geometric'] and HAVE['rdkit'] and HAVE['sklearn']):
        return unittest.skip(PENDING.format('torch, torch_geometric, rdkit and sklearn'))
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        return unittest.skip('PENDIENTE EN CIAI: requires a Slurm GPU allocation (never the login node)')
    return lambda cls: cls


@needs('torch', 'torch_geometric')
class Versions(unittest.TestCase):
    """Check pinned Torch/PyG versions and the GraphMultisetTransformer API."""
    def test_pinned_versions_keep_the_original_gmt_api(self):
        import torch
        import torch_geometric
        from torch_geometric.nn import GraphMultisetTransformer
        self.assertTrue(torch.__version__.startswith('1.11.0'), torch.__version__)
        self.assertEqual(torch_geometric.__version__, '2.0.4')
        params = inspect.signature(GraphMultisetTransformer.__init__).parameters
        # The GMT used by models/gcn.py: (in_channels, hidden_channels, out_channels, Conv, ..., num_heads)
        for name in ('in_channels', 'hidden_channels', 'out_channels', 'Conv', 'pooling_ratio', 'num_heads'):
            self.assertIn(name, params)


@needs('torch', 'torch_geometric', 'rdkit', 'yaml')
class Featurization(unittest.TestCase):
    """Check representative graphs with 37 node and six edge features, plus 1024-bit ECFPs."""
    def test_graph_and_ecfp_dimensions(self):
        from MoleculeACE.benchmark.featurization import Featurizer
        graphs = Featurizer().graphs(['CCO', 'c1ccccc1C(=O)N'])
        self.assertEqual([g.x.shape[1] for g in graphs], [37, 37])
        self.assertEqual([g.edge_attr.shape[1] for g in graphs], [6, 6])
        self.assertEqual(Featurizer().ecfp(['CCO', 'CCN']).shape, (2, 1024))


@needs('sklearn', 'yaml')
class Folds(unittest.TestCase):
    """Check deterministic, disjoint internal folds using only the published training set."""
    def test_upstream_split_partitions_the_training_set(self):
        from MoleculeACE.benchmark.utils import Data, molecular_folds
        data = Data('CHEMBL2034_Ki')
        folds = molecular_folds(data.y_train, n_folds=5, seed=42)
        again = molecular_folds(data.y_train, n_folds=5, seed=42)
        n = len(data.y_train)
        val = sorted(int(i) for f in folds for i in f['val_idx'])
        self.assertEqual(val, list(range(n)))                      # each molecule validates exactly once
        for f, g in zip(folds, again):
            self.assertEqual(list(f['val_idx']), list(g['val_idx']))   # deterministic
            self.assertFalse(set(f['train_idx']) & set(f['val_idx']))
            self.assertEqual(len(f['train_idx']) + len(f['val_idx']), n)


@needs('rdkit', 'Levenshtein')
class CliffLabels(unittest.TestCase):
    """Recompute one dataset's labels from positive nM concentrations without training."""
    def test_published_cliff_labels_recomputed_on_one_dataset(self):
        import csv
        from MoleculeACE.benchmark.cliffs import ActivityCliffs
        from MoleculeACE.benchmark.campaign import data_path
        with data_path('CHEMBL2034_Ki').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        cliffs = ActivityCliffs([r['smiles'] for r in rows], [float(r['exp_mean [nM]']) for r in rows])
        recomputed = cliffs.get_cliff_molecules(return_smiles=False, similarity=0.9, potency_fold=10)
        mismatches = [i for i, (r, c) in enumerate(zip(rows, recomputed)) if int(r['cliff_mol']) != int(c)]
        self.assertEqual(mismatches, [], f"{len(mismatches)} labels differ from the published cliff_mol")


@needs_gpu_allocation()
class GCNOnGPU(unittest.TestCase):
    """Preserve excluded historical GPU/diagnostic tests; not part of the submission suite."""
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='gcn_test_'))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _small_problem(self, n_train=96, n_val=32, probe_start=0):
        from MoleculeACE.benchmark.const import Descriptors
        from MoleculeACE.benchmark.utils import Data, get_benchmark_config
        data = Data('CHEMBL2034_Ki')
        data.smiles_train, data.y_train = data.smiles_train[:n_train + n_val], data.y_train[:n_train + n_val]
        # These are software-test prediction probes, taken only from the published training partition.
        # Keep Data's existing x_test container, but do not use the published test inputs or labels for debugging.
        probe_stop = min(probe_start + 40, len(data.smiles_train))
        self.assertLess(probe_start, probe_stop, 'Prediction probes must not be empty')
        data.smiles_test = data.smiles_train[probe_start:probe_stop]
        data.y_test = data.y_train[probe_start:probe_stop]
        import csv
        from MoleculeACE.benchmark.campaign import data_path
        with data_path('CHEMBL2034_Ki').open(encoding='utf-8', newline='') as handle:
            train_rows = [(i, row['smiles']) for i, row in enumerate(csv.DictReader(handle))
                          if row['split'] == 'train']
        self.probe_identities = [dict(dataset='CHEMBL2034_Ki', split='train', csv_row=csv_row,
                                      train_position=i, smiles=smiles)
                                 for i, (csv_row, smiles) in
                                 enumerate(train_rows[probe_start:probe_stop], start=probe_start)]
        self.assertEqual([row['smiles'] for row in self.probe_identities], data.smiles_test)
        data.featurize_data(Descriptors.GRAPH)
        hyper = get_benchmark_config('CHEMBL2034_Ki')
        yaml_epochs = hyper.pop('epochs')
        return data, hyper, yaml_epochs, n_train

    def test_upstream_epoch_rule_300(self):
        from MoleculeACE.models import GCN
        _, hyper, yaml_epochs, _ = self._small_problem(8, 4)
        self.assertEqual(yaml_epochs, 61)
        self.assertEqual(GCN(**hyper).epochs, 300)      # constructor default overrides GNN.__init__'s 100

    def test_modes_checkpoint_and_restore_at_epoch_limit(self):
        import torch
        from MoleculeACE.models import GCN
        diagnostic_only = os.environ.get('GCN_REPEATABILITY_DIAGNOSTIC') == '1'
        # Freeze a second, disjoint probe group before validation; both groups are in the published training split.
        probe_start = 0 if diagnostic_only else 40
        data, hyper, yaml_epochs, n = self._small_problem(probe_start=probe_start)
        if not diagnostic_only:
            self.assertEqual([row['train_position'] for row in self.probe_identities], list(range(40, 80)))
        model = GCN(**hyper)
        modes = []
        model.model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
        checkpoint = self.dir / 'best_model.pt'
        cwd_before = set(os.listdir('.'))
        final = self.dir / 'final_model.pt'
        record = model.train(data.x_train[:n], data.y_train[:n], data.x_train[n:], data.y_train[n:],
                             early_stopping_patience=10, epochs=4, checkpoint_path=str(checkpoint),
                             final_checkpoint_path=str(final), print_every_n=100)
        self.assertEqual(record['stop_reason'], 'epoch_limit')
        self.assertTrue(checkpoint.is_file())
        # Official (upstream) state at the epoch limit = last epoch, saved before the best state is restored
        self.assertEqual(final.is_file(), record['official_state'] == 'last_epoch')
        if final.is_file():
            self.assertEqual(torch.load(str(final), map_location='cpu')['epoch'], 4)
        self.assertEqual(set(os.listdir('.')), cwd_before)          # nothing written to the working directory
        self.assertIn(True, modes)                                   # training forward passes
        self.assertIn(False, modes)                                  # validation / prediction forward passes
        self.assertFalse(model.model.training)

        saved = torch.load(str(checkpoint), map_location='cpu')
        self.assertEqual(saved['epoch'], record['best_epoch'])
        for key, value in model.model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), saved['state_dict'][key]), key)

        from MoleculeACE.benchmark.artifacts import atomic_write_json, sha256_file
        from MoleculeACE.benchmark.campaign import config_path, data_path, resolve_project_root
        from MoleculeACE.benchmark.gpu_allocation import allocation_report
        from run_benchmark import git_state, package_versions, source_hashes
        from repeatability_diagnostic import diagnose_repeatability

        root = resolve_project_root()
        job_id = os.environ['SLURM_JOB_ID']
        restart = os.environ.get('SLURM_RESTART_COUNT') or '0'
        if not job_id.isdigit() or not restart.isdigit():
            self.fail('Diagnostic output identity must be a numeric Slurm job/restart pair')
        allocation, problems = allocation_report(torch)
        self.assertEqual(problems, [], 'Repeatability validation requires a valid single-GPU allocation')

        # Engineering budget: half the 0.001 display increment already used by analyze_results._fmt.
        # Chosen after diagnostic job 284025, then frozen for validation on positions 40:80.
        # It is not a universal error bound or a guarantee that rounded results never change.
        absolute_tolerance = 5e-4
        policy = dict(name='repeatability_reporting_budget_v1', absolute_tolerance=absolute_tolerance,
                      relative_tolerance=0, activity_scale='y=-log10(nM)', gpu_repeats=20,
                      statistic='maximum per-molecule prediction span across all original-GPU repeats',
                      rationale='half the existing 0.001 analysis-table display increment',
                      selected_after_diagnostic_job='284025')
        metadata = dict(dataset='CHEMBL2034_Ki', scientific=False, seed=42,
                        check_mode='diagnostic' if diagnostic_only else 'full_checks_validation',
                        best_epoch=record['best_epoch'], training_record=record,
                        hyperparameters=hyper, yaml_epochs_ignored=yaml_epochs,
                        train_positions=list(range(n)), validation_positions=list(range(n, len(data.x_train))),
                        probe_identities=self.probe_identities, acceptance_policy=policy,
                        data_sha256=sha256_file(data_path('CHEMBL2034_Ki')),
                        config_sha256=sha256_file(config_path('CHEMBL2034_Ki')),
                        versions=package_versions(), source_sha256=source_hashes(root),
                        git=git_state(root), allocation=allocation,
                        slurm_job_id=job_id, slurm_restart_count=int(restart))
        output_dir = root / 'results' / 'diagnostics' / f'gcn_repeatability_{job_id}_{restart}'
        print(f'Collecting repeatability evidence -> {output_dir}', flush=True)
        modes.clear()
        report = diagnose_repeatability(model, data.x_test, output_dir, metadata, repeats=20)
        original = report['conditions']['gpu_original']
        max_difference = original['max_pairwise_span']
        outside_tolerance = sum(span > absolute_tolerance for span in original['per_graph_span'])
        passed = max_difference <= absolute_tolerance
        report['acceptance'] = dict(policy=policy, max_pairwise_span=max_difference,
                                    molecules_outside_tolerance=outside_tolerance, passed=passed,
                                    note='This verdict covers the saved probe/model only; the full suite must pass.')
        # Persist the decision even if the assertion below fails. No failed result is relabelled or overwritten.
        atomic_write_json(output_dir / 'report.json', report)
        diagnostic = (f'GCN repeated eval predictions: n={len(data.x_test)} repeats=20 '
                      f'max_pairwise_span={max_difference:.9g} '
                      f'absolute_tolerance={absolute_tolerance:.9g} log10(nM) rtol=0 '
                      f'outside_tolerance={outside_tolerance} passed={passed}')
        print(diagnostic, flush=True)
        print(f'Repeatability evidence saved -> {output_dir}', flush=True)
        self.assertEqual(report['status'], 'completed')
        self.assertTrue(modes)
        self.assertFalse(any(modes), 'Every repeated forward pass must run in eval mode')
        self.assertFalse(model.model.training)
        self.assertEqual(report['restored_tf32'], report['original_tf32'])
        self.assertEqual(report['restored_cpu_threads'], report['original_cpu_threads'])
        self.assertTrue(report['cpu_rng_restored'])
        self.assertTrue(passed, diagnostic)

        y_val = torch.tensor(data.y_train[n:])
        val_loss = float(model.loss_fn(model.predict(data.x_train[n:]), y_val))
        self.assertTrue(math.isclose(val_loss, record['best_val_loss'], rel_tol=1e-4))

    def test_early_stopping_restores_best(self):
        from MoleculeACE.models import GCN
        data, hyper, _, n = self._small_problem()
        hyper['lr'] = 0.05                                            # unstable on purpose: triggers early stopping
        model = GCN(**hyper)
        record = model.train(data.x_train[:n], data.y_train[:n], data.x_train[n:], data.y_train[n:],
                             early_stopping_patience=2, epochs=60, checkpoint_path=str(self.dir / 'b.pt'),
                             final_checkpoint_path=str(self.dir / 'f.pt'), print_every_n=100,
                             early_stopping_mode='paper_nonconsecutive')
        self.assertLessEqual(record['epochs_run'], 60)
        self.assertEqual(record['early_stopping']['mode'], 'paper_nonconsecutive')
        if record['stop_reason'] == 'early_stopping':
            self.assertTrue(record['upstream_would_restore_best'])


if __name__ == '__main__':
    unittest.main()
