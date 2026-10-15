"""Slurm requeue / relaunch of one task: nothing partial is ever 'completed', nothing of an earlier attempt survives,
and the requeue is recorded (local; training is replaced by the synthetic helper, no torch)."""
import os
import random
from unittest import mock

import run_benchmark
from MoleculeACE.benchmark.artifacts import atomic_write_json, read_json, sha256_file
from MoleculeACE.benchmark.campaign import CampaignError, task_dir, validate_task_outputs
from helpers import CampaignTestCase, synthetic_training, write_synthetic_task

STALE_TMP = ('.predictions.csv.0123456789abcdef0123456789abcdef.tmp',
             'best_model.pt.0123456789abcdef0123456789abcdef.tmp')


class Requeue(CampaignTestCase):
    def setUp(self):
        super().setUp()
        self.manifest, rows = self.make_campaign('gcn_pilot_t', 'pilot', ['CHEMBL2835_Ki'], folds=[0, 1],
                                                 max_epochs=20, write=False)
        self.task = rows[0]
        self.folder = task_dir(self.root, self.task)

    def run_task(self, restart_count, train_task):
        env = {'SLURM_JOB_ID': '111'}
        if restart_count is not None:
            env['SLURM_RESTART_COUNT'] = str(restart_count)
        with mock.patch.dict(os.environ, env), mock.patch.object(run_benchmark, 'train_task', train_task):
            if restart_count is None:
                os.environ.pop('SLURM_RESTART_COUNT', None)
            run_benchmark.run(self.root, self.manifest, 'gcn_pilot_t', self.task['task_id'])

    def node_failure_before_completion(self):
        """ First attempt wrote every artifact, then the node died before status 'completed' (and mid-write) """
        write_synthetic_task(self.root, self.task, self.rng)
        atomic_write_json(self.folder / 'status.json',
                          dict(status='running', task=self.task, slurm={'SLURM_JOB_ID': '111'},
                               attempt=dict(restart_count=0)))
        (self.folder / 'final_model.pt').write_bytes(b'stale last-epoch checkpoint of attempt 1')
        for name in STALE_TMP:
            (self.folder / name).write_bytes(b'half-written')
        return sha256_file(self.folder / 'predictions.csv')

    def test_partial_attempt_is_never_valid(self):
        self.node_failure_before_completion()
        with self.assertRaisesRegex(CampaignError, 'not completed'):
            validate_task_outputs(self.root, self.task)

    def test_requeue_starts_clean_and_is_recorded(self):
        stale_predictions = self.node_failure_before_completion()
        seen_at_start = []

        def train_task(root, task, folder):
            seen_at_start.extend(sorted(p.name for p in folder.iterdir()))
            return synthetic_training(folder, task, random.Random(1), tag=b'attempt 2 ')

        self.run_task(1, train_task)

        # Training started with no artifact or temporary file of attempt 1
        self.assertEqual(seen_at_start, ['.lock', 'status.json'])
        self.assertEqual([p.name for p in self.folder.iterdir() if p.name.endswith('.tmp')], [])
        self.assertNotEqual(sha256_file(self.folder / 'predictions.csv'), stale_predictions)
        validate_task_outputs(self.root, self.task)          # completed, and only attempt-2 artifacts hashed

        status = read_json(self.folder / 'status.json')
        record = read_json(self.folder / 'task_record.json')
        self.assertEqual(status['status'], 'completed')
        for attempt in (status['attempt'], record['attempt']):
            self.assertEqual(attempt['restart_count'], 1)
            self.assertEqual(attempt['previous'], dict(status='running', slurm_job_id='111', restart_count=0))
            self.assertTrue({'predictions.csv', 'metrics.json', 'task_record.json', 'best_model.pt',
                             'final_model.pt', *STALE_TMP} <= set(attempt['removed']))
            self.assertEqual(attempt['other_files'], [])
        self.assertEqual(record['provenance']['slurm']['SLURM_RESTART_COUNT'], '1')

    def test_requeue_of_a_completed_task_changes_nothing(self):
        write_synthetic_task(self.root, self.task, self.rng)
        before = {p.name: sha256_file(p) for p in self.folder.iterdir()}

        def train_task(root, task, folder):
            raise AssertionError('a completed and valid task must not be retrained')

        self.run_task(1, train_task)
        after = {p.name: sha256_file(p) for p in self.folder.iterdir() if p.name != '.lock'}
        self.assertEqual(after, before)

    def test_manual_relaunch_after_failure_records_previous_status(self):
        self.folder.mkdir(parents=True)
        atomic_write_json(self.folder / 'status.json', dict(status='failed', task=self.task, slurm_job_id='99'))
        (self.folder / 'best_model.pt').write_bytes(b'stale')
        self.run_task(None, lambda root, task, folder: synthetic_training(folder, task, random.Random(2)))
        record = read_json(self.folder / 'task_record.json')
        self.assertEqual(record['attempt']['restart_count'], 0)
        self.assertEqual(record['attempt']['previous']['status'], 'failed')
        self.assertIn('best_model.pt', record['attempt']['removed'])
        validate_task_outputs(self.root, self.task)

    def test_worker_and_step_identity_survive_a_failed_task_retry(self):
        env = {'SLURM_ARRAY_JOB_ID': '110', 'SLURM_ARRAY_TASK_ID': '1', 'SLURM_STEP_ID': '3'}

        def fail_training(root, task, folder):
            raise RuntimeError('synthetic step failure')

        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(RuntimeError, 'synthetic step failure'):
                self.run_task(None, fail_training)
        failed = read_json(self.folder / 'status.json')
        self.assertEqual(failed['attempt']['manifest_task_id'], self.task['task_id'])
        self.assertEqual(failed['attempt']['slurm_array_task_id'], '1')
        self.assertEqual(failed['attempt']['slurm_step_id'], '3')

        with mock.patch.dict(os.environ, dict(env, SLURM_STEP_ID='7')):
            self.run_task(None, lambda root, task, folder: synthetic_training(folder, task, random.Random(3)))
        record = read_json(self.folder / 'task_record.json')
        status = read_json(self.folder / 'status.json')
        self.assertEqual(record['task']['task_id'], 0)
        for attempt in (record['attempt'], status['attempt']):
            self.assertEqual(attempt['manifest_task_id'], 0)
            self.assertEqual(attempt['slurm_array_job_id'], '110')
            self.assertEqual(attempt['slurm_array_task_id'], '1')
            self.assertEqual(attempt['slurm_step_id'], '7')
            self.assertEqual(attempt['previous']['manifest_task_id'], 0)
            self.assertEqual(attempt['previous']['slurm_array_job_id'], '110')
            self.assertEqual(attempt['previous']['slurm_array_task_id'], '1')
            self.assertEqual(attempt['previous']['slurm_step_id'], '3')
            self.assertEqual(attempt['previous']['status'], 'failed')
        self.assertEqual(record['provenance']['slurm']['SLURM_ARRAY_TASK_ID'], '1')
        self.assertEqual(record['provenance']['slurm']['SLURM_STEP_ID'], '7')
        validate_task_outputs(self.root, self.task)

    def test_invalid_restart_count_fails_clearly(self):
        with self.assertRaisesRegex(CampaignError, 'SLURM_RESTART_COUNT'):
            self.run_task('x', lambda *a: None)


if __name__ == '__main__':
    import unittest
    unittest.main()
