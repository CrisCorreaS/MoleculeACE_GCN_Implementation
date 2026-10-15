"""GPU-worker regressions: mocked driver/processes only; Cris runs these inside the checks job."""
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from MoleculeACE.benchmark.gpu_allocation import allocation_report, worker_gpu_map
from gpu_worker_child import bind_and_exec
from run_gpu_worker import child_environment, cpu_lanes, run_lanes, split_lanes


class WorkerAllocation(unittest.TestCase):
    def setUp(self):
        self.env = dict(SLURM_JOB_ID='123', SLURM_JOB_GPUS='2,7', SLURM_GPUS_ON_NODE='2',
                        SLURM_CPUS_PER_TASK='20', CUDA_VISIBLE_DEVICES='0,1')
        self.mapping = [dict(index='2', uuid='GPU-two'), dict(index='7', uuid='GPU-seven')]

    def test_map_queries_only_slurm_gpus_even_with_remapped_visibility(self):
        with patch.dict(os.environ, self.env, clear=True), \
                patch('MoleculeACE.benchmark.gpu_allocation._nvidia_smi',
                      side_effect=['2, GPU-two', '7, GPU-seven']) as smi:
            self.assertEqual(worker_gpu_map(), self.mapping)
        self.assertEqual([call.args[0][:2] for call in smi.call_args_list], [['-i', '2'], ['-i', '7']])

    def test_missing_duplicate_or_ambiguous_allocations_stop_before_driver(self):
        cases = [dict(SLURM_JOB_GPUS='2,2'), dict(SLURM_JOB_GPUS='2'),
                 dict(SLURM_JOB_GPUS='2-7'), dict(CUDA_VISIBLE_DEVICES='0'),
                 dict(SLURM_GPUS_ON_NODE='1'), dict(SLURM_STEP_ID='2')]
        for update in cases:
            with self.subTest(update=update), patch.dict(os.environ, dict(self.env, **update), clear=True), \
                    patch('MoleculeACE.benchmark.gpu_allocation._nvidia_smi') as smi:
                with self.assertRaises(ValueError):
                    worker_gpu_map()
                smi.assert_not_called()

    def test_duplicate_physical_gpu_or_wrong_driver_index_rejected(self):
        for replies in (['2, GPU-two', '7, GPU-two'], ['3, GPU-two'], ['error: driver failed']):
            with self.subTest(replies=replies), patch.dict(os.environ, self.env, clear=True), \
                    patch('MoleculeACE.benchmark.gpu_allocation._nvidia_smi', side_effect=replies):
                with self.assertRaises(ValueError):
                    worker_gpu_map()

    def test_child_binds_before_exec_and_rejects_unallocated_cpus(self):
        with patch('gpu_worker_child.os.sched_getaffinity', return_value={4, 5}), \
                patch('gpu_worker_child.os.sched_setaffinity') as bind, \
                patch('gpu_worker_child.os.execv') as execute:
            bind_and_exec([4], ['python', 'train.py'])
            bind.assert_called_once_with(0, [4])
            execute.assert_called_once_with('python', ['python', 'train.py'])
            execute.reset_mock()
            with self.assertRaises(ValueError):
                bind_and_exec([9], ['python', 'train.py'])
            execute.assert_not_called()

    def test_cpu_lanes_are_disjoint_and_within_allocation(self):
        with patch.dict(os.environ, self.env, clear=True), \
                patch('run_gpu_worker.os.sched_getaffinity', return_value=set(range(20, 40))):
            self.assertEqual(cpu_lanes(10), [list(range(20, 30)), list(range(30, 40))])
            with self.assertRaises(ValueError):
                cpu_lanes(11)

    def check_lane(self, lane, updates=None, replies=None):
        with patch.dict(os.environ, self.env, clear=True):
            original = dict(os.environ)
            env = child_environment(self.mapping, lane, list(range(lane * 10, lane * 10 + 10)))
            self.assertEqual(dict(os.environ), original)
        env.update(updates or {})
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=Mock(return_value=1)))
        gpu = f'{self.mapping[lane]["index"]}, bus-{lane}, {self.mapping[lane]["uuid"]}, A100, driver, 40960 MiB'
        with patch.dict(os.environ, env, clear=True), \
                patch('MoleculeACE.benchmark.gpu_allocation._nvidia_smi',
                      side_effect=replies or [gpu, f'321, bus-{lane}']) as smi, \
                patch('MoleculeACE.benchmark.gpu_allocation.os.getpid', return_value=321):
            report, problems = allocation_report(torch)
        return report, problems, torch, smi

    def test_children_keep_scheduler_evidence_and_use_distinct_uuids(self):
        for lane in (0, 1):
            report, problems, _, _ = self.check_lane(lane)
            self.assertEqual(problems, [])
            self.assertEqual(report['allocation_scope'], 'hpc_worker_lane')
            self.assertEqual(report['env']['SLURM_JOB_GPUS'], '2,7')
            self.assertEqual(report['env']['SLURM_GPUS_ON_NODE'], '2')
            self.assertEqual(report['env']['CUDA_VISIBLE_DEVICES'], self.mapping[lane]['uuid'])

    def test_wrong_lane_visibility_or_forged_parent_map_rejected_before_cuda(self):
        for update in (dict(CUDA_VISIBLE_DEVICES='GPU-seven'), dict(GCN_GPU_LANE='9'),
                       dict(GCN_GPU_MAP='invalid'), dict(GCN_GPU_WORKER='other'),
                       dict(SLURM_JOB_GPUS='4,7'), dict(GCN_PARENT_VISIBLE_DEVICES='0')):
            with self.subTest(update=update):
                _, problems, torch, smi = self.check_lane(0, update)
                self.assertTrue(problems)
                torch.cuda.device_count.assert_not_called()
                smi.assert_not_called()

    def test_driver_uuid_and_process_context_must_match_selected_lane(self):
        for replies in (['2, bus-0, GPU-other, A100, driver, 40960 MiB'],
                        ['2, bus-0, GPU-two, A100, driver, 40960 MiB', '321, bus-1']):
            _, problems, _, _ = self.check_lane(0, replies=replies)
            self.assertTrue(problems)


class WorkerScheduling(unittest.TestCase):
    def test_sparse_ids_split_once_and_preserve_order(self):
        lanes = split_lanes([149, 7, 12, 0, 31])
        self.assertEqual(lanes, [[149, 12, 31], [7, 0]])
        for tasks in ([1], [1, 1]):
            with self.assertRaises(ValueError):
                split_lanes(tasks)

    def run_mock(self, lanes, failure=None, invalid=None):
        launched = []
        processes = []
        def launch(lane, task):
            launched.append((lane, task))
            process = Mock()
            process.poll.return_value = 1 if task == failure else 0
            processes.append(process)
            return process, (io.StringIO(), io.StringIO())
        def validate(lane, task):
            if task == invalid:
                raise ValueError('missing completion artifacts')
        with patch('run_gpu_worker.time.sleep'):
            result = run_lanes(lanes, launch, validate, 1800)
        return launched, result

    def test_150_tasks_covered_once_across_two_workers_and_four_lanes(self):
        all_launched = []
        for worker in range(2):
            ids = list(range(worker, 150, 2))
            launched, outcome = self.run_mock(split_lanes(ids))
            self.assertFalse(outcome['failed'])
            self.assertFalse(outcome['not_started'])
            all_launched.extend(task for lane, task in launched)
        self.assertEqual(sorted(all_launched), list(range(150)))

    def test_failure_stops_refilling_both_lanes(self):
        launched, result = self.run_mock([[0, 2], [1, 3]], failure=1)
        self.assertEqual(launched, [(0, 0), (1, 1)])
        self.assertEqual(result['completed'], [0])
        self.assertEqual(result['not_started'], [2, 3])
        self.assertEqual(result['failed'][0]['task_id'], 1)

    def test_zero_exit_without_valid_artifacts_is_failure(self):
        launched, result = self.run_mock([[0, 2], [1, 3]], invalid=0)
        self.assertEqual(len(launched), 2)
        self.assertEqual(result['failed'][0]['exit_code'], 0)
        self.assertIn('missing completion', result['failed'][0]['error'])

    def test_timeout_terminates_only_owned_process_and_stops_queue(self):
        process = Mock()
        process.poll.return_value = None
        other = Mock()
        other.poll.return_value = 0
        launch = Mock(side_effect=[(process, ()), (other, ())])
        with patch('run_gpu_worker.time.monotonic', side_effect=[0, 0, 10]), \
                patch('run_gpu_worker.time.sleep'), patch('run_gpu_worker.terminate') as terminate:
            result = run_lanes([[0, 2], [1, 3]], launch, Mock(), timeout=5)
        terminate.assert_called_once_with(process)
        self.assertEqual(result['failed'][0]['exit_code'], 124)
        self.assertEqual(result['not_started'], [2, 3])

    def test_launch_failure_reaps_existing_child_on_interrupt(self):
        process = Mock()
        launch = Mock(side_effect=[(process, ()), KeyboardInterrupt()])
        with patch('run_gpu_worker.terminate') as terminate:
            with self.assertRaises(KeyboardInterrupt):
                run_lanes([[0], [1]], launch, Mock(), 10)
        terminate.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
