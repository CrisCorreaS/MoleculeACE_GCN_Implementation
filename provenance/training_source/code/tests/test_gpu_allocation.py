"""Allocation guard tests using fake torch and driver replies; never access a GPU or Slurm."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from MoleculeACE.benchmark.gpu_allocation import allocation_report


class AllocationChecks(unittest.TestCase):
    def setUp(self):
        self.env = {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '1', 'SLURM_GPUS_ON_NODE': '1'}
        self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=Mock(return_value=1)))
        self.gpu = '1, 00000000:41:00.0, GPU-test, NVIDIA A100, 570.195.03, 40960 MiB'
        self.process = '321, 00000000:41:00.0'

    def check(self, env=None, replies=None, check_process=True):
        with patch.dict(os.environ, self.env if env is None else env, clear=True):
            before = dict(os.environ)
            with patch('MoleculeACE.benchmark.gpu_allocation._nvidia_smi',
                       side_effect=replies or [self.gpu, self.process]) as smi:
                with patch('MoleculeACE.benchmark.gpu_allocation.os.getpid', return_value=321):
                    report, problems = allocation_report(self.torch, check_process=check_process)
            self.assertEqual(dict(os.environ), before, 'Allocation checks must not change the environment')
        return report, problems, smi

    def test_no_slurm_job_stops_before_cuda_or_driver_query(self):
        report, problems, smi = self.check(env={'CUDA_VISIBLE_DEVICES': '1'})
        self.assertTrue(problems)
        self.assertIsNone(report['torch_device_count'])
        self.torch.cuda.device_count.assert_not_called()
        smi.assert_not_called()

    def test_missing_or_multiple_visible_devices_stop_before_cuda(self):
        for visible in (None, '', '0,1'):
            with self.subTest(visible=visible):
                env = dict(self.env)
                if visible is None:
                    del env['CUDA_VISIBLE_DEVICES']
                else:
                    env['CUDA_VISIBLE_DEVICES'] = visible
                _, problems, smi = self.check(env=env)
                self.assertTrue(problems)
                self.torch.cuda.device_count.assert_not_called()
                smi.assert_not_called()

    def test_slurm_multiple_gpus_rejected(self):
        _, problems, smi = self.check(env=dict(self.env, SLURM_GPUS_ON_NODE='2'))
        self.assertTrue(problems)
        self.torch.cuda.device_count.assert_not_called()
        smi.assert_not_called()

    def test_nonstep_job_gpu_list_cannot_hide_multiple_allocated_devices(self):
        _, problems, smi = self.check(env=dict(self.env, SLURM_JOB_GPUS='0,1'))
        self.assertTrue(problems)
        self.torch.cuda.device_count.assert_not_called()
        smi.assert_not_called()

    def test_single_gpu_step_in_larger_job_uses_global_gpu_id(self):
        # Slurm may remap physical GPU 1 to CUDA index 0 inside its step's cgroup.
        for on_node in ('1', '2'):
            with self.subTest(on_node=on_node):
                env = dict(self.env, SLURM_ARRAY_JOB_ID='120', SLURM_ARRAY_TASK_ID='1',
                           SLURM_STEP_ID='3', SLURM_STEP_GPUS='1', SLURM_JOB_GPUS='0,1',
                           SLURM_GPUS_ON_NODE=on_node, CUDA_VISIBLE_DEVICES='0')
                report, problems, smi = self.check(env=env)
                self.assertEqual(problems, [])
                self.assertEqual(report['allocation_scope'], 'step')
                self.assertEqual(report['allocated_gpu_id'], '1')
                self.assertEqual(report['env']['SLURM_ARRAY_TASK_ID'], '1')
                self.assertEqual(report['env']['SLURM_STEP_ID'], '3')
                self.assertEqual(smi.call_args_list[0][0][0][:2], ['-i', '1'])
                self.assertEqual(report['process_gpu_bus_ids'], ['00000000:41:00.0'])

    def test_single_gpu_job_also_uses_global_scheduler_gpu_id(self):
        report, problems, smi = self.check(env=dict(self.env, SLURM_JOB_GPUS='1', CUDA_VISIBLE_DEVICES='0'))
        self.assertEqual(problems, [])
        self.assertEqual(report['allocation_scope'], 'job')
        self.assertEqual(smi.call_args_list[0][0][0][:2], ['-i', '1'])

    def test_step_requires_single_gpu_scheduler_evidence_before_cuda(self):
        for gpu_ids in (None, '', '0,1'):
            with self.subTest(step_gpus=gpu_ids):
                env = dict(self.env, SLURM_STEP_ID='3', SLURM_JOB_GPUS='0,1', SLURM_GPUS_ON_NODE='2')
                if gpu_ids is not None:
                    env['SLURM_STEP_GPUS'] = gpu_ids
                _, problems, smi = self.check(env=env)
                self.assertTrue(problems)
                self.torch.cuda.device_count.assert_not_called()
                smi.assert_not_called()

    def test_step_gpu_list_without_numeric_step_id_does_not_allow_multi_gpu_job(self):
        for step_id in (None, 'batch', 'extern', 'invalid'):
            with self.subTest(step_id=step_id):
                env = dict(self.env, SLURM_STEP_GPUS='1', SLURM_JOB_GPUS='0,1', SLURM_GPUS_ON_NODE='2')
                if step_id is not None:
                    env['SLURM_STEP_ID'] = step_id
                _, problems, smi = self.check(env=env)
                self.assertTrue(problems)
                self.torch.cuda.device_count.assert_not_called()
                smi.assert_not_called()

    def test_step_still_rejects_extra_visible_or_torch_gpus(self):
        env = dict(self.env, SLURM_STEP_ID='3', SLURM_STEP_GPUS='1', SLURM_JOB_GPUS='0,1',
                   SLURM_GPUS_ON_NODE='2', CUDA_VISIBLE_DEVICES='0,1')
        _, problems, smi = self.check(env=env)
        self.assertTrue(problems)
        self.torch.cuda.device_count.assert_not_called()
        smi.assert_not_called()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        self.torch.cuda.device_count.return_value = 2
        _, problems, smi = self.check(env=env)
        self.assertTrue(problems)
        smi.assert_not_called()

    def test_step_process_on_parent_sibling_gpu_rejected(self):
        env = dict(self.env, SLURM_STEP_ID='3', SLURM_STEP_GPUS='1', SLURM_JOB_GPUS='0,1',
                   SLURM_GPUS_ON_NODE='2', CUDA_VISIBLE_DEVICES='0')
        _, problems, _ = self.check(env=env, replies=[self.gpu, '321, 00000000:81:00.0'])
        self.assertTrue(problems)

    def test_extra_cuda_device_rejected_before_driver_query(self):
        self.torch.cuda.device_count.return_value = 2
        _, problems, smi = self.check()
        self.assertTrue(problems)
        smi.assert_not_called()

    def test_preflight_queries_only_the_allocated_gpu(self):
        report, problems, smi = self.check(check_process=False)
        self.assertEqual(problems, [])
        self.assertEqual(report['allocated_gpu']['index'], '1')
        self.assertEqual(smi.call_count, 1)
        self.assertEqual(smi.call_args[0][0][:2], ['-i', '1'])
        self.assertNotIn('process_gpu_bus_ids', report)

    def test_process_on_its_allocated_gpu_passes(self):
        report, problems, smi = self.check(replies=[
            self.gpu, self.process + '\n999, 00000000:81:00.0'])
        self.assertEqual(problems, [])
        self.assertEqual(report['process_gpu_bus_ids'], ['00000000:41:00.0'])
        self.assertEqual(smi.call_count, 2)

    def test_process_on_wrong_or_additional_gpu_rejected(self):
        for apps in ('321, 00000000:81:00.0', self.process + '\n321, 00000000:81:00.0'):
            with self.subTest(apps=apps):
                _, problems, _ = self.check(replies=[self.gpu, apps])
                self.assertTrue(problems)

    def test_missing_process_or_failed_process_query_rejected(self):
        for apps in ('', 'error: driver query unavailable'):
            with self.subTest(apps=apps):
                _, problems, _ = self.check(replies=[self.gpu, apps])
                self.assertTrue(problems)

    def test_failed_or_ambiguous_allocated_gpu_query_rejected(self):
        for gpu in ('error: invalid device', self.gpu + '\n' + self.gpu):
            with self.subTest(gpu=gpu):
                _, problems, smi = self.check(replies=[gpu])
                self.assertTrue(problems)
                self.assertEqual(smi.call_count, 1)


if __name__ == '__main__':
    unittest.main()
