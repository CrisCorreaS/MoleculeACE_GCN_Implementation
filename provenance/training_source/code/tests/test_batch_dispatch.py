"""Direct batch wrapper tests with fake Conda/CUDA/Python; execute only in a Slurm checks job."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Requires Bash on a compute node')
class BatchJobWrapper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gcn_batch_wrapper_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.jobs = self.root / 'jobs'
        self.jobs.mkdir()
        self.code = self.root / 'code'
        self.code.mkdir()
        self.wrapper = self.jobs / 'run_gcn.sh'
        shutil.copyfile(ROOT / 'jobs' / 'run_gcn.sh', self.wrapper)
        conda = self.jobs / 'fake_conda_init.sh'
        conda.write_text('conda() { [[ "$1" == activate ]] || return 96; export CONDA_PREFIX="$2"; }\n',
                         encoding='utf-8')
        (self.jobs / 'load_cuda_module.sh').write_text(':\n', encoding='utf-8')
        self.bin = self.root / 'mock_bin'
        self.bin.mkdir()
        python = self.bin / 'python'
        python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >| "$FAKE_PYTHON_MARKER"\n'
                          'exit "${FAKE_PYTHON_RC:-0}"\n', encoding='utf-8')
        python.chmod(0o700)
        self.marker = self.root / 'python_called.txt'
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ('BASH_ENV', 'ENV', 'GCN_TASK_RECEIPT')
                    and not key.startswith(('BASH_FUNC_', 'SLURM_'))}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''),
                        PROJECT_ROOT=str(self.root), CODE_DIR=str(self.code),
                        MANIFEST=str(self.jobs / 'manifest_demo.csv'), CAMPAIGN='gcn_demo_test',
                        MOLECULEACE_ENV=str(self.root / 'fake_env'), CONDA_INIT=str(conda),
                        CUDA_MODULE='mock-cuda', SLURM_JOB_ID='4242', SLURM_CPUS_PER_TASK='10',
                        FAKE_PYTHON_MARKER=str(self.marker))
        self.bash = shutil.which('bash')

    def _run(self, *args, python_rc=0):
        return subprocess.run([self.bash, str(self.wrapper), *args], cwd=str(self.root),
                              env=dict(self.env, FAKE_PYTHON_RC=str(python_rc)),
                              capture_output=True, text=True, timeout=10)

    def test_production_runs_without_array_or_step_variables(self):
        result = self._run('7')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.marker.read_text(encoding='utf-8').splitlines(),
                         ['run_benchmark.py', '--manifest', self.env['MANIFEST'], '--campaign',
                          self.env['CAMPAIGN'], '--task-id', '7', '--project-root', str(self.root)])
        self.assertIn('job=4242', result.stdout)
        self.assertIn('task_id=7 status=0', result.stdout)

    def test_python_failure_is_the_batch_job_exit_code(self):
        result = self._run('7', python_rc=17)
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertIn('task_id=7 status=17', result.stdout)

    def test_pilot_uses_array_index_without_an_argument(self):
        self.env['SLURM_ARRAY_TASK_ID'] = '1'
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.marker.read_text(encoding='utf-8').splitlines()
        self.assertEqual(args[args.index('--task-id') + 1], '1')

    def test_invalid_task_arguments_stop_before_python(self):
        for args in ((), ('-1',), ('abc',), ('0', '1')):
            with self.subTest(args=args):
                result = self._run(*args)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.marker.exists())

    def test_missing_slurm_job_stops_before_python(self):
        self.env.pop('SLURM_JOB_ID')
        result = self._run('7')
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.marker.exists())

    def test_job_scripts_contain_no_step_launcher(self):
        for path in (ROOT / 'jobs').glob('*.sh'):
            with self.subTest(path=path.name):
                self.assertIsNone(re.search(r'\bsrun\b', path.read_text(encoding='utf-8')))
        self.assertFalse((ROOT / 'jobs' / 'run_gcn_batch.sh').exists())


if __name__ == '__main__':
    unittest.main()
