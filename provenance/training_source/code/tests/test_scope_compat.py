"""GCN-only scope, Python 3.8 compatibility and path rules (static checks, local)."""
import ast
import re
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
PY_FILES = sorted(p for p in CODE.rglob('*.py') if '__pycache__' not in p.parts)
SHELL_FILES = sorted((ROOT / 'jobs').glob('*.sh')) if (ROOT / 'jobs').is_dir() else []

# APIs that do not exist in Python 3.8 (the CIAI environment); ast feature_version only checks syntax
PY39_APIS = [r'\.is_relative_to\(', r'\.removeprefix\(', r'\.removesuffix\(', r'functools\.cache\b', r'\bzoneinfo\b',
             r'math\.lcm\(', r'\bstr\s*\|\s*None', r':\s*(list|dict|tuple|set)\[', r'->\s*(list|dict|tuple|set)\[']
EXCLUDED_IMPORTS = ('tensorflow', 'transformers', 'tokenizers', 'huggingface_hub', 'skopt', 'sklearn.svm')


class PythonCompatibility(unittest.TestCase):
    def test_syntax_is_python_38(self):
        for path in PY_FILES:
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))

    def test_no_python39_only_apis(self):
        for path in PY_FILES:
            if path.name == 'test_scope_compat.py':
                continue
            text = path.read_text(encoding='utf-8')
            for pattern in PY39_APIS:
                with self.subTest(file=path.name, api=pattern):
                    self.assertIsNone(re.search(pattern, text))


class Scope(unittest.TestCase):
    def test_no_excluded_model_dependencies(self):
        for path in PY_FILES:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            names += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            for name in names:
                with self.subTest(file=path.name, module=name):
                    self.assertFalse(name.startswith(EXCLUDED_IMPORTS))

    def test_only_gcn_models_and_configs(self):
        package = CODE / 'MoleculeACE'
        self.assertEqual(sorted(p.name for p in (package / 'models').glob('*.py')),
                         ['__init__.py', 'gcn.py', 'training.py', 'utils.py'])
        configs = sorted(p.relative_to(package / 'Data' / 'configures').as_posix()
                         for p in (package / 'Data' / 'configures').rglob('*.yml'))
        self.assertEqual(len(configs), 31)
        self.assertTrue(all(c.endswith('/GCN_GRAPH.yml') or c == 'default/GCN.yml' for c in configs))
        self.assertEqual(len(list((package / 'Data' / 'benchmark_data').glob('CHEMBL*.csv'))), 30)
        self.assertFalse((CODE / 'optimize.py').exists())


class Paths(unittest.TestCase):
    # Match a drive letter at a word boundary, not the final 's:/' in an HTTPS URL.
    # A single slash/backslash also catches doubled backslashes in Python source text.
    ABSOLUTE = re.compile(r'(\b[A-Za-z]:[\\/]|/nfs-stor|/home/|/apps/|/scratch/)')

    def test_path_guard_accepts_urls_and_relative_paths(self):
        for text in ('https://github.com/molML/MoleculeACE/blob/revision/results.csv',
                     'http://example.org/data.csv', 'https://example.org/unrelated.csv',
                     'code/MoleculeACE/Data', '../jobs/manifest_pilot.csv', 'C:relative'):
            with self.subTest(text=text):
                self.assertIsNone(self.ABSOLUTE.search(text))

    def test_path_guard_still_rejects_absolute_filesystem_paths(self):
        for text in ('C:/Users/research/data.csv', 'z:/data/input.csv',
                     r'C:\Users\research\data.csv', r'z:\data\input.csv',
                     r'path = "C:\\Users\\research\\data.csv"',
                     r"root = r'C:\Users\research'",
                     '/home/research/data.csv', '/nfs-stor/research/data.csv',
                     '/apps/local/library', '/scratch/research/data.csv'):
            with self.subTest(text=text):
                self.assertIsNotNone(self.ABSOLUTE.search(text))

    def test_no_absolute_paths_in_code(self):
        for path in PY_FILES:
            if path.name == 'test_scope_compat.py':
                continue
            with self.subTest(file=path.name):
                self.assertIsNone(self.ABSOLUTE.search(path.read_text(encoding='utf-8')))

    def test_cluster_paths_only_in_the_job_config(self):
        for path in SHELL_FILES:
            with self.subTest(file=path.name):
                self.assertIsNone(re.search(r'/nfs-stor|/apps/', path.read_text(encoding='utf-8')))

    def test_jobs_do_not_use_bash_source_and_are_lf(self):
        for path in SHELL_FILES:
            with self.subTest(file=path.name):
                data = path.read_bytes()
                self.assertNotIn(b'BASH_SOURCE', data)
                self.assertNotIn(b'\r\n', data)


if __name__ == '__main__':
    unittest.main()
