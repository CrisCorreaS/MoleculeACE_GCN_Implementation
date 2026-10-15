"""Test strict metric handling, JSON encoding and atomic artefact writes.

Metric cases include finite/aligned values, binary cliff masks and the absence
of cliff molecules. File cases write only temporary fixtures. Package import
requires NumPy, pandas and PyYAML; no Torch, checkpoint loading or training
occurs. The upstream metric comparison supplies an explicit cliff mask.

From the repository root:
    python -B tools/run_scientific_tests.py --selection test_artifacts

See docs/reproduction.md for environment and temporary-directory options.
"""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from MoleculeACE.benchmark.artifacts import (atomic_write_bytes, atomic_write_json, canonical_json_sha256,
                                              fold_metrics, read_json)


class Metrics(unittest.TestCase):
    def test_rmse_and_cliff_rmse(self):
        m = fold_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 4.0], [0, 0, 1, 1])
        self.assertEqual((m['n_test'], m['n_cliff']), (4, 2))
        self.assertAlmostEqual(m['rmse'], 1.0)                 # sqrt(4/4)
        self.assertAlmostEqual(m['rmse_cliff'], math.sqrt(2))  # sqrt(4/2)

    def test_no_cliffs_is_none_not_zero(self):
        self.assertIsNone(fold_metrics([1.0, 2.0], [1.5, 2.5], [0, 0])['rmse_cliff'])

    def test_invalid_inputs(self):
        for args in [([1.0], [float('nan')], [0]), ([1.0], [1.0, 2.0], [0]), ([], [], []), ([1.0], [1.0], [2])]:
            with self.assertRaises(ValueError):
                fold_metrics(*args)

    def test_matches_upstream_numpy_functions(self):
        try:
            from MoleculeACE.benchmark.utils import calc_rmse, calc_cliff_rmse
        except ImportError as exc:
            self.skipTest(f'numpy/pandas/yaml unavailable: {exc}')
        y, p, c = [0.1, -1.2, 2.3, 0.7, -0.4], [0.3, -1.0, 1.1, 0.9, -0.9], [1, 0, 1, 0, 1]
        m = fold_metrics(y, p, c)
        self.assertAlmostEqual(m['rmse'], float(calc_rmse(y, p)), places=12)
        self.assertAlmostEqual(m['rmse_cliff'], float(calc_cliff_rmse(p, y, cliff_mols_test=c)), places=12)
        self.assertIsNone(calc_cliff_rmse(p, y, cliff_mols_test=[0] * 5))


class AtomicFiles(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_no_temporary_leftovers_and_replacement(self):
        path = self.dir / 'sub' / 'a.json'
        atomic_write_json(path, {'a': 1})
        atomic_write_json(path, {'a': 2})
        self.assertEqual(read_json(path), {'a': 2})
        self.assertEqual([p.name for p in path.parent.iterdir()], ['a.json'])

    def test_nonfinite_json_is_rejected_on_write_and_read(self):
        with self.assertRaises(ValueError):
            atomic_write_json(self.dir / 'x.json', {'rmse': float('nan')})
        self.assertFalse((self.dir / 'x.json').exists())
        atomic_write_bytes(self.dir / 'y.json', b'{"rmse": NaN}')
        with self.assertRaises(ValueError):
            read_json(self.dir / 'y.json')

    def test_canonical_hash_ignores_key_order(self):
        self.assertEqual(canonical_json_sha256({'a': 1, 'b': [1, 2]}), canonical_json_sha256({'b': [1, 2], 'a': 1}))
        self.assertNotEqual(canonical_json_sha256({'a': 1}), canonical_json_sha256({'a': 2}))


if __name__ == '__main__':
    unittest.main()
