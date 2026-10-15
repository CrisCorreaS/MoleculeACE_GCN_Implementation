"""Check training import origins without starting a model or writing task results.

Real temporary source copies exercise entry-point, package and cached-module
mismatches in isolation. No historical snapshot is imported or executed.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_benchmark as runner

ROOT = Path(runner.__file__).resolve().parents[1]


class TrainingSourceRoots(unittest.TestCase):
    """Accept coherent copies and reject mixed checkouts before output creation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source-roots-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.first = self.copy_sources("first")
        self.second = self.copy_sources("second")

    def copy_sources(self, name):
        """Copy only active Python sources needed for a Torch-free preflight."""
        target = self.workspace / name
        sources = [ROOT / "code/run_benchmark.py"]
        sources.extend((ROOT / "code/MoleculeACE").rglob("*.py"))
        for source in sources:
            destination = target / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return target

    def child(self, entry_root, import_root, selected_root, operation="validate"):
        """Import a real entry point with an independently selected package root."""
        code = """
import importlib.util, json, sys
from pathlib import Path
entry_root, import_root, selected_root, operation = sys.argv[1:]
sys.path.insert(0, str(Path(import_root) / 'code'))
spec = importlib.util.spec_from_file_location(
    'source_root_runner', Path(entry_root) / 'code/run_benchmark.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
try:
    if operation == 'run':
        runner.run(Path(selected_root), Path(selected_root) / 'missing.csv', 'new', 0)
    else:
        root = runner.validate_training_source_root(Path(selected_root))
        print(json.dumps({'root': str(root), 'torch_loaded': 'torch' in sys.modules,
                          'package_file': sys.modules['MoleculeACE'].__file__}))
except runner.CampaignError as error:
    print(str(error), file=sys.stderr)
    sys.exit(2)
"""
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                code,
                str(entry_root),
                str(import_root),
                str(selected_root),
                operation,
            ],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_no_results(self):
        for root in (self.first, self.second):
            self.assertFalse((root / "results").exists())
            self.assertEqual({item.name for item in root.iterdir()}, {"code"})

    def test_active_checkout_and_resolved_equivalent_path_are_accepted(self):
        selected = ROOT / "code" / ".."
        self.assertEqual(runner.validate_training_source_root(selected), ROOT)

    def test_clean_copy_accepts_its_own_imports_without_loading_torch(self):
        result = self.child(self.first, self.first, self.first / "code" / "..")
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(Path(evidence["root"]), self.first)
        self.assertEqual(
            Path(evidence["package_file"]),
            self.first / "code/MoleculeACE/__init__.py",
        )
        self.assertFalse(evidence["torch_loaded"])
        self.assert_no_results()

    def test_wrong_project_root_is_rejected_before_manifest_or_output(self):
        result = self.child(self.first, self.first, self.second, "run")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Training source-root mismatch for run_benchmark", result.stderr)
        self.assertNotIn("missing.csv", result.stderr)
        self.assert_no_results()

    def test_foreign_package_is_rejected_even_when_entry_point_matches(self):
        result = self.child(self.first, self.second, self.first, "run")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("MoleculeACE import origin", result.stderr)
        self.assertIn(str(self.second), result.stderr)
        self.assert_no_results()

    def test_foreign_loaded_scientific_module_is_rejected(self):
        name = "MoleculeACE.models.training"
        source = self.second / "code/MoleculeACE/models/training.py"
        spec = importlib.util.spec_from_file_location(name, source)
        foreign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foreign)
        with mock.patch.dict(sys.modules, {name: foreign}):
            with self.assertRaisesRegex(runner.CampaignError, name + " import origin"):
                runner.validate_training_source_root(ROOT)
        self.assert_no_results()

    def test_foreign_package_search_path_is_rejected_before_lazy_imports(self):
        import MoleculeACE.models as models

        search_path = [
            str(ROOT / "code/MoleculeACE/models"),
            str(self.second / "code/MoleculeACE/models"),
        ]
        with mock.patch.object(models, "__path__", search_path):
            with self.assertRaisesRegex(runner.CampaignError, "package search path"):
                runner.validate_training_source_root(ROOT)
        self.assert_no_results()

    def test_cached_foreign_function_is_rejected_with_local_module_registered(self):
        name = "MoleculeACE.benchmark.campaign"
        source = self.second / "code/MoleculeACE/benchmark/campaign.py"
        spec = importlib.util.spec_from_file_location(name, source)
        foreign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foreign)
        with mock.patch.object(runner, "load_manifest", foreign.load_manifest):
            with self.assertRaisesRegex(
                runner.CampaignError, "load_manifest function source"
            ):
                runner.validate_training_source_root(ROOT)
        self.assert_no_results()

    def test_direct_train_call_rejects_foreign_root_before_gpu_import(self):
        with mock.patch("builtins.__import__") as import_module:
            with self.assertRaisesRegex(runner.CampaignError, "run_benchmark"):
                runner.train_task(self.second, {}, self.second / "results/task")
        import_module.assert_not_called()
        self.assert_no_results()


if __name__ == "__main__":
    unittest.main(verbosity=2)
