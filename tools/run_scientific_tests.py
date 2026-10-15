"""Run an explicit scientific test selection from the active submission code.

Cover input data, metrics, stopping rules, aggregation, saved evidence and
figures. Select four scientific classes from the mixed historical GCN test
module, excluding its GPU training class and cluster infrastructure tests.
Nothing is imported or executed from provenance/training_source/.
Missing optional packages are reported as skips; no packages are installed.
Use --selection to run explicitly selected checks in the historical environment
without importing the modern paper-figure module.

Usage: python -B tools/run_scientific_tests.py
Temporary fixtures use the system temporary directory unless --temporary-dir is set.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Sequence

LOGGER = logging.getLogger(__name__)
SCIENTIFIC_SELECTION = (
    "test_data.Datasets.test_all_match_corrected_table1_and_label_scale",
    "test_artifacts",
    "test_training_protocol",
    "test_manifest",
    "test_aggregation",
    "test_analysis",
    "test_local_analysis",
    "test_current_science",
    "test_training_source_roots",
    "test_task_status",
    "test_ciai_gcn.Versions",
    "test_ciai_gcn.Featurization",
    "test_ciai_gcn.Folds",
    "test_ciai_gcn.CliffLabels",
)
ALLOWED_SELECTION = SCIENTIFIC_SELECTION + ("test_plot_paper_figures",)
OPTIONAL_PACKAGES = {
    "numpy",
    "pandas",
    "yaml",
    "matplotlib",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "rdkit",
    "sklearn",
    "Levenshtein",
    "tqdm",
    "scipy",
}


class UnavailableTest(unittest.TestCase):
    """Expose an unavailable optional test selection as an explicit skip."""

    def __init__(self, selection: str, reason: str) -> None:
        super().__init__("runTest")
        self.selection = selection
        self.reason = reason

    def id(self) -> str:
        """Return the unavailable selection's identity in the test report."""
        return self.selection

    def runTest(self) -> None:
        """Record the dependency limitation without claiming an assertion passed."""
        self.skipTest(self.reason)


class ScientificTestResult(unittest.TextTestResult):
    """Report inherited dependency messages in British English."""

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        """Preserve the skipped outcome while translating its historical prefix."""
        reason = reason.replace(
            "PENDIENTE EN CIAI:", "Unavailable in this environment:"
        )
        super().addSkip(test, reason)


def build_suite(
    project_root: Path, selections: Optional[Sequence[str]] = None
) -> unittest.TestSuite:
    """Load explicitly selected active tests and report missing optional dependencies.

    Parameters
    ----------
    project_root : Path
        Submission root containing active ``code/`` and ``code/tests/``.
    selections : sequence of str, optional
        Explicitly selected test names. The default runs the complete scientific selection.

    Returns
    -------
    unittest.TestSuite
        Explicit selection, including dependency-related skip records.
    """
    code_root = (project_root / "code").resolve()
    test_root = code_root / "tests"
    if not test_root.is_dir():
        raise ValueError(f"Active test directory is missing: {test_root}")
    sys.path[:0] = [str(code_root), str(test_root)]
    selected_names: List[str] = (
        list(SCIENTIFIC_SELECTION) if selections is None else list(selections)
    )
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("Test selections must be non-empty and contain no duplicates")
    if any(name not in ALLOWED_SELECTION for name in selected_names):
        raise ValueError("Only the supported scientific test selections are permitted")
    suite = unittest.TestSuite()
    if selections is None:
        if (test_root / "test_plot_paper_figures.py").is_file():
            selected_names.append("test_plot_paper_figures")
        else:
            suite.addTest(
                UnavailableTest(
                    "test_plot_paper_figures",
                    "The supported paper-figure test module is not present in this copy",
                )
            )
    for selection in selected_names:
        module_name = selection.split(".", 1)[0]
        if not (test_root / f"{module_name}.py").is_file():
            raise ValueError(f"Required active test module is missing: {module_name}")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = (error.name or "").split(".", 1)[0]
            if missing not in OPTIONAL_PACKAGES:
                raise
            suite.addTest(
                UnavailableTest(selection, f"Missing optional package: {error.name}")
            )
            continue
        if Path(module.__file__).resolve().parent != test_root:
            raise ValueError(f"Test imported outside active code: {module_name}")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(selection))
    return suite


def main() -> int:
    """Run the selected CPU tests and report failures and skips separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--temporary-dir",
        type=Path,
        help="Test fixture directory; defaults to the system temporary directory",
    )
    parser.add_argument(
        "--selection",
        nargs="+",
        choices=ALLOWED_SELECTION,
        help="Run only these supported checks; omit for the complete scientific suite",
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.dont_write_bytecode = True
    if arguments.temporary_dir is not None:
        temporary_root = arguments.temporary_dir.resolve()
        temporary_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temporary_root)
    try:
        suite = build_suite(arguments.project_root.resolve(), arguments.selection)
    except (OSError, ValueError, ImportError) as error:
        LOGGER.error("Cannot load the scientific suite: %s", error)
        return 2
    LOGGER.info("Active-code suite; GPU training and infrastructure tests excluded")
    result = unittest.TextTestRunner(
        verbosity=1 if arguments.quiet else 2, resultclass=ScientificTestResult
    ).run(suite)
    for test, reason in result.skipped:
        LOGGER.warning("SKIP %s: %s", test.id(), reason)
    LOGGER.info(
        "Ran %d tests: %d skipped, %d failures, %d errors",
        result.testsRun,
        len(result.skipped),
        len(result.failures),
        len(result.errors),
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
