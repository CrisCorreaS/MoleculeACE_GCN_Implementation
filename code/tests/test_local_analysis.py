"""Read-only local validation against synthetic saved evidence; no model imports."""

import csv
import hashlib
import shutil
import unittest

from aggregate_results import aggregate
from analyze_results import COMPARISON_FIELDS, TABLE1_FIELDS, build_tables
from helpers import CampaignTestCase
from MoleculeACE.benchmark.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    read_json,
    sha256_file,
)
from MoleculeACE.benchmark.campaign import CampaignError, default_project_root, task_dir
from validate_local_analysis import (
    _source_evidence,
    load_validated_tables,
    report_output_path,
)


class LocalValidation(CampaignTestCase):
    def setUp(self):
        super().setUp()
        original = default_project_root()
        shutil.copytree(
            original / "references" / "published",
            self.root / "references" / "published",
        )
        shutil.copy(
            original / "references" / "corrected_table1.csv", self.root / "references"
        )
        self.campaign = "gcn_pilot_localtest"
        self.manifest, self.tasks = self.make_campaign(
            self.campaign, "pilot", ["CHEMBL2835_Ki"], folds=[0, 1], max_epochs=20
        )
        (self.root / "recorded_source.py").write_bytes(
            b"# preserved synthetic source\n"
        )
        for task in self.tasks:
            path = task_dir(self.root, task) / "task_record.json"
            record = read_json(path)
            record["provenance"]["source_sha256"] = {
                "recorded_source.py": sha256_file(self.root / "recorded_source.py")
            }
            atomic_write_json(path, record)
        _, per_dataset = aggregate(self.root, self.manifest, self.campaign)
        table1, comparison, verified = build_tables(self.root, per_dataset)
        self.folder = self.root / "results" / self.campaign
        self.comparison_path = self.folder / "_analysis" / "comparison.csv"
        atomic_write_csv(self.comparison_path, comparison, COMPARISON_FIELDS)
        atomic_write_csv(
            self.folder / "_analysis" / "datasets_vs_table1.csv", table1, TABLE1_FIELDS
        )
        atomic_write_json(
            self.folder / "_analysis" / "analysis.json",
            dict(
                campaign=self.campaign,
                scientific=False,
                n_datasets=1,
                manifest_sha256=sha256_file(self.manifest),
                published_provenance_verified=verified,
            ),
        )

    def validate(self):
        return load_validated_tables(self.root, self.campaign, self.manifest)

    def test_recalculation_passes_without_changing_any_input_bytes(self):
        before = {
            p.relative_to(self.root): sha256_file(p)
            for p in self.root.rglob("*")
            if p.is_file()
        }
        result = self.validate()
        self.assertEqual(result["report"]["status"], "passed")
        self.assertEqual(result["report"]["n_tasks"], 2)
        self.assertEqual(len(result["comparison"]), 1)
        after = {
            p.relative_to(self.root): sha256_file(p)
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)

    def test_corrupted_task_prediction_is_rejected_by_hash(self):
        path = task_dir(self.root, self.tasks[0]) / "predictions.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(CampaignError, "Hash mismatch"):
            self.validate()

    def test_saved_summary_hash_is_checked(self):
        path = self.folder / "_summary" / "per_dataset.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(CampaignError, "summary hash mismatch"):
            self.validate()

    def test_duplicate_dataset_in_comparison_is_rejected(self):
        with self.comparison_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        atomic_write_csv(self.comparison_path, rows + rows, COMPARISON_FIELDS)
        with self.assertRaisesRegex(CampaignError, "Duplicate row identity"):
            self.validate()

    def test_changed_comparison_value_is_rejected(self):
        with self.comparison_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["rmse_mean"] = str(float(rows[0]["rmse_mean"]) + 0.1)
        atomic_write_csv(self.comparison_path, rows, COMPARISON_FIELDS)
        with self.assertRaisesRegex(CampaignError, "value differs from recalculation"):
            self.validate()

    def test_effective_configuration_is_rehashed(self):
        path = task_dir(self.root, self.tasks[0]) / "task_record.json"
        record = read_json(path)
        record["effective_config"]["epochs"] += 1
        atomic_write_json(path, record)
        with self.assertRaisesRegex(
            CampaignError, "Effective configuration hash mismatch"
        ):
            self.validate()

    def test_reports_cannot_replace_history_sources_or_existing_reports(self):
        for path in (
            self.comparison_path,
            self.root / "code" / "report.json",
            self.folder / "_analysis" / "new_report.json",
            self.root / "results" / "_local_preservation" / "new_report.json",
            self.root / "results" / "_local_analysis" / "new_report.json",
        ):
            with self.subTest(path=path), self.assertRaises(CampaignError):
                report_output_path(self.root, path)
        path = self.root / "results" / "validation" / "new_report.json"
        self.assertEqual(report_output_path(self.root, path), path)
        atomic_write_json(path, {})
        with self.assertRaisesRegex(CampaignError, "overwrite"):
            report_output_path(self.root, path)

    def test_lf_transport_diagnostic_is_separate_and_does_not_rewrite_source(self):
        path = self.root / "recorded_source.py"
        lf = b"# preserved synthetic source\n"
        crlf = lf.replace(b"\n", b"\r\n")
        path.write_bytes(crlf)
        evidence = _source_evidence(
            self.root, {"recorded_source.py": hashlib.sha256(lf).hexdigest()}
        )
        self.assertEqual(evidence["byte_matches"], 0)
        self.assertEqual(evidence["lf_only_matches"], 1)
        self.assertNotEqual(
            evidence["files"][0]["byte_sha256"], evidence["files"][0]["recorded_sha256"]
        )
        self.assertEqual(path.read_bytes(), crlf)
        path.write_bytes(crlf + b"# substantive change\r\n")
        with self.assertRaisesRegex(CampaignError, "differs beyond CRLF"):
            _source_evidence(
                self.root, {"recorded_source.py": hashlib.sha256(lf).hexdigest()}
            )


if __name__ == "__main__":
    unittest.main()
