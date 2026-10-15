"""Check status listing against malformed synthetic evidence without training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import task_status
from MoleculeACE.benchmark.campaign import CampaignError, task_dir


class TaskStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="task_status_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "code" / "MoleculeACE").mkdir(parents=True)
        self.tasks = [
            dict(task_id=i, dataset=f"dataset_{i}", fold=0, campaign="status_test")
            for i in range(2)
        ]
        for task in self.tasks:
            folder = task_dir(self.root, task)
            folder.mkdir(parents=True)
            (folder / "status.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            self.write_record(task, self.completed_record())

    @staticmethod
    def completed_record():
        return {
            "metrics": {"rmse": 0.25, "rmse_cliff": 0.5},
            "training": {
                "best_epoch": 4,
                "epochs_run": 14,
                "stop_reason": "early_stopping",
            },
            "timing": {"total_seconds": 12.5},
        }

    def write_record(self, task, value):
        path = task_dir(self.root, task) / "task_record.json"
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_listing(self, validate=False):
        args = [
            "task_status.py",
            "--manifest",
            str(self.root / "manifest.csv"),
            "--campaign",
            "status_test",
            "--project-root",
            str(self.root),
        ]
        if validate:
            args.append("--validate")
        output = io.StringIO()
        with (
            patch("sys.argv", args),
            patch.object(task_status, "load_manifest", return_value=self.tasks),
        ):
            with contextlib.redirect_stdout(output):
                status = task_status.main()
        return status, output.getvalue()

    def assert_invalid_then_completed(self):
        status, output = self.run_listing()
        self.assertEqual(status, 1)
        self.assertIn("invalid=1", output)
        self.assertIn("completed=1", output)
        self.assertIn("(total 2)", output)
        self.assertIn("dataset_1", output)
        self.assertIn("rmse=0.250 rmse_cliff=0.500 best_epoch=4/14", output)

    def test_malformed_json_does_not_hide_later_valid_task(self):
        for filename in ("status.json", "task_record.json"):
            with self.subTest(filename=filename):
                folder = task_dir(self.root, self.tasks[0])
                (folder / "status.json").write_text(
                    '{"status": "completed"}', encoding="utf-8"
                )
                self.write_record(self.tasks[0], self.completed_record())
                (folder / filename).write_text("{broken", encoding="utf-8")
                self.assert_invalid_then_completed()

    def test_missing_fields_are_invalid_and_listing_continues(self):
        for missing in ("metrics", "training", "timing"):
            with self.subTest(missing=missing):
                record = self.completed_record()
                del record[missing]
                self.write_record(self.tasks[0], record)
                self.assert_invalid_then_completed()
        record = self.completed_record()
        del record["metrics"]["rmse"]
        self.write_record(self.tasks[0], record)
        self.assert_invalid_then_completed()

    def test_invalid_json_shapes_and_numeric_types_are_reported(self):
        for record in (
            [],
            {"metrics": []},
            dict(self.completed_record(), timing={"total_seconds": "slow"}),
        ):
            with self.subTest(record=record):
                self.write_record(self.tasks[0], record)
                self.assert_invalid_then_completed()

    def test_missing_or_invalid_status_fields_are_reported(self):
        path = task_dir(self.root, self.tasks[0]) / "status.json"
        for record in ([], {}, {"status": 1}, {"status": "unrecognised"}):
            with self.subTest(record=record):
                path.write_text(json.dumps(record), encoding="utf-8")
                self.assert_invalid_then_completed()

    def test_none_cliff_metric_is_a_valid_summary(self):
        record = self.completed_record()
        record["metrics"]["rmse_cliff"] = None
        self.write_record(self.tasks[0], record)
        status, output = self.run_listing()
        self.assertEqual(status, 0)
        self.assertIn("rmse_cliff=None", output)

    def test_missing_completed_record_is_invalid(self):
        (task_dir(self.root, self.tasks[0]) / "task_record.json").unlink()
        self.assert_invalid_then_completed()

    def test_expected_io_error_does_not_hide_later_valid_task(self):
        original = task_status.read_json
        first = task_dir(self.root, self.tasks[0]) / "task_record.json"

        def read(path):
            if path == first:
                raise PermissionError("synthetic read failure")
            return original(path)

        with patch.object(task_status, "read_json", side_effect=read):
            self.assert_invalid_then_completed()

    def test_strict_validation_failure_does_not_hide_later_valid_task(self):
        with patch.object(
            task_status,
            "validate_task_outputs",
            side_effect=[CampaignError("bad hash"), None],
        ) as check:
            status, output = self.run_listing(validate=True)
        self.assertEqual(status, 1)
        self.assertEqual(check.call_count, 2)
        self.assertIn("invalid=1", output)
        self.assertIn("completed=1", output)
        self.assertIn("bad hash", output)

    def test_unexpected_exception_propagates(self):
        with patch.object(
            task_status,
            "validate_task_outputs",
            side_effect=RuntimeError("programming failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programming failure"):
                self.run_listing(validate=True)
        with patch.object(
            task_status, "read_json", side_effect=RuntimeError("reader bug")
        ):
            with self.assertRaisesRegex(RuntimeError, "reader bug"):
                self.run_listing()

    def test_valid_completed_failed_and_pending_states_return_zero(self):
        status, output = self.run_listing()
        self.assertEqual(status, 0)
        self.assertIn("completed=2", output)
        first = task_dir(self.root, self.tasks[0]) / "status.json"
        first.write_text(
            '{"status": "failed", "error": "training failure"}', encoding="utf-8"
        )
        status, output = self.run_listing()
        self.assertEqual(status, 0)
        self.assertIn("failed=1", output)
        self.assertIn("training failure", output)
        first.unlink()
        status, output = self.run_listing()
        self.assertEqual(status, 0)
        self.assertIn("pending=1", output)


if __name__ == "__main__":
    unittest.main()
