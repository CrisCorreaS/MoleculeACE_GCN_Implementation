"""Exercise source-only and task-record-backed checks using temporary fixtures.

No recorded project source is imported or modified. Run with:
python -B -m unittest discover -s tools -p test_source_verification.py -v
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

from check_documentation_only import (
    APPROVED_ARCHIVE_ONLY,
    APPROVED_SNAPSHOT_ONLY,
    EXPECTED_TASKS,
    check_sources,
    main,
    source_sha256_lf,
)


class SourceVerification(unittest.TestCase):
    """Verify manifest handling without depending on private campaign artefacts."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="source_verification_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "provenance" / "training_source"
        self.manifest_path = self.root / "provenance" / "source_manifest.json"
        self.campaign = "gcn_full_v1"
        names = (
            ["code/run_benchmark.py"]
            + [f"code/source_{index:02d}.py" for index in range(30)]
            + sorted(APPROVED_SNAPSHOT_ONLY)
        )
        self.source_hashes: Dict[str, str] = {}
        for index, relative in enumerate(names):
            contents = (f'"""Original documentation."""\nVALUE = {index}\n').encode(
                "utf-8"
            )
            frozen = self.snapshot / relative
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(contents)
            self.source_hashes[relative] = hashlib.sha256(contents).hexdigest()
            if relative not in APPROVED_SNAPSHOT_ONLY:
                active = self.root / relative
                active.parent.mkdir(parents=True, exist_ok=True)
                active.write_bytes(contents)
        self.manifest = {
            "schema_version": 1,
            "campaign": self.campaign,
            "task_count": EXPECTED_TASKS,
            "source_sha256": self.source_hashes,
        }
        self.write_manifest()

    def write_manifest(self, value: Optional[Dict[str, Any]] = None) -> None:
        """Write one synthetic manifest for a verification case."""
        self.manifest_path.write_text(
            json.dumps(self.manifest if value is None else value), encoding="utf-8"
        )

    def write_records(self, count: int = EXPECTED_TASKS) -> None:
        """Create synthetic task metadata without predictions or model files."""
        record = {
            "task": {"campaign": self.campaign},
            "provenance": {"source_sha256": self.source_hashes},
        }
        for index in range(count):
            path = (
                self.root
                / "results"
                / self.campaign
                / "CHEMBL_fixture"
                / f"fold_{index}"
                / "task_record.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")

    def check(self) -> Dict[str, Any]:
        """Check the temporary source distribution."""
        return check_sources(self.root, self.snapshot, self.campaign)

    def test_source_only_mode_preserves_all_source_checks(self) -> None:
        report = self.check()
        self.assertEqual(report["mode"], "sources_only")
        self.assertEqual(report["task_records"], 0)
        self.assertEqual(report["manifest_task_count"], 150)
        self.assertEqual(report["snapshot_hash_matches"], 38)
        self.assertEqual(report["active_executable_matches"], 31)
        self.assertEqual(set(report["snapshot_only"]), APPROVED_SNAPSHOT_ONLY)
        self.assertEqual(report["archive_only"], [])
        self.assertEqual(report["errors"], [])

    def test_complete_records_must_agree_with_manifest(self) -> None:
        self.write_records()
        report = self.check()
        self.assertEqual(report["mode"], "task_records")
        self.assertEqual(report["task_records"], 150)
        self.assertEqual(report["snapshot_hash_matches"], 38)
        self.assertEqual(report["archive_only"], [])
        self.assertEqual(report["errors"], [])

    def test_archive_only_allowlist_contains_only_launcher_test(self) -> None:
        self.assertEqual(
            APPROVED_ARCHIVE_ONLY, frozenset({"code/tests/test_launcher.py"})
        )

    def test_cli_reports_archive_only_without_claiming_all_hashes_match(self) -> None:
        relative = "code/tests/test_launcher.py"
        (self.snapshot / relative).unlink()
        arguments = ["check_documentation_only.py", "--project-root", str(self.root)]
        with patch("sys.argv", arguments):
            with self.assertLogs("check_documentation_only", level="INFO") as logs:
                self.assertEqual(main(), 0)
        output = "\n".join(logs.output)
        self.assertIn("archive-only (identity retained;", output)
        self.assertIn(relative, output)
        self.assertIn("37/38 snapshot hashes match", output)
        self.assertIn("1 archive-only files", output)
        self.assertNotIn("38/38 snapshot hashes match", output)

    def test_source_only_mode_reports_missing_launcher_as_archive_only(self) -> None:
        relative = "code/tests/test_launcher.py"
        (self.snapshot / relative).unlink()
        report = self.check()
        self.assertEqual(report["mode"], "sources_only")
        self.assertEqual(report["recorded_sources"], 38)
        self.assertEqual(report["snapshot_hash_matches"], 37)
        self.assertEqual(report["active_executable_matches"], 31)
        self.assertEqual(report["archive_only"], [relative])
        self.assertEqual(
            set(report["snapshot_only"]), APPROVED_SNAPSHOT_ONLY - {relative}
        )
        self.assertEqual(report["errors"], [])
        saved_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["source_sha256"], self.source_hashes)

    def test_complete_records_require_the_archive_only_launcher_snapshot(self) -> None:
        relative = "code/tests/test_launcher.py"
        (self.snapshot / relative).unlink()
        self.write_records()
        report = self.check()
        self.assertEqual(report["mode"], "task_records")
        self.assertEqual(report["snapshot_hash_matches"], 37)
        self.assertEqual(report["archive_only"], [])
        self.assertEqual(report["errors"], [f"Missing snapshot source: {relative}"])

    def test_other_missing_snapshots_fail_in_both_modes(self) -> None:
        for with_records in (False, True):
            if with_records:
                self.write_records()
            for relative in (
                "code/run_benchmark.py",
                "code/tests/test_requeue.py",
            ):
                with self.subTest(with_records=with_records, path=relative):
                    path = self.snapshot / relative
                    original = path.read_bytes()
                    path.unlink()
                    try:
                        report = self.check()
                        self.assertIn(
                            f"Missing snapshot source: {relative}", report["errors"]
                        )
                        self.assertEqual(report["archive_only"], [])
                    finally:
                        path.write_bytes(original)

    def test_present_launcher_must_match_its_hash_in_both_modes(self) -> None:
        relative = "code/tests/test_launcher.py"
        path = self.snapshot / relative
        path.write_bytes(path.read_bytes() + b"# changed snapshot\n")
        for with_records in (False, True):
            with self.subTest(with_records=with_records):
                if with_records:
                    self.write_records()
                report = self.check()
                self.assertIn(
                    f"Snapshot hash differs from source manifest: {relative}",
                    report["errors"],
                )
                self.assertEqual(report["archive_only"], [])

    def test_active_launcher_cannot_bypass_executable_verification(self) -> None:
        relative = "code/tests/test_launcher.py"
        frozen = self.snapshot / relative
        active = self.root / relative
        active.parent.mkdir(parents=True, exist_ok=True)
        active.write_bytes(frozen.read_bytes())
        frozen.unlink()
        report = self.check()
        self.assertIn(
            "Cannot verify active source without its archive-only "
            f"snapshot: {relative}",
            report["errors"],
        )

    def test_missing_manifest_fails_with_or_without_records(self) -> None:
        self.manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing source manifest"):
            self.check()
        self.write_records()
        with self.assertRaisesRegex(ValueError, "Missing source manifest"):
            self.check()

    def test_partial_records_cannot_fall_back_to_source_only(self) -> None:
        (self.snapshot / "code/tests/test_launcher.py").unlink()
        self.write_records(1)
        with self.assertRaisesRegex(ValueError, "Partial task evidence"):
            self.check()

    def test_record_outside_expected_dataset_names_is_not_ignored(self) -> None:
        path = self.root / "results" / self.campaign / "other" / "task_record.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Partial task evidence"):
            self.check()

    def test_record_source_mismatch_is_rejected(self) -> None:
        self.write_records()
        path = next((self.root / "results").rglob("task_record.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["provenance"]["source_sha256"]["code/run_benchmark.py"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not match the source manifest"):
            self.check()

    def test_record_campaign_mismatch_is_rejected(self) -> None:
        self.write_records()
        path = next((self.root / "results").rglob("task_record.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["task"]["campaign"] = "gcn_full_other"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Task record campaign differs"):
            self.check()

    def test_tampered_manifest_hash_fails_snapshot_verification(self) -> None:
        changed = json.loads(json.dumps(self.manifest))
        changed["source_sha256"]["code/run_benchmark.py"] = "0" * 64
        self.write_manifest(changed)
        report = self.check()
        self.assertIn(
            "Snapshot hash differs from source manifest: code/run_benchmark.py",
            report["errors"],
        )

    def test_invalid_manifest_metadata_is_rejected(self) -> None:
        for field, value in (
            ("schema_version", 2),
            ("schema_version", True),
            ("task_count", 149),
            ("task_count", 150.0),
            ("campaign", "gcn_full_other"),
            ("unexpected_field", "not permitted"),
        ):
            with self.subTest(field=field, value=value):
                self.write_manifest(dict(self.manifest, **{field: value}))
                with self.assertRaises(ValueError):
                    self.check()

    def test_missing_identity_and_unsafe_source_path_are_rejected(self) -> None:
        for unsafe in (None, "../outside.py", "code/../outside.py", "code\\file.py"):
            with self.subTest(path=unsafe):
                changed = json.loads(json.dumps(self.manifest))
                digest = changed["source_sha256"].pop("code/run_benchmark.py")
                if unsafe is not None:
                    changed["source_sha256"][unsafe] = digest
                self.write_manifest(changed)
                with self.assertRaises(ValueError):
                    self.check()

    def test_duplicate_json_fields_are_rejected(self) -> None:
        text = self.manifest_path.read_text(encoding="utf-8")
        self.manifest_path.write_text(
            text.replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            self.check()

    def test_snapshot_hashes_remain_required_in_both_modes(self) -> None:
        path = self.snapshot / "code/run_benchmark.py"
        path.write_bytes(path.read_bytes() + b"# changed snapshot\n")
        for with_records in (False, True):
            with self.subTest(with_records=with_records):
                if with_records:
                    self.write_records()
                self.assertTrue(self.check()["errors"])

    def test_unapproved_missing_active_source_fails(self) -> None:
        (self.root / "code/run_benchmark.py").unlink()
        report = self.check()
        self.assertIn(
            "Missing required active recorded training source: code/run_benchmark.py",
            report["errors"],
        )
        self.assertNotIn("code/run_benchmark.py", report["snapshot_only"])

    def test_documentation_edits_pass_but_executable_edits_fail(self) -> None:
        path = self.root / "code/run_benchmark.py"
        original = path.read_text(encoding="utf-8")
        path.write_text(
            original.replace("Original documentation", "Reviewed documentation")
            + "# Explanation only.\n",
            encoding="utf-8",
        )
        self.assertEqual(self.check()["errors"], [])
        path.write_text(original + "VALUE = 999\n", encoding="utf-8")
        self.assertIn(
            "Executable syntax differs from snapshot: code/run_benchmark.py",
            self.check()["errors"],
        )

    def write_revision(self, change_source: bool = True) -> Path:
        """Register one exact synthetic post-training correction."""
        relative = "code/run_benchmark.py"
        active = self.root / relative
        if change_source:
            active.write_bytes(b'"""Current documentation."""\nVALUE = 999\n')
        revision = {
            "schema_version": 1,
            "historical_campaign": self.campaign,
            "sources": {
                relative: {
                    "historical_sha256": self.source_hashes[relative],
                    "current_sha256_lf": source_sha256_lf(active.read_bytes()),
                    "change": "Change a synthetic value for regression testing.",
                }
            },
        }
        path = self.root / "current_source_revision.json"
        path.write_text(json.dumps(revision), encoding="utf-8")
        return path

    def check_revision(self, path: Path) -> Dict[str, Any]:
        """Use the explicit revision route without changing the default check."""
        return check_sources(
            self.root, self.snapshot, self.campaign, revision_manifest=path
        )

    def test_revision_requires_explicit_opt_in_and_keeps_historical_checks(
        self,
    ) -> None:
        path = self.write_revision()
        for records in (False, True):
            with self.subTest(records=records):
                if records:
                    self.write_records()
                self.assertIn(
                    "Executable syntax differs from snapshot: code/run_benchmark.py",
                    self.check()["errors"],
                )
                report = self.check_revision(path)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["snapshot_hash_matches"], 38)
                self.assertEqual(report["active_executable_matches"], 30)
                self.assertEqual(
                    report["current_revision_matches"], ["code/run_benchmark.py"]
                )

    def test_revision_rejects_an_additional_change_in_a_listed_file(self) -> None:
        path = self.write_revision()
        active = self.root / "code/run_benchmark.py"
        active.write_bytes(active.read_bytes().replace(b"999", b"1000"))
        self.assertIn(
            "Current source differs from explicit revision: code/run_benchmark.py",
            self.check_revision(path)["errors"],
        )

    def test_revision_rejects_unlisted_executable_changes(self) -> None:
        path = self.write_revision()
        (self.root / "code/source_00.py").write_bytes(b"VALUE = 500\n")
        self.assertIn(
            "Executable syntax differs from snapshot: code/source_00.py",
            self.check_revision(path)["errors"],
        )

    def test_revision_allows_only_crlf_transport_for_listed_sources(self) -> None:
        path = self.write_revision()
        active = self.root / "code/run_benchmark.py"
        original = active.read_bytes()
        active.write_bytes(original.replace(b"\n", b"\r\n"))
        self.assertEqual(self.check_revision(path)["errors"], [])
        for changed in (
            original.replace(b"\n", b"\r"),
            b"\xef\xbb\xbf" + original,
            original + b"# Unregistered change.\n",
        ):
            with self.subTest(contents=changed):
                active.write_bytes(changed)
                self.assertTrue(self.check_revision(path)["errors"])

    def test_revision_cannot_bypass_missing_or_modified_sources(self) -> None:
        path = self.write_revision()
        active = self.root / "code/run_benchmark.py"
        frozen = self.snapshot / "code/run_benchmark.py"
        original_active = active.read_bytes()
        original_frozen = frozen.read_bytes()
        active.unlink()
        self.assertTrue(self.check_revision(path)["errors"])
        active.write_bytes(original_active)
        frozen.unlink()
        self.assertTrue(self.check_revision(path)["errors"])
        frozen.write_bytes(original_frozen + b"# Altered history.\n")
        self.assertTrue(self.check_revision(path)["errors"])

    def test_revision_cannot_extend_archive_only_exception(self) -> None:
        path = self.write_revision()
        (self.snapshot / "code/source_00.py").unlink()
        self.assertIn(
            "Missing snapshot source: code/source_00.py",
            self.check_revision(path)["errors"],
        )
        self.write_records()
        (self.snapshot / "code/tests/test_launcher.py").unlink()
        self.assertIn(
            "Missing snapshot source: code/tests/test_launcher.py",
            self.check_revision(path)["errors"],
        )

    def test_revision_rejects_unnecessary_documentation_only_entries(self) -> None:
        path = self.write_revision(change_source=False)
        self.assertIn(
            "Revision lists no executable change: code/run_benchmark.py",
            self.check_revision(path)["errors"],
        )

    def test_revision_metadata_is_strict_and_tied_to_history(self) -> None:
        path = self.write_revision()
        original = json.loads(path.read_text(encoding="utf-8"))
        candidates = [
            dict(original, historical_campaign="other"),
            dict(original, schema_version=True),
            dict(original, sources={}),
            dict(original, unexpected=True),
        ]
        for key, value in (
            ("historical_sha256", "0" * 64),
            ("current_sha256_lf", "not-a-hash"),
            ("change", ""),
        ):
            altered = json.loads(json.dumps(original))
            altered["sources"]["code/run_benchmark.py"][key] = value
            candidates.append(altered)
        for name in ("../outside.py", "code/tests/test_launcher.py"):
            altered = json.loads(json.dumps(original))
            entry = altered["sources"].pop("code/run_benchmark.py")
            altered["sources"][name] = entry
            candidates.append(altered)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.check_revision(path)

    def test_revision_duplicate_fields_are_rejected(self) -> None:
        path = self.write_revision()
        text = path.read_text(encoding="utf-8").replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            self.check_revision(path)

    def test_revision_cli_distinguishes_current_from_historical_sources(self) -> None:
        path = self.write_revision()
        arguments = [
            "check_documentation_only.py",
            "--project-root",
            str(self.root),
            "--revision-manifest",
            str(path),
        ]
        with patch("sys.argv", arguments):
            with self.assertLogs("check_documentation_only", level="INFO") as logs:
                self.assertEqual(main(), 0)
        output = "\n".join(logs.output)
        self.assertIn("POST-TRAINING REVISION (not historical training code)", output)
        self.assertIn("30 active files have unchanged executable syntax", output)
        self.assertIn("1 exact current revisions", output)


if __name__ == "__main__":
    unittest.main()
