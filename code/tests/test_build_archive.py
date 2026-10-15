"""Check the complete archive's contents, checksums and non-overwrite guards.

All fixtures and archives are temporary. No completed campaign or original
project file is modified, and symbolic-link cases run only when supported.
"""

import hashlib
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from build_archive import ARCHIVE_FILENAME, MANIFEST_NAME, build_archive, main


class ArchiveTests(unittest.TestCase):
    """Exercise small synthetic submissions without external dependencies."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / "submission"
        self.root.mkdir()
        self.output = self.parent / ARCHIVE_FILENAME
        self.archive_prefix = Path(ARCHIVE_FILENAME).stem + "/"
        self.manifest_member = self.archive_prefix + MANIFEST_NAME

    def write_file(self, relative: str, contents: bytes) -> Path:
        """Create one file inside the temporary synthetic submission."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_ignored_evidence_is_included_and_private_local_files_are_excluded(
        self,
    ) -> None:
        included = {
            ".gitignore": b"results/\nlogs/\n*.pt\n",
            "readme.md": b"Synthetic submission\n",
            "results/gcn_full_v1/target/fold_0/best_model.pt": b"model bytes",
            "results/gcn_full_v1/_summary/per_fold.csv": b"rmse\n0.9\n",
            "logs/gcn_full_v1/training.out": b"training evidence\n",
            "jobs/cluster.env.example": b"ACCOUNT=placeholder\n",
        }
        excluded = (
            ".git/config",
            "code/__pycache__/model.pyc",
            "code/model.pyo",
            ".pytest_cache/nodeids",
            ".ruff_cache/checks",
            ".mypy_cache/types",
            ".venv-analysis/pyvenv.cfg",
            "venv/library.py",
            "env/library.py",
            ".cache/fonts.json",
            "project.egg-info/PKG-INFO",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            "~$temporary.docx",
            "jobs/cluster.env",
            "jobs/ciai.env",
            "custom_environment/pyvenv.cfg",
            "custom_environment/library.py",
        )
        for relative, contents in included.items():
            self.write_file(relative, contents)
        for relative in excluded:
            self.write_file(relative, b"private or generated content")
        result = build_archive(self.root, self.parent)
        with zipfile.ZipFile(self.output) as archive:
            expected = {self.archive_prefix + name for name in included}
            self.assertEqual(set(archive.namelist()), expected | {self.manifest_member})
            self.assertEqual(archive.namelist()[:-1], sorted(expected))
            for relative, contents in included.items():
                self.assertEqual(archive.read(self.archive_prefix + relative), contents)
        self.assertEqual(result["payload_files"], len(included))
        self.assertEqual(result["uncompressed_bytes"], sum(map(len, included.values())))
        self.assertTrue((self.root / "jobs/cluster.env").is_file())

    def test_assistant_and_editor_artefacts_are_excluded_at_any_depth(self) -> None:
        included = {
            "README.md": b"How to review the experiment.\n",
            "docs/protocol.md": b"Scientific protocol.\n",
            "code/assistant_terms.py": b"# A mention of Codex is not a transcript.\n",
            "results/conversation_metrics.csv": b"metric,value\nrmse,0.9\n",
            "provenance/training_source/code/tests/test_launcher.py": b"VALUE = 1\n",
        }
        excluded = (
            ".agents/settings.json",
            "nested/.codex/session.jsonl",
            ".claude/history.json",
            ".cursor/settings.json",
            ".continue/config.json",
            ".aider/private.txt",
            ".vscode/settings.json",
            "nested/.IDEA/workspace.xml",
            "AGENTS.md",
            "nested/CLAUDE.MD",
            "GEMINI.md",
            "ASSISTANT_GUIDELINES.md",
            "local_codex_handoff.md",
            "nested/AGENTS.override.md",
            "nested/CLAUDE.local.md",
            ".github/copilot-instructions.md",
            ".github/instructions/review.instructions.md",
            ".aider.chat.history.md",
            ".aider.input.history",
            "conversations/session.json",
            "nested/CHAT_HISTORY/session.txt",
            "chat-logs/private.txt",
            "conversation_20261015.json",
            "nested/Conversations.Export.JSONL",
            "chat_history.txt",
            "chat-history-20261015.md",
            "chat_logs.html",
            "chatgpt-export.zip",
            "codex_session_20261015.jsonl",
            "claude-session-20261015.txt",
        )
        for relative, contents in included.items():
            self.write_file(relative, contents)
        for relative in excluded:
            self.write_file(relative, b"private working material")
        build_archive(self.root, self.parent)
        with zipfile.ZipFile(self.output) as archive:
            expected = {self.archive_prefix + name for name in included}
            self.assertEqual(set(archive.namelist()), expected | {self.manifest_member})
            manifest = archive.read(self.manifest_member).decode("utf-8")
            self.assertEqual(
                {line.split("  ", 1)[1] for line in manifest.splitlines()},
                set(included),
            )
        for relative in excluded:
            self.assertEqual(
                (self.root / relative).read_bytes(), b"private working material"
            )

    def test_checksums_describe_every_archived_payload_byte_and_not_themselves(
        self,
    ) -> None:
        payload = {
            "zeta.bin": b"\x00\xff" * 100,
            "folder/with spaces.txt": "Preserved provenance.\n".encode("utf-8"),
            "alpha.txt": b"first\n",
        }
        for relative, contents in payload.items():
            self.write_file(relative, contents)
        result = build_archive(self.root, self.parent)
        with zipfile.ZipFile(self.output) as archive:
            manifest = archive.read(self.manifest_member)
            entries = dict(
                (name, digest)
                for digest, name in (
                    line.split("  ", 1)
                    for line in manifest.decode("utf-8").splitlines()
                )
            )
            self.assertEqual(
                {self.archive_prefix + name for name in entries},
                set(archive.namelist()) - {self.manifest_member},
            )
            self.assertEqual(list(entries), sorted(payload))
            for name, digest in entries.items():
                self.assertEqual(
                    digest,
                    hashlib.sha256(
                        archive.read(self.archive_prefix + name)
                    ).hexdigest(),
                )
        self.assertEqual(
            result["manifest_sha256"], hashlib.sha256(manifest).hexdigest()
        )

    def test_manifest_is_stable_for_the_same_payload(self) -> None:
        self.write_file("b.txt", b"second")
        self.write_file("a.txt", b"first")
        build_archive(self.root, self.parent)
        second_directory = self.parent / "second_export"
        second_directory.mkdir()
        second_output = second_directory / ARCHIVE_FILENAME
        build_archive(self.root, second_directory)
        with (
            zipfile.ZipFile(self.output) as first,
            zipfile.ZipFile(second_output) as second,
        ):
            self.assertEqual(
                first.read(self.manifest_member), second.read(self.manifest_member)
            )
            self.assertEqual(first.namelist(), second.namelist())

    def test_existing_archive_is_never_replaced(self) -> None:
        self.output.write_bytes(b"preserve the original archive")
        with self.assertRaises(FileExistsError):
            build_archive(self.root, self.parent)
        self.assertEqual(self.output.read_bytes(), b"preserve the original archive")

    def test_destination_created_after_validation_is_still_preserved(self) -> None:
        def create_competing_output(root: Path) -> list[Path]:
            self.output.write_bytes(b"created by another process")
            return []

        with patch(
            "build_archive.collect_payload", side_effect=create_competing_output
        ):
            with self.assertRaises(FileExistsError):
                build_archive(self.root, self.parent)
        self.assertEqual(self.output.read_bytes(), b"created by another process")

    def test_output_inside_source_is_rejected_before_creation(self) -> None:
        output = self.root / ARCHIVE_FILENAME
        with self.assertRaisesRegex(ValueError, "outside"):
            build_archive(self.root, self.root)
        self.assertFalse(output.exists())

    def test_missing_or_non_directory_destination_does_not_create_output(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_archive(self.root, self.parent / "missing")
        self.assertFalse((self.parent / "missing").exists())
        non_directory = self.parent / "existing.txt"
        non_directory.write_bytes(b"preserve existing file")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            build_archive(self.root, non_directory)
        self.assertEqual(non_directory.read_bytes(), b"preserve existing file")

    def test_source_manifest_name_cannot_be_silently_overwritten(self) -> None:
        self.write_file(MANIFEST_NAME, b"received manifest\n")
        with self.assertRaisesRegex(ValueError, "Reserved generated manifest"):
            build_archive(self.root, self.parent)
        self.assertFalse(self.output.exists())
        self.assertEqual(
            (self.root / MANIFEST_NAME).read_bytes(), b"received manifest\n"
        )

    def test_empty_submission_still_has_an_explicit_empty_manifest(self) -> None:
        result = build_archive(self.root, self.parent)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.namelist(), [self.manifest_member])
            self.assertEqual(archive.read(self.manifest_member), b"")
        self.assertEqual(result["payload_files"], 0)

    def test_failed_payload_write_removes_only_its_new_partial_archive(self) -> None:
        source = self.write_file("evidence.txt", b"keep scientific evidence")
        with patch(
            "zipfile.ZipFile.open", side_effect=OSError("synthetic write failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic write failure"):
                build_archive(self.root, self.parent)
        self.assertFalse(self.output.exists())
        self.assertEqual(source.read_bytes(), b"keep scientific evidence")

    def test_file_symlink_to_outside_is_rejected(self) -> None:
        external = self.parent / "external.txt"
        external.write_bytes(b"must never enter the archive")
        link = self.root / "linked.txt"
        try:
            link.symlink_to(external)
        except (OSError, NotImplementedError) as error:
            self.skipTest(
                f"File symlinks require unavailable platform privileges: {error}"
            )
        with self.assertRaisesRegex(ValueError, "Links and reparse points"):
            build_archive(self.root, self.parent)
        self.assertFalse(self.output.exists())

    def test_windows_reparse_attribute_is_rejected_without_link_privileges(
        self,
    ) -> None:
        source = self.write_file("evidence.txt", b"ordinary fixture")
        original_lstat = Path.lstat

        def reparse_information(path: Path) -> object:
            if path == source:
                return SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o644,
                    st_file_attributes=getattr(
                        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
                    ),
                )
            return original_lstat(path)

        with patch.object(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024, create=True):
            with patch.object(
                Path, "lstat", autospec=True, side_effect=reparse_information
            ):
                with self.assertRaisesRegex(ValueError, "Links and reparse points"):
                    build_archive(self.root, self.parent)
        self.assertFalse(self.output.exists())

    def test_directory_symlink_to_outside_is_rejected(self) -> None:
        external = self.parent / "external"
        external.mkdir()
        (external / "private.txt").write_bytes(b"must never enter the archive")
        link = self.root / "linked_directory"
        try:
            link.symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(
                f"Directory symlinks require unavailable platform privileges: {error}"
            )
        with self.assertRaisesRegex(ValueError, "Links and reparse points"):
            build_archive(self.root, self.parent)
        self.assertFalse(self.output.exists())

    def test_default_destination_and_extraction_produce_one_named_root(self) -> None:
        self.write_file("readme.md", b"submission contents")
        self.write_file("results/evidence.json", b"{}")
        self.assertEqual(main(["--root", str(self.root)]), 0)
        extraction = self.parent / "extracted"
        with zipfile.ZipFile(self.output) as archive:
            archive.extractall(extraction)
        extracted_root = extraction / Path(ARCHIVE_FILENAME).stem
        self.assertEqual(list(extraction.iterdir()), [extracted_root])
        self.assertEqual(
            (extracted_root / "readme.md").read_bytes(), b"submission contents"
        )
        self.assertEqual((extracted_root / "results/evidence.json").read_bytes(), b"{}")
        self.assertTrue((extracted_root / MANIFEST_NAME).is_file())

    def test_cli_output_directory_retains_the_fixed_filename(self) -> None:
        directory = self.parent / "chosen_destination"
        directory.mkdir()
        self.assertEqual(
            main(["--root", str(self.root), "--output-dir", str(directory)]), 0
        )
        self.assertEqual(list(directory.iterdir()), [directory / ARCHIVE_FILENAME])
        self.assertFalse(self.output.exists())

    def test_cli_rejects_custom_output_filenames(self) -> None:
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                main(["--root", str(self.root), "--output", str(self.output)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(list(self.parent.iterdir()), [self.root])


if __name__ == "__main__":
    unittest.main()
