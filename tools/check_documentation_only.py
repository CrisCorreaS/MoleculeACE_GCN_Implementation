"""Verify the 38-source snapshot and documentation-only edits to training sources.

Read the source manifest derived from the 150 full-campaign task records and
verify the preserved hashes and executable syntax of active recorded training sources.
If task records are supplied, require all 150 and check their source identities
against the manifest. Otherwise report sources-only verification: this checks
the source distribution, not predictions, metrics or full experiment evidence.
Only seven explicitly listed infrastructure files may be absent from active code.
In sources-only distributions, the frozen launcher test may be archive-only;
its identity remains in the manifest, and full task evidence requires its bytes.
No project source is imported or run. By default, any executable difference
fails. An explicit --revision-manifest additionally checks the exact current
source identities listed for the bounded post-training corrections; it never
relaxes the historical hashes or permits unlisted executable changes.

Usage: python -B tools/check_documentation_only.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
EXPECTED_TASKS = 150
EXPECTED_SOURCES = 38
MANIFEST_FIELDS = {"schema_version", "campaign", "task_count", "source_sha256"}
APPROVED_SNAPSHOT_ONLY = frozenset(
    {
        "code/tests/repeatability_diagnostic.py",
        "code/tests/test_batch_dispatch.py",
        "code/tests/test_gpu_allocation.py",
        "code/tests/test_gpu_workers.py",
        "code/tests/test_launcher.py",
        "code/tests/test_requeue.py",
        "code/tests/test_scope_compat.py",
    }
)
APPROVED_ARCHIVE_ONLY = frozenset({"code/tests/test_launcher.py"})


class StripDocstrings(ast.NodeTransformer):
    """Remove only leading docstrings in modules, classes and functions."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit nested definitions and strip the module docstring."""
        self.generic_visit(node)
        self.remove_docstring(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visit methods and strip the class docstring."""
        self.generic_visit(node)
        self.remove_docstring(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Strip the function docstring without discarding other literals."""
        self.generic_visit(node)
        self.remove_docstring(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        """Strip the asynchronous function's docstring."""
        self.generic_visit(node)
        self.remove_docstring(node)
        return node

    @staticmethod
    def remove_docstring(node: Any) -> None:
        """Remove a leading string expression; retain every other statement."""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]


def unique_json_fields(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject repeated JSON keys instead of silently replacing earlier values."""
    fields: Dict[str, Any] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError(f"Duplicate JSON field: {name}")
        fields[name] = value
    return fields


def validate_source_map(value: Any, origin: str) -> Dict[str, str]:
    """Validate the exact source count, portable paths and SHA-256 syntax.

    Parameters
    ----------
    value : object
        Candidate mapping from recorded relative Python paths to digest strings.
    origin : str
        Description of the input used in any failure message.

    Returns
    -------
    dict
        The validated source identity mapping.
    """
    if not isinstance(value, dict) or len(value) != EXPECTED_SOURCES:
        raise ValueError(f"Expected {EXPECTED_SOURCES} source identities in {origin}")
    for relative, expected in value.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"Source paths and hashes must be strings in {origin}")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != "code"
            or relative_path.suffix != ".py"
            or relative_path.as_posix() != relative
            or "\\" in relative
            or ":" in relative
        ):
            raise ValueError(f"Invalid recorded source path in {origin}: {relative}")
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Invalid SHA-256 in {origin}: {relative}")
    return value


def source_sha256_lf(contents: bytes) -> str:
    """Hash source bytes with only CRLF-to-LF transport normalisation."""
    return hashlib.sha256(contents.replace(b"\r\n", b"\n")).hexdigest()


def read_revision(
    path: Path, recorded: Dict[str, str], campaign: str
) -> Dict[str, Any]:
    """Read explicit current identities tied to the historical source map."""
    revision = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_fields
    )
    fields = {"schema_version", "historical_campaign", "sources"}
    if not isinstance(revision, dict) or set(revision) != fields:
        raise ValueError("Current-source revision fields do not match schema version 1")
    if type(revision["schema_version"]) is not int or revision["schema_version"] != 1:
        raise ValueError("Unsupported current-source revision schema")
    if revision["historical_campaign"] != campaign:
        raise ValueError("Current-source revision campaign differs")
    sources = revision["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Current-source revision must list explicit source identities")
    for relative, entry in sources.items():
        if relative not in recorded or relative in APPROVED_SNAPSHOT_ONLY:
            raise ValueError(
                f"Revision path is not an active recorded source: {relative}"
            )
        if not isinstance(entry, dict) or set(entry) != {
            "historical_sha256",
            "current_sha256_lf",
            "change",
        }:
            raise ValueError(f"Invalid current-source revision entry: {relative}")
        if entry["historical_sha256"] != recorded[relative]:
            raise ValueError(f"Revision historical identity differs: {relative}")
        digest = entry["current_sha256_lf"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid current-source SHA-256: {relative}")
        if not isinstance(entry["change"], str) or not entry["change"].strip():
            raise ValueError(f"Missing revision change description: {relative}")
    return sources


def check_sources(
    project_root: Path,
    source_root: Path,
    campaign: str,
    manifest_path: Optional[Path] = None,
    revision_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify snapshot bytes and executable syntax of recorded sources.

    Parameters
    ----------
    project_root : Path
        Submission root containing active code and optional campaign results.
    source_root : Path
        Snapshot root containing the recorded relative ``code/...`` paths.
    campaign : str
        Full-campaign identifier, expressed as a single directory name.
    manifest_path : Path, optional
        Derived source manifest; defaults to ``provenance/source_manifest.json``.
    revision_manifest : Path, optional
        Explicit identities for the post-training corrections. Without this
        argument, all executable differences from the snapshot remain errors.

    Returns
    -------
    dict
        Check counts, snapshot-only and archive-only paths, and failures.
    """
    project_root = project_root.resolve()
    source_root = source_root.resolve()
    if not campaign or campaign in (".", "..") or "/" in campaign or "\\" in campaign:
        raise ValueError("Campaign must be a single directory name")
    manifest_path = manifest_path or (
        project_root / "provenance" / "source_manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_json_fields,
    )
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("Source manifest fields do not match schema version 1")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported source manifest schema version")
    if manifest["campaign"] != campaign:
        raise ValueError(
            "Source manifest campaign does not match the requested campaign"
        )
    if (
        type(manifest["task_count"]) is not int
        or manifest["task_count"] != EXPECTED_TASKS
    ):
        raise ValueError(
            f"Source manifest must declare {EXPECTED_TASKS} original tasks"
        )
    recorded = validate_source_map(manifest["source_sha256"], "source manifest")

    revisions = (
        read_revision(revision_manifest, recorded, campaign)
        if revision_manifest is not None
        else {}
    )
    records = sorted((project_root / "results" / campaign).rglob("task_record.json"))
    if records and len(records) != EXPECTED_TASKS:
        raise ValueError(
            f"Partial task evidence: expected {EXPECTED_TASKS} records, "
            f"found {len(records)}"
        )
    for path in records:
        relative_record = path.relative_to(project_root)
        record = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_fields
        )
        if record["task"]["campaign"] != campaign:
            raise ValueError(f"Task record campaign differs: {relative_record}")
        current = validate_source_map(
            record["provenance"]["source_sha256"], str(relative_record)
        )
        if current != recorded:
            raise ValueError(
                "Task source identities do not match the source manifest: "
                f"{relative_record}"
            )

    errors: List[str] = []
    snapshot_only: List[str] = []
    archive_only: List[str] = []
    snapshot_matches = 0
    executable_matches = 0
    revision_matches: List[str] = []
    for relative, expected in sorted(recorded.items()):
        relative_path = Path(relative)
        frozen_path = (source_root / relative_path).resolve()
        active_path = (project_root / relative_path).resolve()
        try:
            frozen_path.relative_to(source_root)
            active_path.relative_to(project_root)
        except ValueError:
            errors.append(f"Recorded source path escapes its source root: {relative}")
            continue
        if not frozen_path.is_file():
            if not records and relative in APPROVED_ARCHIVE_ONLY:
                archive_only.append(relative)
                if active_path.is_file():
                    errors.append(
                        "Cannot verify active source without its archive-only "
                        f"snapshot: {relative}"
                    )
            else:
                errors.append(f"Missing snapshot source: {relative}")
            continue
        frozen_bytes = frozen_path.read_bytes()
        if hashlib.sha256(frozen_bytes).hexdigest() != expected:
            errors.append(f"Snapshot hash differs from source manifest: {relative}")
            continue
        snapshot_matches += 1
        if not active_path.is_file():
            if relative in APPROVED_SNAPSHOT_ONLY:
                snapshot_only.append(relative)
            else:
                errors.append(
                    f"Missing required active recorded training source: {relative}"
                )
            continue
        try:
            frozen_tree = StripDocstrings().visit(
                ast.parse(frozen_bytes.decode("utf-8-sig"), filename=str(frozen_path))
            )
            active_bytes = active_path.read_bytes()
            active_tree = StripDocstrings().visit(
                ast.parse(
                    active_bytes.decode("utf-8-sig"),
                    filename=str(active_path),
                )
            )
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"Cannot parse {relative}: {error}")
            continue
        same_syntax = ast.dump(frozen_tree, include_attributes=False) == ast.dump(
            active_tree, include_attributes=False
        )
        if relative in revisions:
            if same_syntax:
                errors.append(f"Revision lists no executable change: {relative}")
            elif (
                source_sha256_lf(active_bytes)
                != revisions[relative]["current_sha256_lf"]
            ):
                errors.append(
                    f"Current source differs from explicit revision: {relative}"
                )
            else:
                revision_matches.append(relative)
        elif not same_syntax:
            errors.append(f"Executable syntax differs from snapshot: {relative}")
        else:
            executable_matches += 1
    return {
        "mode": "task_records" if records else "sources_only",
        "task_records": len(records),
        "manifest_task_count": manifest["task_count"],
        "recorded_sources": len(recorded),
        "snapshot_hash_matches": snapshot_matches,
        "active_executable_matches": executable_matches,
        "current_revision_matches": revision_matches,
        "snapshot_only": snapshot_only,
        "archive_only": archive_only,
        "errors": errors,
    }


def main() -> int:
    """Read command-line paths and report the verification result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--source-root", type=Path, help="Default: provenance/training_source/"
    )
    parser.add_argument(
        "--source-manifest", type=Path, help="Default: provenance/source_manifest.json"
    )
    parser.add_argument(
        "--revision-manifest",
        type=Path,
        help="Explicit identities of the bounded post-training corrections",
    )
    parser.add_argument("--campaign", default="gcn_full_v1")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source_root = arguments.source_root or (
        arguments.project_root / "provenance" / "training_source"
    )
    try:
        report = check_sources(
            arguments.project_root,
            source_root,
            arguments.campaign,
            arguments.source_manifest,
            arguments.revision_manifest,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.error("Source verification failed: %s", error)
        return 1
    if report["mode"] == "sources_only":
        LOGGER.info(
            "SOURCES ONLY: task records are absent; saved predictions, metrics "
            "and experiment evidence are not validated by this check"
        )
    else:
        LOGGER.info("All 150 task source maps match the derived source manifest")
    for relative in report["snapshot_only"]:
        LOGGER.info("SNAPSHOT ONLY (not active executable code): %s", relative)
    for relative in report["archive_only"]:
        LOGGER.info(
            "archive-only (identity retained; snapshot bytes require the "
            "complete evidence archive): %s",
            relative,
        )
    for relative in report["current_revision_matches"]:
        LOGGER.info(
            "POST-TRAINING REVISION (not historical training code): %s", relative
        )
    for error in report["errors"]:
        LOGGER.error("%s", error)
    LOGGER.info(
        "%d task records; %d/%d snapshot hashes match; "
        "%d active files have unchanged executable syntax; %d snapshot-only files; "
        "%d archive-only files",
        report["task_records"],
        report["snapshot_hash_matches"],
        report["recorded_sources"],
        report["active_executable_matches"],
        len(report["snapshot_only"]),
        len(report["archive_only"]),
    )
    if report["errors"]:
        return 1
    if arguments.revision_manifest is None:
        LOGGER.info("PASS: source snapshot and documentation-only checks")
    else:
        LOGGER.info(
            "PASS: historical snapshot, unchanged source syntax and %d exact current revisions",
            len(report["current_revision_matches"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
