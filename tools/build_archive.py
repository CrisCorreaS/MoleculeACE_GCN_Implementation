"""Build a complete submission ZIP independently of Git's ignored-file rules.

Run from the submission root, optionally choosing an existing external directory:
    python tools/build_archive.py --output-dir ..

The fixed archive filename is defined by ARCHIVE_FILENAME below. Its stem names
the single top-level directory in the ZIP, including the checksum manifest.
Results, logs, checkpoints and other evidence are included even when Git ignores
them. Git metadata, caches, local environments, system cruft, private cluster
configuration, editor settings and named assistant artefacts are excluded.
Transcript exclusions use explicit filenames, not a scan of scientific content.
Symbolic links, junctions and special files are
rejected. Payload members are ordered by their relative POSIX names; SHA256SUMS
is written last and lists every payload file, excluding itself to avoid a
self-referential checksum. Checksum paths are relative to that top-level
directory. Its UTF-8 contents are deterministic for a fixed payload. An existing
destination is never replaced.

Only the standard library is required. Input files are read without modification;
the output must be outside the submission directory and have an existing parent.
"""

import argparse
from fnmatch import fnmatchcase
import hashlib
import logging
import os
from pathlib import Path
import stat
import zipfile


ARCHIVE_FILENAME = "MoleculeACE_GCN_Implementation.zip"
MANIFEST_NAME = "SHA256SUMS"
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".ipynb_checkpoints",
        ".venv",
        "venv",
        "env",
        ".agents",
        ".codex",
        ".claude",
        ".cursor",
        ".continue",
        ".aider",
        ".vscode",
        ".idea",
        "conversations",
        "chat_history",
        "chat-history",
        "chat_logs",
        "chat-logs",
    }
)
EXCLUDED_FILE_NAMES = frozenset(
    {
        ".git",
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
        ".coverage",
        "agents.md",
        "claude.md",
        "gemini.md",
        "assistant_guidelines.md",
        "local_codex_handoff.md",
        "copilot-instructions.md",
        ".aider.chat.history.md",
        ".aider.input.history",
    }
)
PRIVATE_CONFIGURATION_PATHS = frozenset({"jobs/cluster.env", "jobs/ciai.env"})
PRIVATE_INSTRUCTION_PATTERNS = (
    "agents.*.md",
    "claude.*.md",
    "gemini.*.md",
    "*.instructions.md",
)
PRIVATE_TRANSCRIPT_STEMS = (
    "conversation*",
    "chat_history*",
    "chat-history*",
    "chat_log*",
    "chat-log*",
    "chatgpt*",
    "codex_session*",
    "codex-session*",
    "claude_session*",
    "claude-session*",
)
PRIVATE_TRANSCRIPT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".md", ".txt", ".html", ".htm", ".zip"}
)
CHUNK_SIZE = 1024 * 1024


def validate_paths(root: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve the input root and a new ZIP destination outside that root.

    Parameters
    ----------
    root : Path
        Submission directory to package.
    output_dir : Path or None
        Existing external destination directory; None uses the root's parent.

    Returns
    -------
    tuple[Path, Path]
        Absolute input and output paths.

    Raises
    ------
    ValueError
        If the destination is inside the input tree or is not a directory.
    FileExistsError
        If any directory entry already occupies the destination.
    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Submission root is not a directory: {root}")
    output_dir = root.parent if output_dir is None else output_dir.resolve(strict=True)
    if not output_dir.is_dir():
        raise ValueError(f"Output destination is not a directory: {output_dir}")
    output = output_dir / ARCHIVE_FILENAME
    if output == root or root in output.parents:
        raise ValueError("The archive must be outside the submission directory")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to replace the existing output: {output}")
    return root, output


def collect_payload(root: Path) -> list[Path]:
    """Select ordinary submission files without consulting Git or .gitignore.

    Parameters
    ----------
    root : Path
        Resolved submission root.

    Returns
    -------
    list[Path]
        Files sorted by their relative POSIX names.

    Raises
    ------
    ValueError
        If an included path is a link, a special file, outside the root, has an
        ambiguous checksum name, or would conflict with the generated manifest.
    """
    payload = []
    pending_directories = [root]
    while pending_directories:
        directory = pending_directories.pop()
        for path in directory.iterdir():
            relative = path.relative_to(root)
            relative_name = relative.as_posix()
            lower_name = path.name.lower()
            if lower_name in EXCLUDED_FILE_NAMES or any(
                fnmatchcase(lower_name, pattern)
                for pattern in PRIVATE_INSTRUCTION_PATTERNS
            ):
                continue
            if path.suffix.lower() in PRIVATE_TRANSCRIPT_SUFFIXES and any(
                fnmatchcase(path.stem.lower(), pattern)
                for pattern in PRIVATE_TRANSCRIPT_STEMS
            ):
                continue
            if relative_name.lower() in PRIVATE_CONFIGURATION_PATHS:
                continue
            information = path.lstat()
            if stat.S_ISLNK(information.st_mode) or (
                getattr(information, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ):
                raise ValueError(
                    f"Links and reparse points are not archived: {relative_name}"
                )
            resolved = path.resolve(strict=True)
            if root not in resolved.parents:
                raise ValueError(
                    f"Input path escapes the submission root: {relative_name}"
                )
            if stat.S_ISDIR(information.st_mode):
                if (
                    lower_name in EXCLUDED_DIRECTORY_NAMES
                    or lower_name.startswith((".venv", ".cache"))
                    or lower_name.endswith(".egg-info")
                    or (path / "pyvenv.cfg").is_file()
                ):
                    continue
                pending_directories.append(path)
            elif stat.S_ISREG(information.st_mode):
                if path.suffix.lower() in {".pyc", ".pyo"} or path.name.startswith(
                    "~$"
                ):
                    continue
                if relative_name.casefold() == MANIFEST_NAME.casefold():
                    raise ValueError(
                        f"Reserved generated manifest already exists: {relative_name}"
                    )
                if any(character in relative_name for character in ("\r", "\n", "\\")):
                    raise ValueError(
                        f"Filename cannot be represented safely in SHA256SUMS: {relative_name!r}"
                    )
                payload.append(path)
            else:
                raise ValueError(f"Special files are not archived: {relative_name}")
    return sorted(payload, key=lambda path: path.relative_to(root).as_posix())


def build_archive(root: Path, output_dir: Path | None = None) -> dict[str, int | str]:
    """Stream selected files into a new ZIP and hash the same uncompressed bytes.

    Parameters
    ----------
    root : Path
        Submission directory, including any Git-ignored scientific evidence.
    output_dir : Path or None
        Existing external directory for the fixed archive name; None uses the
        submission directory's parent.

    Returns
    -------
    dict[str, int | str]
        Output location, payload count, uncompressed bytes and manifest hash.

    Raises
    ------
    OSError, ValueError
        For unsafe paths, unreadable inputs, concurrent input changes or failed
        archive writes. A partial archive created by this call is removed.
    """
    root, output = validate_paths(root, output_dir)
    archive_root = Path(ARCHIVE_FILENAME).stem
    payload = collect_payload(root)
    manifest_lines = []
    uncompressed_bytes = 0
    output_handle = output.open("xb")
    try:
        with (
            output_handle,
            zipfile.ZipFile(
                output_handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive,
        ):
            for path in payload:
                relative_name = path.relative_to(root).as_posix()
                before = path.lstat()
                if (
                    not stat.S_ISREG(before.st_mode)
                    or (
                        getattr(before, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    )
                    or root not in path.resolve(strict=True).parents
                ):
                    raise ValueError(
                        f"Input is no longer an ordinary in-root file: {relative_name}"
                    )
                member = zipfile.ZipInfo(
                    f"{archive_root}/{relative_name}", date_time=(1980, 1, 1, 0, 0, 0)
                )
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                digest = hashlib.sha256()
                bytes_read = 0
                with (
                    path.open("rb") as source,
                    archive.open(
                        member,
                        mode="w",
                        force_zip64=True,
                    ) as target,
                ):
                    opened = os.fstat(source.fileno())
                    if not os.path.samestat(before, opened):
                        raise ValueError(
                            f"Input changed before it could be opened: {relative_name}"
                        )
                    for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                        target.write(block)
                        digest.update(block)
                        bytes_read += len(block)
                    after = os.fstat(source.fileno())
                if (
                    bytes_read != before.st_size
                    or after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns
                ):
                    raise ValueError(f"Input changed during archiving: {relative_name}")
                manifest_lines.append(f"{digest.hexdigest()}  {relative_name}\n")
                uncompressed_bytes += bytes_read
            manifest_bytes = "".join(manifest_lines).encode("utf-8")
            member = zipfile.ZipInfo(
                f"{archive_root}/{MANIFEST_NAME}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, manifest_bytes)
    except BaseException:
        output_handle.close()
        output.unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "payload_files": len(payload),
        "uncompressed_bytes": uncompressed_bytes,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse the optional destination directory and report the completed export.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; None reads the process arguments.

    Returns
    -------
    int
        Zero after successful archive creation.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Existing directory outside --root; defaults to the root's parent",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = build_archive(arguments.root, arguments.output_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Archive creation failed: {error}\n")
    logging.info(
        "Created %s: %s payload files, %s uncompressed bytes, plus SHA256SUMS",
        result["output"],
        result["payload_files"],
        result["uncompressed_bytes"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
