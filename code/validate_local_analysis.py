"""Validate saved campaigns for local figures without changing historical artefacts.

No model is loaded and no training is performed. Existing validators recalculate
metrics from saved predictions; this module additionally checks saved summary and
comparison tables. New reports are confined to results/validation/.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

from aggregate_results import collect, summarise
from analyze_results import build_tables
from MoleculeACE.benchmark.artifacts import (
    canonical_json_sha256,
    read_json,
    sha256_file,
)
from MoleculeACE.benchmark.campaign import CampaignError, default_project_root, task_dir


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows or len(set(reader.fieldnames or [])) != len(
            reader.fieldnames or []
        ):
            raise CampaignError("Empty table or duplicate columns: {}".format(path))
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise CampaignError("Malformed CSV row: {}".format(path))
        return reader.fieldnames, rows


def _compare_table(path, expected, keys):
    """Check identities and every field, permitting only floating-point round-off."""
    fields, observed = _read_csv(path)
    if set(fields) != set(expected[0]):
        raise CampaignError(
            "Table columns differ from recalculated fields: {}".format(path)
        )
    index = {}
    for row in observed:
        identity = tuple(row[key] for key in keys)
        if identity in index:
            raise CampaignError(
                "Duplicate row identity {} in {}".format(identity, path)
            )
        index[identity] = row
    expected_keys = {tuple(str(row[key]) for key in keys) for row in expected}
    if set(index) != expected_keys or len(observed) != len(expected):
        raise CampaignError(
            "Table identities differ from the validated campaign: {}".format(path)
        )
    for row in expected:
        identity = tuple(str(row[key]) for key in keys)
        for field, value in row.items():
            actual = index[identity][field]
            if isinstance(value, float):
                try:
                    numeric = float(actual)
                except ValueError as exc:
                    raise CampaignError(
                        "Invalid number in {}: {}".format(path, field)
                    ) from exc
                same = math.isfinite(numeric) and math.isclose(
                    numeric, value, rel_tol=1e-12, abs_tol=1e-12
                )
            else:
                same = actual == ("" if value is None else str(value))
            if not same:
                raise CampaignError(
                    "Table value differs from recalculation: {} {} {}".format(
                        path, identity, field
                    )
                )


def _source_evidence(source_root, recorded):
    """Report byte identity separately from an in-memory LF diagnostic; never edit."""
    entries = []
    for relative, expected in sorted(recorded.items()):
        path = (source_root / relative).resolve()
        if not _inside(path, source_root):
            raise CampaignError(
                "Recorded source path escapes the source root: {}".format(relative)
            )
        if not path.is_file():
            raise CampaignError("Missing recorded source: {}".format(relative))
        contents = path.read_bytes()
        actual = hashlib.sha256(contents).hexdigest()
        lf_hash = hashlib.sha256(contents.replace(b"\r\n", b"\n")).hexdigest()
        status = (
            "byte_match"
            if actual == expected
            else ("lf_only_match" if lf_hash == expected else "different")
        )
        entries.append(
            dict(
                path=relative,
                recorded_sha256=expected,
                byte_sha256=actual,
                lf_diagnostic_sha256=lf_hash,
                status=status,
            )
        )
    different = [entry["path"] for entry in entries if entry["status"] == "different"]
    if different:
        raise CampaignError(
            "Historical source differs beyond CRLF transport: {}".format(
                ", ".join(different)
            )
        )
    return dict(
        files=entries,
        n_files=len(entries),
        byte_matches=sum(e["status"] == "byte_match" for e in entries),
        lf_only_matches=sum(e["status"] == "lf_only_match" for e in entries),
        policy="Exact bytes checked first; LF hashes are diagnostic only. No files are normalised.",
    )


def _check_metadata(path, expected):
    metadata = read_json(path)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CampaignError("Metadata mismatch in {}: {}".format(path, key))
    return metadata


def load_validated_tables(root, campaign, manifest=None, source_root=None):
    """Return validated tables and evidence; read only, with no torch dependency.

    ``root`` locates campaign/reference files. Dataset/configuration access follows
    the existing package location, whose bytes are checked against the manifest.
    ``source_root`` can identify the preserved received copy, since later local
    analysis code need not be identical to the code that trained the campaign.
    """
    root = Path(root).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else root
    if (
        not campaign
        or campaign in (".", "..")
        or Path(campaign).name != campaign
        or "/" in campaign
        or "\\" in campaign
    ):
        raise CampaignError("Campaign must be a single directory name")
    folder = root / "results" / campaign
    summary_path = folder / "_summary" / "summary.json"
    summary = read_json(summary_path)
    manifest = (
        Path(manifest).resolve()
        if manifest is not None
        else root / "jobs" / ("manifest_{}.csv".format(summary["stage"]))
    )
    tasks, results = collect(root, manifest, campaign)
    per_fold, per_dataset = summarise(tasks, results)
    table1, comparison, verified = build_tables(root, per_dataset)
    if not verified:
        raise CampaignError("Published reference provenance could not be verified")
    if any(not row["matches_table1"] for row in table1):
        raise CampaignError("Dataset verification differs from corrected Table 1")
    if any(
        row["published_rmse"] is None or row["published_rmse_cliff"] is None
        for row in comparison
    ):
        raise CampaignError("Published GCN values are missing for a campaign dataset")

    first = tasks[0]
    manifest_hash = sha256_file(manifest)
    _check_metadata(
        summary_path,
        dict(
            campaign=campaign,
            stage=first["stage"],
            scientific=bool(first["scientific"]),
            n_tasks=len(tasks),
            n_datasets=len(per_dataset),
            manifest_sha256=manifest_hash,
            max_epochs=first["max_epochs"],
            early_stopping_mode=first["early_stopping_mode"],
        ),
    )
    analysis_path = folder / "_analysis" / "analysis.json"
    _check_metadata(
        analysis_path,
        dict(
            campaign=campaign,
            scientific=bool(first["scientific"]),
            n_datasets=len(per_dataset),
            manifest_sha256=manifest_hash,
            published_provenance_verified=True,
        ),
    )
    tables = [
        (folder / "_summary" / "per_fold.csv", per_fold, ("dataset", "fold")),
        (folder / "_summary" / "per_dataset.csv", per_dataset, ("dataset",)),
        (folder / "_analysis" / "comparison.csv", comparison, ("dataset",)),
        (folder / "_analysis" / "datasets_vs_table1.csv", table1, ("dataset",)),
    ]
    for path, expected, keys in tables:
        if (
            path.parent.name == "_summary"
            and sha256_file(path) != summary[path.stem + "_sha256"]
        ):
            raise CampaignError("Saved summary hash mismatch: {}".format(path))
        _compare_table(path, expected, keys)

    records = {}
    for task in tasks:
        record_path = task_dir(root, task) / "task_record.json"
        record = read_json(record_path)
        if (
            canonical_json_sha256(record["effective_config"])
            != record["effective_config_sha256"]
        ):
            raise CampaignError(
                "Effective configuration hash mismatch: {}".format(record_path)
            )
        records[record_path.relative_to(root).as_posix()] = sha256_file(record_path)
    sources = _source_evidence(source_root, results[0]["provenance"]["source_sha256"])
    paths = (
        [summary_path, analysis_path, manifest]
        + [t[0] for t in tables]
        + [
            root / "references" / "corrected_table1.csv",
            root / "references" / "published" / "MoleculeACE_results.csv",
            root / "references" / "published" / "provenance.json",
        ]
    )
    input_hashes = {
        (p.relative_to(root).as_posix() if _inside(p, root) else p.name): sha256_file(p)
        for p in paths
    }
    report = dict(
        schema_version=1,
        status=(
            "passed_with_source_transport_differences"
            if sources["lf_only_matches"]
            else "passed"
        ),
        campaign=campaign,
        stage=first["stage"],
        scientific=bool(first["scientific"]),
        n_tasks=len(tasks),
        n_datasets=len(per_dataset),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root_is_project_root=source_root == root,
        source_evidence=sources,
        input_sha256=input_hashes,
        task_record_sha256=records,
        checks=[
            "manifest and input hashes",
            "all task artefact hashes",
            "prediction identities and recalculated official/secondary metrics",
            "early-stopping histories and campaign provenance consistency",
            "effective configuration hashes",
            "saved summary hashes",
            "all saved summary/comparison table values",
            "published reference provenance",
            "corrected Table 1",
            "historical source identity",
        ],
        table_float_tolerance=dict(relative=1e-12, absolute=1e-12),
        limitations=[
            "Checkpoints are hashed, never deserialised or executed.",
            "This validates saved evidence; it does not repeat training or inference.",
            "Fold SD is descriptive; folds share the test set and are not independent seeds.",
        ],
    )
    return dict(
        tasks=tasks,
        per_fold=per_fold,
        per_dataset=per_dataset,
        table1=table1,
        comparison=comparison,
        report=report,
    )


def report_output_path(root, output):
    """Accept only a new JSON report under the dedicated validation directory."""
    allowed = (Path(root).resolve() / "results" / "validation").resolve()
    output = Path(output).resolve()
    if (
        not _inside(output, allowed)
        or output == allowed
        or output.suffix.lower() != ".json"
    ):
        raise CampaignError("Output must be a new JSON file under results/validation/")
    if output.exists():
        raise CampaignError(
            "Refusing to overwrite an existing report: {}".format(output)
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = report_output_path(args.project_root, args.output)
    except CampaignError as exc:
        parser.exit(2, "ERROR: {}\n".format(exc))
    try:
        report = load_validated_tables(
            args.project_root, args.campaign, args.manifest, args.source_root
        )["report"]
        exit_code = 0
    except (CampaignError, OSError, ValueError, KeyError) as exc:
        report = dict(
            schema_version=1, status="failed", campaign=args.campaign, error=str(exc)
        )
        exit_code = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second invocation from replacing evidence.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print("{}: {}".format(report["status"], output))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
