# Reproduction commands and output contracts

Start with the [README](../README.md) to install the CPU analysis environment and
validate saved evidence. Commands below start in the repository root unless
stated otherwise. Use active `code/`, never the frozen snapshot as executable code.
Choose new destinations: preparation refuses work directories that already exist.

The [README](../README.md#github-repository-and-complete-submission-zip) explains
which material is public and which requires the complete archive. Source-only
checks do not authenticate omitted task evidence. Strict validation and
recalculation require the full tasks, including checkpoint bytes for hash checks.

## Verify the current source revision

With the CPU analysis environment active, from the project root:

```text
python -B tools/check_documentation_only.py --revision-manifest tools/current_source_revision.json
python -B tools/run_scientific_tests.py
```

The explicit source record identifies five corrections made after training.
Historical snapshot hashes and all supplied task source maps remain mandatory;
other active recorded files must retain their historical executable syntax.
Omitting `--revision-manifest` deliberately runs the stricter documentation-only
comparison, which rejects the five executable differences. Neither mode executes
the historical snapshot. Saved-result validation below is a separate check.

To run the metric/threshold regressions in the isolated chemistry environment:

```text
python -B tools/run_scientific_tests.py --selection test_current_science
```

The real mixed-graph batch checks require RDKit, Torch and PyG together. They are
reported as skipped in the supplied local analysis/chemistry runtimes, which lack
Torch/PyG; no replacement dependency versions are installed by the tests.

## Recalculate aggregation and analysis

Use the **complete archive**, its CPU analysis environment and enough space for
a temporary copy of the 150 tasks. This does not train or load checkpoints.
The aggregation and analysis CLIs have no independent output-directory option
and may replace derived files. Prepare a separate project from the complete
archive, in PowerShell:

```powershell
@'
from pathlib import Path
import csv
import shutil

source = Path.cwd().resolve()
scratch = source.parent / 'moleculeace_recalculation'
manifest = source / 'jobs/manifest_full.csv'
with manifest.open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 150 and {r['campaign'] for r in rows} == {'gcn_full_v1'}
datasets = sorted({r['dataset'] for r in rows})
assert len(datasets) == 30
assert all((source / 'results/gcn_full_v1' / d).is_dir() for d in datasets)
scratch.mkdir(exist_ok=False)
for name in ('code', 'references', 'provenance', 'tools'):
    shutil.copytree(source / name, scratch / name,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(scratch / 'jobs').mkdir()
shutil.copy2(manifest, scratch / 'jobs/manifest_full.csv')
for dataset in datasets:
    shutil.copytree(source / 'results/gcn_full_v1' / dataset,
                    scratch / 'results/gcn_full_v1' / dataset)
print(scratch)
'@ | python -B -
```

On POSIX shells, wrap the same Python lines with `python -B - <<'PY'` and a closing
`PY` instead of the PowerShell wrapper. The scratch contains active code,
references, historical sources, manifest and tasks, initially without summary or
analysis outputs. Run its own scripts so that imported data/configuration paths
also refer to that scratch:

```text
python -B ../moleculeace_recalculation/code/aggregate_results.py --project-root ../moleculeace_recalculation --manifest ../moleculeace_recalculation/jobs/manifest_full.csv --campaign gcn_full_v1
python -B ../moleculeace_recalculation/code/analyze_results.py --project-root ../moleculeace_recalculation --manifest ../moleculeace_recalculation/jobs/manifest_full.csv --campaign gcn_full_v1
python -B ../moleculeace_recalculation/code/validate_local_analysis.py --project-root ../moleculeace_recalculation --campaign gcn_full_v1 --manifest ../moleculeace_recalculation/jobs/manifest_full.csv --source-root ../moleculeace_recalculation/provenance/training_source --output ../moleculeace_recalculation/results/validation/evidence_recheck.json
```

Expected scratch outputs are `_summary/per_fold.csv` (150 rows), `per_dataset.csv`
(30 rows), `summary.json`, and `_analysis/comparison.csv`, `datasets_vs_table1.csv`,
`analysis.json`. Strict validation must report 30 datasets and 150 tasks.

Compare all four regenerated CSVs with the supplied tables. In the verified
Python 3.14 runtime, `per_fold.csv` and `datasets_vs_table1.csv` were byte-identical;
`per_dataset.csv` and `comparison.csv` differed only in the final binary rounding
of ten SD values: at most one representable float step, with maximum absolute
difference `6.94e-18`. Python's standard-library SD implementation accounts for
nine differences when compared directly with Python 3.8.20; the remaining
historical-runtime difference is not localised. All values pass the strict
validator's existing `1e-12` absolute/relative tolerance. Preserve the historical
files; do not replace them with the scratch versions.

JSON creation timestamps and the scratch manifest location differ. Derived-file
hashes must match their corresponding regenerated files, so `per_dataset_sha256`
can differ after this rounding. Compare every other scientific field and the
unchanged input hashes. `figure_data.csv` remains byte-identical when figures are
regenerated from the supplied historical tables.

Historical analysis also regenerates `analysis.md` and `figure_rmse_cliff.png/.pdf`
with Spanish text. Strict validation does not require these outputs. Use
`plot_paper_figures.py` for the English presentation figures.

## Optional isolated chemistry runtime on Windows

This recipe reproduces the local verification runtime: CPython 3.8.20, uv 0.13.0,
RDKit 2022.09.5 (`rdkit-pypi==2022.9.5`) and Levenshtein 0.21.1. The cluster used
Python 3.8.18. Sixteen distribution pins match training requirements; Windows
additionally needs colorama 0.4.6 for tqdm. Torch/PyG are not installed.

Requirements: Windows AMD64, the Python 3.14 launcher used for the installer,
network access to obtain uv/Python/wheels, and a new external work directory.
Existing analysis/training environments remain unchanged. Use the short external
path below: these historical Windows wheels can fail to load DLLs from deeply
nested paths. From the repository root in PowerShell:

```powershell
$cliffWork = Join-Path $env:LOCALAPPDATA 'Temp/moleculeace_cliff_check'
if (Test-Path -LiteralPath $cliffWork) { throw 'Choose a new cliff work directory' }
New-Item -ItemType Directory -Path $cliffWork | Out-Null
py -3.14 -m pip install --target "$cliffWork/tools" uv==0.13.0
if ($LASTEXITCODE -ne 0) { throw 'uv installation failed' }
$uv = Join-Path $cliffWork 'tools/bin/uv.exe'
$env:UV_CACHE_DIR = Join-Path $cliffWork '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $cliffWork 'python'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
& $uv venv --python 3.8.20 "$cliffWork/environment"
if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
@'
numpy==1.23.5
scipy==1.10.1
pandas==1.5.3
scikit-learn==1.1.3
PyYAML==6.0.1
tqdm==4.66.1
rdkit-pypi==2022.9.5
python-Levenshtein==0.21.1
Levenshtein==0.21.1
rapidfuzz==3.5.2
joblib==1.3.2
threadpoolctl==3.2.0
python-dateutil==2.8.2
pytz==2023.3.post1
six==1.16.0
Pillow==10.1.0
colorama==0.4.6
'@ | Set-Content -LiteralPath "$cliffWork/requirements.txt" -Encoding ascii
$cliffPython = Join-Path $cliffWork 'environment/Scripts/python.exe'
& $uv pip install --python $cliffPython -r "$cliffWork/requirements.txt"
if ($LASTEXITCODE -ne 0) { throw 'Chemistry installation failed' }
& $uv pip check --python $cliffPython
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
Copy-Item -LiteralPath code -Destination "$cliffWork/code" -Recurse
& $cliffPython -B -s "$cliffWork/code/verify_cliffs.py" --all --project-root $cliffWork
if ($LASTEXITCODE -ne 0) { throw 'Cliff verification failed; retain reports and investigate' }
```

The verifier reads the copied, unchanged CSVs and writes 30
`results/data_check/cliffs_<dataset>.json` reports under the external root.
Exit zero and `n_differ=0` in every report indicate agreement. Investigate any
difference without replacing published labels. The supplied reports cover
48,714 molecule-dataset rows and 18,831 cliffs. Keep the six original cluster
reports unchanged; elapsed times need not reproduce.

From the repository root, using the same PowerShell session:

```powershell
& $cliffPython -B -s tools/run_scientific_tests.py --selection test_ciai_gcn.Versions test_ciai_gcn.Featurization test_ciai_gcn.Folds test_ciai_gcn.CliffLabels
if ($LASTEXITCODE -ne 0) { throw 'Selected scientific tests failed' }
```

`Folds` and `CliffLabels` can run here. `Versions` and `Featurization` explicitly
skip without Torch/PyG. This selection avoids the modern plotting module, which
is not a Python 3.8 component. CPU chemistry checks do not establish GPU numerical
repeatability. This Windows recipe is separate from historical Linux training.

## Historical Linux/CUDA training route

Use a separate checkout/campaign for new training, preserving `gcn_full_v1`.
A receiving Slurm site needs Conda, compatible CUDA 11.3 libraries and the recorded
Torch/PyG wheels. Site policy and hardware must be checked there; local Windows
checks do not certify them. From the project root on that configured site:

```bash
cp jobs/cluster.env.example jobs/cluster.env
# Edit every site-specific placeholder in the private jobs/cluster.env.
bash jobs/run_setup.sh
bash jobs/run_launch.sh --cpu env_check
bash jobs/run_launch.sh --checks
```

Review each job's completion before proceeding. Setup installs the pinned training
environment and writes dated conda/pip exports under `results/environment/`; an
existing environment prefix is refused. The CPU check writes an environment
report. `--checks` checks the environment/GPU allocation, not the local scientific
suite. Logs go to the launcher-selected directories under `logs/`.

After site checks pass, generate and review a new manifest:

```bash
bash jobs/run_launch.sh --cpu make_manifest --stage full --campaign gcn_full_reproduction --out ../jobs/reproduction_manifest.csv
# Wait for and review manifest generation before submitting workers.
bash jobs/run_launch.sh --workers jobs/reproduction_manifest.csv gcn_full_reproduction
```

CPU-tool relative arguments are resolved from `code/`; the worker launcher accepts
its manifest from the project root. Keep `PROJECT_ROOT`, scripts and imported
package in the same checkout; the delivered runner rejects inconsistent resolved
source/import paths before task output creation. A coherent clean copy is valid.
New evidence goes under
`results/gcn_full_reproduction/` and new logs under `logs/gcn_full_reproduction/`.
Review completion and validate all tasks before aggregation. New training records
its own sources; the preserved historical hashes identify the completed campaign.
