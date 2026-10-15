# GCN reproduction of the corrected MoleculeACE benchmark

This repository contains a bounded reproduction of the **GCN** model using the corrected MoleculeACE data and published per-dataset configurations. The completed `gcn_full_v1` campaign comprises **30 dataset–response combinations and 150 tasks** (five internal folds per dataset). 
The delivered results are complete; validating them does not retrain a model.

Only GCN was trained. Other strategies shown in the figures are corrected **published reference results**, not additional implementations or experiments in this repository. There is no new hyperparameter search.

## Start here

| Purpose | Entry point | Requirements |
|---|---|---|
| Read results | `results/gcn_full_v1/_summary/per_dataset.csv`, `_analysis/comparison.csv` and `figures/figure_catalogue.pdf` | No installation |
| Validate saved evidence | `code/validate_local_analysis.py` | Complete archive and CPU analysis environment |
| Regenerate presentation figures | `code/plot_paper_figures.py` | CPU environment; authenticate inputs with the complete archive |
| Run the selected scientific tests | `tools/run_scientific_tests.py` | CPU environment; additional chemistry/Torch tests require their historical dependencies |
| Recompute activity-cliff labels | `code/verify_cliffs.py` | RDKit and Levenshtein in the historical environment |
| Repeat training in a separate campaign | `code/run_benchmark.py`, full manifest and `jobs/` | Historical Linux/CUDA environment and configured Slurm allocation |

The scientific comparisons are descriptive. The available published results do not provide fold-level uncertainty, and this reproduction does not establish statistical equivalence or reproduce the original multi-strategy conclusions.

For a first review, read the result tables and figure catalogue, then follow
[run_benchmark.py](code/run_benchmark.py) into
[gcn.py](code/MoleculeACE/models/gcn.py) and
[models/utils.py](code/MoleculeACE/models/utils.py).
The [protocol](docs/protocol.md) explains units, folds, stopping and metrics;
[upstream changes](docs/upstream_changes.md) separates inherited code from the
adaptations. [Reproduction commands](docs/reproduction.md) specify prerequisites,
outputs and scratch preparation. Use active `code/`; `provenance/` is evidence.

## GitHub repository and complete submission ZIP

Public repository: [MoleculeACE GCN implementation](https://github.com/CrisCorreaS/MoleculeACE_GCN_Implementation).

The GitHub repository contains source code, corrected inputs, published references, summary/comparison tables, figures and the frozen source snapshot except for the archive-only infrastructure test described under Provenance. Logs, per-task evidence and checkpoints, the aggregation metadata containing its original path, node-specific environment reports and numerical diagnostics are **ZIP-only**. They preserve the complete historical evidence without publishing execution metadata or large model files. The source checker and synthetic scientific tests run on the repository alone; strict campaign validation requires the complete ZIP.

The ZIP is built from this folder independently of Git's ignore rules, with a `SHA256SUMS` list for its payload. Git metadata, caches, local environments and private `jobs/cluster.env` and recognised local working files are excluded by the builder. These filename rules do not identify an arbitrarily renamed private document; review the payload before release. With the Python 3.14 analysis environment active, run `python -B tools/build_archive.py` to create the archive in the parent directory, or choose another external directory with `--output-dir`. The filename is defined once in the builder as `MoleculeACE_GCN_Implementation.zip`; extracting it produces only `MoleculeACE_GCN_Implementation/`, containing the delivery and `SHA256SUMS`.

## Repository map

| Path | Responsibility |
|---|---|
| `code/MoleculeACE/Data/` | Corrected datasets, metadata and published GCN configurations |
| `code/MoleculeACE/benchmark/` | Data loading, featurisation, cliffs, metrics, manifests and evidence validation |
| `code/MoleculeACE/models/` | GCN, Graph Multiset Transformer pooling and training protocol |
| `code/tests/` | Selected scientific tests and synthetic fixtures |
| `jobs/` | Historical Slurm execution routes, full manifest and site-configuration example |
| `references/` | Corrected Table 1, published comparison CSV and its provenance |
| `results/gcn_full_v1/` | The 150 completed tasks, aggregation tables and analysis data |
| `results/data_check/` | Shared corrected-input checks and cliff reports for all 30 datasets |
| `results/environment/` | Historical package exports and the pre-full environment check |
| `results/diagnostics/` | The numerical check associated with pre-full job `285912` |
| `results/validation/` | Saved validation evidence; subsequent reports remain local |
| `logs/` | Full campaign logs and the associated checks, aggregation and analysis logs |
| `figures/` | Submission figures, captions, plotted data, catalogue and export manifest |
| `provenance/training_source/` | Exact recorded training sources; data for hash/AST comparison only |
| `tools/` | Source/documentation checking, selected test runner and archive builder |
| `docs/` | Protocol, upstream differences and detailed reproduction commands |
| `environments/` | Separate pinned training and CPU analysis requirements |
| `.github/` | Public CPU workflow, dependency-update policy and issue/PR templates |

The principal training entry point, `run_benchmark.py`, executes **one manifest
row**, not all 150 tasks in one Python process. The full manifest and worker
orchestration cover all 30 datasets. Shared stage-handling code remains intact;
historical pilot/demo campaigns and manifests are not included.

| Script in `code/` | What it does |
|---|---|
| `run_benchmark.py` | Loads and validates a dataset–fold task, trains/evaluates GCN and records its evidence |
| `run_gpu_worker.py` | Coordinates the two-GPU worker and its independent task lanes |
| `gpu_worker_child.py` | Starts a child with its assigned GPU and CPU affinity |
| `make_manifest.py` | Builds a protocol-checked manifest for a new campaign |
| `task_status.py` | Inspects task completion and saved evidence |
| `env_check.py` | Checks package versions and, when requested, GPU allocation |
| `verify_datasets.py` | Checks corrected Table 1, CSV identity and response scale |
| `verify_cliffs.py` | Recomputes cliff labels without changing the benchmark CSVs |
| `aggregate_results.py` | Validates tasks and calculates per-fold/per-dataset tables |
| `analyze_results.py` | Checks the published reference and calculates comparison tables |
| `validate_local_analysis.py` | Recalculates metrics/tables and verifies historical source and artefact identities |
| `plot_paper_figures.py` | Produces all presentation figures from saved results and corrected reference data |

## Environments

Keep training and analysis environments separate. The historical training runtime used Python 3.8.18, Torch 1.11.0+cu113, torch-geometric 2.0.4 and RDKit 2022.09.5. `environments/training/environment.yml`, `environments/training/requirements.txt` and `jobs/run_setup.sh` record the installation requirements. Exact installed packages are retained in `results/environment/conda_explicit_20260925_202858.txt` and `results/environment/pip_freeze_20260925_202858.txt`.

The local analysis environment uses Python 3.14 and the pins in `environments/analysis/requirements.txt`. For a new Windows installation, from the repository root:

```powershell
py -3.14 -m venv .venv_analysis
.\.venv_analysis\Scripts\Activate.ps1
python -m pip install -r environments/analysis/requirements.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
```

On Linux, use `python3.14 -m venv .venv_analysis` and
`source .venv_analysis/bin/activate`, followed by the same installation command.
The runners work directly from `code/`; optional packaging uses `pyproject.toml` and
does not replace either pinned environment.

## Validate and regenerate figures

From the repository root, with the analysis environment active:

```text
python -B tools/check_documentation_only.py --revision-manifest tools/current_source_revision.json
python -B tools/run_scientific_tests.py
```

The runner selects scientific tests explicitly. It never imports or executes anything under `provenance/`. Four scientific classes from the active `code/tests/test_ciai_gcn.py` are selected; its GPU training diagnostic is excluded. Unavailable dependencies are reported as skips, not passes. Synthetic stage tests need no historical pilot/demo results. Do not use blanket test discovery: the mixed test module retains its historical GPU test with unchanged behaviour. The delivered `jobs/run_checks.sh` checks the environment/GPU allocation only; it does not replace the selected scientific runner.

With the complete archive, from `code/`:

```text
python -B validate_local_analysis.py --project-root .. --campaign gcn_full_v1 --manifest ../jobs/manifest_full.csv --source-root ../provenance/training_source --output ../results/validation/evidence_recheck.json
python -B plot_paper_figures.py --published ../references/published/MoleculeACE_results.csv --benchmark-dir MoleculeACE/Data/benchmark_data --comparison ../results/gcn_full_v1/_analysis/comparison.csv --per-fold ../results/gcn_full_v1/_summary/per_fold.csv --output-dir ../figures_regenerated
```

The delivered validation report is `results/validation/evidence.json`; choose a new report filename for each subsequent check. The figure script writes directly into `--output-dir` and refuses **any existing output directory**, including an empty one. Delivered figures are in `figures/`; regenerate into a new directory.

With complete task evidence, **run the strict validator before the figure script**. A public source download can regenerate figures from the included tables, but cannot authenticate them against the omitted task/checkpoint evidence. The figure script checks schemas, coverage and some numerical consistency, but does not independently validate all task/source hashes or the pinned provenance of the published table. It checks finite errors and non-negative SDs, exact dataset/fold coverage, and means/SDs including the paired cliff-error difference against per-fold inputs. Use the same inputs as the successful strict validation. Its optional `--demo` mode creates watermarked synthetic layout data and is not a historical campaign.

## Scientific protocol and adaptations

The response is **y = -log10(concentration in nM)**; pKi/pEC50 is `y + 9`. Five median-stratified folds with seed 42 partition the published training set; all five models use the unchanged published test. Graphs carry 37 node features and six bond features, but the GCN consumes node features and connectivity: **its forward pass does not use `edge_attr`**. Cliff labels are evaluation labels.

The implementation retains the GCN constructor's 300-epoch limit, published hyperparameters and dropout zero. `upstream_consecutive` stopping differs from the correction's non-consecutive wording. Official checkpoint selection preserves the upstream epoch-ceiling rule. Means are averages of fold RMSEs; SDs are sample SDs, with paired within-fold differences for RMSE_cliff minus RMSE. See the [protocol](docs/protocol.md) for exact definitions and limitations.

The [upstream-change record](docs/upstream_changes.md) explains package reduction, explicit model modes, per-task checkpoints, logging and provenance adaptations. Configuration readers reject unsupported formats and empty/non-mapping files; this is not exhaustive validation of every possible model keyword.

## Supporting checks and their coverage

`results/data_check/datasets.csv` and `datasets.json` cover all 30 datasets (48,714 rows); their input hashes agree with the full manifest. The 30 `cliffs_*.json` reports show zero label differences. Six original cluster reports are retained unchanged (`CHEMBL2835_Ki`, `CHEMBL4203_Ki`, `CHEMBL4616_EC50`, `CHEMBL2147_Ki`, `CHEMBL214_Ki` and `CHEMBL244_Ki`); the other 24 were added from local recomputation of all 30 datasets. The six historical reports checked **shared, byte-identical inputs** for an earlier stage selection. Their chemistry versions are inferred from the recorded environment, rather than stamped in each cliff report. The [verification summary](results/data_check/cliff_verification_summary.csv) records the complete coverage: 48,714 molecule–dataset rows, including 18,831 rows labelled as cliffs, and zero label differences.

`results/environment/env_check_*.json` and `logs/checks/gcn_checks_285912.out` document the pre-full environment check. The historical log records 151 tests; that is distinct from the selected local submission suite. Training-node and package details for each actual full task are also recorded in its `task_record.json`.

The [numerical-diagnostic explanation](docs/protocol.md#numerical-diagnostic) identifies the scope and limitations of the `5e-4` engineering check, including its model and probe graphs. These files are not additional full-campaign tasks.

## Recalculation and optional training

Follow [Reproduction commands](docs/reproduction.md) for an autonomous scratch project built from the complete archive, the exact isolated Windows chemistry recipe, and the historical Linux/CUDA training route. Aggregation/analysis can overwrite derived outputs, so their examples use a new external scratch. New training must use a new campaign and a configured site.

The delivered `_analysis/` contains `comparison.csv`, `datasets_vs_table1.csv` and `analysis.json`. Historical analysis also generates Spanish Markdown and figure outputs in scratch space; use `plot_paper_figures.py` for the presentation figures.

Local cliff verification used Python 3.8.20, RDKit 2022.09.5 and Levenshtein 0.21.1, with historical package pins plus Windows colorama. `Folds` and `CliffLabels` passed; `Versions` and `Featurization` require the absent Torch/PyG dependencies. The [reproduction guide](docs/reproduction.md) specifies the requirements and commands for recalculation, chemistry checks and optional training.

## Provenance, naming exceptions and limitations

The complete ZIP preserves all 38 Python sources recorded by all 150 tasks under `provenance/training_source/code/`. One infrastructure test, `test_launcher.py`, is ZIP-only because it contains cluster configuration; its hash remains recorded in `provenance/source_manifest.json`. The repository checker explicitly reports that single missing file as `archive-only`; every other missing source is an error. The complete ZIP must pass all 38 source-hash checks. These frozen files are never imported or executed. The strict validator uses them through `--source-root`; the documentation checker parses them as data. Infrastructure tests remain there because their identities are part of the historical record, even though they are excluded from active test selection.

The delivered code includes five bounded **post-training corrections**. They did not produce the saved campaign, and no retraining accompanies them. `tools/current_source_revision.json` ties each corrected file to its historical hash and exact current source identity. The command above verifies those explicit identities and requires unchanged executable syntax for every other recorded active source. Without `--revision-manifest`, the documentation-only checker still rejects executable changes, including these corrections. Historical source hashes and task records remain independently checked. See [source provenance](docs/upstream_changes.md#frozen-sources-and-active-code) for the comparison rules.

`gcn_full_v1`, `_summary`, `_analysis`, package/data identifiers and historical job filenames retain their original names because code or evidence requires them. Absolute cluster paths inside saved evidence describe the original execution; they are not installation instructions. Shared executable stage logic and synthetic tests can mention pilot/demo without depending on those campaigns.

The full log index records attempts **285913 and 285914**, but their individual log files were not received. They have not been reconstructed. Saved evidence validation verifies hashes, identities, metrics and protocol histories; it does not deserialise checkpoints or repeat training/inference. Historical commit IDs do not imply that the original Git objects are included.

If results are supplied as a separate supplement, extract the complete hierarchy at the paths shown here before validation. Checkpoints are required for the strict artefact-hash check even though they are never loaded as models by that validator.

The current revision requires an explicit mask for `calc_cliff_rmse`, makes the pairwise cliff threshold strictly greater than ten, uses six edge-feature columns for bondless graphs, rejects mixed training checkouts and continues status inspection past invalid task evidence. The completed evaluation used published cliff masks; its labels, predictions, stopping and checkpoint selection are unchanged. The real mixed-graph batch test remains pending because the available local environments lack Torch/PyG. See [post-training corrections](docs/upstream_changes.md#post-training-corrections) for interfaces, tests and remaining runtime limitations.

The 158 `.pt` files in the complete archive are **157 model checkpoints and one probe-graph archive**: 150 best states, six additional final states, one diagnostic model and `probe_graphs.pt`. The last two support the numerical diagnostic and remain ZIP-only with their recorded hashes. They are not two extra benchmark runs.

## Figure traceability

All 24 figures below are generated by `code/plot_paper_figures.py`, in PDF, SVG and PNG. Inputs are the corrected published CSV, corrected benchmark CSVs, `_analysis/comparison.csv` and `_summary/per_fold.csv`. `figure_data.csv` records the plotted dataset table; `figure_manifest.json` records all 30 benchmark CSVs, other tabular inputs, individual exports, supporting products, script identity and environment. The manifest excludes its own hash to avoid a circular identity; the final archive checksum inventory can cover it. Captions and catalogue use the same names.

Each individual figure fits within 16 cm wide by 22 cm high, with base font sizes of at least 7 pt; mathematical subscripts retain their usual relative size.

| Figure stem | Content |
|---|---|
| `paired_cliff_rmse` | Paired published/reproduced cliff RMSE |
| `paired_rmse` | Paired overall RMSE |
| `paired_error_metrics` | Three paired error measures |
| `cliff_rmse_strategy_context` | Cliff RMSE with published strategy context |
| `rmse_cliff_comparison` | Overall/cliff RMSE comparison |
| `rmse_cliff_comparison_overlay` | Comparison with published context overlay |
| `rmse_cliff_shifts_zoomed` | Published-to-reproduced shifts, zoomed |
| `rmse_cliff_shifts_context` | Shifts in broader published context |
| `cliff_error_difference_strategy_context` | Cliff-error difference with published context |
| `cliff_error_difference_by_dataset` | GCN cliff-error difference by dataset |
| `dataset_errors_mean_sd` | Dataset mean errors and fold SD |
| `dataset_errors_by_fold` | Individual fold errors |
| `cliff_error_difference_training_size` | Cliff-error difference against training size |
| `cliff_error_difference_training_size_comparison` | Published/reproduced size comparison |
| `published_reproduced_agreement` | Agreement scatter plots |
| `bland_altman_agreement` | Bland–Altman comparison |
| `published_reproduced_differences_by_dataset` | Reproduced minus published errors by dataset |
| `published_reproduced_difference_distribution` | Distribution of those differences |
| `published_strategy_ranking_context` | Published strategy ranking context |
| `activity_cliff_example_datasets` | Selected example datasets in published context |
| `activity_cliff_overview` | Composite activity-cliff comparison |
| `dataset_comparison_overview` | Composite dataset comparison |
| `gcn_results_overview` | Compact results overview |
| `dataset_error_comparison_table` | All 30 dataset identifiers, training/test-cliff molecule counts and two GCN error comparisons |

## Attribution, citation and licence

The implementation derives from [MoleculeACE](https://github.com/molML/MoleculeACE), by Derek van Tilborg and collaborators. The inherited licence is preserved unchanged in [LICENCE](LICENCE).

This repository is distributed under the original MoleculeACE MIT licence (see [LICENCE](LICENCE)), and the modifications and new code by Cristina Correa Segade are released under the same terms.

The bioactivity data originate from [ChEMBL](https://www.ebi.ac.uk/chembl/) and are distributed under the [Creative Commons Attribution-ShareAlike 3.0 Unported licence (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/), as stated by [ChEMBL](https://chembl.github.io/chembl-licensing/). This repository uses the version curated and corrected by MoleculeACE.

- Van Tilborg, Alenicheva and Grisoni, *Exposing the Limitations of Molecular
  Machine Learning with Activity Cliffs* (2022),
  [doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073).
- Early-stopping correction,
  [doi:10.1021/acs.jcim.3c00423](https://doi.org/10.1021/acs.jcim.3c00423).
- Corrected data, splits and results,
  [doi:10.1021/acs.jcim.3c01576](https://doi.org/10.1021/acs.jcim.3c01576).

The published-reference commit and hash are retained in `references/published/provenance.json`. The source experiment began from upstream commit `7e6de0bd2968c56589c580f2a397f01c531ede26`; the README's protocol history is based on the received project evidence, with the early-stopping inference explicitly distinguished from an observed upstream execution.
