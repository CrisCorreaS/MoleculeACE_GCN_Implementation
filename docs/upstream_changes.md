# Relationship to MoleculeACE and historical sources

The [protocol](protocol.md) defines computations; the [README](../README.md)
provides navigation and attribution. This document distinguishes inherited
behaviour, recorded experiment adaptations and later submission tooling.

## Reference implementation and inputs

The project began from MoleculeACE commit
`7e6de0bd2968c56589c580f2a397f01c531ede26`. The corrected published comparison CSV
is pinned to `45cf46d77f7681a7758d6447ffc3f239b349f406` by
[its provenance record](../references/published/provenance.json). Verification
permits LF/CRLF transport only, not modified values or row order. These identifiers
describe historical/external evidence; the original Git objects are not supplied.

The original article is DOI [10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073),
the stopping correction [10.1021/acs.jcim.3c00423](https://doi.org/10.1021/acs.jcim.3c00423),
and the corrected data/results [10.1021/acs.jcim.3c01576](https://doi.org/10.1021/acs.jcim.3c01576).
The latter appeared online in October 2023 and in the 2024 journal volume.

| Component | Relationship to MoleculeACE |
|---|---|
| Corrected CSVs, metadata and per-dataset YAMLs | Published inputs retained; no new download or optimisation |
| GCN and Graph Multiset Transformer | Inherited architecture and feature construction |
| Internal folds | Inherited median-stratified split with seed 42, executed as separate tasks |
| Epoch handling | Retains upstream removal of YAML epochs and the GCN limit of 300 |
| Stopping/state selection | Retains the consecutive implementation and epoch-ceiling boundary; the correction's wording differs |
| Package scope | Reduced to GCN_GRAPH; ECFP retained for cliff verification |

Upstream commit `141ebc9` reset the counter on improvement in March 2023; the
corrected reference results were updated later. That chronology supports, but
does not prove, which rule generated those scores. Published execution manifests
are not supplied. GPU reductions can be nondeterministic: separate processes with
the same seed do not establish bitwise reproduction across devices.

## Recorded adaptations

| Adaptation | Reason and consequence |
|---|---|
| Explicit train/evaluation modes | Makes handling explicit; zero dropout means no change is expected for these settings, without a universal numerical guarantee |
| Per-task state dictionaries and atomic replacement | Avoids a shared checkpoint; retains official upstream state and labelled secondary best-validation output |
| Batch-mean training-loss logging | Changes the logged statistic, not optimisation |
| Manifests, workers, locks, histories and hashes | Coordinates independent tasks and detects incomplete or mixed evidence |
| Configuration-format checks | Rejects unsupported formats and empty/non-mapping files; not every constructor keyword is validated |
| Empty cliff subset yields `None` | Avoids reporting zero error for no evaluated cliff molecules |
| Reduced models/imports | Removes unrelated models; data access still needs lightweight package dependencies |

## Frozen sources and active code

[`source_manifest.json`](../provenance/source_manifest.json) retains all 38 source
identities recorded by all 150 tasks. `provenance/training_source/code/` is evidence
of training: **never import or execute it**. Use active `code/` for current reading
and commands. Seven infrastructure files are retained only in the snapshot.

The delivered code contains the post-training corrections listed below. They are
separate from the sources that produced `gcn_full_v1`; no later training campaign
was run for this revision. Other active counterparts retain the historical
executable syntax and differ only in comments or docstrings.

`tools/check_documentation_only.py` still rejects every executable difference by
default. It compares abstract syntax trees (ASTs), removing only leading module,
class and function docstrings. All other strings, expressions and instructions
remain checked.

For the current revision, pass
`--revision-manifest tools/current_source_revision.json`. That bounded record
lists the five changed source files, their original historical hashes and exact
current SHA-256 identities. Current identities normalise only CRLF to LF; every
other source byte is checked, including documentation. An additional change
requires a new explicit identity and review. Unlisted sources must still have
unchanged ASTs; missing or altered historical sources still fail. The workflow
uses this explicit mode. Regression tests also prove that the default
documentation-only mode rejects these executable changes.

Only `provenance/training_source/code/tests/test_launcher.py` is ZIP-only: it embeds
real historical partition, QoS, account and node configuration. Its hash remains
in the complete manifest. In a sources-only public copy the checker reports that
one absent file as `archive-only`; every other missing frozen file or wrong hash
is an error. With complete task evidence, all 38 source files are mandatory.

The strict result validator uses `--source-root provenance/training_source` to
check historical identities. The source checker separately verifies the delivered
counterparts and the explicit correction identities. Neither executes the snapshot.
The local validator, figure generator and packaging tools have separate source
identities, which are not substituted into historical task records. Figure
provenance identifies its own generator and products.

## Post-training corrections

These changes affect future calls to the delivered code, not the historical
experiment or its saved evidence. The source record is
[`tools/current_source_revision.json`](../tools/current_source_revision.json).

| Source | Current contract | Historical relevance and verification |
|---|---|---|
| `benchmark/utils.py` | `calc_cliff_rmse` requires an explicit cliff mask; omission raises `ValueError`. Legacy optional arguments remain accepted for call compatibility but cannot request recomputation. | Existing callers supply a mask. The completed runner uses `artifacts.fold_metrics`; tests cover valid/empty masks and missing masks, including legacy arguments. |
| `benchmark/cliffs.py` | Pairwise `is_cliff` requires a concentration ratio strictly greater than `potency_fold` (ten by default). | Matches the existing bulk `ActivityCliffs` definition. Tests compare below, equal and above-threshold cases. Published labels remain unchanged. |
| `benchmark/featurization.py` | A bondless graph has `edge_attr` shape `(0, 6)`, consistent with ordinary bonds. | The real RDKit/Torch/PyG mixed-graph batch test is included but remains unexecuted locally because Torch/PyG are absent. This fix is not yet fully runtime-validated. |
| `run_benchmark.py` | The entry script, actual loaded scientific modules/functions and import origins must resolve to the selected checkout before task outputs are created; lazy scientific imports are checked again. | Coherent clean copies and equivalent resolved paths are supported. Regression tests reject mixed checkouts, foreign imports and cached foreign functions without training. Historical records still identify the original source tree. |
| `task_status.py` | Expected file, JSON and required-field errors are reported per task; later tasks are still listed. Exit 1 means invalid evidence, 2 a manifest/campaign error, and 0 a valid listing. Unexpected exceptions propagate. | Synthetic corrupt-record tests and the complete saved-task listing exercise this behaviour. A valid stored `failed` state is distinct from malformed evidence. |

The focused tests are `code/tests/test_current_science.py`,
`test_training_source_roots.py` and `test_task_status.py`. The explicit scientific
runner selects them alongside the existing suite. Source-verification tests
separately exercise exact revision identities and unchanged historical checks.

Keep each training checkout unchanged throughout a task: the source-root guard
checks path consistency, not concurrent edits to source files.

No GPU training or new Slurm execution validates this revision. The retained
early-stopping and checkpoint-selection protocol is unchanged. CPU checks do not
certify arbitrary new datasets, cluster configurations or GPU numerical behaviour.
