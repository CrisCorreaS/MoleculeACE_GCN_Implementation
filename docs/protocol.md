# Scientific and execution protocol

This document specifies the completed `gcn_full_v1` experiment. The [README](../README.md)
provides the reading route; [reproduction](reproduction.md) contains commands and
[upstream changes](upstream_changes.md) distinguishes inherited and adapted behaviour.

## Inputs, units and folds

The scope is GCN_GRAPH on 30 corrected dataset-response combinations, with five
internal folds each. No other strategy was trained and no new hyperparameter
search was performed. The published CSVs retain row order, train/test assignments
and molecule-level cliff labels. No new ChEMBL download, deduplication or scaffold
split is part of the experiment; datasets need not represent distinct proteins.

| Field | Meaning |
|---|---|
| `smiles` | Molecular string used for fixed graph featurisation |
| `exp_mean [nM]` | Positive experimental concentration in nanomolar units |
| `y` | Regression response: minus log10 of concentration in nM |
| `y [pEC50/pKi]` | Corresponding conventional response, `y + 9` |
| `split` | Published `train` or `test` membership |
| `cliff_mol` | Published binary molecule label, used for evaluation only |

Predictions use the `y` scale. Adding nine to observations and predictions leaves
RMSE unchanged. `benchmark.utils.molecular_folds` applies
`StratifiedKFold(5, shuffle=True, random_state=42)` within published training data,
stratified below/above its response median; equality belongs to the upper group.
Each molecule validates once across the five folds. Validation is excluded from
that fold's optimisation. All models use the same published test set, whose labels
do not select hyperparameters, stopping times or checkpoints. Featurisation is
not fitted to test responses. The split is not claimed to be scaffold-independent.

## Representation and model

`benchmark.featurization` produces `(n_atoms, 37)` node features, connectivity of
shape `(2, n_directed_edges)` and normally six features per edge. Nodes encode
element, atomic weight, valence, hydrogen count, degree, hybridisation, partial
charge, aromaticity and ring membership. Bonds encode type and conjugation.

`models.gcn.GCNmodel.forward` consumes **node features, connectivity and batch
membership**. It accepts `edge_attr` but does not use it in the convolutions or
pooling. The graph representation therefore contains bond attributes that this
GCN does not consume. The historical bondless-graph branch used ten empty
edge-feature columns; the delivered post-training revision corrects this to
`(0, 6)`. Its real mixed-graph batch check remains pending Torch/PyG availability;
see [upstream changes](upstream_changes.md#post-training-corrections).

GCN convolutions and ReLU feed eight-head Graph Multiset Transformer pooling,
fully connected layers and a scalar linear regression output. Adam minimises
MSE. Published settings come from
`Data/configures/benchmark/<dataset>/GCN_GRAPH.yml`; all use dropout zero.
The default `GCN.yml` is contextual material, not a search executed here.

## Stopping and official state

The upstream benchmark removed YAML `epochs` before constructing GCN, so the
effective limit is the constructor's **300 epochs**. The ignored value is retained
as `yaml_epochs_ignored`. Model initialisation and internal splitting use seed 42;
the training loader does not shuffle.

`upstream_consecutive` uses patience ten. A validation loss less than or equal to
the best loss counts as improvement and resets the counter. The 2023 correction
describes ten non-consecutive non-improving epochs. These rules differ. Attribution
of corrected published scores to the consecutive implementation is inferred from
repository chronology, not proved by published execution manifests. The protocol
was not changed to improve numerical agreement.

When stopping takes effect before the ceiling, the official result uses the best
validation state. At the ceiling it uses the last state, including the boundary
case where patience is exhausted on that final allowed epoch. When last and best
coincide, no additional final checkpoint is needed. Secondary best-validation
predictions come from the same training, without test-based result selection.

There are 139 tasks stopped before the ceiling and eleven reaching it. Five have
their best state at epoch 300; six require a separate final state. Thus the 150
tasks contain 150 `best_model.pt` and six `final_model.pt` files.

## Cliffs, metrics and uncertainty

`ActivityCliffs` combines ECFP/Tanimoto, generic-scaffold Tanimoto and normalised
SMILES/Levenshtein similarity using OR. Similarity must be >= 0.9 by at least one
measure; the concentration ratio must be **strictly > 10**. This operates on nM
concentrations, not ratios of logarithmic responses, across the complete dataset
when checking published labels. Each cliff molecule counts once, irrespective
of its number of cliff partners. Evaluation always uses the published mask.

`benchmark.artifacts.fold_metrics` calculates RMSE over test molecules and
RMSE_cliff over their cliff subset. An empty subset yields a missing value, not
zero. Dataset estimates are arithmetic means of five per-fold RMSEs, not RMSEs
of averaged predictions. SD is sample SD (`ddof=1`). `delta_mean` and `delta_sd`
use paired within-fold differences `RMSE_cliff - RMSE`, retaining covariance.
Folds overlap in training and share the test; SD is not an independent-seed
estimate or confidence interval. Published scores lack fold-level dispersion;
comparisons with them and other published strategies are descriptive.

## Execution boundary and artefacts

`run_benchmark.py` executes one manifest row; `jobs/manifest_full.csv` defines all
150 historical tasks. The GPU worker coordinates two allocated GPU lanes, with
one GPU and ten CPU threads per child. Scheduler values belong in private
`jobs/cluster.env`; the example uses placeholders. A receiving site must validate
its own policy, CUDA libraries and resource allocation. Local CPU checks do not
repeat training or certify a new Slurm installation.

Keep the training script, imported package and `PROJECT_ROOT` in the same checkout.
The historical runner did not fully enforce that correspondence. The delivered
post-training revision checks resolved entry-point and scientific import paths
before creating task outputs, and checks lazy scientific imports again. The
separate-root scratch recipe applies to post-hoc checks, not mixed-checkout
training. Historical source validation always uses the frozen snapshot.

Tasks preserve official/secondary predictions and metrics, checkpoints, histories,
fold indices, input/source/artefact hashes, environment and status. CSV/JSON writes
are atomic and synchronised. Checkpoint replacement is atomic but does not call
`fsync`; it does not provide the same explicit storage-durability guarantee.
Strict validation hashes checkpoints without loading models.

## Numerical diagnostic

`results/diagnostics/gcn_repeatability_285912_0/` is an engineering check, not a
151st scientific task. `model_checkpoint.pt` is its four-epoch model;
`probe_graphs.pt` stores input graphs, **not another checkpoint**. `report.json`
records both hashes. The complete delivery therefore contains 158 `.pt` files:
157 model checkpoints and one graph archive. Retain both diagnostic files with
their report in the complete archive; neither belongs in the public repository.

The diagnostic uses 40 training molecules at positions 40-79 and 20 GPU predictions
at original precision. Its maximum per-molecule span was
0.000022411346435546875, below the absolute `5e-4` engineering budget. TF32/RNG
restoration is recorded. The budget was selected after an earlier diagnostic
and before this check; it is half the analysis table's 0.001 display step. It is
not a theoretical bound, identical-rounding guarantee, training-variation tolerance
or proof for every future molecule/checkpoint. Benchmark precision was unchanged.
