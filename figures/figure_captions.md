# Figure captions

Common text for every caption:

Each dataset is a target–response combination from the corrected MoleculeACE benchmark (n = 30). Reproduced values are arithmetic means of five official fold RMSEs, each evaluated on the same published test set; they are not RMSEs of averaged predictions. Where shown, reproduced error bars are the sample SD across those folds (ddof = 1): folds share the test set and overlap in their training data, so the SD is neither a confidence interval nor seed-to-seed variation. Published values are corrected dataset-level scores (upstream commit 45cf46d; DOI 10.1021/acs.jcim.3c01576) with no fold dispersion available. Errors are in log10 units of the response y = −log10(concentration in nM). Indigo circles: published GCN; crimson diamonds: reproduced GCN; grey: other published strategies (external data, not reproduced).

## paired_cliff_rmse — Fig. 4a adaptation, paired GCN boxes (cliff)

GCN-only adaptation of corrected Fig. 4a. RMSE_cliff of the published and the reproduced GCN across datasets. Boxes show the median and interquartile range; whiskers extend to 1.5 × IQR; thin grey lines join the two values of the same dataset.

## paired_rmse — Fig. 4a adaptation, paired GCN boxes (rmse)

As paired_cliff_rmse for whole-test RMSE.

## paired_error_metrics — Paired GCN boxes for RMSE, RMSE_cliff and their difference

(a) RMSE, (b) RMSE_cliff and (c) RMSE_cliff − RMSE for the published and reproduced GCN; lines join datasets. The difference in (c) is computed within folds before averaging and therefore preserves the dependence between the cliff subset and the whole test.

## cliff_rmse_strategy_context — Fig. 4a with the reproduction inserted among published deep-learning strategies

Corrected Fig. 4a with the reproduced GCN inserted beside the published GCN (shaded). Grey boxes are the other published deep-learning strategies and the two ECFP reference models (MLP, SVM), in the order of the corrected figure; they were not retrained here. The equally spaced GCN categories are labelled publ. (published) and repr. (reproduced).

## rmse_cliff_comparison — Fig. 4c: published and reproduced GCN in separate panels

Corrected Fig. 4c adaptation. RMSE_cliff against RMSE for (a) the published and (b) the reproduced GCN, on identical axes. Grey points are all other published strategy–dataset pairs. The black dashed line is RMSE_cliff = RMSE; grey dashed lines mark ±0.5 log units. Fold SD bars are omitted so that both panels show the same quantity.

## rmse_cliff_comparison_overlay — Fig. 4c: published and reproduced GCN in separate panels (with ghost)

As rmse_cliff_comparison, with the other source drawn as hollow markers in each panel to make the shift visible.

## rmse_cliff_shifts_zoomed — Published-to-reproduced shift per dataset in the RMSE plane

Shift of each dataset from its published (circle) to its reproduced (diamond) position in the RMSE–RMSE_cliff plane; axes are zoomed to the GCN range. The four largest shifts are labelled descriptively; labels do not imply a cause. Shifts shorter than 5 pt at the exported size use a thin connector without an arrowhead, drawn above the marker fills at the true endpoints; circles and diamonds identify the published and reproduced positions.

## rmse_cliff_shifts_context — Published-to-reproduced shift per dataset in the RMSE plane

As rmse_cliff_shifts_zoomed on the full axes of corrected Fig. 4c, with other published strategies in grey. Shifts shorter than 5 pt at the exported size use a thin connector without an arrowhead at the true endpoints; circles and diamonds identify the published and reproduced positions.

## cliff_error_difference_strategy_context — Fig. 5a adaptation: RMSE_cliff − RMSE per dataset

Adaptation of corrected Fig. 5a. RMSE_cliff − RMSE per dataset. Grey boxes summarise the 23 other published strategies (box: IQR and median; whiskers: lowest and highest value, as in the paper). Circles: published GCN; diamonds: reproduced GCN ± fold SD of the within-fold difference. Rows and the r column follow the corrected figure: Pearson r between RMSE and RMSE_cliff across the 24 published strategies, recomputed from the published table.

## cliff_error_difference_by_dataset — Fig. 5a adaptation: RMSE_cliff − RMSE per dataset

As cliff_error_difference_strategy_context without the across-strategy boxes.

## dataset_errors_mean_sd — Per-dataset published versus reproduced RMSE and RMSE_cliff

Published and reproduced (a) RMSE and (b) RMSE_cliff per dataset, joined by a line; reproduced values show ± fold SD. Rows follow the corrected Fig. 5a order, which is fixed by published data and independent of the reproduction.

## dataset_errors_by_fold — Per-dataset published versus reproduced RMSE and RMSE_cliff

As dataset_errors_mean_sd, with the five individual reproduced fold scores (hollow diamonds, offset below each row) instead of the SD.

## cliff_error_difference_training_size — Fig. 5d adaptation: RMSE_cliff − RMSE versus training-set size

Adaptation of corrected Fig. 5d. RMSE_cliff − RMSE against the number of molecules in the published training partition (before the internal fold division). Vertical grey segments join the two values of a dataset. The relation is descriptive, not causal.

## cliff_error_difference_training_size_comparison — Fig. 5d adaptation: RMSE_cliff − RMSE versus training-set size

As cliff_error_difference_training_size with (a) published and (b) reproduced GCN in separate panels.

## published_reproduced_agreement — Reproduced versus published dataset means

Reproduced against published dataset means for (a) RMSE, (b) RMSE_cliff and (c) RMSE_cliff − RMSE; dashed line: identity. Summary statistics are descriptive over the 30 datasets; no paired fold test or equivalence claim is made, because published fold scores are unavailable.

## bland_altman_agreement — Bland–Altman agreement plots (descriptive)

Bland–Altman plots for (a) RMSE and (b) RMSE_cliff: reproduced minus published against their mean. Solid line: mean difference; dashed lines: mean ± 1.96 SD of the 30 dataset differences (descriptive limits of agreement across datasets, not fold-level uncertainty).

## published_reproduced_differences_by_dataset — Reproduced minus published, per dataset

Reproduced minus published (a) RMSE, (b) RMSE_cliff and (c) RMSE_cliff − RMSE per dataset. Positive values indicate larger reproduced errors in (a, b), and a larger cliff-versus-whole-test error gap in (c). Dark horizontal segments connect each difference estimate to zero as a visual guide. Light-grey capped bars show ± reproduced fold SD around the difference from the fixed published value, not the uncertainty of a difference between two estimated means.

## published_reproduced_difference_distribution — Distribution of reproduced minus published differences

Distribution over the 30 datasets of reproduced minus published scores. Violins are kernel density estimates; the vertical bar marks the median.

## published_strategy_ranking_context — Where the reproduced GCN sits among published strategies

(a) Mean RMSE_cliff over the 30 datasets for the 24 published strategies, with the reproduced GCN added. (b) Rank of GCN among the 24 strategies on each dataset when the published GCN is used and when it is replaced by the reproduction; lines join datasets. Other strategies are external published results and were not retrained.

## activity_cliff_example_datasets — Fig. 5b,c adaptation: 5-HT1A and CLK4 with all published strategies

Adaptation of corrected Fig. 5b,c: RMSE_cliff against RMSE of all published strategies (grey) on (a) 5-HT1A and (b) CLK4, the examples discussed in the paper, with the published and reproduced GCN highlighted. Reproduced bars: ± fold SD.

## activity_cliff_overview — Composite in the layout of corrected Figure 4

Composite in the layout of corrected Fig. 4: (a) Fig. 4a with the reproduction inserted; (b) reproduced against published RMSE_cliff; (c, d) Fig. 4c for published and reproduced GCN; (e) RMSE_cliff − RMSE against training-set size for both sources. Error bars in (b, e) show ±1 SD across the five reproduced folds; in (e), the SD is calculated from each fold's RMSE_cliff − RMSE. In (a), publ. denotes the published GCN and repr. denotes the reproduced GCN.

## dataset_comparison_overview — Composite in the layout of corrected Figure 5

Composite in the layout of corrected Fig. 5: (a) Fig. 5a with the published GCN and the reproduction; (b, c) 5-HT1A and CLK4; (d) Fig. 5d for both sources; (e) reproduced against published RMSE_cliff − RMSE. The r column reports the Pearson correlation between RMSE and RMSE_cliff across published strategies for each dataset. Error bars show ±1 SD across the five reproduced folds; those for RMSE_cliff − RMSE use the per-fold differences.

## gcn_results_overview — README summary banner

Summary for the README: published and reproduced GCN in the Fig. 4c layout, and reproduced against published RMSE_cliff.

## dataset_error_comparison_table — Dataset counts and GCN error comparisons

All 30 target–response datasets, ordered by the size of the published training partition (n_train, before internal fold subdivision), then by dataset identifier. The table gives the full ChEMBL dataset identifier, target abbreviation and n_cliff: the number of published test molecules labelled as activity-cliff molecules, with each molecule counted once rather than counting cliff pairs. The response type (Ki or EC50) remains in the full dataset identifier. (a) RMSE_cliff − RMSE for the corrected published GCN (indigo circles) and the reproduction (crimson diamonds). Reproduced bars are ± the sample SD of the five paired within-fold differences, retaining the covariance of the two errors; no published fold dispersion is available. (b) Reproduced minus published RMSE_cliff (grey squares); positive values indicate a larger reproduced cliff error. Dark horizontal segments connect each difference estimate to zero as a visual guide. Light-grey capped bars show ± reproduced RMSE_cliff fold SD around the difference from the fixed published value, not uncertainty in the difference between two estimated means. These SDs describe the five overlapping training folds evaluated on the same published test set, not confidence intervals or seed-to-seed variation. Errors and differences are in log10 units of the response y = −log10(concentration in nM).
