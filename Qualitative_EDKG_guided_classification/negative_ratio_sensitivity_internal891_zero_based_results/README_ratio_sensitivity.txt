Negative-ratio sensitivity analysis completed for the INTERNAL 891-compound cohort.

Validated graph IDs
-------------------
- Internal EDCs: 0–80 (n = 81)
- Internal non-EDCs: 87–896 (n = 810)
- Total internal ratio-sensitivity cohort: 891 graphs

Explicitly excluded from this analysis
--------------------------------------
- External EDCs: 81–86 (n = 6)
- External non-EDCs: 897–902 (n = 6)
- These 12 external graphs were never sampled as decoys, never entered any
  train/validation/test split, and never contributed to reported metrics.

Ratios
------
1:1, 1:2, 1:5, and 1:10 EDC:decoy datasets.

Sampling
--------
- All 81 internal EDCs (IDs 0–80) were retained in every analysis.
- For each ratio, 50 balanced decoy-sampling rounds were run.
- Each parent group contained one EDC plus the selected number of its own
  DUD-E decoys.
- Parent groups were never split across train and outer-test data.

Model configuration
-------------------
Manual fallback hyperparameters.
Fixed hyperparameters:
{'hidden_size1': 60, 'hidden_size2': 40, 'hidden_size3': 20, 'lr': 0.0005, 'weight_decay': 0.0005, 'dropout': 0.5}

Validation
----------
- For every ratio and sampling round, five-fold outer group CV was used.
- A group-preserving validation split inside each outer training set selected
  early stopping and the EDC positive-class F1-score threshold.
- Outer-test data were never used for model, epoch, or threshold selection.

Reporting
---------
- cohort_membership_and_group_mapping.csv explicitly documents all 903 graph
  IDs, their cohort membership, and their internal parent-group assignment.
- sampling_round_metrics.csv contains one combined evaluation per ratio-round.
- ratio_metric_summary.csv reports distributions across the 50
  sampling rounds, including bootstrap 95% CIs for mean metrics.
- outer_fold_metrics.csv and outer_fold_predictions.csv retain all raw outer
  fold-level results.
- selected_decoys_audit.csv and decoy_coverage_audit.csv document every decoy
  selection and its reuse across rounds.

Interpretation
--------------
This analysis tests whether performance within the internal 891-compound cohort
changes materially when the number of negative decoys per EDC changes. It
should be interpreted alongside the primary repeated group-nested CV analysis,
not as an external validation dataset. The independent 12-compound external
cohort is evaluated separately.
