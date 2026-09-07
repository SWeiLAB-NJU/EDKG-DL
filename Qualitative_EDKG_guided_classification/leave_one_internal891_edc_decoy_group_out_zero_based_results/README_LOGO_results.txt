Leave-one-internal-EDC-decoy-group-out validation completed.

Validated graph IDs
-------------------
- Internal EDCs: 0–80 (n = 81)
- Internal non-EDCs: 87–896 (n = 810)
- Total internal LOGO cohort: 891 graphs

Explicitly excluded from this LOGO analysis
--------------------------------------------
- External EDCs: 81–86 (n = 6)
- External non-EDCs: 897–902 (n = 6)
- These 12 external graphs did not enter any LOGO train/test split, model
  fitting, pooled OOF table, group bootstrap sample, or reported metric.

Core design
-----------
- 81 validation rounds were performed.
- In each round, one internal EDC and its 10 paired internal DUD-E decoys were
  withheld as a complete 11-chemical test group.
- The model was trained on the remaining 80 internal EDCs and 800 internal
  decoys.
- Fixed hyperparameters were specified before LOGO validation:
  {'hidden_size1': 60, 'hidden_size2': 40, 'hidden_size3': 20, 'lr': 0.0005, 'weight_decay': 0.0005, 'batch_size': 64, 'dropout': 0.5}
- Hyperparameter source:
  Used manual fallback hyperparameters.
- Weighted loss:
  False
- Training epochs per LOGO fold:
  200
- Fixed classification threshold:
  0.5

Pooled OOF result
-----------------
- Every graph received exactly one prediction from a model that did not include
  its parent EDC-decoy group in classifier training.
- Correctly classified held-out EDCs at threshold 0.5:
  56/81
- Held-out EDCs ranked first within their own 11-compound EDC-decoy group:
  73/81

Main files
----------
- cohort_membership_and_group_mapping.csv
    Auditable membership of all 903 graph IDs, including the 891 internal LOGO
    graphs and the 12 external IDs explicitly excluded from this analysis.
- leave_one_group_predictions.csv
    Complete out-of-fold predictions for the 891 internal graph IDs only:
    0–80 and 87–896.
- pooled_oof_metrics.csv
    Overall pooled OOF PR-AUC, ROC-AUC, recall, specificity, balanced accuracy,
    MCC, positive-class F1-score, and accuracy.
- group_bootstrap_metric_ci.csv
    Group-level bootstrap 95% confidence intervals, obtained by resampling the
    81 parent EDC-decoy groups with replacement.
- per_edc_heldout_summary.csv
    Per-EDC held-out probabilities, fixed-threshold predictions, and rankings
    relative to the corresponding 10 withheld decoys.
- leave_one_group_metrics.csv
    Fold-level descriptive metrics. Per-fold AUCs should not be used as the
    primary reported evidence because each fold has only one EDC.
- training_history.csv
    200-epoch training loss trace for every LOGO fold.

Scope
-----
This script establishes classifier-level generalization to unseen internal
EDC-decoy groups. It does not cross-fit the upstream 74 entity-specific QSAR
annotation models. That separate analysis would be needed to rule out upstream
model-training overlap completely. Independent external validation is conducted
separately using graph IDs 81–86 and 897–902.
