Repeated group-nested CV completed for the INTERNAL 891-compound cohort.

Validated graph IDs
-------------------
- Internal EDCs: 0–80 (n = 81)
- Internal non-EDCs: 87–896 (n = 810)
- Total internal nested-CV cohort: 891 graphs

Explicitly excluded from this nested-CV analysis
------------------------------------------------
- External EDCs: 81–86 (n = 6)
- External non-EDCs: 897–902 (n = 6)
- These 12 external graphs were never used in the outer splits, inner CV,
  hyperparameter selection, threshold selection, model fitting, or metrics.

Design
------
- 20 repeats × 5 outer group folds = 100 outer evaluations.
- Each internal parent group contains one EDC and 10 paired decoys.
- All samples from one internal parent group are kept in either train or test.
- Hyperparameters, early stopping, and classification thresholds are selected
  only within the internal outer-training groups through 4-fold inner CV.

Main output files
-----------------
- cohort_membership_and_group_mapping.csv
    Explicitly records all 903 graph IDs, their labels, whether they belong to
    internal nested-CV or external-excluded cohorts, and internal parent groups.
- outer_fold_metrics.csv
    Metrics from every outer test evaluation.
- metric_summary_bootstrap_ci.csv
    Mean, median, standard deviation, range, and bootstrap 95% CI for the
    mean metric across outer evaluations.
- outer_fold_predictions.csv
    Out-of-fold predicted probabilities and labels for every graph.
- selected_hyperparameters_per_outer_fold.csv
    Hyperparameters, epochs, and thresholds selected within each outer fold.
- inner_cv_hyperparameter_results.csv
    Full nested inner-CV tuning results.

Interpretation
--------------
These results quantify robustness only within the 891-compound internal cohort.
PR-AUC, recall, specificity, balanced accuracy, MCC, and F1-score should be
emphasized in the manuscript. Accuracy is retained as a secondary metric only.
The 12 external compounds require separate, one-time independent evaluation.
