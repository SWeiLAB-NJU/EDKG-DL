Internal 891-compound training plus independent external validation completed.

Zero-based cohort definition
----------------------------
Internal modeling cohort (n = 891)
- EDCs: graph IDs 0–80 (n = 81)
- non-EDCs: graph IDs 87–896 (n = 810)

Independent external cohort (n = 12)
- EDCs: graph IDs 81–86 (n = 6)
- non-EDCs: graph IDs 897–902 (n = 6)

Internal model development and evaluation
-----------------------------------------
- Only the internal 891 compounds were organized into 81 parent groups.
- Each parent group contained one EDC and 10 paired DUD-E decoys.
- A group-preserving 80/20 split assigned 65 groups (715 graphs) to internal
  development and 16 groups (176 graphs) to an untouched internal test set.
- Hyperparameters were selected only with 10-fold group CV inside the 715
  internal development graphs.
- Selection criterion: pooled inner out-of-fold positive-class F1-score,
  followed by PR-AUC, balanced accuracy, and ROC-AUC.
- No external sample was used in splitting, tuning, model selection, or the
  internal 20% holdout assessment.

Independent external validation
-------------------------------
- After model selection, the selected hyperparameters were refit on all 891
  internal graphs.
- The resulting model was evaluated once on the independent external 12 graphs.
- The external dataset was never used for hyperparameter selection or training.

Selected model
--------------
{'hidden_size1': 40, 'hidden_size2': 50, 'hidden_size3': 100, 'lr': 0.0007162378850072, 'weight_decay': 0.0005040261196391, 'batch_size': 50, 'dropout': 0.5}

Loss and decision rule
----------------------
- Class-weighted loss: False
- Classification threshold: 0.5
- F1-score: positive-class F1-score for EDCs (label = 1)

Key files
---------
Internal evaluation:
- internal_20pct_test_metrics.csv
- internal_20pct_test_predictions.csv
- internal_80pct_model.pt

Independent external evaluation:
- independent_external_12_metrics.csv
- independent_external_12_predictions.csv
- all_internal_891_model_for_external_validation.pt

Training/tuning audit:
- all_903_dataset_roles_and_internal_split.csv
- candidate_hyperparameters.csv
- hyperparameter_trial_summary.csv
- best_internal_inner_cv_oof_predictions.csv
- selected_model.json
