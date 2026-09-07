![](https://img.shields.io/badge/version-1.1.0-blue)

# EDKG-DL: Causality-Integrated Graph Learning for Multi-Endpoint Toxicity Prediction

This repository is the official implementation of **EDKG-DL**, described in our manuscript: [**Causality-Integrated Graph Learning for Multi-Endpoint Toxicity Prediction**]. The paper has been accepted for publication in the *Proceedings of the National Academy of Sciences of the United States of America* (***PNAS***).

🔑 **Keywords**: *Causality-Integrated Graph Learning*; *Multi-Endpoint Toxicity Prediction*; *Endocrine-Disrupting Chemical*; *Knowledge Graph*; *Chemical Safety Assessment*

# 💖 Brief introduction
We propose an **D**eep **L**earning framework with causality-integrated **E**ndocrine **D**isruption **K**nowledge **G**raph (**EDKG-DL**), designed to enable efficient, interpretable, and sustainable screening of endocrine-disrupting chemicals (EDCs).

This repository provides the **core code and modeling data required to reproduce the main analyses of EDKG-DL**. Due to the large scale of the complete EDKG-DL framework and its associated datasets, the full knowledge graph, toxicological data resources, and integrated prediction system are not hosted directly on GitHub. These resources, together with the interactive prediction platform, are available through the EDKG-DL website 👉 https://www.edkgdl.com/.


<div align="center">
  <img src="images/EDKG-DL.png" alt="EDKG-DL platform" width="1000"/>
</div>

# 🎥 EDKG-DL Video Demonstration
https://github.com/user-attachments/assets/3000b196-a187-4507-bcde-b2eeed067cc7


## 🧭 Note on Prediction Modes
The current EDKG-DL web platform processes one compound at a time, primarily to enable interactive visualization and mechanistic interpretation of its compound-specific virtual perturbation map. A high-throughput batch prediction tool is currently under development, and its API will be made publicly available in a future release. Because batch prediction is designed for large-scale chemical screening, it will return structured prediction results without visualizing the virtual perturbation map for each individual compound. Users should therefore select the appropriate prediction mode according to whether their primary need is detailed mechanistic visualization or high-throughput screening.

# 🤖 Model

1. **Graph-informed data preparation**  
   Assay outcomes from 26 *in vitro* and *in vivo* tests are mapped onto EDKG biological entities, generating two toxicity-informed graph representations:  
   - a **qualitative graph** (74 entities with binary annotations), and  
   - a **quantitative graph** (52 entities with potency values, including QSAR-based predictions).  

2. **Qualitative EDKG-guided classification**  
   The qualitative graph is embedded into a three-layer edge-based **graph convolutional network (GCN)**. This module captures mechanistic interactions to predict whether a compound exhibits overall endocrine-disrupting potential, outputting a binary EDC/non-EDC label.  

3. **Quantitative EDKG-guided prioritization**  
   For compounds classified as EDCs, the quantitative graph is used to estimate **adverse outcome (AO)-specific potency** and predict **NOAEL** values. By tracing upstream causal pathways, this module identifies the most sensitive AO and its mechanistic drivers, supporting evidence-based prioritization and risk assessment.  

<div align="center">
  <img src="images/overview.png" alt="Overview of EDKG-DL" width="1000"/>
</div>


# 🔬 Requirements

All experiments were run in **Python** environment.

To run our code, please install dependency packages.
```
python          3.12.0
torch           2.8.0+cu126
torch-geometric 2.6.1
numpy           2.3.1
pandas          2.3.0
scikit-learn    1.6.1
torchvision     0.23.0+cu126
wandb           0.20.1
xgboost         3.0.2
```

# 📚 Overview

This project mainly contains the following parts.

```
├── Graph_informed_data_preparation/                 # Data and demos for biological entity-specific classification and regression
│
│   ├── qualitative_data_for_modeling/               # Raw modeling data for 74 biological entities, represented by PaDEL-derived molecular descriptors/fingerprints
│   │   ├── event_7-abortion_finger.csv              # Event 7: abortion
│   │   ├── event_47-adrenal_histop_finger.csv       # Event 47: adrenal histopathology
│   │   ├── event_<ID>-<biological_entity>_finger.csv
│   │   │                                            # Naming format: event_ID-biological_entity_finger.csv
│   │   └── ...                                      # One file per biological entity
│
│   ├── demo_data_cleaning_pipeline/                 # Complete example of the data-cleaning and preprocessing workflow using biological entity 7 (abortion)
│   │   ├── edkgdl_data_pipeline.py                  # Data cleaning, feature selection, standardization, train/test splitting, and resampling pipeline
│   │   ├── event_7-abortion_finger-2.csv            # Feature matrix after data cleaning and feature selection
│   │   ├── event_7-abortion_finger-3.csv            # Corresponding target labels after data cleaning
│   │   ├── event_7-abortion_finger-Xtrain.csv       # Standardized training features before SMOTE
│   │   ├── event_7-abortion_finger-Xtrain_.csv      # SMOTE-resampled training features
│   │   ├── event_7-abortion_finger-Xtest.csv        # Standardized held-out test features
│   │   ├── event_7-abortion_finger-Ytrain.csv       # Training labels before SMOTE
│   │   ├── event_7-abortion_finger-Ytrain_.csv      # SMOTE-resampled training labels
│   │   └── event_7-abortion_finger-Ytest.csv        # Held-out test labels
│
│   ├── quantitative_data_for_modeling/              # Cleaned and preprocessed modeling data for 52 biological entity-specific regression tasks, ready for direct model training
│   │   ├── Event_47-Xtrain.csv                      # Training features for biological entity 47
│   │   ├── Event_47-Ytrain.csv                      # Training targets for biological entity 47
│   │   ├── Event_47-Xtest.csv                       # Test features for biological entity 47
│   │   ├── Event_47-Ytest.csv                       # Test targets for biological entity 47
│   │   ├── Event_<ID>-Xtrain.csv                    # Naming format for training features
│   │   ├── Event_<ID>-Ytrain.csv                    # Naming format for training targets
│   │   ├── Event_<ID>-Xtest.csv                     # Naming format for test features
│   │   ├── Event_<ID>-Ytest.csv                     # Naming format for test targets
│   │   └── ...                                      # Four files for each of the 52 biological entities with quantitative data
│
│   ├── demo_biological_entity_classifier/           # Complete example of biological entity-specific qualitative classification using processed demo data
│   │   ├── edkgdl_biological_entity_classifier.py   # Training, hyperparameter optimization, evaluation, and model-export code for five classifiers
│   │   ├── predictive performance.csv               # Held-out test-set performance summary for the five classification algorithms
│   │   └── model_outputs/                           # Cross-validation results and trained models for each classifier
│   │       ├── cv_results_RandomForest.csv          # Grid-search cross-validation results for Random Forest
│   │       ├── cv_results_LinearSVC.csv             # Grid-search cross-validation results for Linear SVC
│   │       ├── cv_results_DecisionTree.csv          # Grid-search cross-validation results for Decision Tree
│   │       ├── cv_results_GaussianNB.csv            # Cross-validation results for Gaussian Naive Bayes
│   │       ├── cv_results_KNeighbors.csv            # Grid-search cross-validation results for K-Nearest Neighbors
│   │       ├── model_RandomForest.pkl               # Best trained Random Forest model
│   │       ├── model_LinearSVC.pkl                  # Best trained Linear SVC model
│   │       ├── model_DecisionTree.pkl               # Best trained Decision Tree model
│   │       ├── model_GaussianNB.pkl                 # Best trained Gaussian Naive Bayes model
│   │       └── model_KNeighbors.pkl                 # Best trained K-Nearest Neighbors model
│
│   └── demo_biological_entity_regressor/            # Complete example of biological entity-specific quantitative regression using processed Event 47 data
│       ├── edkgdl_biological_entity_regressor.py    # Training, hyperparameter optimization, evaluation, visualization, and model-export code for five regressors
│       ├── predictive performance (regression).csv  # Held-out test-set performance summary for the five regression algorithms
│       └── regression_outputs/                      # Cross-validation results, trained models, and parity plots for each regressor
│           ├── cv_results_RandomForestRegressor.csv
│           │                                        # Grid-search cross-validation results for Random Forest regression
│           ├── cv_results_DecisionTreeRegressor.csv
│           │                                        # Grid-search cross-validation results for Decision Tree regression
│           ├── cv_results_XGBRegressor.csv          # Grid-search cross-validation results for XGBoost regression
│           ├── cv_results_KNeighborsRegressor.csv   # Grid-search cross-validation results for K-Nearest Neighbors regression
│           ├── cv_results_SVR.csv                   # Grid-search cross-validation results for Support Vector Regression
│           ├── model_RandomForestRegressor.pkl      # Best trained Random Forest regression model
│           ├── model_DecisionTreeRegressor.pkl      # Best trained Decision Tree regression model
│           ├── model_XGBRegressor.pkl               # Best trained XGBoost regression model
│           ├── model_KNeighborsRegressor.pkl        # Best trained K-Nearest Neighbors regression model
│           ├── model_SVR.pkl                        # Best trained Support Vector Regression model
│           ├── parity_RandomForestRegressor.png     # Measured-versus-predicted plots for training and test sets
│           ├── parity_DecisionTreeRegressor.png     # Measured-versus-predicted plots for training and test sets
│           ├── parity_XGBRegressor.png              # Measured-versus-predicted plots for training and test sets
│           ├── parity_KNeighborsRegressor.png       # Measured-versus-predicted plots for training and test sets
│           └── parity_SVR.png                       # Measured-versus-predicted plots for training and test sets
│
├── Qualitative_EDKG_guided_classification/          # Causality-integrated graph learning for qualitative EDC prediction
│
│   ├── edkgdl_train_with_external_validation.py     # Primary EDKG-DL training, hyperparameter optimization, internal holdout evaluation, and independent external validation
│   │
│   ├── validation_1_repeated_group_nested_cv_analysis.py
│   │                                                # Repeated group-preserving nested cross-validation for model robustness assessment
│   │
│   ├── validation_2_negative_sample_ratio_sensitivity_analysis.py
│   │                                                # Sensitivity analysis across different EDC:non-EDC ratios
│   │
│   ├── validation_3_leave_one_edc_decoy_group_analysis.py
│   │                                                # Leave-one-EDC-decoy-group-out validation for strict group-level generalization assessment
│
│   ├── edkgdl_all_data/                             # Compound-specific graph inputs for 903 chemicals: 891 for internal model development and 12 for independent external validation
│   │   ├── 0/                                       # Internal confirmed EDC
│   │   │   ├── Graph_index.txt                      # Nodes and biological entity activation labels
│   │   │   └── Graph_edge_index_direct.txt          # Directed edges, attributes, and confidence
│   │   ├── 1/                                       # Internal confirmed EDC
│   │   │   ├── Graph_index.txt
│   │   │   └── Graph_edge_index_direct.txt
│   │   ├── ...
│   │   ├── 80/                                      # Internal confirmed EDC
│   │   ├── 81/                                      # External confirmed EDC
│   │   ├── ...
│   │   ├── 86/                                      # External confirmed EDC
│   │   ├── 87/                                      # Internal confirmed non-EDC
│   │   ├── ...
│   │   ├── 896/                                     # Internal confirmed non-EDC
│   │   ├── 897/                                     # External confirmed non-EDC
│   │   ├── ...
│   │   └── 902/                                     # External confirmed non-EDC
│
│   ├── EDKGDL_results/                              # Outputs from the primary EDKG-DL training and external-validation workflow
│   │   ├── internal_20pct_test_metrics.csv          # Performance on the held-out internal test set from the 891-compound modeling cohort
│   │   ├── independent_external_12_metrics.csv      # Performance on the independent external validation set of 12 compounds
│   │   └── ...                                      # Additional generated files, including predictions, trained models, training histories, hyperparameter-search results, dataset-role assignments, and run configurations
│
│   ├── EDKG_repeated_group_nested_cv_results/       # Outputs from repeated group-preserving nested cross-validation
│   │   ├── outer_fold_metrics.csv                   # Performance metrics from all repeated outer-fold evaluations
│   │   ├── metric_summary_bootstrap_ci.csv          # Summary statistics and bootstrap 95% confidence intervals
│   │   └── ...                                      # Additional generated files, including inner-CV results, selected hyperparameters, group mappings, and run configurations
│
│   ├── negative_ratio_sensitivity_internal891_zero_based_results/
│   │   # Outputs from EDC:non-EDC ratio sensitivity analysis
│   │   ├── outer_fold_metrics.csv                   # Outer-fold performance across different EDC:non-EDC ratios and sampling rounds
│   │   ├── ratio_metric_summary.csv                 # Overall performance summary for each EDC:non-EDC ratio
│   │   └── ...                                      # Additional generated files, including decoy-selection audits, coverage audits, group mappings, and run configurations
│
│   └── leave_one_internal891_edc_decoy_group_out_zero_based_results/
│       # Outputs from leave-one-EDC-decoy-group-out validation
│       ├── group_bootstrap_metric_ci.csv             # Group-level bootstrap 95% confidence intervals for LOGO validation
│       ├── pooled_oof_metrics.csv                    # Pooled performance metrics across all 891 out-of-fold predictions
│       └── ...                                      # Additional generated files, including training histories, bootstrap samples, group mappings, and run configurations
│
└── Quantitative_EDKG_guided_regression/
    └── README.txt                                   # AO prioritization and mechanistic backtracking
```


### 📑 Data schema (per compound folder)

Each compound folder (e.g., `0/`) contains two files describing its EDKG subset:

---

#### `Graph_index.txt`  
Two columns (whitespace or tab-delimited):

1. **node_id** *(int)*: node index in the EDKG  
2. **activated** *(int)*: node activation label  
   - `1` = activated  
   - `0` = not activated  

**Example**  
```
0 1
1 0
2 1
```
#### `Graph_edge_index_direct.txt`  

Four columns (whitespace or tab-delimited):

1. **src** *(int)*: source node id  
2. **dst** *(int)*: target node id  
3. **edge_attr** *(float)*: edge activation attribute  
   - both nodes activated → `1`  
   - exactly one node activated → `0.5`  
   - neither node activated → `0`  
4. **confidence** *(float)*: confidence score (larger = higher confidence)  

**Example**  
```
49 21 0 3
17 49 0 5
17 37 0 5
```


# 📝 Note
All training data were generated by first converting **SMILES strings** into **molecular descriptors/fingerprints**.  
The conversion was performed using **PaDEL-Descriptor**, which can be downloaded from:  
[http://yapcwsoft.com/dd/padeldescriptor/](http://yapcwsoft.com/dd/padeldescriptor/)

# About
Should you have any questions, please feel free to contact **Dr. Haoyue Tan**   at njutanhaoyue@nju.edu.cn.
