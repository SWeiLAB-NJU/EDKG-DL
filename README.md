![](https://img.shields.io/badge/version-1.0.0-blue)

# EDKG-DL: An Interpretabile Endocrine Disruption Knowledge Graph-Augmented Deep Learning Model for Rapid and Accurate Prediction of Endocrine-Disrupting Chemicals

This repository is the official implementation of **EDKG-DL**, proposed in our unpublished manuscript:  
[**Causality-Integrated Graph Learning for Multi-Endpoint Toxicity Prediction**].  
The paper has been submitted and is currently under peer review.

🔑 **Keywords**: endocrine disruption, knowledge graph, deep learning, causality, toxicity prediction

# 💖 Brief introduction
We propose an **D**eep **L**earning framework with causality-integrated **E**ndocrine **D**isruption **K**nowledge **G**raph (**EDKG-DL**), designed to enable efficient, interpretable, and sustainable screening of endocrine-disrupting chemicals (EDCs).  
This repository provides the **core code and modeling data** used in our unpublished manuscript (currently under peer review). For the **complete model**, including the endocrine disruption knowledge graph, toxicology datasets, and interactive prediction platform, please visit 👉 [https://www.edkgdl.com/#/](https://www.edkgdl.com/).

<div align="center">
  <img src="images/EDKG-DL.png" alt="EDKG-DL platform" width="1000"/>
</div>

## 🤖 Model

1. **Graph-informed data preparation**  
   Assay outcomes from 26 in vitro and in vivo tests are mapped onto EDKG biological elements, generating two toxicity-informed graph representations:  
   - a **qualitative graph** (74 elements with binary annotations), and  
   - a **quantitative graph** (52 elements with potency values, including (Q)SAR-based predictions).  

2. **Qualitative EDKG-guided classification**  
   The qualitative graph is embedded into a three-layer edge-based **graph convolutional network (GCN)**. This module captures mechanistic interactions to predict whether a compound exhibits overall endocrine-disrupting potential, outputting a binary EDC/non-EDC label.  

3. **Quantitative EDKG-guided prioritization**  
   For compounds classified as EDCs, the quantitative graph is used to estimate **adverse outcome (AO)-specific potency** and predict **NOAEL** values. By tracing upstream causal pathways, this module identifies the most sensitive AO and its mechanistic drivers, supporting evidence-based prioritization and risk assessment.  

<div align="center">
  <img src="images/overview.png" alt="Overview of EDKG-DL" width="1000"/>
</div>


# 🔬 Requirements

All experiments were run in **Jupyter Notebook** environment.

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
├── Graph-informed data preparation/
│   │                                                # Data and demos for biological entity-specific classification and regression
│   │
│   ├── qualitative_data_for_modeling/               # Data for entity-specific classifiers (active/inactive)
│   │   └── event_X/                                 # One folder per biological entity
│   │       ├── Xtrain.csv                           # Training features
│   │       ├── Ytrain.csv                           # Training labels
│   │       ├── Xtest.csv                            # Test features
│   │       └── Ytest.csv                            # Test labels
│   │
│   ├── quantitative_data_for_modeling/              # Data for entity-specific regressors (NOAEL values)
│   │   └── event_Y/                                 # One folder per biological entity
│   │       ├── xtrain.csv                           # Training features
│   │       ├── ytrain.csv                           # Training targets
│   │       ├── xtest.csv                            # Test features
│   │       └── ytest.csv                            # Test targets
│   │
│   ├── demo_data_cleaning_pipeline/                 # Example data-preprocessing workflow
│   │   ├── edkgdl_data_pipeline.py                  # Data-cleaning and preprocessing code
│   │   └── event_7-abortion_finger.csv              # Example raw biological entity dataset
│   │
│   ├── demo_biological_entity_classifier/           # Example entity-specific classification workflow
│   │   ├── edkgdl_biological_entity_classifier.py   # Classification model training code
│   │   ├── Xtrain.csv                               # Example training features
│   │   ├── Ytrain.csv                               # Example training labels
│   │   ├── Xtest.csv                                # Example test features
│   │   └── Ytest.csv                                # Example test labels
│   │
│   └── demo_biological_entity_regressor/            # Example entity-specific regression workflow
│       ├── edkgdl_biological_entity_regressor.py    # Regression model training code
│       ├── xtrain.csv                               # Example training features
│       ├── ytrain.csv                               # Example training targets
│       ├── xtest.csv                                # Example test features
│       └── ytest.csv                                # Example test targets
│
├── Qualitative EDKG-guided classification/
│   │                                                # Causality-integrated graph learning for qualitative EDC prediction
│   │
│   ├── edkgdl_train_with_external_validation.py     # Model optimization, training, and validation
│   │
│   ├── validation_1_repeated_group_nested_cv_analysis.py
│   │                                                # Repeated group-nested cross-validation
│   │
│   ├── validation_2_negative_sample_ratio_sensitivity_analysis.py
│   │                                                # Negative-sample ratio sensitivity analysis
│   │
│   ├── validation_3_leave_one_edc_decoy_group_analysis.py
│   │                                                # Leave-one-EDC-decoy-group-out validation
│   │
│   ├── EDKGDL_results/                              # Final model and performance results
│   │   ├── edkg_dl_model.pt                         # Final trained EDKG-DL model
│   │   ├── internal_test_set_performance_metrics.csv
│   │   │                                            # Internal test-set performance
│   │   └── independent_external_validation_performance_metrics.csv
│   │                                                # External validation results for 12 compounds
│   │
│   └── edkgdl_all_data/                             # Graph data for training, validation, and testing
│       ├── 0/                                       # Compound 0
│       │   ├── Graph_index.txt                      # Nodes and biological entity activation labels
│       │   └── Graph_edge_index_direct.txt          # Directed edges, attributes, and confidence
│       ├── 1/                                       # Compound 1
│       │   ├── Graph_index.txt
│       │   └── Graph_edge_index_direct.txt
│       ├── 2/                                       # Compound 2
│       │   ├── Graph_index.txt
│       │   └── Graph_edge_index_direct.txt
│       └── ...                                      # One folder per compound
│
└── Quantitative EDKG-guided regression/
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
