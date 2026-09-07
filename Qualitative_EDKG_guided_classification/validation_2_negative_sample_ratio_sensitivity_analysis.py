#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Negative-sample ratio sensitivity analysis for the qualitative EDKG-DL classifier.

Purpose
-------
This script evaluates whether classifier performance is sensitive to the
EDC:non-EDC ratio. It uses all 81 confirmed EDCs and samples 1, 2, 5, or 10
paired DUD-E decoys per EDC, producing 1:1, 1:2, 1:5, and 1:10 datasets.

Key design features
-------------------
1. Group preservation:
   Each EDC and its selected decoys are assigned together to one parent group.
   Thus, paired EDC-decoy samples can never be split between training and
   outer-test sets.

2. Balanced decoy coverage:
   For 50 sampling rounds, each decoy receives balanced reuse across rounds.
   For example, under the 1:1 analysis each of the 10 decoys for each EDC is
   sampled exactly 5 times across 50 rounds.

3. Ratio-isolation strategy:
   Model hyperparameters are held fixed across all four negative-sample ratios.
   By default, they are automatically read from the primary repeated nested-CV
   output. This prevents ratio-specific re-tuning from confounding the
   sensitivity analysis.

4. Strict outer-test protection:
   Within each outer training set, a group-preserving validation split is used
   only for early stopping and F1-score threshold selection. The outer test
   fold is never used for model selection or threshold optimization.

5. Metrics:
   PR-AUC, ROC-AUC, Recall, Specificity, Balanced Accuracy, MCC, positive-class
   F1-score, and Accuracy are calculated. F1-score uses EDC = 1 as the
   positive class.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Parameter
from torch.utils.data import Subset

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


# ----- File locations -----
GRAPH_DATA_DIR = "edkgdl_all_data"
PROCESSED_ROOT = "."                       # cache in current working directory
OUTPUT_DIR = "negative_ratio_sensitivity_internal891_zero_based_results"

LABEL_FILE = None
LABEL_FILE_CANDIDATES = ["Graph_label.txt"]

GROUP_MAPPING_CSV = None

REPROCESS_DATASET = False

N_EDCS = 81
DECOYS_PER_EDC = 10

INTERNAL_EDC_IDS = np.arange(0, 81, dtype=int)        # 0–80
INTERNAL_NON_EDC_IDS = np.arange(87, 897, dtype=int)  # 87–896
INTERNAL_GRAPH_IDS = np.concatenate(
    [INTERNAL_EDC_IDS, INTERNAL_NON_EDC_IDS]
)  # 891 graph IDs

EXTERNAL_EDC_IDS = np.arange(81, 87, dtype=int)       # 81–86
EXTERNAL_NON_EDC_IDS = np.arange(897, 903, dtype=int) # 897–902
EXTERNAL_GRAPH_IDS = np.concatenate(
    [EXTERNAL_EDC_IDS, EXTERNAL_NON_EDC_IDS]
)  # 12 graph IDs

EXPECTED_INTERNAL_GRAPHS = N_EDCS * (1 + DECOYS_PER_EDC)  # 891
EXPECTED_TOTAL_GRAPHS = EXPECTED_INTERNAL_GRAPHS + len(EXTERNAL_GRAPH_IDS)  # 903

NEGATIVE_RATIOS = [1, 2, 5, 10]

N_SAMPLING_ROUNDS = 50

OUTER_FOLDS = 5
VALIDATION_GROUP_FRACTION = 0.20

MAX_EPOCHS = 150
MIN_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 25
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 0
USE_CLASS_WEIGHTED_LOSS = True

BASE_SEED = 20260624
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUTO_LOAD_PRIMARY_HYPERPARAMETERS = True
PRIMARY_HPARAMETER_FILE = (
    "EDKG_repeated_group_nested_cv_results_v3/"
    "selected_hyperparameters_per_outer_fold.csv"
)

MANUAL_HYPERPARAMETERS = {
    "hidden_size1": 60,
    "hidden_size2": 40,
    "hidden_size3": 20,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "dropout": 0.5,
}

RESUME_IF_OUTPUT_EXISTS = True

BOOTSTRAP_ITERATIONS = 5000

def seed_everything(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EDKGGraphDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        graph_data_dir: str,
        label_file: str,
        n_graphs: int,
        transform=None,
        pre_transform=None,
    ):
        self.graph_data_dir = Path(graph_data_dir)
        self.label_file = Path(label_file)
        self.n_graphs = int(n_graphs)
        super().__init__(root, transform, pre_transform)
        try:
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False
            )
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["Graph_label.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["edkg_ratio_sensitivity_graphs.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        labels = pd.read_csv(self.label_file, header=None)
        if labels.shape[0] != self.n_graphs:
            raise ValueError(
                f"Label file has {labels.shape[0]} rows, expected {self.n_graphs}."
            )

        graph_list: List[Data] = []

        for graph_id in range(self.n_graphs):
            graph_dir = self.graph_data_dir / str(graph_id)
            node_path = graph_dir / "Graph_index.txt"
            edge_path = graph_dir / "Graph_edge_index_direct.txt"

            if not node_path.exists():
                raise FileNotFoundError(f"Missing node file: {node_path}")
            if not edge_path.exists():
                raise FileNotFoundError(f"Missing edge file: {edge_path}")

            node_df = pd.read_csv(node_path, header=None)
            edge_df = pd.read_csv(edge_path, header=None)

            if edge_df.shape[1] < 3:
                raise ValueError(
                    f"{edge_path} must contain source, target, and edge features."
                )

            x = torch.tensor(node_df.values, dtype=torch.float)
            edge_index = torch.tensor(
                edge_df.iloc[:, :2].T.values, dtype=torch.long
            )
            edge_attr = torch.tensor(
                edge_df.iloc[:, 2:].values, dtype=torch.float
            )

            y_value = int(labels.iloc[graph_id, 1])
            y = torch.tensor([y_value], dtype=torch.long)

            graph_list.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    graph_id=torch.tensor([graph_id], dtype=torch.long),
                )
            )

        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])


def load_labels(label_file: Path) -> np.ndarray:
    """Load and validate graph labels."""
    label_df = pd.read_csv(label_file, header=None)

    if label_df.shape[1] < 2:
        raise ValueError(
            "Graph_label.txt must contain at least two columns: graph_id,label."
        )

    graph_ids = label_df.iloc[:, 0].astype(int).to_numpy()
    labels = label_df.iloc[:, 1].astype(int).to_numpy()

    if not np.array_equal(graph_ids, np.arange(len(labels))):
        raise ValueError(
            "The first Graph_label.txt column must contain sequential graph IDs "
            "from 0 to N-1."
        )

    return labels


def resolve_label_file(graph_data_dir: Path) -> Path:
    if LABEL_FILE is not None:
        return Path(LABEL_FILE)

    for filename in LABEL_FILE_CANDIDATES:
        candidate = graph_data_dir / filename
        if candidate.exists():
            return candidate

    tried = [str(graph_data_dir / x) for x in LABEL_FILE_CANDIDATES]
    raise FileNotFoundError(
        "No Graph_label file found. Tried: " + "; ".join(tried)
    )


def build_parent_groups(labels: np.ndarray) -> np.ndarray:
    n_graphs = len(labels)
    if n_graphs != EXPECTED_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GRAPHS} total graphs, found {n_graphs}."
        )

    groups = np.full(n_graphs, -1, dtype=int)

    if GROUP_MAPPING_CSV is not None:
        mapping_path = Path(GROUP_MAPPING_CSV)
        if not mapping_path.exists():
            raise FileNotFoundError(f"GROUP_MAPPING_CSV not found: {mapping_path}")

        mapping = pd.read_csv(mapping_path)
        required = {"graph_id", "parent_edc_id"}
        if not required.issubset(mapping.columns):
            raise ValueError(
                "GROUP_MAPPING_CSV must contain columns: graph_id,parent_edc_id."
            )

        mapping = mapping[
            mapping["graph_id"].astype(int).isin(INTERNAL_GRAPH_IDS)
        ].copy()

        if mapping["graph_id"].duplicated().any():
            raise ValueError(
                "GROUP_MAPPING_CSV contains duplicate graph IDs in the internal cohort."
            )

        mapping = mapping.sort_values("graph_id").reset_index(drop=True)
        if not np.array_equal(
            mapping["graph_id"].astype(int).to_numpy(),
            np.sort(INTERNAL_GRAPH_IDS),
        ):
            raise ValueError(
                "GROUP_MAPPING_CSV must include every internal graph ID exactly "
                "once: EDCs 0–80 and non-EDCs 87–896."
            )

        groups[mapping["graph_id"].astype(int).to_numpy()] = (
            mapping["parent_edc_id"].astype(int).to_numpy()
        )

    else:
        # One internal EDC per parent group.
        groups[INTERNAL_EDC_IDS] = np.arange(N_EDCS)

        # Ten consecutive internal decoys per parent group.
        groups[INTERNAL_NON_EDC_IDS] = np.repeat(
            np.arange(N_EDCS),
            DECOYS_PER_EDC,
        )

    # The external cohort must never receive parent groups in this analysis.
    if np.any(groups[EXTERNAL_GRAPH_IDS] != -1):
        raise RuntimeError(
            "External IDs 81–86 and 897–902 must remain excluded (group = -1)."
        )

    # Validate each internal parent group has exactly 1 EDC and 10 decoys.
    for parent_group in range(N_EDCS):
        group_ids = INTERNAL_GRAPH_IDS[
            groups[INTERNAL_GRAPH_IDS] == parent_group
        ]
        group_y = labels[group_ids]
        n_pos = int((group_y == 1).sum())
        n_neg = int((group_y == 0).sum())

        if n_pos != 1 or n_neg != DECOYS_PER_EDC:
            raise ValueError(
                f"Internal parent group {parent_group} has {n_pos} EDCs and "
                f"{n_neg} non-EDCs. Expected 1 and {DECOYS_PER_EDC}. "
                "Check decoy ordering or provide GROUP_MAPPING_CSV."
            )

    return groups

class GCNConvEdge(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr="add")
        self.lin_node = Linear(in_channels, out_channels, bias=False)
        self.lin_edge = Linear(edge_channels, out_channels, bias=False)
        self.bias = Parameter(torch.zeros(2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        self.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index_with_loops, _ = add_self_loops(
            edge_index,
            num_nodes=x.size(0),
        )

        x_proj = self.lin_node(x)
        edge_proj = self.lin_edge(edge_attr)

        self_loop_edge_emb = torch.zeros(
            (x_proj.size(0), edge_proj.size(1)),
            dtype=edge_proj.dtype,
            device=edge_proj.device,
        )
        edge_proj_with_loops = torch.cat(
            [edge_proj, self_loop_edge_emb],
            dim=0,
        )

        row, col = edge_index_with_loops
        deg = degree(col, x_proj.size(0), dtype=x_proj.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index_with_loops,
            x=x_proj,
            norm=norm,
            edge_emb=edge_proj_with_loops,
        )
        out = out + self.bias
        return out, edge_proj

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        return norm.view(-1, 1) * torch.cat([x_j, edge_emb], dim=1)


class EDKGDLClassifier(nn.Module):

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = GCNConvEdge(node_channels, hidden_size1, edge_channels)
        self.conv2 = GCNConvEdge(
            2 * hidden_size1,
            hidden_size2,
            hidden_size1,
        )
        self.conv3 = GCNConvEdge(
            2 * hidden_size2,
            hidden_size3,
            hidden_size2,
        )
        self.dropout = float(dropout)
        self.readout = Linear(2 * hidden_size3, 2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_emb = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x, edge_emb = self.conv2(x, edge_index, edge_emb)
        x = F.relu(x)

        x, _ = self.conv3(x, edge_index, edge_emb)

        graph_embedding = global_mean_pool(x, batch)
        graph_embedding = F.dropout(
            graph_embedding,
            p=self.dropout,
            training=self.training,
        )
        return self.readout(graph_embedding)


@dataclass(frozen=True)
class HyperParameters:
    hidden_size1: int
    hidden_size2: int
    hidden_size3: int
    lr: float
    weight_decay: float
    dropout: float


def resolve_fixed_hyperparameters() -> Tuple[HyperParameters, str]:

    primary_path = Path(PRIMARY_HPARAMETER_FILE)
    required_columns = {
        "hidden_size1",
        "hidden_size2",
        "hidden_size3",
        "lr",
        "weight_decay",
        "dropout",
    }

    if AUTO_LOAD_PRIMARY_HYPERPARAMETERS and primary_path.exists():
        hp_df = pd.read_csv(primary_path)

        if required_columns.issubset(hp_df.columns):
            hp_cols = [
                "hidden_size1",
                "hidden_size2",
                "hidden_size3",
                "lr",
                "weight_decay",
                "dropout",
            ]

            grouped = (
                hp_df.groupby(hp_cols, dropna=False)
                .agg(
                    n_selected=("repeat", "size"),
                    mean_inner_pr_auc=(
                        "inner_best_pr_auc",
                        "mean",
                    )
                    if "inner_best_pr_auc" in hp_df.columns
                    else ("repeat", "size"),
                )
                .reset_index()
            )

            grouped = grouped.sort_values(
                ["n_selected", "mean_inner_pr_auc"],
                ascending=[False, False],
            ).reset_index(drop=True)

            chosen = grouped.iloc[0]
            hp = HyperParameters(
                hidden_size1=int(chosen["hidden_size1"]),
                hidden_size2=int(chosen["hidden_size2"]),
                hidden_size3=int(chosen["hidden_size3"]),
                lr=float(chosen["lr"]),
                weight_decay=float(chosen["weight_decay"]),
                dropout=float(chosen["dropout"]),
            )

            source = (
                "Automatically selected from the primary nested-CV output: "
                "most frequently inner-CV-selected hyperparameter configuration."
            )
            return hp, source

        print(
            "Warning: primary hyperparameter file exists but does not contain "
            "all required columns. Falling back to MANUAL_HYPERPARAMETERS."
        )

    hp = HyperParameters(**MANUAL_HYPERPARAMETERS)
    source = "Manual fallback hyperparameters."
    return hp, source

def group_kfold_splits(
    sample_indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    sample_indices = np.asarray(sample_indices, dtype=int)
    unique_groups = np.unique(groups[sample_indices])

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} parent groups available for {n_splits} folds."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    fold_group_sets = np.array_split(shuffled_groups, n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for test_groups in fold_group_sets:
        is_test = np.isin(groups[sample_indices], test_groups)
        test_idx = sample_indices[is_test]
        train_idx = sample_indices[~is_test]

        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Parent-group leakage detected.")

        splits.append((train_idx, test_idx))

    return splits


def group_train_validation_split(
    outer_train_indices: np.ndarray,
    groups: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    unique_groups = np.unique(groups[outer_train_indices])

    if len(unique_groups) < 5:
        raise ValueError("Too few parent groups for a validation split.")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_validation_groups = max(1, int(round(len(unique_groups) * validation_fraction)))
    n_validation_groups = min(n_validation_groups, len(unique_groups) - 1)

    val_groups = shuffled[:n_validation_groups]
    is_val = np.isin(groups[outer_train_indices], val_groups)

    train_idx = outer_train_indices[~is_val]
    val_idx = outer_train_indices[is_val]

    if set(groups[train_idx]).intersection(set(groups[val_idx])):
        raise RuntimeError("Parent-group leakage detected in train-validation split.")

    return train_idx, val_idx


def make_balanced_decoy_plan(
    groups: np.ndarray,
    ratio: int,
    n_rounds: int,
    seed: int,
) -> Dict[int, List[np.ndarray]]:

    if ratio not in NEGATIVE_RATIOS:
        raise ValueError(f"Unsupported ratio: {ratio}")

    chunks_per_permutation = DECOYS_PER_EDC // ratio
    if DECOYS_PER_EDC % ratio != 0:
        raise ValueError(
            f"ratio={ratio} must divide DECOYS_PER_EDC={DECOYS_PER_EDC}."
        )

    rng = np.random.default_rng(seed)
    plan: Dict[int, List[np.ndarray]] = {}

    all_graph_ids = np.arange(len(groups), dtype=int)
    internal_decoy_mask = np.isin(all_graph_ids, INTERNAL_NON_EDC_IDS)

    for parent_group in range(N_EDCS):
        decoy_ids = np.where(
            (groups == parent_group) & internal_decoy_mask
        )[0]
        if len(decoy_ids) != DECOYS_PER_EDC:
            raise RuntimeError(
                f"Parent group {parent_group} has {len(decoy_ids)} decoys; "
                f"expected {DECOYS_PER_EDC}."
            )

        selections: List[np.ndarray] = []
        while len(selections) < n_rounds:
            permuted = rng.permutation(decoy_ids)
            for start in range(0, DECOYS_PER_EDC, ratio):
                selections.append(permuted[start:start + ratio])
                if len(selections) == n_rounds:
                    break

        plan[parent_group] = selections

    return plan


def indices_for_ratio_round(
    ratio: int,
    sampling_round: int,
    groups: np.ndarray,
    decoy_plan: Dict[int, List[np.ndarray]],
) -> np.ndarray:

    selected = INTERNAL_EDC_IDS.astype(int).tolist()

    for parent_group in range(N_EDCS):
        selected.extend(decoy_plan[parent_group][sampling_round].tolist())

    selected = np.asarray(sorted(selected), dtype=int)

    expected_size = N_EDCS * (1 + ratio)
    if len(selected) != expected_size:
        raise RuntimeError(
            f"Ratio {ratio}, round {sampling_round}: expected {expected_size} "
            f"samples but selected {len(selected)}."
        )

    return selected


def save_decoy_selection_audit(
    output_dir: Path,
    groups: np.ndarray,
) -> None:
    audit_rows = []

    for ratio in NEGATIVE_RATIOS:
        plan = make_balanced_decoy_plan(
            groups=groups,
            ratio=ratio,
            n_rounds=N_SAMPLING_ROUNDS,
            seed=BASE_SEED + 1000 * ratio,
        )

        for parent_group in range(N_EDCS):
            for sampling_round in range(N_SAMPLING_ROUNDS):
                chosen_decoys = plan[parent_group][sampling_round]
                for decoy_id in chosen_decoys:
                    audit_rows.append(
                        {
                            "ratio": ratio,
                            "sampling_round": sampling_round + 1,
                            "parent_edc_group": parent_group,
                            "decoy_graph_id": int(decoy_id),
                        }
                    )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / "selected_decoys_audit.csv", index=False)

    coverage = (
        audit_df.groupby(["ratio", "parent_edc_group", "decoy_graph_id"])
        .size()
        .reset_index(name="times_selected")
    )
    coverage.to_csv(output_dir / "decoy_coverage_audit.csv", index=False)


def create_loader(
    dataset: InMemoryDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(
    dataset: InMemoryDataset,
    hp: HyperParameters,
) -> EDKGDLClassifier:
    return EDKGDLClassifier(
        node_channels=dataset.num_node_features,
        edge_channels=dataset.num_edge_features,
        hidden_size1=hp.hidden_size1,
        hidden_size2=hp.hidden_size2,
        hidden_size3=hp.hidden_size3,
        dropout=hp.dropout,
    ).to(DEVICE)


def class_weights(labels: np.ndarray, train_indices: np.ndarray) -> torch.Tensor:
    """Compute ratio-aware class weights from the current inner training data."""
    y = labels[train_indices]
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training partition must contain both EDCs and non-EDCs.")

    return torch.tensor(
        [1.0, n_neg / n_pos],
        dtype=torch.float,
        device=DEVICE,
    )


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    graph_ids_list = []
    labels_list = []
    probs_list = []

    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1]

        graph_ids_list.append(data.graph_id.view(-1).cpu().numpy())
        labels_list.append(data.y.view(-1).cpu().numpy())
        probs_list.append(probs.cpu().numpy())

    return (
        np.concatenate(graph_ids_list),
        np.concatenate(labels_list).astype(int),
        np.concatenate(probs_list),
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = (recall + specificity) / 2

    denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0

    positive_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "f1_score": positive_f1,
        "accuracy": float((y_true == y_pred).mean()),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_f1_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_key = None

    for threshold in candidates:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        key = (
            metrics["f1_score"],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = criterion(logits, data.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * data.num_graphs
        total_graphs += int(data.num_graphs)

    return total_loss / max(total_graphs, 1)


def fit_outer_fold_model(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    outer_train_indices: np.ndarray,
    groups: np.ndarray,
    hp: HyperParameters,
    seed: int,
) -> Tuple[nn.Module, float, int, float, np.ndarray, np.ndarray]:
    seed_everything(seed)

    inner_train_indices, validation_indices = group_train_validation_split(
        outer_train_indices=outer_train_indices,
        groups=groups,
        validation_fraction=VALIDATION_GROUP_FRACTION,
        seed=seed + 19,
    )

    model = build_model(dataset, hp)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )

    if USE_CLASS_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights(labels, inner_train_indices)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = create_loader(
        dataset,
        inner_train_indices,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = create_loader(
        dataset,
        validation_indices,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    best_state = None
    best_epoch = 1
    best_val_pr_auc = -np.inf
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion)

        _, y_val, p_val = get_predictions(model, validation_loader)
        val_pr_auc = average_precision_score(y_val, p_val)

        if val_pr_auc > best_val_pr_auc + 1e-8:
            best_val_pr_auc = float(val_pr_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= MIN_EPOCHS and patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)

    _, y_val, p_val = get_predictions(model, validation_loader)
    threshold = select_f1_threshold(y_val, p_val)

    return (
        model,
        threshold,
        best_epoch,
        best_val_pr_auc,
        inner_train_indices,
        validation_indices,
    )

PRIMARY_METRICS = [
    "pr_auc",
    "roc_auc",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1_score",
]
SECONDARY_METRICS = ["accuracy"]


def repeat_round_metrics_from_folds(
    fold_df: pd.DataFrame,
) -> Dict[str, float]:
    weights = fold_df["n_test_graphs"].to_numpy(dtype=float)

    tp = int(fold_df["tp"].sum())
    tn = int(fold_df["tn"].sum())
    fp = int(fold_df["fp"].sum())
    fn = int(fold_df["fn"].sum())

    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = (recall + specificity) / 2

    denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0

    f1_denom = 2 * tp + fp + fn
    f1_score_value = 2 * tp / f1_denom if f1_denom > 0 else 0.0

    return {
        "n_outer_folds": int(len(fold_df)),
        "n_samples": int(total),
        "n_edcs": int(tp + fn),
        "n_non_edcs": int(tn + fp),
        "pr_auc": float(np.average(fold_df["pr_auc"], weights=weights)),
        "roc_auc": float(np.average(fold_df["roc_auc"], weights=weights)),
        "ranking_metric_aggregation": "test-size-weighted outer-fold mean",
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "f1_score": f1_score_value,
        "accuracy": (tp + tn) / total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold_mean": float(np.average(fold_df["threshold"], weights=weights)),
        "threshold_sd": float(fold_df["threshold"].std(ddof=1)),
    }


def bootstrap_ci(
    values: np.ndarray,
    metric_name: str,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan

    seed = BASE_SEED + sum(ord(c) for c in metric_name)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)

    for i in range(BOOTSTRAP_ITERATIONS):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()

    return (
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def create_ratio_summary(
    round_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    output_rows = []

    for ratio in NEGATIVE_RATIOS:
        subset = round_metrics_df[round_metrics_df["ratio"] == ratio].copy()

        if len(subset) != N_SAMPLING_ROUNDS:
            raise ValueError(
                f"Ratio 1:{ratio} has {len(subset)} completed sampling rounds, "
                f"expected {N_SAMPLING_ROUNDS}."
            )

        for metric in PRIMARY_METRICS + SECONDARY_METRICS:
            values = subset[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(
                values,
                metric_name=f"ratio_{ratio}_{metric}",
            )

            output_rows.append(
                {
                    "negative_ratio": f"1:{ratio}",
                    "decoys_per_edc": ratio,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "bootstrap_mean_ci_2.5%": ci_low,
                    "bootstrap_mean_ci_97.5%": ci_high,
                    "n_sampling_rounds": int(len(values)),
                }
            )

    return pd.DataFrame(output_rows)


def load_existing_results(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = output_dir / "outer_fold_metrics.csv"
    predictions_path = output_dir / "outer_fold_predictions.csv"

    if RESUME_IF_OUTPUT_EXISTS and metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame()

    if RESUME_IF_OUTPUT_EXISTS and predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
    else:
        predictions_df = pd.DataFrame()

    return metrics_df, predictions_df


def already_completed(
    metrics_df: pd.DataFrame,
    ratio: int,
    sampling_round: int,
    outer_fold: int,
) -> bool:
    if metrics_df.empty:
        return False

    mask = (
        (metrics_df["ratio"] == ratio)
        & (metrics_df["sampling_round"] == sampling_round)
        & (metrics_df["outer_fold"] == outer_fold)
    )
    return int(mask.sum()) == 1


def append_and_save(
    output_dir: Path,
    existing_metrics: pd.DataFrame,
    existing_predictions: pd.DataFrame,
    metric_row: Dict,
    prediction_rows: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    new_metric_df = pd.DataFrame([metric_row])
    new_prediction_df = pd.DataFrame(prediction_rows)

    metrics_df = pd.concat(
        [existing_metrics, new_metric_df],
        ignore_index=True,
    )
    predictions_df = pd.concat(
        [existing_predictions, new_prediction_df],
        ignore_index=True,
    )

    metrics_df.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    predictions_df.to_csv(
        output_dir / "outer_fold_predictions.csv",
        index=False,
    )

    return metrics_df, predictions_df

def main() -> None:
    seed_everything(BASE_SEED)

    print(f"Using device: {DEVICE}")

    graph_data_dir = Path(GRAPH_DATA_DIR)

    if not graph_data_dir.exists():
        raise FileNotFoundError(f"GRAPH_DATA_DIR not found: {graph_data_dir}")

    label_file = resolve_label_file(graph_data_dir)
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    labels = load_labels(label_file)
    if len(labels) != EXPECTED_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GRAPHS} labels after adding the external "
            f"cohort, found {len(labels)}."
        )

    expected_labels = np.zeros(EXPECTED_TOTAL_GRAPHS, dtype=int)
    expected_labels[INTERNAL_EDC_IDS] = 1
    expected_labels[EXTERNAL_EDC_IDS] = 1

    if not np.array_equal(labels, expected_labels):
        mismatch_ids = np.where(labels != expected_labels)[0]
        raise ValueError(
            "Graph labels do not match the required zero-based layout. "
            "Expected positives at 0–86 and negatives at 87–902. "
            f"Mismatch IDs: {mismatch_ids[:20].tolist()}"
        )

    internal_labels = labels[INTERNAL_GRAPH_IDS]
    external_labels = labels[EXTERNAL_GRAPH_IDS]
    if (
        int((internal_labels == 1).sum()) != N_EDCS
        or int((internal_labels == 0).sum()) != N_EDCS * DECOYS_PER_EDC
    ):
        raise ValueError(
            "Internal sensitivity-analysis cohort must be: EDCs 0–80 and "
            "non-EDCs 87–896."
        )
    if int((external_labels == 1).sum()) != 6 or int((external_labels == 0).sum()) != 6:
        raise ValueError(
            "External excluded cohort must be: EDCs 81–86 and "
            "non-EDCs 897–902."
        )

    parent_groups = build_parent_groups(labels)

    processed_file = Path(PROCESSED_ROOT) / "processed" / "edkg_ratio_sensitivity_graphs.pt"
    if REPROCESS_DATASET and processed_file.exists():
        processed_file.unlink()

    dataset = EDKGGraphDataset(
        root=PROCESSED_ROOT,
        graph_data_dir=str(graph_data_dir),
        label_file=str(label_file),
        n_graphs=len(labels),
    )

    if len(dataset) != len(labels):
        raise RuntimeError(
            f"Dataset has {len(dataset)} graphs but labels have {len(labels)} rows."
        )

    print(
        f"Loaded {len(dataset)} total graphs | "
        f"internal ratio-sensitivity cohort: {len(INTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 0–80; non-EDCs 87–896) | "
        f"external excluded cohort: {len(EXTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 81–86; non-EDCs 897–902)"
    )

    hp, hp_source = resolve_fixed_hyperparameters()
    print("\nFixed hyperparameters for all ratios:")
    print(asdict(hp))
    print(hp_source)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "graph_data_dir": str(graph_data_dir.resolve()),
        "label_file": str(label_file.resolve()),
        "device": str(DEVICE),
        "total_graphs_loaded": EXPECTED_TOTAL_GRAPHS,
        "internal_ratio_sensitivity_cohort": {
            "edc_graph_ids": "0-80",
            "non_edc_graph_ids": "87-896",
            "n_graphs": EXPECTED_INTERNAL_GRAPHS,
            "n_edcs": N_EDCS,
            "n_non_edcs": N_EDCS * DECOYS_PER_EDC,
        },
        "external_cohort_excluded_from_ratio_sensitivity": {
            "edc_graph_ids": "81-86",
            "non_edc_graph_ids": "897-902",
            "n_graphs": len(EXTERNAL_GRAPH_IDS),
            "n_edcs": 6,
            "n_non_edcs": 6,
        },
        "negative_ratios": NEGATIVE_RATIOS,
        "n_sampling_rounds": N_SAMPLING_ROUNDS,
        "outer_folds": OUTER_FOLDS,
        "validation_group_fraction": VALIDATION_GROUP_FRACTION,
        "class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "base_seed": BASE_SEED,
        "fixed_hyperparameters": asdict(hp),
        "hyperparameter_source": hp_source,
        "grouping_rule": (
            "One EDC and its selected paired decoys were retained in the same "
            "parent group in all train/test splits."
        ),
        "negative_sampling_rule": (
            "Balanced cyclic sampling of decoys, with no duplicate decoy "
            "within any parent EDC group in a sampling round."
        ),
    }
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    cohort_audit = pd.DataFrame(
        {
            "graph_id": np.arange(EXPECTED_TOTAL_GRAPHS, dtype=int),
            "label": labels,
            "cohort_role": np.where(
                np.isin(np.arange(EXPECTED_TOTAL_GRAPHS), INTERNAL_GRAPH_IDS),
                "internal_ratio_sensitivity",
                "external_excluded",
            ),
            "parent_edc_group": parent_groups,
        }
    )
    cohort_audit.to_csv(
        output_dir / "cohort_membership_and_group_mapping.csv",
        index=False,
    )

    save_decoy_selection_audit(output_dir, parent_groups)

    existing_metrics, existing_predictions = load_existing_results(output_dir)

    all_indices = np.sort(INTERNAL_GRAPH_IDS.astype(int))

    for ratio in NEGATIVE_RATIOS:
        print(f"\n{'=' * 78}\nNegative-ratio analysis: 1:{ratio}\n{'=' * 78}")

        decoy_plan = make_balanced_decoy_plan(
            groups=parent_groups,
            ratio=ratio,
            n_rounds=N_SAMPLING_ROUNDS,
            seed=BASE_SEED + 1000 * ratio,
        )

        for sampling_round_zero_based in range(N_SAMPLING_ROUNDS):
            sampling_round = sampling_round_zero_based + 1
            selected_indices = indices_for_ratio_round(
                ratio=ratio,
                sampling_round=sampling_round_zero_based,
                groups=parent_groups,
                decoy_plan=decoy_plan,
            )

            expected_total = N_EDCS * (1 + ratio)
            assert len(selected_indices) == expected_total

            if np.intersect1d(selected_indices, EXTERNAL_GRAPH_IDS).size > 0:
                raise RuntimeError(
                    "External IDs 81–86 or 897–902 were selected in the "
                    "internal ratio sensitivity analysis."
                )
            if not np.all(np.isin(selected_indices, INTERNAL_GRAPH_IDS)):
                raise RuntimeError(
                    "A sampled ratio dataset contains graph IDs outside the "
                    "internal 891-compound cohort."
                )

            outer_splits = group_kfold_splits(
                sample_indices=selected_indices,
                groups=parent_groups,
                n_splits=OUTER_FOLDS,
                seed=BASE_SEED + 100000 * ratio + sampling_round,
            )

            print(
                f"\nRatio 1:{ratio} | sampling round "
                f"{sampling_round}/{N_SAMPLING_ROUNDS} | "
                f"{len(selected_indices)} graphs"
            )

            for outer_fold_zero_based, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
                outer_fold = outer_fold_zero_based + 1

                if already_completed(
                    metrics_df=existing_metrics,
                    ratio=ratio,
                    sampling_round=sampling_round,
                    outer_fold=outer_fold,
                ):
                    print(f"  Fold {outer_fold}/{OUTER_FOLDS}: already completed, skipping.")
                    continue

                model_seed = (
                    BASE_SEED
                    + 10000000 * ratio
                    + 10000 * sampling_round
                    + 100 * outer_fold
                )

                (
                    model,
                    threshold,
                    best_epoch,
                    best_validation_pr_auc,
                    inner_train_idx,
                    validation_idx,
                ) = fit_outer_fold_model(
                    dataset=dataset,
                    labels=labels,
                    outer_train_indices=outer_train_idx,
                    groups=parent_groups,
                    hp=hp,
                    seed=model_seed,
                )

                test_loader = create_loader(
                    dataset,
                    outer_test_idx,
                    batch_size=EVAL_BATCH_SIZE,
                    shuffle=False,
                )
                graph_ids, y_test, p_test = get_predictions(model, test_loader)
                metrics = calculate_metrics(y_test, p_test, threshold)

                # Verify graph-label consistency after loading test batches.
                if not np.array_equal(
                    y_test.astype(int),
                    labels[graph_ids.astype(int)].astype(int),
                ):
                    raise RuntimeError(
                        "Graph IDs and labels from the test loader do not agree."
                    )

                metric_row = {
                    "ratio": ratio,
                    "negative_ratio": f"1:{ratio}",
                    "sampling_round": sampling_round,
                    "outer_fold": outer_fold,
                    "n_selected_total": int(len(selected_indices)),
                    "n_train_graphs": int(len(outer_train_idx)),
                    "n_validation_graphs": int(len(validation_idx)),
                    "n_test_graphs": int(len(outer_test_idx)),
                    "n_train_edcs": int((labels[outer_train_idx] == 1).sum()),
                    "n_train_non_edcs": int((labels[outer_train_idx] == 0).sum()),
                    "n_validation_edcs": int((labels[validation_idx] == 1).sum()),
                    "n_validation_non_edcs": int((labels[validation_idx] == 0).sum()),
                    "n_test_edcs": int((labels[outer_test_idx] == 1).sum()),
                    "n_test_non_edcs": int((labels[outer_test_idx] == 0).sum()),
                    "best_epoch": int(best_epoch),
                    "best_validation_pr_auc": float(best_validation_pr_auc),
                    **asdict(hp),
                    **metrics,
                }

                y_pred = (p_test >= threshold).astype(int)
                prediction_rows = []
                for graph_id, y_true, probability, pred in zip(
                    graph_ids,
                    y_test,
                    p_test,
                    y_pred,
                ):
                    prediction_rows.append(
                        {
                            "ratio": ratio,
                            "negative_ratio": f"1:{ratio}",
                            "sampling_round": sampling_round,
                            "outer_fold": outer_fold,
                            "graph_id": int(graph_id),
                            "parent_edc_group": int(parent_groups[int(graph_id)]),
                            "y_true": int(y_true),
                            "predicted_probability_edc": float(probability),
                            "selected_threshold": float(threshold),
                            "y_pred": int(pred),
                        }
                    )

                existing_metrics, existing_predictions = append_and_save(
                    output_dir=output_dir,
                    existing_metrics=existing_metrics,
                    existing_predictions=existing_predictions,
                    metric_row=metric_row,
                    prediction_rows=prediction_rows,
                )

                print(
                    f"  Fold {outer_fold}/{OUTER_FOLDS} | "
                    f"PR-AUC={metrics['pr_auc']:.3f}, "
                    f"ROC-AUC={metrics['roc_auc']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"BalancedAcc={metrics['balanced_accuracy']:.3f}, "
                    f"MCC={metrics['mcc']:.3f}, "
                    f"F1={metrics['f1_score']:.3f}"
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    final_metrics_df = pd.read_csv(output_dir / "outer_fold_metrics.csv")

    expected_rows = len(NEGATIVE_RATIOS) * N_SAMPLING_ROUNDS * OUTER_FOLDS
    if len(final_metrics_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} outer-fold rows after completion but found "
            f"{len(final_metrics_df)}. The run may be incomplete."
        )

    round_rows = []
    for ratio in NEGATIVE_RATIOS:
        for sampling_round in range(1, N_SAMPLING_ROUNDS + 1):
            fold_subset = final_metrics_df[
                (final_metrics_df["ratio"] == ratio)
                & (final_metrics_df["sampling_round"] == sampling_round)
            ].copy()

            if len(fold_subset) != OUTER_FOLDS:
                raise RuntimeError(
                    f"Ratio 1:{ratio}, sampling round {sampling_round} has "
                    f"{len(fold_subset)} folds, expected {OUTER_FOLDS}."
                )

            round_rows.append(
                {
                    "ratio": ratio,
                    "negative_ratio": f"1:{ratio}",
                    "sampling_round": sampling_round,
                    **repeat_round_metrics_from_folds(fold_subset),
                }
            )

    round_metrics_df = pd.DataFrame(round_rows)
    summary_df = create_ratio_summary(round_metrics_df)

    round_metrics_df.to_csv(
        output_dir / "sampling_round_metrics.csv",
        index=False,
    )
    summary_df.to_csv(
        output_dir / "ratio_metric_summary.csv",
        index=False,
    )

    readme = f"""Negative-ratio sensitivity analysis completed for the INTERNAL 891-compound cohort.

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
- All {N_EDCS} internal EDCs (IDs 0–80) were retained in every analysis.
- For each ratio, {N_SAMPLING_ROUNDS} balanced decoy-sampling rounds were run.
- Each parent group contained one EDC plus the selected number of its own
  DUD-E decoys.
- Parent groups were never split across train and outer-test data.

Model configuration
-------------------
{hp_source}
Fixed hyperparameters:
{asdict(hp)}

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
- ratio_metric_summary.csv reports distributions across the {N_SAMPLING_ROUNDS}
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
"""
    (output_dir / "README_ratio_sensitivity.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("Negative-ratio sensitivity analysis complete.")
    print("=" * 78)
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs are saved in:\n{output_dir.resolve()}")


if __name__ == "__main__":
    main()
