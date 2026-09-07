#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repeated group-nested cross-validation for the qualitative EDKG-DL classifier.

Purpose
-------
This script evaluates the EDKG-DL classifier using repeated, group-preserving,
nested cross-validation. Each group contains one EDC and its ten paired
DUD-E decoys, so an EDC and its paired decoys can never be split between
training and test sets.

Design
------
- 20 repeated outer 5-fold group CV runs (100 outer test evaluations)
- 4-fold group CV inside each outer training set for hyperparameter selection
- The classification threshold is selected only from inner-CV out-of-fold
  predictions; the outer test set is never used for tuning.
- The final model for each outer fold is trained on the full outer-training
  set for the median best epoch selected by the inner folds.
- Main output metrics: PR-AUC, ROC-AUC, recall, specificity, balanced
  accuracy, MCC, macro-F1, and secondary accuracy.
- Bootstrap 95% CIs are calculated for the mean metric across outer folds.

IMPORTANT ASSUMPTION ABOUT GROUPS
---------------------------------
The default grouping assumes:
  graph IDs 0-80      : 81 EDCs
  graph IDs 81-890    : 810 decoys arranged in consecutive blocks of 10
                         (decoys 81-90 belong to EDC 0,
                          91-100 belong to EDC 1, etc.).
"""

from __future__ import annotations

import os
import json
import copy
import random
import shutil
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Parameter
from torch.utils.data import Subset

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import add_self_loops, degree


# ----- Input/output paths -----
GRAPH_DATA_DIR = "edkgdl_all_data"
PROCESSED_ROOT = "EDKG_nested_cv_processed"
OUTPUT_DIR = "EDKG_repeated_group_nested_cv_results"

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

N_REPEATS = 20
OUTER_FOLDS = 5
INNER_FOLDS = 4
HYPERPARAM_TRIALS = 6

MAX_EPOCHS = 150
MIN_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 25
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 0

USE_CLASS_WEIGHTED_LOSS = True

BASE_SEED = 20260623
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HYPERPARAM_SPACE = {
    "hidden_size1": [32, 48, 64, 80],
    "hidden_size2": [16, 32, 48, 64],
    "hidden_size3": [16, 24, 32, 48],
    "lr": [1e-3, 5e-4, 3e-4],
    "weight_decay": [1e-3, 5e-4, 1e-4],
    "dropout": [0.2, 0.4, 0.5],
}

def seed_everything(seed: int) -> None:
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
        return ["edkg_graphs.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        labels = pd.read_csv(self.label_file, header=None)
        if labels.shape[0] != self.n_graphs:
            raise ValueError(
                f"Label file has {labels.shape[0]} rows, but n_graphs={self.n_graphs}."
            )

        data_list: List[Data] = []
        for graph_id in range(self.n_graphs):
            graph_dir = self.graph_data_dir / str(graph_id)
            node_path = graph_dir / "Graph_index.txt"
            edge_path = graph_dir / "Graph_edge_index_direct.txt"

            if not node_path.exists():
                raise FileNotFoundError(f"Missing node file: {node_path}")
            if not edge_path.exists():
                raise FileNotFoundError(f"Missing edge file: {edge_path}")

            node_df = pd.read_csv(node_path, header=None)
            x = torch.tensor(node_df.values, dtype=torch.float)

            edge_df = pd.read_csv(edge_path, header=None)
            if edge_df.shape[1] < 3:
                raise ValueError(
                    f"{edge_path} must contain source, target, and >=1 edge attribute."
                )

            edge_index = torch.tensor(
                edge_df.iloc[:, :2].T.values, dtype=torch.long
            )
            edge_attr = torch.tensor(
                edge_df.iloc[:, 2:].values, dtype=torch.float
            )

            y_value = int(labels.iloc[graph_id, 1])
            y = torch.tensor([y_value], dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                graph_id=torch.tensor([graph_id], dtype=torch.long),
            )
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_labels(label_file: str) -> np.ndarray:
    labels_df = pd.read_csv(label_file, header=None)
    if labels_df.shape[1] < 2:
        raise ValueError("Graph_label.txt must have at least two columns: graph_id,label.")
    graph_ids = labels_df.iloc[:, 0].astype(int).to_numpy()
    labels = labels_df.iloc[:, 1].astype(int).to_numpy()

    expected_ids = np.arange(len(labels))
    if not np.array_equal(graph_ids, expected_ids):
        raise ValueError(
            "The first label-file column must be sequential graph IDs 0..N-1."
        )
    return labels


def resolve_label_file(graph_data_dir: Path) -> Path:
    if LABEL_FILE is not None:
        return Path(LABEL_FILE)

    for candidate in LABEL_FILE_CANDIDATES:
        candidate_path = graph_data_dir / candidate
        if candidate_path.exists():
            return candidate_path

    tried = [str(graph_data_dir / x) for x in LABEL_FILE_CANDIDATES]
    raise FileNotFoundError(
        "No label file was found. Tried: " + "; ".join(tried)
    )


def build_parent_groups(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    if n != EXPECTED_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GRAPHS} total graphs, received {n}."
        )

    groups = np.full(n, -1, dtype=int)

    if GROUP_MAPPING_CSV is not None:
        mapping = pd.read_csv(GROUP_MAPPING_CSV)
        required = {"graph_id", "parent_edc_id"}
        if not required.issubset(mapping.columns):
            raise ValueError(
                f"{GROUP_MAPPING_CSV} must contain columns: {sorted(required)}"
            )

        mapping = mapping[mapping["graph_id"].isin(INTERNAL_GRAPH_IDS)].copy()
        if mapping["graph_id"].duplicated().any():
            raise ValueError("GROUP_MAPPING_CSV contains duplicated internal graph IDs.")

        mapping = mapping.sort_values("graph_id").reset_index(drop=True)
        if not np.array_equal(
            mapping["graph_id"].to_numpy(dtype=int),
            np.sort(INTERNAL_GRAPH_IDS),
        ):
            raise ValueError(
                "GROUP_MAPPING_CSV must cover every internal graph ID exactly "
                "once: 0–80 and 87–896."
            )

        groups[mapping["graph_id"].astype(int).to_numpy()] = (
            mapping["parent_edc_id"].astype(int).to_numpy()
        )
    else:
        groups[INTERNAL_EDC_IDS] = np.arange(N_EDCS)

        groups[INTERNAL_NON_EDC_IDS] = np.repeat(
            np.arange(N_EDCS),
            DECOYS_PER_EDC,
        )

    if np.any(groups[EXTERNAL_GRAPH_IDS] != -1):
        raise RuntimeError(
            "External graph IDs 81–86 and 897–902 must have parent group -1."
        )
    
    for group_id in range(N_EDCS):
        graph_ids = INTERNAL_GRAPH_IDS[groups[INTERNAL_GRAPH_IDS] == group_id]
        group_y = labels[graph_ids]
        n_pos = int((group_y == 1).sum())
        n_neg = int((group_y == 0).sum())

        if n_pos != 1 or n_neg != DECOYS_PER_EDC:
            raise ValueError(
                f"Internal group {group_id}: expected 1 EDC and "
                f"{DECOYS_PER_EDC} non-EDCs; found {n_pos} EDCs and "
                f"{n_neg} non-EDCs. Check decoy ordering or GROUP_MAPPING_CSV."
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
            edge_index, num_nodes=x.size(0)
        )

        node_proj = self.lin_node(x)
        edge_proj = self.lin_edge(edge_attr)

        zero_self_loop_edges = torch.zeros(
            (node_proj.size(0), edge_proj.size(1)),
            device=edge_proj.device,
            dtype=edge_proj.dtype,
        )
        edge_proj_with_loops = torch.cat(
            [edge_proj, zero_self_loop_edges], dim=0
        )

        row, col = edge_index_with_loops
        deg = degree(col, node_proj.size(0), dtype=node_proj.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index_with_loops,
            x=node_proj,
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
        node_plus_edge = torch.cat([x_j, edge_emb], dim=1)
        return norm.view(-1, 1) * node_plus_edge


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
        self.conv2 = GCNConvEdge(2 * hidden_size1, hidden_size2, hidden_size1)
        self.conv3 = GCNConvEdge(2 * hidden_size2, hidden_size3, hidden_size2)
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

        graph_emb = global_mean_pool(x, batch)
        graph_emb = F.dropout(graph_emb, p=self.dropout, training=self.training)
        return self.readout(graph_emb)


@dataclass(frozen=True)
class HyperParams:
    hidden_size1: int
    hidden_size2: int
    hidden_size3: int
    lr: float
    weight_decay: float
    dropout: float


def group_kfold_splits(
    indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.asarray(indices, dtype=int)
    unique_groups = np.unique(groups[indices])
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} groups available for {n_splits}-fold CV."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)
    fold_groups = np.array_split(shuffled_groups, n_splits)

    splits = []
    for heldout_groups in fold_groups:
        test_mask = np.isin(groups[indices], heldout_groups)
        test_idx = indices[test_mask]
        train_idx = indices[~test_mask]

        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        if train_groups.intersection(test_groups):
            raise RuntimeError("Group leakage detected in CV split.")

        splits.append((train_idx, test_idx))
    return splits


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


def class_weights_from_indices(labels: np.ndarray, train_idx: np.ndarray) -> torch.Tensor:
    train_y = labels[train_idx]
    n_neg = int((train_y == 0).sum())
    n_pos = int((train_y == 1).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training split must contain both classes.")

    return torch.tensor([1.0, n_neg / n_pos], dtype=torch.float, device=DEVICE)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_prob = [], []

    for data in loader:
        data = data.to(DEVICE)
        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_y.append(data.y.view(-1).detach().cpu().numpy())
        all_prob.append(probs.detach().cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_prob)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    result = {
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": float((y_pred == y_true).mean()),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return result


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_metrics = None
    best_key = None

    for threshold in candidates:
        metrics = compute_metrics(y_true, probabilities, float(threshold))
        key = (
            metrics["f1_score"],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def build_model(hp: HyperParams, node_channels: int, edge_channels: int) -> nn.Module:
    return EDKGDLClassifier(
        node_channels=node_channels,
        edge_channels=edge_channels,
        hidden_size1=hp.hidden_size1,
        hidden_size2=hp.hidden_size2,
        hidden_size3=hp.hidden_size3,
        dropout=hp.dropout,
    ).to(DEVICE)


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


def fit_with_early_stopping(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hp: HyperParams,
    seed: int,
) -> Tuple[nn.Module, int, float]:
    seed_everything(seed)
    node_channels = dataset.num_node_features
    edge_channels = dataset.num_edge_features

    model = build_model(hp, node_channels, edge_channels)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )

    if USE_CLASS_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_from_indices(labels, train_idx)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = create_loader(dataset, train_idx, TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = create_loader(dataset, val_idx, EVAL_BATCH_SIZE, shuffle=False)

    best_score = -np.inf
    best_epoch = 1
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, criterion)
        y_val, p_val = predict_probabilities(model, val_loader)
        val_pr_auc = average_precision_score(y_val, p_val)

        if val_pr_auc > best_score + 1e-8:
            best_score = float(val_pr_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= MIN_EPOCHS and patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_score


def fit_fixed_epochs(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    train_idx: np.ndarray,
    hp: HyperParams,
    epochs: int,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    node_channels = dataset.num_node_features
    edge_channels = dataset.num_edge_features

    model = build_model(hp, node_channels, edge_channels)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )

    if USE_CLASS_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_from_indices(labels, train_idx)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = create_loader(dataset, train_idx, TRAIN_BATCH_SIZE, shuffle=True)
    for _ in range(int(epochs)):
        train_one_epoch(model, train_loader, optimizer, criterion)
    return model


def sample_hyperparameters(n_trials: int, seed: int) -> List[HyperParams]:
    all_configs = [
        HyperParams(*values)
        for values in product(
            HYPERPARAM_SPACE["hidden_size1"],
            HYPERPARAM_SPACE["hidden_size2"],
            HYPERPARAM_SPACE["hidden_size3"],
            HYPERPARAM_SPACE["lr"],
            HYPERPARAM_SPACE["weight_decay"],
            HYPERPARAM_SPACE["dropout"],
        )
    ]

    rng = np.random.default_rng(seed)
    n_trials = min(n_trials, len(all_configs))
    sampled_ids = rng.choice(len(all_configs), size=n_trials, replace=False)
    return [all_configs[int(i)] for i in sampled_ids]


def nested_tune(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train_idx: np.ndarray,
    repeat_id: int,
    outer_fold_id: int,
) -> Tuple[HyperParams, float, int, pd.DataFrame]:
    configs = sample_hyperparameters(
        HYPERPARAM_TRIALS,
        seed=BASE_SEED + 100000 * repeat_id + 1000 * outer_fold_id,
    )

    inner_splits = group_kfold_splits(
        outer_train_idx,
        groups,
        n_splits=INNER_FOLDS,
        seed=BASE_SEED + 50000 * repeat_id + 500 * outer_fold_id,
    )

    tuning_rows = []

    best_choice = None
    best_key = None

    for config_id, hp in enumerate(configs):
        inner_y_all, inner_p_all, best_epochs = [], [], []

        for inner_fold_id, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            fit_seed = (
                BASE_SEED
                + 1000000 * repeat_id
                + 10000 * outer_fold_id
                + 100 * config_id
                + inner_fold_id
            )

            model, best_epoch, _ = fit_with_early_stopping(
                dataset=dataset,
                labels=labels,
                train_idx=inner_train_idx,
                val_idx=inner_val_idx,
                hp=hp,
                seed=fit_seed,
            )

            val_loader = create_loader(
                dataset, inner_val_idx, EVAL_BATCH_SIZE, shuffle=False
            )
            y_val, p_val = predict_probabilities(model, val_loader)

            inner_y_all.append(y_val)
            inner_p_all.append(p_val)
            best_epochs.append(best_epoch)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        y_oof = np.concatenate(inner_y_all)
        p_oof = np.concatenate(inner_p_all)
        threshold, inner_metrics = choose_threshold(y_oof, p_oof)
        selected_epochs = int(np.median(best_epochs))

        row = {
            "repeat": repeat_id,
            "outer_fold": outer_fold_id,
            "config_id": config_id,
            **asdict(hp),
            "inner_selected_threshold": threshold,
            "inner_selected_epochs_median": selected_epochs,
            **{f"inner_{k}": v for k, v in inner_metrics.items()},
        }
        tuning_rows.append(row)

        key = (
            inner_metrics["pr_auc"],
            inner_metrics["f1_score"],
            inner_metrics["balanced_accuracy"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_choice = (hp, threshold, selected_epochs)

    tuning_df = pd.DataFrame(tuning_rows)
    best_hp, best_threshold, final_epochs = best_choice
    return best_hp, float(best_threshold), int(final_epochs), tuning_df

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


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = BASE_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        boot_sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = boot_sample.mean()

    return (
        float(np.quantile(boot_means, 0.025)),
        float(np.quantile(boot_means, 0.975)),
    )


def summarize_outer_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        values = metrics_df[metric].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(
            values, seed=BASE_SEED + hash(metric) % 10000
        )
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "bootstrap_mean_ci_2.5%": ci_low,
                "bootstrap_mean_ci_97.5%": ci_high,
                "n_outer_evaluations": int(len(values)),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    seed_everything(BASE_SEED)
    print(f"Using device: {DEVICE}")

    graph_data_dir = Path(GRAPH_DATA_DIR)

    if not graph_data_dir.exists():
        raise FileNotFoundError(f"GRAPH_DATA_DIR does not exist: {graph_data_dir}")

    label_file = resolve_label_file(graph_data_dir)
    if not label_file.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_file}")

    labels = load_labels(str(label_file))
    if len(labels) != EXPECTED_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_GRAPHS} labels after adding the external "
            f"cohort, but found {len(labels)}."
        )

    expected_labels = np.zeros(EXPECTED_TOTAL_GRAPHS, dtype=int)
    expected_labels[INTERNAL_EDC_IDS] = 1
    expected_labels[EXTERNAL_EDC_IDS] = 1

    if not np.array_equal(labels, expected_labels):
        mismatch_ids = np.where(labels != expected_labels)[0]
        raise ValueError(
            "Graph labels do not match the required zero-based cohort layout. "
            "Expected EDCs at 0–86 and non-EDCs at 87–902, with nested CV "
            "restricted to EDCs 0–80 and non-EDCs 87–896. "
            f"Mismatch graph IDs: {mismatch_ids[:20].tolist()}"
        )

    # Explicitly verify internal and external cohort class counts.
    internal_labels = labels[INTERNAL_GRAPH_IDS]
    external_labels = labels[EXTERNAL_GRAPH_IDS]
    if (
        int((internal_labels == 1).sum()) != N_EDCS
        or int((internal_labels == 0).sum()) != N_EDCS * DECOYS_PER_EDC
    ):
        raise ValueError(
            "Internal nested-CV cohort must contain 81 EDCs (0–80) and "
            "810 non-EDCs (87–896)."
        )
    if int((external_labels == 1).sum()) != 6 or int((external_labels == 0).sum()) != 6:
        raise ValueError(
            "External cohort must contain 6 EDCs (81–86) and 6 non-EDCs "
            "(897–902)."
        )

    groups = build_parent_groups(labels)

    processed_root = Path(PROCESSED_ROOT)
    processed_file = processed_root / "processed" / "edkg_graphs.pt"
    if REPROCESS_DATASET and processed_file.exists():
        processed_file.unlink()

    dataset = EDKGGraphDataset(
        root=str(processed_root),
        graph_data_dir=str(graph_data_dir),
        label_file=str(label_file),
        n_graphs=len(labels),
    )

    if len(dataset) != len(labels):
        raise RuntimeError(
            f"Dataset has {len(dataset)} graphs but label file has {len(labels)} rows."
        )

    print(
        f"Loaded {len(dataset)} total graphs | "
        f"internal nested-CV cohort: {len(INTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 0–80; non-EDCs 87–896) | "
        f"external excluded cohort: {len(EXTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 81–86; non-EDCs 897–902) | "
        f"node features: {dataset.num_node_features} | "
        f"edge features: {dataset.num_edge_features} | "
        f"internal parent groups: {N_EDCS}"
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_export = {
        "graph_data_dir": str(graph_data_dir.resolve()),
        "label_file": str(label_file.resolve()),
        "group_mapping_csv": GROUP_MAPPING_CSV,
        "total_graphs_loaded": EXPECTED_TOTAL_GRAPHS,
        "internal_validation_cohort": {
            "edc_graph_ids": "0-80",
            "non_edc_graph_ids": "87-896",
            "n_graphs": EXPECTED_INTERNAL_GRAPHS,
            "n_edcs": N_EDCS,
            "n_non_edcs": N_EDCS * DECOYS_PER_EDC,
        },
        "external_cohort_excluded_from_nested_cv": {
            "edc_graph_ids": "81-86",
            "non_edc_graph_ids": "897-902",
            "n_graphs": len(EXTERNAL_GRAPH_IDS),
            "n_edcs": 6,
            "n_non_edcs": 6,
        },
        "decoys_per_edc": DECOYS_PER_EDC,
        "n_repeats": N_REPEATS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "hyperparam_trials": HYPERPARAM_TRIALS,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "base_seed": BASE_SEED,
        "device": str(DEVICE),
        "grouping_assumption": (
            "Each group contains one EDC and its 10 paired decoys. "
            "Default uses block ordering after graph ID 80."
        ),
    }
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as f:
        json.dump(config_export, f, ensure_ascii=False, indent=2)

    cohort_audit = pd.DataFrame(
        {
            "graph_id": np.arange(EXPECTED_TOTAL_GRAPHS, dtype=int),
            "label": labels,
            "cohort_role": np.where(
                np.isin(np.arange(EXPECTED_TOTAL_GRAPHS), INTERNAL_GRAPH_IDS),
                "internal_nested_cv",
                "external_excluded",
            ),
            "parent_edc_group": groups,
        }
    )
    cohort_audit.to_csv(
        output_dir / "cohort_membership_and_group_mapping.csv",
        index=False,
    )

    all_indices = np.sort(INTERNAL_GRAPH_IDS.astype(int))
    outer_metric_rows = []
    outer_prediction_rows = []
    selected_hp_rows = []
    all_tuning_dfs = []

    for repeat_id in range(N_REPEATS):
        print(f"\n========== Repeat {repeat_id + 1}/{N_REPEATS} ==========")

        outer_splits = group_kfold_splits(
            all_indices,
            groups,
            n_splits=OUTER_FOLDS,
            seed=BASE_SEED + repeat_id,
        )

        for outer_fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
            train_groups = np.unique(groups[outer_train_idx])
            test_groups = np.unique(groups[outer_test_idx])

            print(
                f"Repeat {repeat_id + 1}, outer fold {outer_fold_id + 1}/{OUTER_FOLDS} | "
                f"train: {len(outer_train_idx)} graphs / {len(train_groups)} groups | "
                f"test: {len(outer_test_idx)} graphs / {len(test_groups)} groups"
            )

            best_hp, threshold, final_epochs, tuning_df = nested_tune(
                dataset=dataset,
                labels=labels,
                groups=groups,
                outer_train_idx=outer_train_idx,
                repeat_id=repeat_id,
                outer_fold_id=outer_fold_id,
            )
            all_tuning_dfs.append(tuning_df)

            final_seed = (
                BASE_SEED + 9000000 + 100000 * repeat_id + 1000 * outer_fold_id
            )
            final_model = fit_fixed_epochs(
                dataset=dataset,
                labels=labels,
                train_idx=outer_train_idx,
                hp=best_hp,
                epochs=final_epochs,
                seed=final_seed,
            )

            outer_test_loader = create_loader(
                dataset, outer_test_idx, EVAL_BATCH_SIZE, shuffle=False
            )
            y_test, p_test = predict_probabilities(final_model, outer_test_loader)
            metrics = compute_metrics(y_test, p_test, threshold)

            if not np.array_equal(y_test.astype(int), labels[outer_test_idx].astype(int)):
                raise RuntimeError(
                    "Test-loader label order did not match outer_test_idx order."
                )

            metric_row = {
                "repeat": repeat_id + 1,
                "outer_fold": outer_fold_id + 1,
                "n_train_graphs": int(len(outer_train_idx)),
                "n_test_graphs": int(len(outer_test_idx)),
                "n_train_groups": int(len(train_groups)),
                "n_test_groups": int(len(test_groups)),
                "n_train_edcs": int((labels[outer_train_idx] == 1).sum()),
                "n_test_edcs": int((labels[outer_test_idx] == 1).sum()),
                "n_train_non_edcs": int((labels[outer_train_idx] == 0).sum()),
                "n_test_non_edcs": int((labels[outer_test_idx] == 0).sum()),
                "selected_epochs": int(final_epochs),
                **asdict(best_hp),
                **metrics,
            }
            outer_metric_rows.append(metric_row)

            selected_hp_rows.append(
                {
                    "repeat": repeat_id + 1,
                    "outer_fold": outer_fold_id + 1,
                    "selected_threshold": threshold,
                    "selected_epochs": int(final_epochs),
                    **asdict(best_hp),
                    "inner_best_pr_auc": float(
                        tuning_df["inner_pr_auc"].max()
                    ),
                }
            )

            y_pred = (p_test >= threshold).astype(int)
            for graph_id, group_id, y_true, probability, pred in zip(
                outer_test_idx, groups[outer_test_idx], y_test, p_test, y_pred
            ):
                outer_prediction_rows.append(
                    {
                        "repeat": repeat_id + 1,
                        "outer_fold": outer_fold_id + 1,
                        "graph_id": int(graph_id),
                        "parent_edc_group": int(group_id),
                        "y_true": int(y_true),
                        "predicted_probability_edc": float(probability),
                        "selected_threshold": float(threshold),
                        "y_pred": int(pred),
                    }
                )

            print(
                "  Selected inner-CV model | "
                f"PR-AUC={tuning_df['inner_pr_auc'].max():.3f}, "
                f"threshold={threshold:.3f}, epochs={final_epochs}"
            )
            print(
                "  Outer test | "
                f"PR-AUC={metrics['pr_auc']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, "
                f"Recall={metrics['recall']:.3f}, Specificity={metrics['specificity']:.3f}, "
                f"Balanced accuracy={metrics['balanced_accuracy']:.3f}, "
                f"MCC={metrics['mcc']:.3f}, F1-score={metrics['f1_score']:.3f}"
            )

            del final_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(outer_metric_rows)
    predictions_df = pd.DataFrame(outer_prediction_rows)
    selected_hp_df = pd.DataFrame(selected_hp_rows)
    tuning_df = pd.concat(all_tuning_dfs, ignore_index=True)
    summary_df = summarize_outer_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "outer_fold_predictions.csv", index=False)
    selected_hp_df.to_csv(
        output_dir / "selected_hyperparameters_per_outer_fold.csv", index=False
    )
    tuning_df.to_csv(output_dir / "inner_cv_hyperparameter_results.csv", index=False)
    summary_df.to_csv(output_dir / "metric_summary_bootstrap_ci.csv", index=False)

    readme_text = f"""Repeated group-nested CV completed for the INTERNAL 891-compound cohort.

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
- {N_REPEATS} repeats × {OUTER_FOLDS} outer group folds = {N_REPEATS * OUTER_FOLDS} outer evaluations.
- Each internal parent group contains one EDC and {DECOYS_PER_EDC} paired decoys.
- All samples from one internal parent group are kept in either train or test.
- Hyperparameters, early stopping, and classification thresholds are selected
  only within the internal outer-training groups through {INNER_FOLDS}-fold inner CV.

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
"""
    (output_dir / "README_results.txt").write_text(readme_text, encoding="utf-8")

    print("\n========== Analysis complete ==========")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
