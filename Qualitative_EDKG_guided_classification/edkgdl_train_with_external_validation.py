#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline internal 80/20 holdout training with inner 10-fold group CV
plus independent external validation for the qualitative EDKG-DL classifier.

Zero-based graph-ID design
--------------------------
Internal 891-compound modeling cohort (including 81 confirmed EDCs-collected up to April 2023 and 810 DUD-E-derived non-EDCs):
  - EDCs:     graph IDs 0-80 (n=81)
  - non-EDCs: graph IDs 87-896 (n=810)

Independent external validation cohort (including 6 confirmed EDCs and 6 confirmed non-EDCs collected during April 2023 and April 2025):
  - EDCs:     graph IDs 81-86 (n=6)
  - non-EDCs: graph IDs 897-902 (n=6)

The internal 891 compounds alone are used for:
  1) group-preserving 80/20 internal train/test splitting;
  2) 10-fold group CV inside the 80% development set;
  3) hyperparameter selection.

After hyperparameters are selected, the script:
  - trains one model on the 80% internal development set and evaluates its
    untouched 20% internal test set;
  - independently refits the selected configuration on all 891 internal
    compounds, then evaluates the 12 external compounds exactly once.

The external 12 compounds are excluded from all internal splitting, tuning,
model selection, and training steps.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    precision_score,
    roc_auc_score,
)

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


# ----- Paths -----
GRAPH_DATA_DIR = "edkgdl_all_data"
PROCESSED_ROOT = "."
OUTPUT_DIR = "EDKGDL_results"
LABEL_FILE = None
GROUP_MAPPING_CSV = None
REPROCESS_DATASET = False

# ----- Zero-based internal and external data design -----
N_INTERNAL_EDCS = 81
DECOYS_PER_EDC = 10
N_INTERNAL_NON_EDCS = N_INTERNAL_EDCS * DECOYS_PER_EDC  # 810
N_INTERNAL_GRAPHS = N_INTERNAL_EDCS + N_INTERNAL_NON_EDCS  # 891

N_EXTERNAL_EDCS = 6
N_EXTERNAL_NON_EDCS = 6
N_EXTERNAL_GRAPHS = N_EXTERNAL_EDCS + N_EXTERNAL_NON_EDCS  # 12
N_TOTAL_GRAPHS = N_INTERNAL_GRAPHS + N_EXTERNAL_GRAPHS  # 903

INTERNAL_EDC_GRAPH_IDS = np.arange(0, 81, dtype=int)
EXTERNAL_EDC_GRAPH_IDS = np.arange(81, 87, dtype=int)
INTERNAL_NON_EDC_GRAPH_IDS = np.arange(87, 897, dtype=int)
EXTERNAL_NON_EDC_GRAPH_IDS = np.arange(897, 903, dtype=int)

INTERNAL_GRAPH_IDS = np.sort(
    np.concatenate([INTERNAL_EDC_GRAPH_IDS, INTERNAL_NON_EDC_GRAPH_IDS])
)
EXTERNAL_GRAPH_IDS = np.sort(
    np.concatenate([EXTERNAL_EDC_GRAPH_IDS, EXTERNAL_NON_EDC_GRAPH_IDS])
)

N_EDCS = N_INTERNAL_EDCS
EXPECTED_N_GRAPHS = N_INTERNAL_GRAPHS

# ----- 80/20 outer holdout -----
OUTER_TEST_GROUPS = 16
OUTER_SPLIT_SEED = 12345

# ----- Inner cross-validation -----
INNER_FOLDS = 10
INNER_SPLIT_SEED = 23456

# ----- Offline hyperparameter search -----
N_HYPERPARAMETER_TRIALS = 100
HYPERPARAMETER_SEED = 34567

HIDDEN_CHANNEL_OPTIONS = [100, 80, 60, 50, 40, 30, 20, 10]
BATCH_SIZE_OPTIONS = [300, 200, 100, 50, 20]
LR_MIN = 1e-4
LR_MAX = 1e-3
WEIGHT_DECAY_MEAN = 5e-4
WEIGHT_DECAY_SD = 1e-5
DROPOUT = 0.5

# ----- Training -----
MAX_EPOCHS = 200   # matches the original notebook
TRAINING_SEED = 45678
USE_CLASS_WEIGHTED_LOSS = False
CLASSIFICATION_THRESHOLD = 0.5

# ----- Data loading / device -----
NUM_WORKERS = 0
EVAL_BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Resume behavior -----
RESUME_IF_OUTPUT_EXISTS = True

# 2. REPRODUCIBILITY AND BASIC FILE UTILITIES

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible model initialization and shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def append_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Append a DataFrame to a CSV, writing header only on first creation."""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def write_json(path: Path, payload: Dict) -> None:
    """Write an indented JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# 3. DATASET

class EDKGGraphDataset(InMemoryDataset):
    """Read graph files using the original EDKG-DL folder structure."""

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
                self.processed_paths[0],
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch.
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["Graph_label.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["edkgdl_903_internal_external_graphs.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        label_df = pd.read_csv(self.label_file, header=None)

        if label_df.shape[0] != self.n_graphs:
            raise ValueError(
                f"Graph_label.txt contains {label_df.shape[0]} rows, "
                f"but n_graphs={self.n_graphs}."
            )

        graphs: List[Data] = []

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
                    f"{edge_path} must contain source, target, and at least "
                    "one edge feature column."
                )

            x = torch.tensor(node_df.values, dtype=torch.float)
            edge_index = torch.tensor(
                edge_df.iloc[:, :2].T.values,
                dtype=torch.long,
            )
            edge_attr = torch.tensor(
                edge_df.iloc[:, 2:].values,
                dtype=torch.float,
            )

            y_value = int(label_df.iloc[graph_id, 1])
            y = torch.tensor([y_value], dtype=torch.long)

            graphs.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    graph_id=torch.tensor([graph_id], dtype=torch.long),
                )
            )

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


def load_labels(label_path: Path) -> np.ndarray:
    """Load labels and check graph-ID order."""
    df = pd.read_csv(label_path, header=None)

    if df.shape[1] < 2:
        raise ValueError(
            "Graph_label.txt must have at least two columns: graph_id,label."
        )

    graph_ids = df.iloc[:, 0].astype(int).to_numpy()
    labels = df.iloc[:, 1].astype(int).to_numpy()

    if not np.array_equal(graph_ids, np.arange(len(labels))):
        raise ValueError(
            "The first Graph_label.txt column must be sequential graph IDs "
            "from 0 to N-1."
        )

    return labels


def resolve_label_path(graph_data_dir: Path) -> Path:
    """Resolve a label file robustly for Graph_label.txt or Graph_label(1).txt."""
    if LABEL_FILE is not None:
        candidate = Path(LABEL_FILE)
        if not candidate.exists():
            raise FileNotFoundError(
                f"LABEL_FILE was set but does not exist: {candidate.resolve()}"
            )
        return candidate

    candidates = [
        graph_data_dir / "Graph_label.txt",
        graph_data_dir / "Graph_label(1).txt",
        Path("Graph_label.txt"),
        Path("Graph_label(1).txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Graph_label.txt or Graph_label(1).txt. "
        "Set LABEL_FILE explicitly at the top of the script."
    )


def validate_zero_based_labels(labels: np.ndarray) -> None:
    """Confirm the requested 891-internal + 12-external split."""
    if len(labels) != N_TOTAL_GRAPHS:
        raise ValueError(
            f"Expected {N_TOTAL_GRAPHS} labels (graph IDs 0–902), "
            f"found {len(labels)}."
        )

    expected = np.zeros(N_TOTAL_GRAPHS, dtype=int)
    expected[INTERNAL_EDC_GRAPH_IDS] = 1
    expected[EXTERNAL_EDC_GRAPH_IDS] = 1

    if not np.array_equal(labels, expected):
        mismatch = np.where(labels != expected)[0]
        raise ValueError(
            "Graph labels do not match the required zero-based design. "
            "Expected EDC IDs 0–86 and non-EDC IDs 87–902, with internal "
            "EDCs 0–80, external EDCs 81–86, internal non-EDCs 87–896, and "
            "external non-EDCs 897–902. Mismatches at graph IDs: "
            f"{mismatch[:20].tolist()}"
        )

    if len(INTERNAL_GRAPH_IDS) != N_INTERNAL_GRAPHS:
        raise RuntimeError("Internal graph-ID configuration is not 891 samples.")
    if len(EXTERNAL_GRAPH_IDS) != N_EXTERNAL_GRAPHS:
        raise RuntimeError("External graph-ID configuration is not 12 samples.")
    if np.intersect1d(INTERNAL_GRAPH_IDS, EXTERNAL_GRAPH_IDS).size:
        raise RuntimeError("Internal and external graph-ID sets overlap.")


def build_parent_groups(labels: np.ndarray) -> np.ndarray:
    """
    Build parent-group assignments for the INTERNAL 891-compound cohort only.

    Parent group i (i=0..80) contains:
      - internal EDC graph ID i;
      - internal decoy graph IDs 87 + 10*i through 96 + 10*i.

    External graph IDs 81–86 and 897–902 receive parent_group = -1 and are
    never passed into internal splitting or inner cross-validation.
    """
    validate_zero_based_labels(labels)

    if GROUP_MAPPING_CSV is not None:
        map_path = Path(GROUP_MAPPING_CSV)
        if not map_path.exists():
            raise FileNotFoundError(f"Cannot find GROUP_MAPPING_CSV: {map_path}")

        mapping = pd.read_csv(map_path)
        required = {"graph_id", "parent_edc_id"}
        if not required.issubset(mapping.columns):
            raise ValueError(
                "GROUP_MAPPING_CSV must contain graph_id and parent_edc_id."
            )

        mapping = mapping[mapping["graph_id"].isin(INTERNAL_GRAPH_IDS)].copy()
        mapping = mapping.sort_values("graph_id").reset_index(drop=True)

        if not np.array_equal(
            mapping["graph_id"].astype(int).to_numpy(),
            INTERNAL_GRAPH_IDS,
        ):
            raise ValueError(
                "GROUP_MAPPING_CSV must contain every internal graph ID exactly "
                "once (0–80 and 87–896)."
            )

        groups = np.full(N_TOTAL_GRAPHS, -1, dtype=int)
        groups[INTERNAL_GRAPH_IDS] = mapping["parent_edc_id"].astype(int).to_numpy()

    else:
        groups = np.full(N_TOTAL_GRAPHS, -1, dtype=int)
        groups[INTERNAL_EDC_GRAPH_IDS] = np.arange(N_INTERNAL_EDCS)
        groups[INTERNAL_NON_EDC_GRAPH_IDS] = np.repeat(
            np.arange(N_INTERNAL_EDCS),
            DECOYS_PER_EDC,
        )

    internal_groups = groups[INTERNAL_GRAPH_IDS]
    unique_groups = np.unique(internal_groups)
    if not np.array_equal(unique_groups, np.arange(N_INTERNAL_EDCS)):
        raise ValueError(
            f"Expected parent groups 0–{N_INTERNAL_EDCS - 1}, "
            f"found {unique_groups.tolist()}."
        )

    for group_id in unique_groups:
        graph_ids = INTERNAL_GRAPH_IDS[groups[INTERNAL_GRAPH_IDS] == group_id]
        group_labels = labels[graph_ids]
        n_pos = int((group_labels == 1).sum())
        n_neg = int((group_labels == 0).sum())

        if n_pos != 1 or n_neg != DECOYS_PER_EDC:
            raise ValueError(
                f"Internal parent group {group_id} has {n_pos} EDC(s) and "
                f"{n_neg} non-EDC(s); expected 1 and {DECOYS_PER_EDC}."
            )

    return groups


# 4. EDGE-AWARE GCN MODEL

class GCNConvEdge(MessagePassing):
    """
    Edge-aware graph convolution consistent with the original EDKG-DL notebook.

    Node embeddings are iteratively updated. Directed edge embeddings are
    projected at each layer and concatenated to incoming messages.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr="add")
        self.lin_node = Linear(in_channels, out_channels, bias=False)
        self.lin_edge = Linear(edge_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(2 * out_channels))
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

        node_embeddings = self.lin_node(x)
        edge_embeddings = self.lin_edge(edge_attr)

        # Newly added self-loops receive zero-valued edge embeddings.
        zero_self_loops = torch.zeros(
            (node_embeddings.shape[0], edge_embeddings.shape[1]),
            dtype=edge_embeddings.dtype,
            device=edge_embeddings.device,
        )
        extended_edge_embeddings = torch.cat(
            [edge_embeddings, zero_self_loops],
            dim=0,
        )

        row, col = edge_index_with_loops
        deg = degree(col, node_embeddings.size(0), dtype=node_embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index_with_loops,
            x=node_embeddings,
            norm=norm,
            edge_embeddings=extended_edge_embeddings,
        )
        out = out + self.bias

        # Return the original-edge embeddings, without added self-loop rows,
        # for use by the next graph layer.
        return out, edge_embeddings

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        message = torch.cat([x_j, edge_embeddings], dim=1)
        return norm.view(-1, 1) * message


class EDKGDLClassifier(nn.Module):
    """Three-layer directed edge-aware GCN plus global mean pooling and FCNN."""

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

        self.conv1 = GCNConvEdge(
            node_channels,
            hidden_size1,
            edge_channels,
        )
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
        self.classifier = Linear(2 * hidden_size3, 2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_embeddings = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x, edge_embeddings = self.conv2(x, edge_index, edge_embeddings)
        x = F.relu(x)

        x, _ = self.conv3(x, edge_index, edge_embeddings)

        graph_embeddings = global_mean_pool(x, batch)
        graph_embeddings = F.dropout(
            graph_embeddings,
            p=self.dropout,
            training=self.training,
        )

        return self.classifier(graph_embeddings)


# 5. SPLITTING, HYPERPARAMETERS, LOADERS

@dataclass(frozen=True)
class HyperParameters:
    hidden_size1: int
    hidden_size2: int
    hidden_size3: int
    lr: float
    weight_decay: float
    batch_size: int
    dropout: float = DROPOUT


def make_outer_holdout_split(
    groups: np.ndarray,
    candidate_graph_ids: np.ndarray,
    test_group_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a group-preserving 80/20 split using only the INTERNAL 891 cohort.

    The 12 external graphs are not eligible for any internal split.
    """
    candidate_graph_ids = np.asarray(candidate_graph_ids, dtype=int)
    candidate_groups = groups[candidate_graph_ids]

    if np.any(candidate_groups < 0):
        raise ValueError(
            "candidate_graph_ids includes external or otherwise ungrouped graphs."
        )

    unique_groups = np.unique(candidate_groups)
    if test_group_count <= 0 or test_group_count >= len(unique_groups):
        raise ValueError(
            "OUTER_TEST_GROUPS must be between 1 and number of internal groups - 1."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    test_groups = np.sort(shuffled_groups[:test_group_count])
    train_groups = np.sort(shuffled_groups[test_group_count:])

    is_test = np.isin(candidate_groups, test_groups)
    test_indices = candidate_graph_ids[is_test]
    train_indices = candidate_graph_ids[~is_test]

    if set(groups[train_indices]).intersection(set(groups[test_indices])):
        raise RuntimeError("Parent-group leakage was detected in internal outer split.")

    return train_indices, test_indices, train_groups, test_groups


def make_group_kfold_splits(
    train_indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create random group-preserving folds for inner 10-fold CV.

    Every parent group has the same 1:10 EDC:decoy composition, so fold-level
    class balance remains approximately identical.
    """
    train_indices = np.asarray(train_indices, dtype=int)
    unique_groups = np.unique(groups[train_indices])

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} groups for {n_splits}-fold CV."
        )

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    fold_group_arrays = np.array_split(shuffled_groups, n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for validation_groups in fold_group_arrays:
        is_validation = np.isin(groups[train_indices], validation_groups)

        inner_train_indices = train_indices[~is_validation]
        inner_validation_indices = train_indices[is_validation]

        if set(groups[inner_train_indices]).intersection(
            set(groups[inner_validation_indices])
        ):
            raise RuntimeError("Parent-group leakage in inner CV.")

        splits.append((inner_train_indices, inner_validation_indices))

    return splits


def sample_hyperparameter_candidates(
    n_trials: int,
    seed: int,
) -> List[HyperParameters]:
    """
    Sample candidates offline using the original W&B parameter ranges.

    Hidden sizes and batch sizes are sampled from their original discrete
    choices. Learning rate is uniformly sampled between 1e-4 and 1e-3.
    Weight decay follows the original normal distribution centered at 5e-4.
    """
    rng = np.random.default_rng(seed)
    candidates: List[HyperParameters] = []

    for _ in range(n_trials):
        weight_decay = max(
            1e-8,
            float(rng.normal(WEIGHT_DECAY_MEAN, WEIGHT_DECAY_SD)),
        )

        candidates.append(
            HyperParameters(
                hidden_size1=int(rng.choice(HIDDEN_CHANNEL_OPTIONS)),
                hidden_size2=int(rng.choice(HIDDEN_CHANNEL_OPTIONS)),
                hidden_size3=int(rng.choice(HIDDEN_CHANNEL_OPTIONS)),
                lr=float(rng.uniform(LR_MIN, LR_MAX)),
                weight_decay=weight_decay,
                batch_size=int(rng.choice(BATCH_SIZE_OPTIONS)),
                dropout=float(DROPOUT),
            )
        )

    return candidates


def create_loader(
    dataset: InMemoryDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyG DataLoader for a selected list of graph IDs."""
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(
    dataset: InMemoryDataset,
    hp: HyperParameters,
) -> EDKGDLClassifier:
    """Initialize one EDKG-DL model."""
    return EDKGDLClassifier(
        node_channels=dataset.num_node_features,
        edge_channels=dataset.num_edge_features,
        hidden_size1=hp.hidden_size1,
        hidden_size2=hp.hidden_size2,
        hidden_size3=hp.hidden_size3,
        dropout=hp.dropout,
    ).to(DEVICE)


def make_criterion(
    labels: np.ndarray,
    train_indices: np.ndarray,
) -> nn.Module:
    """Create unweighted or class-weighted cross entropy as configured."""
    if not USE_CLASS_WEIGHTED_LOSS:
        return nn.CrossEntropyLoss()

    train_y = labels[train_indices]
    n_negative = int((train_y == 0).sum())
    n_positive = int((train_y == 1).sum())

    if n_positive == 0 or n_negative == 0:
        raise ValueError("Both classes must be present in an inner training fold.")

    class_weights = torch.tensor(
        [1.0, n_negative / n_positive],
        dtype=torch.float,
        device=DEVICE,
    )
    return nn.CrossEntropyLoss(weight=class_weights)


# 6. METRICS AND MODEL FITTING

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate all binary metrics, with EDC encoded as the positive class (1).

    The default threshold is 0.5, matching the original `argmax(dim=1)`
    prediction rule.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Both classes are required to calculate all metrics.")

    y_pred = (probabilities >= float(threshold)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2

    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "f1_score": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "accuracy": float((y_pred == y_true).mean()),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one training epoch and return mean graph-level loss."""
    model.train()

    loss_sum = 0.0
    n_graphs = 0

    for data in loader:
        data = data.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = criterion(logits, data.y.view(-1))
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * int(data.num_graphs)
        n_graphs += int(data.num_graphs)

    return loss_sum / max(n_graphs, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return graph IDs, true labels, and EDC probabilities."""
    model.eval()

    graph_ids_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    probability_list: List[np.ndarray] = []

    for data in loader:
        data = data.to(DEVICE)

        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        graph_ids_list.append(data.graph_id.view(-1).cpu().numpy())
        y_list.append(data.y.view(-1).cpu().numpy())
        probability_list.append(probabilities.cpu().numpy())

    return (
        np.concatenate(graph_ids_list).astype(int),
        np.concatenate(y_list).astype(int),
        np.concatenate(probability_list).astype(float),
    )


def fit_inner_fold(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    hp: HyperParameters,
    inner_train_indices: np.ndarray,
    validation_indices: np.ndarray,
    trial_id: int,
    fold_id: int,
    seed: int,
) -> Tuple[nn.Module, List[Dict], Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Train one inner-CV fold for the fixed 200 epochs.

    The entire epoch history is saved in memory and returned for offline export.
    No outer test data are accessed here.
    """
    seed_everything(seed)

    model = build_model(dataset, hp)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )
    criterion = make_criterion(labels, inner_train_indices)

    train_loader = create_loader(
        dataset,
        inner_train_indices,
        batch_size=hp.batch_size,
        shuffle=True,
    )
    validation_loader = create_loader(
        dataset,
        validation_indices,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    history_rows: List[Dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        # Validation assessment is confined to the current inner-CV fold.
        _, y_validation, p_validation = predict(model, validation_loader)
        val_metrics = calculate_metrics(
            y_validation,
            p_validation,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        history_rows.append(
            {
                "trial_id": int(trial_id),
                "inner_fold": int(fold_id),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_pr_auc": val_metrics["pr_auc"],
                "validation_roc_auc": val_metrics["roc_auc"],
                "validation_f1_score": val_metrics["f1_score"],
                "validation_recall": val_metrics["recall"],
                "validation_specificity": val_metrics["specificity"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_mcc": val_metrics["mcc"],
                "validation_accuracy": val_metrics["accuracy"],
            }
        )

    graph_ids, y_validation, p_validation = predict(model, validation_loader)
    final_metrics = calculate_metrics(
        y_validation,
        p_validation,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    return (
        model,
        history_rows,
        final_metrics,
        graph_ids,
        y_validation,
        p_validation,
    )


def train_final_model(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    hp: HyperParameters,
    outer_train_indices: np.ndarray,
    seed: int,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Train the selected final model on the complete 80% outer-training set.

    The holdout test set is not evaluated during these 200 epochs.
    """
    seed_everything(seed)

    model = build_model(dataset, hp)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )
    criterion = make_criterion(labels, outer_train_indices)

    train_loader = create_loader(
        dataset,
        outer_train_indices,
        batch_size=hp.batch_size,
        shuffle=True,
    )

    history_rows: List[Dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        # Record only training loss; never inspect outer test performance here.
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
            }
        )

    return model, history_rows


# 7. OFFLINE INNER-CV HYPERPARAMETER SEARCH

def choose_best_trial(trial_summary_df: pd.DataFrame) -> pd.Series:
    """
    Choose the model using only inner-CV out-of-fold performance.

    Selection hierarchy:
    1. Positive-class F1-score (primary; matches original sweep objective)
    2. PR-AUC
    3. Balanced accuracy
    4. ROC-AUC
    """
    ranked = trial_summary_df.sort_values(
        [
            "inner_oof_f1_score",
            "inner_oof_pr_auc",
            "inner_oof_balanced_accuracy",
            "inner_oof_roc_auc",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return ranked.iloc[0]


def run_inner_cv_for_trial(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    hp: HyperParameters,
    trial_id: int,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Evaluate one hyperparameter candidate through inner 10-fold group CV.

    All history, fold metrics, and validation predictions are written locally.
    """
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_graph_ids: List[np.ndarray] = []

    fold_metric_rows: List[Dict] = []
    all_epoch_rows: List[Dict] = []
    all_prediction_rows: List[Dict] = []

    for fold_id, (inner_train_indices, validation_indices) in enumerate(
        inner_splits,
        start=1,
    ):
        fold_seed = TRAINING_SEED + trial_id * 1000 + fold_id

        (
            model,
            epoch_rows,
            fold_metrics,
            graph_ids,
            y_validation,
            p_validation,
        ) = fit_inner_fold(
            dataset=dataset,
            labels=labels,
            hp=hp,
            inner_train_indices=inner_train_indices,
            validation_indices=validation_indices,
            trial_id=trial_id,
            fold_id=fold_id,
            seed=fold_seed,
        )

        all_epoch_rows.extend(epoch_rows)

        fold_metric_rows.append(
            {
                "trial_id": int(trial_id),
                "inner_fold": int(fold_id),
                "n_inner_train": int(len(inner_train_indices)),
                "n_validation": int(len(validation_indices)),
                "n_inner_train_edcs": int(
                    (labels[inner_train_indices] == 1).sum()
                ),
                "n_validation_edcs": int(
                    (labels[validation_indices] == 1).sum()
                ),
                **asdict(hp),
                **fold_metrics,
            }
        )

        y_pred = (p_validation >= CLASSIFICATION_THRESHOLD).astype(int)

        for graph_id, y_true, probability, pred in zip(
            graph_ids,
            y_validation,
            p_validation,
            y_pred,
        ):
            all_prediction_rows.append(
                {
                    "trial_id": int(trial_id),
                    "inner_fold": int(fold_id),
                    "graph_id": int(graph_id),
                    "y_true": int(y_true),
                    "predicted_probability_edc": float(probability),
                    "threshold": float(CLASSIFICATION_THRESHOLD),
                    "y_pred": int(pred),
                    **asdict(hp),
                }
            )

        all_y.append(y_validation)
        all_p.append(p_validation)
        all_graph_ids.append(graph_ids)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # All development-set samples appear once as inner out-of-fold predictions.
    y_oof = np.concatenate(all_y)
    p_oof = np.concatenate(all_p)
    graph_ids_oof = np.concatenate(all_graph_ids)

    if len(np.unique(graph_ids_oof)) != len(graph_ids_oof):
        raise RuntimeError(
            f"Trial {trial_id}: duplicate inner out-of-fold graph IDs detected."
        )

    oof_metrics = calculate_metrics(
        y_oof,
        p_oof,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    append_dataframe(
        output_dir / "inner_epoch_history.csv",
        pd.DataFrame(all_epoch_rows),
    )
    append_dataframe(
        output_dir / "inner_fold_metrics.csv",
        pd.DataFrame(fold_metric_rows),
    )
    append_dataframe(
        output_dir / "inner_oof_predictions_all_trials.csv",
        pd.DataFrame(all_prediction_rows),
    )

    summary = {
        "trial_id": int(trial_id),
        "n_inner_folds": int(len(inner_splits)),
        "n_inner_oof_samples": int(len(y_oof)),
        "n_inner_oof_edcs": int((y_oof == 1).sum()),
        "n_inner_oof_non_edcs": int((y_oof == 0).sum()),
        **asdict(hp),
        **{f"inner_oof_{key}": value for key, value in oof_metrics.items()},
    }

    return summary


# 8. MAIN

def main() -> None:
    """
    Run internal 891-compound model development/testing, then independent
    12-compound external validation.
    """
    seed_everything(TRAINING_SEED)

    print(f"Device: {DEVICE}")

    graph_data_dir = Path(GRAPH_DATA_DIR)
    if not graph_data_dir.exists():
        raise FileNotFoundError(
            f"Cannot find GRAPH_DATA_DIR: {graph_data_dir.resolve()}"
        )

    label_path = resolve_label_path(graph_data_dir)
    labels = load_labels(label_path)
    validate_zero_based_labels(labels)

    parent_groups = build_parent_groups(labels)

    processed_path = (
        Path(PROCESSED_ROOT)
        / "processed"
        / "edkgdl_903_internal_external_graphs.pt"
    )
    if REPROCESS_DATASET and processed_path.exists():
        processed_path.unlink()

    # Load all 903 graph objects, but select only INTERNAL_GRAPH_IDS for every
    # split, tuning, and training operation before the external validation step.
    dataset = EDKGGraphDataset(
        root=PROCESSED_ROOT,
        graph_data_dir=str(graph_data_dir),
        label_file=str(label_path),
        n_graphs=N_TOTAL_GRAPHS,
    )

    if len(dataset) != N_TOTAL_GRAPHS:
        raise RuntimeError(
            f"Dataset length {len(dataset)} differs from expected "
            f"{N_TOTAL_GRAPHS} graphs."
        )

    print(
        f"Loaded {len(dataset)} graph objects | "
        f"internal cohort={len(INTERNAL_GRAPH_IDS)} | "
        f"external cohort={len(EXTERNAL_GRAPH_IDS)} | "
        f"node features={dataset.num_node_features} | "
        f"edge features={dataset.num_edge_features}"
    )

    (
        outer_train_indices,
        outer_test_indices,
        outer_train_groups,
        outer_test_groups,
    ) = make_outer_holdout_split(
        groups=parent_groups,
        candidate_graph_ids=INTERNAL_GRAPH_IDS,
        test_group_count=OUTER_TEST_GROUPS,
        seed=OUTER_SPLIT_SEED,
    )

    if len(outer_train_indices) != 65 * (1 + DECOYS_PER_EDC):
        raise RuntimeError(
            f"Unexpected internal development-set size: {len(outer_train_indices)}."
        )
    if len(outer_test_indices) != OUTER_TEST_GROUPS * (1 + DECOYS_PER_EDC):
        raise RuntimeError(
            f"Unexpected internal test-set size: {len(outer_test_indices)}."
        )
    if np.intersect1d(outer_train_indices, EXTERNAL_GRAPH_IDS).size:
        raise RuntimeError("External graphs entered internal development set.")
    if np.intersect1d(outer_test_indices, EXTERNAL_GRAPH_IDS).size:
        raise RuntimeError("External graphs entered internal test set.")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "graph_data_dir": str(graph_data_dir.resolve()),
        "label_file": str(label_path.resolve()),
        "device": str(DEVICE),
        "n_total_graphs_loaded": int(len(dataset)),
        "internal_cohort": {
            "edcs": "graph IDs 0–80",
            "non_edcs": "graph IDs 87–896",
            "n_total": N_INTERNAL_GRAPHS,
        },
        "external_cohort": {
            "edcs": "graph IDs 81–86",
            "non_edcs": "graph IDs 897–902",
            "n_total": N_EXTERNAL_GRAPHS,
        },
        "outer_train_groups": int(len(outer_train_groups)),
        "outer_test_groups": int(len(outer_test_groups)),
        "outer_train_graphs": int(len(outer_train_indices)),
        "outer_test_graphs": int(len(outer_test_indices)),
        "outer_split_seed": OUTER_SPLIT_SEED,
        "inner_folds": INNER_FOLDS,
        "inner_split_seed": INNER_SPLIT_SEED,
        "hyperparameter_trials": N_HYPERPARAMETER_TRIALS,
        "hyperparameter_seed": HYPERPARAMETER_SEED,
        "max_epochs": MAX_EPOCHS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
        "model_selection_priority": [
            "inner_oof_f1_score",
            "inner_oof_pr_auc",
            "inner_oof_balanced_accuracy",
            "inner_oof_roc_auc",
        ],
        "internal_grouping_rule": (
            "Each internal EDC graph ID 0–80 and its 10 paired internal "
            "DUD-E decoys from graph IDs 87–896 were assigned to one parent group."
        ),
        "external_validation_rule": (
            "The external graphs (81–86 and 897–902) were excluded from all "
            "internal split, hyperparameter-selection, and training steps. "
            "After internal model selection, the selected configuration was "
            "refit using all 891 internal compounds and evaluated once on "
            "the 12 external compounds."
        ),
    }
    write_json(output_dir / "run_configuration.json", configuration)

    all_graph_ids = np.arange(N_TOTAL_GRAPHS, dtype=int)
    dataset_role = np.full(N_TOTAL_GRAPHS, "external", dtype=object)
    dataset_role[INTERNAL_GRAPH_IDS] = "internal"

    partition = np.full(N_TOTAL_GRAPHS, "external_not_used_in_internal_cv", dtype=object)
    partition[outer_train_indices] = "internal_outer_train"
    partition[outer_test_indices] = "internal_outer_test"

    split_df = pd.DataFrame(
        {
            "graph_id": all_graph_ids,
            "label": labels,
            "dataset_role": dataset_role,
            "parent_edc_group": parent_groups,
            "partition": partition,
        }
    )
    split_df.to_csv(output_dir / "all_903_dataset_roles_and_internal_split.csv", index=False)

    # --- Create reproducible offline hyperparameter candidates ---
    candidates = sample_hyperparameter_candidates(
        N_HYPERPARAMETER_TRIALS,
        HYPERPARAMETER_SEED,
    )
    candidate_df = pd.DataFrame(
        [
            {"trial_id": trial_id, **asdict(hp)}
            for trial_id, hp in enumerate(candidates, start=1)
        ]
    )
    candidate_path = output_dir / "candidate_hyperparameters.csv"

    if candidate_path.exists() and RESUME_IF_OUTPUT_EXISTS:
        existing_candidates = pd.read_csv(candidate_path)
        if not existing_candidates.equals(candidate_df):
            raise RuntimeError(
                "Existing candidate_hyperparameters.csv differs from the "
                "current deterministic candidate list. Use a new OUTPUT_DIR "
                "or delete the old output folder."
            )
    else:
        candidate_df.to_csv(candidate_path, index=False)

    # Inner CV sees only the 715 internal-development graph IDs.
    inner_splits = make_group_kfold_splits(
        train_indices=outer_train_indices,
        groups=parent_groups,
        n_splits=INNER_FOLDS,
        seed=INNER_SPLIT_SEED,
    )

    trial_summary_path = output_dir / "hyperparameter_trial_summary.csv"
    if trial_summary_path.exists() and RESUME_IF_OUTPUT_EXISTS:
        completed_trials_df = pd.read_csv(trial_summary_path)
        completed_trial_ids = set(completed_trials_df["trial_id"].astype(int))
    else:
        completed_trial_ids = set()

    print(
        f"Internal outer holdout: {len(outer_train_indices)} development graphs "
        f"({int((labels[outer_train_indices] == 1).sum())} EDCs) | "
        f"{len(outer_test_indices)} untouched test graphs "
        f"({int((labels[outer_test_indices] == 1).sum())} EDCs)"
    )
    print(
        f"Internal inner CV: {INNER_FOLDS}-fold group CV within the "
        f"{len(outer_train_indices)}-graph development set | "
        f"offline trials: {N_HYPERPARAMETER_TRIALS}"
    )
    print(
        f"External cohort reserved: {len(EXTERNAL_GRAPH_IDS)} graphs "
        f"({int((labels[EXTERNAL_GRAPH_IDS] == 1).sum())} EDCs), "
        "not accessed until final independent evaluation."
    )

    # --- Internal-only 10-fold CV hyperparameter search ---
    for trial_id, hp in enumerate(candidates, start=1):
        if trial_id in completed_trial_ids:
            print(
                f"Trial {trial_id}/{N_HYPERPARAMETER_TRIALS}: already saved, skipping."
            )
            continue

        print(
            f"\nTrial {trial_id}/{N_HYPERPARAMETER_TRIALS} | "
            f"h=({hp.hidden_size1}, {hp.hidden_size2}, {hp.hidden_size3}), "
            f"lr={hp.lr:.6g}, wd={hp.weight_decay:.6g}, batch={hp.batch_size}"
        )

        summary = run_inner_cv_for_trial(
            dataset=dataset,
            labels=labels,
            inner_splits=inner_splits,
            hp=hp,
            trial_id=trial_id,
            output_dir=output_dir,
        )
        append_dataframe(trial_summary_path, pd.DataFrame([summary]))

        print(
            f"  Internal inner OOF | F1={summary['inner_oof_f1_score']:.4f}, "
            f"PR-AUC={summary['inner_oof_pr_auc']:.4f}, "
            f"ROC-AUC={summary['inner_oof_roc_auc']:.4f}, "
            f"Recall={summary['inner_oof_recall']:.4f}, "
            f"BalancedAcc={summary['inner_oof_balanced_accuracy']:.4f}"
        )

    # --- Choose the configuration solely from internal inner-CV results ---
    trial_summary_df = pd.read_csv(trial_summary_path)
    if len(trial_summary_df) != N_HYPERPARAMETER_TRIALS:
        raise RuntimeError(
            f"Expected {N_HYPERPARAMETER_TRIALS} completed trial summaries, "
            f"found {len(trial_summary_df)}."
        )
    if trial_summary_df["trial_id"].duplicated().any():
        duplicated = trial_summary_df.loc[
            trial_summary_df["trial_id"].duplicated(), "trial_id"
        ].tolist()
        raise RuntimeError(f"Duplicate trial IDs found: {duplicated}")

    best_trial = choose_best_trial(trial_summary_df)
    best_trial_id = int(best_trial["trial_id"])
    best_hp = HyperParameters(
        hidden_size1=int(best_trial["hidden_size1"]),
        hidden_size2=int(best_trial["hidden_size2"]),
        hidden_size3=int(best_trial["hidden_size3"]),
        lr=float(best_trial["lr"]),
        weight_decay=float(best_trial["weight_decay"]),
        batch_size=int(best_trial["batch_size"]),
        dropout=float(best_trial["dropout"]),
    )

    all_oof_path = output_dir / "inner_oof_predictions_all_trials.csv"
    all_oof_df = pd.read_csv(all_oof_path)
    best_oof_df = all_oof_df[all_oof_df["trial_id"] == best_trial_id].copy()
    best_oof_df.to_csv(output_dir / "best_internal_inner_cv_oof_predictions.csv", index=False)

    selected_payload = {
        "selected_trial_id": best_trial_id,
        "selected_hyperparameters": asdict(best_hp),
        "selection_basis": (
            "Highest pooled internal 10-fold out-of-fold positive-class F1-score; "
            "ties resolved by PR-AUC, balanced accuracy, then ROC-AUC."
        ),
        "inner_oof_metrics": {
            key.replace("inner_oof_", ""): (
                int(best_trial[key])
                if key in {
                    "inner_oof_tp", "inner_oof_tn",
                    "inner_oof_fp", "inner_oof_fn",
                }
                else float(best_trial[key])
            )
            for key in best_trial.index
            if key.startswith("inner_oof_")
        },
        "external_data_used_for_selection": False,
    }
    write_json(output_dir / "selected_model.json", selected_payload)

    print("\n" + "=" * 80)
    print("Selected configuration from INTERNAL inner 10-fold CV only:")
    print(f"Trial {best_trial_id}: {asdict(best_hp)}")
    print(
        f"Inner OOF F1={best_trial['inner_oof_f1_score']:.4f}, "
        f"PR-AUC={best_trial['inner_oof_pr_auc']:.4f}, "
        f"ROC-AUC={best_trial['inner_oof_roc_auc']:.4f}"
    )
    print("=" * 80)

    # ==============================================================
    # A. Internal 80/20 holdout assessment
    #    Train on 715 internal development samples, test on 176 unseen
    #    internal samples. External data remain untouched.
    # ==============================================================
    internal_holdout_model, internal_history = train_final_model(
        dataset=dataset,
        labels=labels,
        hp=best_hp,
        outer_train_indices=outer_train_indices,
        seed=TRAINING_SEED + 999999,
    )
    pd.DataFrame(internal_history).to_csv(
        output_dir / "internal_80pct_final_training_history.csv",
        index=False,
    )

    outer_test_loader = create_loader(
        dataset,
        outer_test_indices,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    graph_ids_test, y_test, p_test = predict(internal_holdout_model, outer_test_loader)

    if not np.array_equal(y_test, labels[graph_ids_test]):
        raise RuntimeError("Internal outer-test graph IDs and labels are inconsistent.")

    internal_holdout_metrics = calculate_metrics(
        y_test, p_test, threshold=CLASSIFICATION_THRESHOLD
    )
    y_pred_test = (p_test >= CLASSIFICATION_THRESHOLD).astype(int)
    pd.DataFrame(
        {
            "graph_id": graph_ids_test,
            "parent_edc_group": parent_groups[graph_ids_test],
            "y_true": y_test,
            "predicted_probability_edc": p_test,
            "threshold": CLASSIFICATION_THRESHOLD,
            "y_pred": y_pred_test,
            "evaluation_set": "internal_20pct_holdout",
        }
    ).to_csv(output_dir / "internal_20pct_test_predictions.csv", index=False)

    pd.DataFrame(
        [
            {
                "n_internal_test_graphs": int(len(y_test)),
                "n_internal_test_edcs": int((y_test == 1).sum()),
                "n_internal_test_non_edcs": int((y_test == 0).sum()),
                "selected_trial_id": best_trial_id,
                **asdict(best_hp),
                **internal_holdout_metrics,
            }
        ]
    ).to_csv(output_dir / "internal_20pct_test_metrics.csv", index=False)

    torch.save(
        {
            "model_state_dict": internal_holdout_model.state_dict(),
            "hyperparameters": asdict(best_hp),
            "training_graph_ids": outer_train_indices.tolist(),
            "evaluation_graph_ids": outer_test_indices.tolist(),
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "model_role": "internal_80pct_model_for_internal_20pct_holdout",
        },
        output_dir / "internal_80pct_model.pt",
    )

    del internal_holdout_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==============================================================
    # B. Independent external validation
    #    Refit the selected configuration using all 891 INTERNAL samples,
    #    then evaluate the external 12 only once.
    # ==============================================================
    external_validation_model, all_internal_history = train_final_model(
        dataset=dataset,
        labels=labels,
        hp=best_hp,
        outer_train_indices=INTERNAL_GRAPH_IDS,
        seed=TRAINING_SEED + 202606,
    )
    pd.DataFrame(all_internal_history).to_csv(
        output_dir / "all_internal_891_training_history_for_external_validation.csv",
        index=False,
    )

    external_loader = create_loader(
        dataset,
        EXTERNAL_GRAPH_IDS,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    graph_ids_external, y_external, p_external = predict(
        external_validation_model,
        external_loader,
    )

    if not np.array_equal(y_external, labels[graph_ids_external]):
        raise RuntimeError("External graph IDs and labels are inconsistent.")

    external_metrics = calculate_metrics(
        y_external,
        p_external,
        threshold=CLASSIFICATION_THRESHOLD,
    )
    y_pred_external = (p_external >= CLASSIFICATION_THRESHOLD).astype(int)

    external_prediction_df = pd.DataFrame(
        {
            "graph_id": graph_ids_external,
            "y_true": y_external,
            "predicted_probability_edc": p_external,
            "threshold": CLASSIFICATION_THRESHOLD,
            "y_pred": y_pred_external,
            "evaluation_set": "independent_external_12",
        }
    ).sort_values("graph_id").reset_index(drop=True)
    external_prediction_df.to_csv(
        output_dir / "independent_external_12_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "n_external_graphs": int(len(y_external)),
                "n_external_edcs": int((y_external == 1).sum()),
                "n_external_non_edcs": int((y_external == 0).sum()),
                "selected_trial_id": best_trial_id,
                **asdict(best_hp),
                **external_metrics,
            }
        ]
    ).to_csv(
        output_dir / "independent_external_12_metrics.csv",
        index=False,
    )

    torch.save(
        {
            "model_state_dict": external_validation_model.state_dict(),
            "hyperparameters": asdict(best_hp),
            "training_graph_ids": INTERNAL_GRAPH_IDS.tolist(),
            "evaluation_graph_ids": EXTERNAL_GRAPH_IDS.tolist(),
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "model_role": "all_internal_891_model_for_independent_external_12",
        },
        output_dir / "all_internal_891_model_for_external_validation.pt",
    )

    del external_validation_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Documentation ---
    readme = f"""Internal 891-compound training plus independent external validation completed.

Zero-based cohort definition
----------------------------
Internal modeling cohort (n = {N_INTERNAL_GRAPHS})
- EDCs: graph IDs 0–80 (n = {N_INTERNAL_EDCS})
- non-EDCs: graph IDs 87–896 (n = {N_INTERNAL_NON_EDCS})

Independent external cohort (n = {N_EXTERNAL_GRAPHS})
- EDCs: graph IDs 81–86 (n = {N_EXTERNAL_EDCS})
- non-EDCs: graph IDs 897–902 (n = {N_EXTERNAL_NON_EDCS})

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
{asdict(best_hp)}

Loss and decision rule
----------------------
- Class-weighted loss: {USE_CLASS_WEIGHTED_LOSS}
- Classification threshold: {CLASSIFICATION_THRESHOLD}
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
"""
    (output_dir / "README_internal_plus_external_validation.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Internal 20% holdout performance:")
    for metric in [
        "pr_auc", "roc_auc", "precision", "recall", "specificity",
        "balanced_accuracy", "mcc", "f1_score", "accuracy",
    ]:
        print(f"  {metric}: {internal_holdout_metrics[metric]:.4f}")
    print(
        f"  TP={internal_holdout_metrics['tp']}, "
        f"TN={internal_holdout_metrics['tn']}, "
        f"FP={internal_holdout_metrics['fp']}, "
        f"FN={internal_holdout_metrics['fn']}"
    )

    print("\nIndependent external 12-compound performance:")
    for metric in [
        "pr_auc", "roc_auc", "precision", "recall", "specificity",
        "balanced_accuracy", "mcc", "f1_score", "accuracy",
    ]:
        print(f"  {metric}: {external_metrics[metric]:.4f}")
    print(
        f"  TP={external_metrics['tp']}, TN={external_metrics['tn']}, "
        f"FP={external_metrics['fp']}, FN={external_metrics['fn']}"
    )

    print(f"\nAll offline records saved to:\n{output_dir.resolve()}")


if __name__ == "__main__":
    main()
