#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leave-one-EDC-decoy-group-out (LOGO) validation for the qualitative EDKG-DL
classifier.

Purpose
-------
This script performs 81 leave-one-group-out validation rounds. In each round,
one confirmed EDC and its 10 paired DUD-E decoys are completely withheld from
classifier training. The model is trained on the remaining 80 EDCs and 800
decoys, then predicts the held-out 11-compound parent group.

The final out-of-fold (OOF) prediction table contains exactly one prediction
for every one of the 891 graph inputs. Pooled OOF metrics are then calculated,
and 95% confidence intervals are estimated by resampling the 81 parent groups
with replacement (group-level bootstrap).

Scientific design
-----------------
- Held-out test unit:
      1 confirmed EDC + 10 paired decoys
- Training unit in every round:
      80 confirmed EDCs + 800 paired decoys
- Hyperparameters:
      fixed before LOGO validation, preferably loaded from the primary
      80/20 + inner-10-fold analysis
- Training:
      fixed number of epochs (default 200), without early stopping and without
      looking at the held-out group
- Decision rule:
      fixed EDC probability threshold of 0.5, matching the original
      argmax(dim=1) classifier behavior
- No held-out EDC or its paired decoys appear in the corresponding
      classifier-training set.

This specifically tests:
"Each EDC was predicted when both that EDC and its paired decoys were absent
from classifier training."
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Parameter
from torch.utils.data import Subset

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


GRAPH_DATA_DIR = "edkgdl_all_data"
PROCESSED_ROOT = "."

OUTPUT_DIR = "leave_one_internal891_edc_decoy_group_out_zero_based_results"

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

PRIMARY_SELECTED_MODEL_JSON = (
    "EDKGDL_80_20_holdout_10fold_offline_results/selected_model.json"
)
PRIMARY_RUN_CONFIGURATION_JSON = (
    "EDKGDL_80_20_holdout_10fold_offline_results/run_configuration.json"
)

PRIMARY_NESTED_HP_CSV = (
    "EDKG_repeated_group_nested_cv_results_v3/"
    "selected_hyperparameters_per_outer_fold.csv"
)

AUTO_LOAD_PRIMARY_HYPERPARAMETERS = True

MANUAL_HYPERPARAMETERS = {
    "hidden_size1": 60,
    "hidden_size2": 40,
    "hidden_size3": 20,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "batch_size": 64,
    "dropout": 0.5,
}

MANUAL_USE_CLASS_WEIGHTED_LOSS = False

MAX_EPOCHS = 200
TRAINING_SEED = 20260625
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 0

CLASSIFICATION_THRESHOLD = 0.5

GROUP_BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260626

SAVE_FOLD_MODELS = False  # Saves 81 model state files if True.
RESUME_IF_OUTPUT_EXISTS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


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
                self.processed_paths[0],
                weights_only=False,
            )
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["Graph_label.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["edkg_logo_validation_graphs.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        labels_df = pd.read_csv(self.label_file, header=None)

        if labels_df.shape[0] != self.n_graphs:
            raise ValueError(
                f"Label file has {labels_df.shape[0]} rows, "
                f"but n_graphs={self.n_graphs}."
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
                    f"{edge_path} needs source, target, and >=1 edge feature."
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

            y = torch.tensor(
                [int(labels_df.iloc[graph_id, 1])],
                dtype=torch.long,
            )

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


def load_labels(label_path: Path) -> np.ndarray:
    df = pd.read_csv(label_path, header=None)

    if df.shape[1] < 2:
        raise ValueError(
            "Graph_label.txt requires at least two columns: graph_id,label."
        )

    graph_ids = df.iloc[:, 0].astype(int).to_numpy()
    labels = df.iloc[:, 1].astype(int).to_numpy()

    if not np.array_equal(graph_ids, np.arange(len(labels))):
        raise ValueError(
            "Graph_label.txt first column must be sequential IDs 0..N-1."
        )

    return labels


def resolve_label_file(graph_data_dir: Path) -> Path:
    if LABEL_FILE is not None:
        return Path(LABEL_FILE)

    for candidate_name in LABEL_FILE_CANDIDATES:
        candidate_path = graph_data_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

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

    parent_groups = np.full(n_graphs, -1, dtype=int)

    if GROUP_MAPPING_CSV is not None:
        mapping_path = Path(GROUP_MAPPING_CSV)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"GROUP_MAPPING_CSV not found: {mapping_path}"
            )

        mapping = pd.read_csv(mapping_path)
        needed = {"graph_id", "parent_edc_id"}
        if not needed.issubset(mapping.columns):
            raise ValueError(
                "GROUP_MAPPING_CSV must contain graph_id,parent_edc_id."
            )

        mapping = mapping[
            mapping["graph_id"].astype(int).isin(INTERNAL_GRAPH_IDS)
        ].copy()
        if mapping["graph_id"].duplicated().any():
            raise ValueError(
                "GROUP_MAPPING_CSV contains duplicated internal graph IDs."
            )

        mapping = mapping.sort_values("graph_id").reset_index(drop=True)
        if not np.array_equal(
            mapping["graph_id"].astype(int).to_numpy(),
            np.sort(INTERNAL_GRAPH_IDS),
        ):
            raise ValueError(
                "GROUP_MAPPING_CSV must cover exactly the internal validation "
                "IDs: EDCs 0–80 and non-EDCs 87–896."
            )

        parent_groups[mapping["graph_id"].astype(int).to_numpy()] = (
            mapping["parent_edc_id"].astype(int).to_numpy()
        )
    else:
        parent_groups[INTERNAL_EDC_IDS] = np.arange(N_EDCS)
        parent_groups[INTERNAL_NON_EDC_IDS] = np.repeat(
            np.arange(N_EDCS),
            DECOYS_PER_EDC,
        )

    if np.any(parent_groups[EXTERNAL_GRAPH_IDS] != -1):
        raise RuntimeError(
            "External IDs 81–86 and 897–902 must remain excluded (group = -1)."
        )

    for group_id in range(N_EDCS):
        group_ids = INTERNAL_GRAPH_IDS[
            parent_groups[INTERNAL_GRAPH_IDS] == group_id
        ]
        group_labels = labels[group_ids]
        n_positive = int((group_labels == 1).sum())
        n_negative = int((group_labels == 0).sum())

        if n_positive != 1 or n_negative != DECOYS_PER_EDC:
            raise ValueError(
                f"Internal parent group {group_id}: {n_positive} EDCs and "
                f"{n_negative} decoys, expected 1 and {DECOYS_PER_EDC}. "
                "Check decoy ordering or provide GROUP_MAPPING_CSV."
            )

    return parent_groups


class GCNConvEdge(MessagePassing):

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
        edge_index_loops, _ = add_self_loops(
            edge_index,
            num_nodes=x.size(0),
        )

        node_emb = self.lin_node(x)
        edge_emb = self.lin_edge(edge_attr)

        zero_loop_edges = torch.zeros(
            (node_emb.shape[0], edge_emb.shape[1]),
            device=edge_emb.device,
            dtype=edge_emb.dtype,
        )
        extended_edge_emb = torch.cat([edge_emb, zero_loop_edges], dim=0)

        row, col = edge_index_loops
        deg = degree(col, node_emb.size(0), dtype=node_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index_loops,
            x=node_emb,
            norm=norm,
            edge_emb=extended_edge_emb,
        )
        out = out + self.bias

        return out, edge_emb

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
        x, edge_emb = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x, edge_emb = self.conv2(x, edge_index, edge_emb)
        x = F.relu(x)

        x, _ = self.conv3(x, edge_index, edge_emb)

        graph_emb = global_mean_pool(x, batch)
        graph_emb = F.dropout(
            graph_emb,
            p=self.dropout,
            training=self.training,
        )
        return self.classifier(graph_emb)

@dataclass(frozen=True)
class HyperParameters:
    hidden_size1: int
    hidden_size2: int
    hidden_size3: int
    lr: float
    weight_decay: float
    batch_size: int
    dropout: float


def resolve_fixed_settings() -> Tuple[HyperParameters, bool, str]:

    selected_path = Path(PRIMARY_SELECTED_MODEL_JSON)
    run_config_path = Path(PRIMARY_RUN_CONFIGURATION_JSON)

    if AUTO_LOAD_PRIMARY_HYPERPARAMETERS and selected_path.exists():
        with open(selected_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        hp_data = payload.get("selected_hyperparameters", {})
        required = {
            "hidden_size1",
            "hidden_size2",
            "hidden_size3",
            "lr",
            "weight_decay",
            "batch_size",
            "dropout",
        }

        if required.issubset(hp_data):
            hp = HyperParameters(
                hidden_size1=int(hp_data["hidden_size1"]),
                hidden_size2=int(hp_data["hidden_size2"]),
                hidden_size3=int(hp_data["hidden_size3"]),
                lr=float(hp_data["lr"]),
                weight_decay=float(hp_data["weight_decay"]),
                batch_size=int(hp_data["batch_size"]),
                dropout=float(hp_data["dropout"]),
            )

            use_weighted_loss = MANUAL_USE_CLASS_WEIGHTED_LOSS
            if run_config_path.exists():
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_payload = json.load(f)
                use_weighted_loss = bool(
                    run_payload.get(
                        "use_class_weighted_loss",
                        MANUAL_USE_CLASS_WEIGHTED_LOSS,
                    )
                )

            return (
                hp,
                use_weighted_loss,
                "Loaded fixed hyperparameters from primary 80/20 inner-10-fold analysis.",
            )

    nested_path = Path(PRIMARY_NESTED_HP_CSV)
    if AUTO_LOAD_PRIMARY_HYPERPARAMETERS and nested_path.exists():
        df = pd.read_csv(nested_path)
        hp_cols = [
            "hidden_size1",
            "hidden_size2",
            "hidden_size3",
            "lr",
            "weight_decay",
            "dropout",
        ]

        if set(hp_cols).issubset(df.columns):
            summary = (
                df.groupby(hp_cols, dropna=False)
                .agg(
                    n_selected=("repeat", "size"),
                    mean_inner_pr_auc=(
                        "inner_best_pr_auc",
                        "mean",
                    )
                    if "inner_best_pr_auc" in df.columns
                    else ("repeat", "size"),
                )
                .reset_index()
                .sort_values(
                    ["n_selected", "mean_inner_pr_auc"],
                    ascending=[False, False],
                )
                .reset_index(drop=True)
            )
            chosen = summary.iloc[0]

            hp = HyperParameters(
                hidden_size1=int(chosen["hidden_size1"]),
                hidden_size2=int(chosen["hidden_size2"]),
                hidden_size3=int(chosen["hidden_size3"]),
                lr=float(chosen["lr"]),
                weight_decay=float(chosen["weight_decay"]),
                batch_size=int(MANUAL_HYPERPARAMETERS["batch_size"]),
                dropout=float(chosen["dropout"]),
            )
            return (
                hp,
                True,
                "Loaded most frequently selected hyperparameters from primary "
                "repeated nested-CV output; used class-weighted loss consistent "
                "with that primary script.",
            )

    hp = HyperParameters(**MANUAL_HYPERPARAMETERS)
    return (
        hp,
        MANUAL_USE_CLASS_WEIGHTED_LOSS,
        "Used manual fallback hyperparameters.",
    )


def create_loader(
    dataset: InMemoryDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
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
    return EDKGDLClassifier(
        node_channels=dataset.num_node_features,
        edge_channels=dataset.num_edge_features,
        hidden_size1=hp.hidden_size1,
        hidden_size2=hp.hidden_size2,
        hidden_size3=hp.hidden_size3,
        dropout=hp.dropout,
    ).to(DEVICE)


def create_criterion(
    labels: np.ndarray,
    train_indices: np.ndarray,
    use_weighted_loss: bool,
) -> nn.Module:
    if not use_weighted_loss:
        return nn.CrossEntropyLoss()

    y_train = labels[train_indices]
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required for weighted loss.")

    weights = torch.tensor(
        [1.0, n_neg / n_pos],
        dtype=torch.float,
        device=DEVICE,
    )
    return nn.CrossEntropyLoss(weight=weights)


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

        total_loss += float(loss.item()) * int(data.num_graphs)
        total_graphs += int(data.num_graphs)

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_ids: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []

    for data in loader:
        data = data.to(DEVICE)

        logits = model(data.x, data.edge_index, data.batch, data.edge_attr)
        p_edc = torch.softmax(logits, dim=1)[:, 1]

        all_ids.append(data.graph_id.view(-1).detach().cpu().numpy())
        all_y.append(data.y.view(-1).detach().cpu().numpy())
        all_p.append(p_edc.detach().cpu().numpy())

    return (
        np.concatenate(all_ids).astype(int),
        np.concatenate(all_y).astype(int),
        np.concatenate(all_p).astype(float),
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Both classes are needed for complete metric calculation.")

    y_pred = (probabilities >= float(threshold)).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2

    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
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


def run_one_logo_fold(
    dataset: InMemoryDataset,
    labels: np.ndarray,
    parent_groups: np.ndarray,
    heldout_group: int,
    hp: HyperParameters,
    use_weighted_loss: bool,
) -> Tuple[List[Dict], Dict, List[Dict], Optional[Dict]]:

    all_indices = np.sort(INTERNAL_GRAPH_IDS.astype(int))
    test_indices = all_indices[parent_groups[all_indices] == heldout_group]
    train_indices = all_indices[parent_groups[all_indices] != heldout_group]

    if len(test_indices) != 1 + DECOYS_PER_EDC:
        raise RuntimeError(
            f"Held-out internal group {heldout_group} has {len(test_indices)} graphs; "
            f"expected {1 + DECOYS_PER_EDC}."
        )
    if len(train_indices) != EXPECTED_INTERNAL_GRAPHS - (1 + DECOYS_PER_EDC):
        raise RuntimeError(
            "Unexpected internal LOGO training-set size. Expected 880 graphs."
        )

    if np.intersect1d(test_indices, EXTERNAL_GRAPH_IDS).size > 0:
        raise RuntimeError("External graphs leaked into the held-out LOGO group.")
    if np.intersect1d(train_indices, EXTERNAL_GRAPH_IDS).size > 0:
        raise RuntimeError("External graphs leaked into the LOGO training set.")

    if set(parent_groups[train_indices]).intersection(set(parent_groups[test_indices])):
        raise RuntimeError("Parent group leakage detected.")

    seed_everything(TRAINING_SEED)

    model = build_model(dataset, hp)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )
    criterion = create_criterion(
        labels,
        train_indices,
        use_weighted_loss,
    )

    train_loader = create_loader(
        dataset,
        train_indices,
        batch_size=hp.batch_size,
        shuffle=True,
    )
    test_loader = create_loader(
        dataset,
        test_indices,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    epoch_rows: List[Dict] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )
        epoch_rows.append(
            {
                "heldout_parent_group": int(heldout_group),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
            }
        )

    graph_ids, y_test, p_test = predict_probabilities(model, test_loader)

    if not np.array_equal(y_test, labels[graph_ids]):
        raise RuntimeError(
            f"Held-out group {heldout_group}: graph-label mismatch."
        )

    fold_metrics = calculate_metrics(
        y_test,
        p_test,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    heldout_edc_mask = y_test == 1
    edc_probability = float(p_test[heldout_edc_mask][0])
    edc_prediction = int(
        (p_test[heldout_edc_mask][0] >= CLASSIFICATION_THRESHOLD)
    )

    edc_rank_within_group = int(
        1 + np.sum(p_test > edc_probability)
    )

    fold_row = {
        "heldout_parent_group": int(heldout_group),
        "heldout_edc_graph_id": int(graph_ids[heldout_edc_mask][0]),
        "n_train_graphs": int(len(train_indices)),
        "n_train_edcs": int((labels[train_indices] == 1).sum()),
        "n_train_non_edcs": int((labels[train_indices] == 0).sum()),
        "n_test_graphs": int(len(test_indices)),
        "n_test_edcs": int((y_test == 1).sum()),
        "n_test_non_edcs": int((y_test == 0).sum()),
        "edc_probability": edc_probability,
        "edc_prediction": edc_prediction,
        "edc_rank_within_group": edc_rank_within_group,
        "n_decoys_predicted_edc": int(
            ((y_test == 0) & (p_test >= CLASSIFICATION_THRESHOLD)).sum()
        ),
        **asdict(hp),
        **fold_metrics,
    }

    y_pred = (p_test >= CLASSIFICATION_THRESHOLD).astype(int)
    prediction_rows: List[Dict] = []
    for graph_id, y_true, probability, pred in zip(
        graph_ids,
        y_test,
        p_test,
        y_pred,
    ):
        prediction_rows.append(
            {
                "heldout_parent_group": int(heldout_group),
                "graph_id": int(graph_id),
                "y_true": int(y_true),
                "predicted_probability_edc": float(probability),
                "threshold": float(CLASSIFICATION_THRESHOLD),
                "y_pred": int(pred),
            }
        )

    model_state = None
    if SAVE_FOLD_MODELS:
        model_state = {
            "model_state_dict": model.state_dict(),
            "heldout_parent_group": int(heldout_group),
            "hyperparameters": asdict(hp),
            "threshold": float(CLASSIFICATION_THRESHOLD),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return epoch_rows, fold_row, prediction_rows, model_state

METRIC_NAMES = [
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "f1_score",
    "accuracy",
]


def group_bootstrap_metrics(
    predictions_df: pd.DataFrame,
    n_iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    required = {
        "heldout_parent_group",
        "graph_id",
        "y_true",
        "predicted_probability_edc",
        "threshold",
        "y_pred",
    }
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError(
            f"Predictions file missing required columns: {sorted(missing)}"
        )

    unique_groups = np.sort(predictions_df["heldout_parent_group"].unique())
    if len(unique_groups) != N_EDCS:
        raise ValueError(
            f"Expected {N_EDCS} held-out groups, found {len(unique_groups)}."
        )

    group_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for group_id in unique_groups:
        group_df = predictions_df[
            predictions_df["heldout_parent_group"] == group_id
        ].sort_values("graph_id")

        if len(group_df) != 1 + DECOYS_PER_EDC:
            raise ValueError(
                f"Group {group_id} has {len(group_df)} predictions, expected "
                f"{1 + DECOYS_PER_EDC}."
            )
        if int((group_df["y_true"] == 1).sum()) != 1:
            raise ValueError(f"Group {group_id} does not contain exactly 1 EDC.")

        group_cache[int(group_id)] = (
            group_df["y_true"].to_numpy(dtype=int),
            group_df["predicted_probability_edc"].to_numpy(dtype=float),
        )

    rng = np.random.default_rng(seed)
    bootstrap_rows: List[Dict] = []

    for iteration in range(1, n_iterations + 1):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        y_parts = [group_cache[int(g)][0] for g in sampled_groups]
        p_parts = [group_cache[int(g)][1] for g in sampled_groups]

        y_boot = np.concatenate(y_parts)
        p_boot = np.concatenate(p_parts)

        metrics = calculate_metrics(
            y_boot,
            p_boot,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        bootstrap_rows.append(
            {
                "bootstrap_iteration": int(iteration),
                **metrics,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    summary_rows = []
    for metric in METRIC_NAMES:
        values = bootstrap_df[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "point_estimate": np.nan,  # filled below
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_ci_2.5%": float(np.quantile(values, 0.025)),
                "bootstrap_ci_97.5%": float(np.quantile(values, 0.975)),
                "n_group_bootstrap_iterations": int(n_iterations),
            }
        )

    return bootstrap_df, pd.DataFrame(summary_rows)


def create_pooled_summary(
    predictions_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    predictions_df = predictions_df.sort_values("graph_id").reset_index(drop=True)

    if len(predictions_df) != EXPECTED_INTERNAL_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_INTERNAL_GRAPHS} internal OOF predictions, found "
            f"{len(predictions_df)}."
        )

    if predictions_df["graph_id"].duplicated().any():
        duplicates = predictions_df.loc[
            predictions_df["graph_id"].duplicated(),
            "graph_id",
        ].tolist()
        raise ValueError(f"Duplicate OOF graph IDs detected: {duplicates[:10]}")

    observed_internal_ids = np.sort(
        predictions_df["graph_id"].to_numpy(dtype=int)
    )
    if not np.array_equal(observed_internal_ids, np.sort(INTERNAL_GRAPH_IDS)):
        raise ValueError(
            "OOF prediction table must contain exactly the internal validation "
            "graph IDs: 0–80 and 87–896. External IDs 81–86 and 897–902 "
            "must be absent."
        )

    y_true = predictions_df["y_true"].to_numpy(dtype=int)
    probabilities = predictions_df["predicted_probability_edc"].to_numpy(
        dtype=float
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    edc_summary_rows = []
    for group_id, group_df in predictions_df.groupby(
        "heldout_parent_group",
        sort=True,
    ):
        group_df = group_df.sort_values("graph_id")
        edc_row = group_df[group_df["y_true"] == 1].iloc[0]

        edc_probability = float(edc_row["predicted_probability_edc"])
        rank = int(
            1 + np.sum(
                group_df["predicted_probability_edc"].to_numpy()
                > edc_probability
            )
        )

        edc_summary_rows.append(
            {
                "heldout_parent_group": int(group_id),
                "edc_graph_id": int(edc_row["graph_id"]),
                "edc_probability": edc_probability,
                "edc_prediction": int(edc_row["y_pred"]),
                "edc_rank_within_11_compound_group": rank,
                "n_decoys_predicted_edc": int(
                    (
                        (group_df["y_true"] == 0)
                        & (group_df["y_pred"] == 1)
                    ).sum()
                ),
            }
        )

    return metrics, pd.DataFrame(edc_summary_rows)

def check_or_write_run_configuration(
    output_dir: Path,
    payload: Dict,
) -> None:
    path = output_dir / "run_configuration.json"

    if path.exists() and RESUME_IF_OUTPUT_EXISTS:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        required_match_keys = [
            "fixed_hyperparameters",
            "use_class_weighted_loss",
            "max_epochs",
            "classification_threshold",
            "grouping_rule",
        ]
        for key in required_match_keys:
            if existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Existing output configuration differs for '{key}'. "
                    "Use a new OUTPUT_DIR or remove the previous result folder."
                )
    else:
        write_json(path, payload)


def load_completed_groups(output_dir: Path) -> set:
    metrics_path = output_dir / "leave_one_group_metrics.csv"
    predictions_path = output_dir / "leave_one_group_predictions.csv"

    if (
        not RESUME_IF_OUTPUT_EXISTS
        or not metrics_path.exists()
        or not predictions_path.exists()
    ):
        return set()

    metrics_df = pd.read_csv(metrics_path)
    predictions_df = pd.read_csv(predictions_path)

    completed = set()
    for group_id in metrics_df["heldout_parent_group"].astype(int).unique():
        metric_count = int(
            (metrics_df["heldout_parent_group"].astype(int) == group_id).sum()
        )
        prediction_count = int(
            (
                predictions_df["heldout_parent_group"].astype(int) == group_id
            ).sum()
        )
        if metric_count == 1 and prediction_count == 1 + DECOYS_PER_EDC:
            completed.add(group_id)

    return completed

def main() -> None:
    seed_everything(TRAINING_SEED)

    print(f"Using device: {DEVICE}")

    graph_dir = Path(GRAPH_DATA_DIR)

    if not graph_dir.exists():
        raise FileNotFoundError(
            f"GRAPH_DATA_DIR not found: {graph_dir.resolve()}"
        )

    label_path = resolve_label_file(graph_dir)
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path.resolve()}"
        )

    labels = load_labels(label_path)

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
            "Graph labels do not match the required zero-based cohort layout. "
            "Expected EDCs 0–86 and non-EDCs 87–902, with internal LOGO "
            "restricted to EDCs 0–80 and non-EDCs 87–896. "
            f"Mismatch IDs: {mismatch_ids[:20].tolist()}"
        )

    internal_labels = labels[INTERNAL_GRAPH_IDS]
    external_labels = labels[EXTERNAL_GRAPH_IDS]
    if (
        int((internal_labels == 1).sum()) != N_EDCS
        or int((internal_labels == 0).sum()) != N_EDCS * DECOYS_PER_EDC
    ):
        raise ValueError(
            "Internal LOGO cohort must contain EDCs 0–80 and non-EDCs 87–896."
        )
    if int((external_labels == 1).sum()) != 6 or int((external_labels == 0).sum()) != 6:
        raise ValueError(
            "External excluded cohort must contain EDCs 81–86 and "
            "non-EDCs 897–902."
        )

    parent_groups = build_parent_groups(labels)

    processed_path = (
        Path(PROCESSED_ROOT)
        / "processed"
        / "edkg_logo_validation_graphs.pt"
    )
    if REPROCESS_DATASET and processed_path.exists():
        processed_path.unlink()

    dataset = EDKGGraphDataset(
        root=PROCESSED_ROOT,
        graph_data_dir=str(graph_dir),
        label_file=str(label_path),
        n_graphs=len(labels),
    )

    if len(dataset) != len(labels):
        raise RuntimeError("Dataset length differs from label length.")

    hp, use_weighted_loss, hp_source = resolve_fixed_settings()

    print(
        f"Loaded {len(dataset)} total graphs | "
        f"internal LOGO cohort: {len(INTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 0–80; non-EDCs 87–896) | "
        f"external excluded cohort: {len(EXTERNAL_GRAPH_IDS)} graphs "
        f"(EDCs 81–86; non-EDCs 897–902) | "
        f"node features={dataset.num_node_features} | "
        f"edge features={dataset.num_edge_features}"
    )
    print("\nFixed hyperparameters:")
    print(asdict(hp))
    print(f"Weighted loss: {use_weighted_loss}")
    print(f"Source: {hp_source}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "graph_data_dir": str(graph_dir.resolve()),
        "label_file": str(label_path.resolve()),
        "device": str(DEVICE),
        "total_graphs_loaded": int(len(dataset)),
        "internal_logo_cohort": {
            "edc_graph_ids": "0-80",
            "non_edc_graph_ids": "87-896",
            "n_graphs": EXPECTED_INTERNAL_GRAPHS,
            "n_edcs": N_EDCS,
            "n_non_edcs": N_EDCS * DECOYS_PER_EDC,
        },
        "external_cohort_excluded_from_logo": {
            "edc_graph_ids": "81-86",
            "non_edc_graph_ids": "897-902",
            "n_graphs": len(EXTERNAL_GRAPH_IDS),
            "n_edcs": 6,
            "n_non_edcs": 6,
        },
        "n_parent_groups": int(N_EDCS),
        "heldout_group_size": int(1 + DECOYS_PER_EDC),
        "training_group_size": int(EXPECTED_INTERNAL_GRAPHS - (1 + DECOYS_PER_EDC)),
        "max_epochs": int(MAX_EPOCHS),
        "classification_threshold": float(CLASSIFICATION_THRESHOLD),
        "fixed_hyperparameters": asdict(hp),
        "hyperparameter_source": hp_source,
        "use_class_weighted_loss": bool(use_weighted_loss),
        "grouping_rule": (
            "Each held-out group contained one EDC and its 10 paired decoys; "
            "the complete group was excluded from classifier training."
        ),
        "scope_note": (
            "This is classifier-level leave-one-parent-group-out validation. "
            "It does not retrain upstream entity-specific QSAR models for each "
            "held-out group."
        ),
    }
    check_or_write_run_configuration(output_dir, config)

    all_graph_ids = np.arange(EXPECTED_TOTAL_GRAPHS, dtype=int)
    group_map_df = pd.DataFrame(
        {
            "graph_id": all_graph_ids,
            "label": labels,
            "cohort_role": np.where(
                np.isin(all_graph_ids, INTERNAL_GRAPH_IDS),
                "internal_logo_validation",
                "external_excluded",
            ),
            "parent_edc_group": parent_groups,
            "is_edc": labels == 1,
        }
    )
    group_map_df.to_csv(
        output_dir / "cohort_membership_and_group_mapping.csv",
        index=False,
    )

    completed_groups = load_completed_groups(output_dir)

    for heldout_group in range(N_EDCS):
        if heldout_group in completed_groups:
            print(
                f"Held-out group {heldout_group + 1}/{N_EDCS}: "
                "already complete, skipping."
            )
            continue

        print(
            f"\n{'=' * 78}\n"
            f"LOGO fold {heldout_group + 1}/{N_EDCS} | "
            f"held-out parent EDC group = {heldout_group}\n"
            f"{'=' * 78}"
        )

        (
            epoch_rows,
            fold_row,
            prediction_rows,
            model_state,
        ) = run_one_logo_fold(
            dataset=dataset,
            labels=labels,
            parent_groups=parent_groups,
            heldout_group=heldout_group,
            hp=hp,
            use_weighted_loss=use_weighted_loss,
        )

        append_csv(
            output_dir / "training_history.csv",
            pd.DataFrame(epoch_rows),
        )
        append_csv(
            output_dir / "leave_one_group_metrics.csv",
            pd.DataFrame([fold_row]),
        )
        append_csv(
            output_dir / "leave_one_group_predictions.csv",
            pd.DataFrame(prediction_rows),
        )

        if SAVE_FOLD_MODELS and model_state is not None:
            model_dir = output_dir / "fold_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model_state,
                model_dir / f"heldout_group_{heldout_group:02d}.pt",
            )

        print(
            f"  Held-out EDC graph ID: {fold_row['heldout_edc_graph_id']} | "
            f"EDC probability={fold_row['edc_probability']:.4f} | "
            f"prediction={fold_row['edc_prediction']} | "
            f"rank in group={fold_row['edc_rank_within_group']}/11"
        )
        print(
            f"  Group diagnostic only | "
            f"PR-AUC={fold_row['pr_auc']:.4f}, "
            f"ROC-AUC={fold_row['roc_auc']:.4f}, "
            f"F1={fold_row['f1_score']:.4f}, "
            f"BalancedAcc={fold_row['balanced_accuracy']:.4f}"
        )

    prediction_path = output_dir / "leave_one_group_predictions.csv"
    fold_metrics_path = output_dir / "leave_one_group_metrics.csv"

    if not prediction_path.exists() or not fold_metrics_path.exists():
        raise RuntimeError("No LOGO outputs found after validation.")

    predictions_df = pd.read_csv(prediction_path)
    fold_metrics_df = pd.read_csv(fold_metrics_path)

    if (
        predictions_df["heldout_parent_group"].nunique() != N_EDCS
        or len(predictions_df) != EXPECTED_INTERNAL_GRAPHS
    ):
        raise RuntimeError(
            "LOGO validation is incomplete. Expected one 11-sample OOF "
            "prediction group for each of 81 parent groups."
        )
    if (
        fold_metrics_df["heldout_parent_group"].nunique() != N_EDCS
        or len(fold_metrics_df) != N_EDCS
    ):
        raise RuntimeError("Expected exactly one metric row per held-out group.")

    pooled_metrics, edc_summary_df = create_pooled_summary(predictions_df)

    bootstrap_df, bootstrap_summary_df = group_bootstrap_metrics(
        predictions_df=predictions_df,
        n_iterations=GROUP_BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )

    # Insert pooled OOF estimates next to bootstrap CIs.
    bootstrap_summary_df["point_estimate"] = bootstrap_summary_df["metric"].map(
        pooled_metrics
    )

    pd.DataFrame([pooled_metrics]).to_csv(
        output_dir / "pooled_oof_metrics.csv",
        index=False,
    )
    edc_summary_df.to_csv(
        output_dir / "per_edc_heldout_summary.csv",
        index=False,
    )
    bootstrap_df.to_csv(
        output_dir / "group_bootstrap_metric_samples.csv",
        index=False,
    )
    bootstrap_summary_df.to_csv(
        output_dir / "group_bootstrap_metric_ci.csv",
        index=False,
    )

    # A concise reusable text file for result interpretation.
    correct_edcs = int(edc_summary_df["edc_prediction"].sum())
    rank1_edcs = int(
        (edc_summary_df["edc_rank_within_11_compound_group"] == 1).sum()
    )

    readme = f"""Leave-one-internal-EDC-decoy-group-out validation completed.

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
  {asdict(hp)}
- Hyperparameter source:
  {hp_source}
- Weighted loss:
  {use_weighted_loss}
- Training epochs per LOGO fold:
  {MAX_EPOCHS}
- Fixed classification threshold:
  {CLASSIFICATION_THRESHOLD}

Pooled OOF result
-----------------
- Every graph received exactly one prediction from a model that did not include
  its parent EDC-decoy group in classifier training.
- Correctly classified held-out EDCs at threshold {CLASSIFICATION_THRESHOLD}:
  {correct_edcs}/{N_EDCS}
- Held-out EDCs ranked first within their own 11-compound EDC-decoy group:
  {rank1_edcs}/{N_EDCS}

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
"""
    (output_dir / "README_LOGO_results.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("LOGO validation complete.")
    print("=" * 78)
    print(
        f"Correctly classified held-out EDCs: "
        f"{correct_edcs}/{N_EDCS}"
    )
    print(
        f"Held-out EDCs ranked first within own group: "
        f"{rank1_edcs}/{N_EDCS}"
    )
    print("\nPooled out-of-fold metrics:")
    for metric in METRIC_NAMES:
        print(f"  {metric}: {pooled_metrics[metric]:.4f}")

    print("\nGroup-level bootstrap 95% CIs:")
    print(
        bootstrap_summary_df[
            [
                "metric",
                "point_estimate",
                "bootstrap_ci_2.5%",
                "bootstrap_ci_97.5%",
            ]
        ].to_string(index=False)
    )
    print(f"\nAll outputs saved to:\n{output_dir.resolve()}")


if __name__ == "__main__":
    main()
