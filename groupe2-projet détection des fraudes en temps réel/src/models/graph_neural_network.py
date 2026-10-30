"""Phase 5.2 - Graph Neural Network model (GCN) with PyTorch Geometric.

This module provides a 2-layer GCN for node-level fraud classification.
It keeps the same public API names used by the project:
- GCNModel
- GNNTrainingArtifacts
- compute_balanced_class_weight
- train_gcn
- predict_proba
- generate_embeddings
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


@dataclass
class GNNTrainingArtifacts:
    """Training outputs for analysis and reuse."""

    losses: list[float]
    class_weight: dict[int, float]


class GCNModel(nn.Module):
    """2-layer GCN with an embedding head and a fraud logit head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.3) -> None:
        super().__init__()
        emb_dim = hidden_dim // 2
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        self.dropout_rate = float(dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        logits = self.head(h2).squeeze(-1)
        return logits, h2

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"GCNModel(params={total:,}, layers=2)"


def _edge_index_from_dense_adjacency(adj: np.ndarray) -> np.ndarray:
    row, col = np.where(adj > 0)
    return np.stack([row, col], axis=0).astype(np.int64)


def _build_data(x: np.ndarray, edge_like: np.ndarray, y: np.ndarray) -> Data:
    if edge_like.ndim == 2 and edge_like.shape[0] == 2:
        edge_index = edge_like.astype(np.int64)
    elif edge_like.ndim == 2 and edge_like.shape[0] == edge_like.shape[1]:
        edge_index = _edge_index_from_dense_adjacency(edge_like)
    else:
        raise ValueError("edge input must be COO edge_index [2,E] or dense adjacency [N,N]")

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y.astype(np.int64), dtype=torch.long),
    )


def compute_balanced_class_weight(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights without external sklearn dependency."""
    y = y.astype(np.int64)
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    # balanced formula: n_samples / (n_classes * n_samples_class)
    w0 = n / (2.0 * max(neg, 1))
    w1 = n / (2.0 * max(pos, 1))
    return {0: float(w0), 1: float(w1)}


def train_gcn(
    model: GCNModel,
    x: np.ndarray,
    a_norm_dense: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    learning_rate: float = 5e-4,
    class_weight: dict[int, float] | None = None,
) -> GNNTrainingArtifacts:
    """Train GCN on full graph in transductive setting.

    The second matrix argument keeps its historic name for compatibility.
    It can be either dense adjacency [N,N] or edge_index [2,E].
    """
    data = _build_data(x, a_norm_dense, y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    if class_weight is None:
        class_weight = compute_balanced_class_weight(y)

    sample_w = torch.where(
        data.y == 1,
        torch.tensor(class_weight[1], dtype=torch.float32),
        torch.tensor(class_weight[0], dtype=torch.float32),
    )

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = loss_fn(logits, data.y.float())
        loss = (loss * sample_w).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return GNNTrainingArtifacts(losses=losses, class_weight=class_weight)


def predict_proba(model: GCNModel, x: np.ndarray, a_norm_dense: np.ndarray) -> np.ndarray:
    """Predict fraud probability per node.

    The second matrix argument can be dense adjacency [N,N] or edge_index [2,E].
    """
    data = _build_data(x, a_norm_dense, np.zeros(x.shape[0], dtype=np.int64))
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        return torch.sigmoid(logits).cpu().numpy()


def generate_embeddings(model: GCNModel, x: np.ndarray, a_norm_dense: np.ndarray) -> np.ndarray:
    """Generate embeddings for each transaction node.

    The second matrix argument can be dense adjacency [N,N] or edge_index [2,E].
    """
    data = _build_data(x, a_norm_dense, np.zeros(x.shape[0], dtype=np.int64))
    model.eval()
    with torch.no_grad():
        _, emb = model(data.x, data.edge_index)
        return emb.cpu().numpy()
