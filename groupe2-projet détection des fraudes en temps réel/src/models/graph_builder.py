"""Phase 5.1 - Transaction Graph Construction.

Build a graph from the credit card fraud dataset where:
  - Nodes  : transactions (one node per transaction)
  - Node features : V1-V28 features + normalised Amount + Time
  - Edges  : two transactions are connected if they share a similar
             time window AND similar amount (likely from the same card/merchant)
  - Labels : 0 = normal, 1 = fraud

The output is in PyTorch Geometric-compatible format (COO edge_index) but uses
only numpy/scipy/networkx so it works without a working torch_geometric install.
"""

from __future__ import annotations

import os
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Data container (mirrors PyG Data structure without the dependency)
# ---------------------------------------------------------------------------

class GraphData:
    """Simple graph data container.

    Attributes:
        x          : Node feature matrix  [N, F]
        edge_index : COO edge indices      [2, E]  (source, destination)
        y          : Node labels           [N]
        node_ids   : Original row indices in the input matrix [N]
        num_nodes  : Total number of nodes
        num_edges  : Total number of edges
        adj_sparse : Normalised adjacency matrix as scipy csr_matrix [N, N]
    """
    def __init__(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        y: np.ndarray,
        node_ids: np.ndarray,
        num_nodes: int,
        num_edges: int,
        adj_sparse: Any = None,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.node_ids = node_ids
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.adj_sparse = adj_sparse

    def __repr__(self) -> str:
        return (
            f"GraphData(num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"fraud={int(self.y.sum())})"
        )


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _normalised_adjacency(edge_index: np.ndarray, n: int) -> csr_matrix:
    """Return D^(-1/2) A_hat D^(-1/2) where A_hat = A + I.

    Used in the GCN spectral convolution formula:
        H' = sigma(A_hat_norm * H * W)
    """
    row, col = edge_index[0], edge_index[1]
    # Build A + I (add self-loops)
    rows_all = np.concatenate([row, col, np.arange(n)])
    cols_all = np.concatenate([col, row, np.arange(n)])
    data_all = np.ones(len(rows_all), dtype=np.float32)

    A = csr_matrix((data_all, (rows_all, cols_all)), shape=(n, n))
    # Remove duplicates by summing
    A = A.minimum(1.0)

    # Degree matrix D
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    # D^(-1/2) A D^(-1/2)  (diagonal scaling)
    D_inv_sqrt = csr_matrix((deg_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


def build_transaction_graph(
    X: np.ndarray,
    y: np.ndarray,
    time_window: float = 0.05,
    amount_window: float = 0.05,
    max_edges_per_node: int = 10,
    subsample: int | None = 10_000,
    random_state: int = 42,
) -> GraphData:
    """Build a transaction graph from feature matrix X and labels y.

    Edges are created between two transactions if:
      - Their normalised Time difference <= time_window
      - Their normalised Amount difference <= amount_window

    Args:
        X              : Feature matrix [N, F] (already normalised).
        y              : Labels [N] (0=normal, 1=fraud).
        time_window    : Max |ΔTime| to link two nodes.
        amount_window  : Max |ΔAmount| to link two nodes.
        max_edges_per_node : Cap on neighbours per node (KNN-style).
        subsample      : Use only this many samples (None = all). Keeps
                         the graph tractable on large datasets.
        random_state   : Random seed for subsampling.

    Returns:
        GraphData object with normalised adjacency ready for GCN.
    """
    rng = np.random.default_rng(random_state)

    if subsample is not None and subsample < len(y):
        # Stratified sub-sample to preserve fraud ratio
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]

        n_fraud = min(len(fraud_idx), subsample // 10)   # keep ~10% fraud
        n_normal = subsample - n_fraud

        chosen_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        chosen_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
        idx = np.concatenate([chosen_fraud, chosen_normal])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(y))

    X_sub = X[idx].astype(np.float32)
    y_sub = y[idx].astype(np.int64)
    n = len(idx)

    # Features: assume column 0 = Time (normalised), column -1 = Amount (normalised)
    # (as produced by the preprocessing pipeline)
    time_col = X_sub[:, 0]
    amount_col = X_sub[:, -1]

    print(f"  Building graph: {n} nodes  (fraud={y_sub.sum()})")

    # ---- Edge construction via batched comparison ----
    # Process in chunks to avoid O(N^2) memory blow-up
    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []

    chunk = 500  # process 500 nodes at a time
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dt = np.abs(time_col[start:end, None] - time_col[None, :])   # [chunk, N]
        da = np.abs(amount_col[start:end, None] - amount_col[None, :])  # [chunk, N]

        # Connected if both within window
        connected = (dt <= time_window) & (da <= amount_window)

        # Exclude self-loops
        for local_i, global_i in enumerate(range(start, end)):
            connected[local_i, global_i] = False

        rows_i, cols_i = np.where(connected)
        rows_i += start  # convert local to global

        # Cap max_edges_per_node
        if max_edges_per_node is not None:
            for local_i in range(end - start):
                mask = rows_i == (local_i + start)
                if mask.sum() > max_edges_per_node:
                    keep = rng.choice(np.where(mask)[0], size=max_edges_per_node, replace=False)
                    drop = np.where(mask)[0]
                    drop = np.setdiff1d(drop, keep)
                    rows_i = np.delete(rows_i, drop)
                    cols_i = np.delete(cols_i, drop)

        src_list.append(rows_i)
        dst_list.append(cols_i)

    if src_list:
        src = np.concatenate(src_list).astype(np.int64)
        dst = np.concatenate(dst_list).astype(np.int64)
    else:
        src = np.array([], dtype=np.int64)
        dst = np.array([], dtype=np.int64)

    edge_index = np.stack([src, dst], axis=0)  # [2, E]

    # Remove duplicate edges
    if edge_index.shape[1] > 0:
        pairs = edge_index[0] * n + edge_index[1]
        _, unique_idx = np.unique(pairs, return_index=True)
        edge_index = edge_index[:, unique_idx]

    adj = _normalised_adjacency(edge_index, n)

    num_edges = edge_index.shape[1]
    print(f"  Graph built: {n} nodes, {num_edges} edges  "
          f"(avg degree={num_edges/n:.1f})")

    return GraphData(
        x=X_sub,
        edge_index=edge_index,
        y=y_sub,
        node_ids=idx.astype(np.int64),
        num_nodes=n,
        num_edges=num_edges,
        adj_sparse=adj,
    )


def graph_stats(g: GraphData) -> dict:
    """Return basic graph statistics."""
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(g.num_nodes))
    if g.num_edges > 0:
        edges = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        nx_g.add_edges_from(edges)

    degrees = [d for _, d in nx_g.degree()]

    return {
        "num_nodes": g.num_nodes,
        "num_edges": g.num_edges,
        "num_fraud": int(g.y.sum()),
        "fraud_rate": float(g.y.mean()),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "num_isolated": int(sum(1 for d in degrees if d == 0)),
        "num_components": int(nx.number_connected_components(nx_g)),
    }


def save_graph(g: GraphData, path: str) -> None:
    """Save GraphData to a .npz file."""
    np.savez_compressed(
        path,
        x=g.x,
        edge_index=g.edge_index,
        y=g.y,
        node_ids=g.node_ids,
        num_nodes=np.array([g.num_nodes]),
        num_edges=np.array([g.num_edges]),
    )
    print(f"  Saved graph to {path}")


def load_graph(path: str) -> GraphData:
    """Load GraphData from a .npz file."""
    data = np.load(path)
    x = data["x"]
    edge_index = data["edge_index"]
    y = data["y"]
    node_ids = data["node_ids"] if "node_ids" in data.files else np.arange(len(y))
    n = int(data["num_nodes"][0])
    adj = _normalised_adjacency(edge_index, n)
    return GraphData(
        x=x,
        edge_index=edge_index,
        y=y,
        node_ids=node_ids,
        num_nodes=n,
        num_edges=int(data["num_edges"][0]),
        adj_sparse=adj,
    )
