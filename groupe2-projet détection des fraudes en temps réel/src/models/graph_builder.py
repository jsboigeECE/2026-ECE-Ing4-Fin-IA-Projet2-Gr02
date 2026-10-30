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

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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
        node_type: np.ndarray | None = None,
        tx_mask: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.node_ids = node_ids
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.adj_sparse = adj_sparse
        self.node_type = node_type
        self.tx_mask = tx_mask
        self.metadata = metadata or {}

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
        node_type=np.zeros(n, dtype=np.int64),
        tx_mask=np.ones(n, dtype=bool),
        metadata={"graph_kind": "homogeneous_transaction"},
    )


def _stable_bucket_ids(features: np.ndarray, n_buckets: int, seed: int = 0) -> np.ndarray:
    """Create deterministic pseudo-IDs by hashing feature vectors into buckets."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be > 0")
    z = np.round(features.astype(np.float64), 3)
    coeff = np.arange(1, z.shape[1] + 1, dtype=np.float64) + float(seed)
    h = np.abs(np.sum(z * coeff[None, :], axis=1) * 10_000.0).astype(np.int64)
    return h % int(n_buckets)


def build_hetero_transaction_graph(
    X: np.ndarray,
    y: np.ndarray,
    account_ids: np.ndarray | None = None,
    merchant_ids: np.ndarray | None = None,
    n_account_buckets: int = 3_000,
    n_merchant_buckets: int = 1_500,
    add_tx_similarity_edges: bool = True,
    time_window: float = 0.03,
    amount_window: float = 0.03,
    max_tx_edges_per_node: int = 8,
    subsample: int | None = 8_000,
    random_state: int = 42,
) -> GraphData:
    """Build a heterogeneous graph with transaction/account/merchant nodes.

    Node types:
      0 = transaction
      1 = account
      2 = merchant

    If account_ids / merchant_ids are not provided by the dataset, deterministic
    pseudo-IDs are generated from transaction features so the pipeline remains
    executable on the Kaggle credit card dataset.
    """
    rng = np.random.default_rng(random_state)

    if subsample is not None and subsample < len(y):
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        n_fraud = min(len(fraud_idx), subsample // 10)
        n_normal = subsample - n_fraud
        chosen_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        chosen_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
        idx = np.concatenate([chosen_fraud, chosen_normal])
        rng.shuffle(idx)
    else:
        idx = np.arange(len(y))

    X_tx = X[idx].astype(np.float32)
    y_tx = y[idx].astype(np.int64)
    n_tx = len(idx)

    if account_ids is None:
        # Pseudo account ID from latent transaction profile.
        account_raw = _stable_bucket_ids(X_tx[:, :8], n_account_buckets, seed=17)
        account_source = "pseudo_from_features"
    else:
        account_raw = account_ids[idx].astype(np.int64)
        account_source = "provided"

    if merchant_ids is None:
        # Pseudo merchant ID from amount/time behavioral signature.
        merchant_feats = np.stack([X_tx[:, 0], X_tx[:, -1], X_tx[:, 1], X_tx[:, 2]], axis=1)
        merchant_raw = _stable_bucket_ids(merchant_feats, n_merchant_buckets, seed=53)
        merchant_source = "pseudo_from_features"
    else:
        merchant_raw = merchant_ids[idx].astype(np.int64)
        merchant_source = "provided"

    account_unique, account_inv = np.unique(account_raw, return_inverse=True)
    merchant_unique, merchant_inv = np.unique(merchant_raw, return_inverse=True)

    n_acc = len(account_unique)
    n_mer = len(merchant_unique)
    acc_offset = n_tx
    mer_offset = n_tx + n_acc

    x_acc = np.zeros((n_acc, X_tx.shape[1]), dtype=np.float32)
    x_mer = np.zeros((n_mer, X_tx.shape[1]), dtype=np.float32)
    for i in range(n_acc):
        m = account_inv == i
        if np.any(m):
            x_acc[i] = X_tx[m].mean(axis=0)
    for i in range(n_mer):
        m = merchant_inv == i
        if np.any(m):
            x_mer[i] = X_tx[m].mean(axis=0)

    x_all = np.vstack([X_tx, x_acc, x_mer]).astype(np.float32)
    y_all = np.concatenate([
        y_tx,
        np.zeros(n_acc, dtype=np.int64),
        np.zeros(n_mer, dtype=np.int64),
    ])

    tx_node_ids = idx.astype(np.int64)
    non_tx_ids = np.full(n_acc + n_mer, -1, dtype=np.int64)
    node_ids = np.concatenate([tx_node_ids, non_tx_ids])

    node_type = np.concatenate([
        np.zeros(n_tx, dtype=np.int64),
        np.ones(n_acc, dtype=np.int64),
        np.full(n_mer, 2, dtype=np.int64),
    ])
    tx_mask = np.zeros(len(y_all), dtype=bool)
    tx_mask[:n_tx] = True

    tx_nodes = np.arange(n_tx, dtype=np.int64)
    acc_nodes = acc_offset + account_inv.astype(np.int64)
    mer_nodes = mer_offset + merchant_inv.astype(np.int64)

    src_parts: list[np.ndarray] = [tx_nodes, acc_nodes, tx_nodes, mer_nodes]
    dst_parts: list[np.ndarray] = [acc_nodes, tx_nodes, mer_nodes, tx_nodes]

    if add_tx_similarity_edges:
        time_col = X_tx[:, 0]
        amount_col = X_tx[:, -1]
        chunk = 500
        tx_src_list: list[np.ndarray] = []
        tx_dst_list: list[np.ndarray] = []
        for start in range(0, n_tx, chunk):
            end = min(start + chunk, n_tx)
            dt = np.abs(time_col[start:end, None] - time_col[None, :])
            da = np.abs(amount_col[start:end, None] - amount_col[None, :])
            connected = (dt <= time_window) & (da <= amount_window)
            for local_i, global_i in enumerate(range(start, end)):
                connected[local_i, global_i] = False
            rows_i, cols_i = np.where(connected)
            rows_i += start

            if max_tx_edges_per_node is not None:
                for local_i in range(end - start):
                    mask = rows_i == (local_i + start)
                    if mask.sum() > max_tx_edges_per_node:
                        keep = rng.choice(np.where(mask)[0], size=max_tx_edges_per_node, replace=False)
                        drop = np.where(mask)[0]
                        drop = np.setdiff1d(drop, keep)
                        rows_i = np.delete(rows_i, drop)
                        cols_i = np.delete(cols_i, drop)

            tx_src_list.append(rows_i.astype(np.int64))
            tx_dst_list.append(cols_i.astype(np.int64))

        if tx_src_list:
            tx_src = np.concatenate(tx_src_list)
            tx_dst = np.concatenate(tx_dst_list)
            src_parts.extend([tx_src, tx_dst])
            dst_parts.extend([tx_dst, tx_src])

    src = np.concatenate(src_parts).astype(np.int64)
    dst = np.concatenate(dst_parts).astype(np.int64)
    edge_index = np.stack([src, dst], axis=0)

    if edge_index.shape[1] > 0:
        n_total = x_all.shape[0]
        pairs = edge_index[0] * n_total + edge_index[1]
        _, unique_idx = np.unique(pairs, return_index=True)
        edge_index = edge_index[:, unique_idx]

    adj = _normalised_adjacency(edge_index, x_all.shape[0])
    num_edges = int(edge_index.shape[1])

    print(
        "  Hetero graph built: "
        f"tx={n_tx}, accounts={n_acc}, merchants={n_mer}, "
        f"nodes={x_all.shape[0]}, edges={num_edges}"
    )

    return GraphData(
        x=x_all,
        edge_index=edge_index,
        y=y_all,
        node_ids=node_ids,
        num_nodes=x_all.shape[0],
        num_edges=num_edges,
        adj_sparse=adj,
        node_type=node_type,
        tx_mask=tx_mask,
        metadata={
            "graph_kind": "heterogeneous_tx_account_merchant",
            "num_transactions": int(n_tx),
            "num_accounts": int(n_acc),
            "num_merchants": int(n_mer),
            "account_id_source": account_source,
            "merchant_id_source": merchant_source,
        },
    )


def graph_stats(g: GraphData) -> dict:
    """Return basic graph statistics."""
    n = g.num_nodes
    if g.num_edges > 0:
        row = g.edge_index[0]
        col = g.edge_index[1]
        data = np.ones(len(row), dtype=np.float32)
        adj = csr_matrix((data, (row, col)), shape=(n, n))
        # Work with an undirected support graph for topology metrics.
        adj = (adj + adj.T).minimum(1.0)
    else:
        adj = csr_matrix((n, n), dtype=np.float32)

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    n_components, _ = connected_components(adj, directed=False, return_labels=True)

    stats = {
        "num_nodes": g.num_nodes,
        "num_edges": g.num_edges,
        "num_fraud": int(g.y.sum()),
        "fraud_rate": float(g.y.mean()),
        "avg_degree": float(np.mean(degrees)) if degrees.size > 0 else 0.0,
        "max_degree": int(np.max(degrees)) if degrees.size > 0 else 0,
        "num_isolated": int(np.sum(degrees == 0)),
        "num_components": int(n_components),
    }
    if g.node_type is not None:
        stats["num_tx_nodes"] = int(np.sum(g.node_type == 0))
        stats["num_account_nodes"] = int(np.sum(g.node_type == 1))
        stats["num_merchant_nodes"] = int(np.sum(g.node_type == 2))
    if g.tx_mask is not None and np.any(g.tx_mask):
        y_tx = g.y[g.tx_mask]
        stats["num_tx_fraud"] = int(y_tx.sum())
        stats["tx_fraud_rate"] = float(y_tx.mean())
    return stats


def save_graph(g: GraphData, path: str) -> None:
    """Save GraphData to a .npz file."""
    save_payload: dict[str, Any] = {
        "x": g.x,
        "edge_index": g.edge_index,
        "y": g.y,
        "node_ids": g.node_ids,
        "num_nodes": np.array([g.num_nodes]),
        "num_edges": np.array([g.num_edges]),
    }
    if g.node_type is not None:
        save_payload["node_type"] = g.node_type
    if g.tx_mask is not None:
        save_payload["tx_mask"] = g.tx_mask.astype(np.int8)
    np.savez_compressed(
        path,
        **save_payload,
    )
    print(f"  Saved graph to {path}")


def load_graph(path: str) -> GraphData:
    """Load GraphData from a .npz file."""
    data = np.load(path)
    x = data["x"]
    edge_index = data["edge_index"]
    y = data["y"]
    node_ids = data["node_ids"] if "node_ids" in data.files else np.arange(len(y))
    node_type = data["node_type"] if "node_type" in data.files else None
    tx_mask = (data["tx_mask"].astype(np.int64) > 0) if "tx_mask" in data.files else None
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
        node_type=node_type,
        tx_mask=tx_mask,
    )
