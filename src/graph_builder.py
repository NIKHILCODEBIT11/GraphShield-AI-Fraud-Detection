import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def build_graph_data(
    accounts_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Data:
    """
    Constructs a PyTorch Geometric Data object with engineered behavioral node features,
    bidirectional transaction edges, and stratified node masks.
    """
    num_nodes = len(accounts_df)

    # Outgoing transaction aggregations
    out_agg = tx_df.groupby("src").agg(
        out_count=("amount", "count"),
        out_sum=("amount", "sum"),
        out_mean=("amount", "mean"),
        out_std=("amount", "std"),
        out_max=("amount", "max"),
        unique_dst=("dst", "nunique"),
    )

    # Incoming transaction aggregations
    in_agg = tx_df.groupby("dst").agg(
        in_count=("amount", "count"),
        in_sum=("amount", "sum"),
        in_mean=("amount", "mean"),
        in_std=("amount", "std"),
        in_max=("amount", "max"),
        unique_src=("src", "nunique"),
    )

    # Temporal burst velocity (min inter-transaction time per sender)
    tx_sorted = tx_df.sort_values(["src", "timestamp"])
    tx_sorted["dt"] = tx_sorted.groupby("src")["timestamp"].diff()
    vel_agg = tx_sorted.groupby("src").agg(
        min_inter_tx_sec=("dt", "min"),
        median_inter_tx_sec=("dt", "median"),
    )

    features = accounts_df.set_index("account_id")[
        ["account_age_days", "kyc_verified", "avg_monthly_balance"]
    ].copy()

    features = features.join(out_agg).join(in_agg).join(vel_agg)
    features["out_std"] = features["out_std"].fillna(0.0)
    features["in_std"] = features["in_std"].fillna(0.0)
    features["min_inter_tx_sec"] = features["min_inter_tx_sec"].fillna(86400.0)
    features["median_inter_tx_sec"] = features["median_inter_tx_sec"].fillna(86400.0)
    features = features.fillna(0.0)

    # Domain-specific ratio features
    features["in_out_amount_ratio"] = (features["in_sum"] + 1.0) / (features["out_sum"] + 1.0)
    features["pass_through_ratio"] = np.minimum(features["in_sum"], features["out_sum"]) / (
        np.maximum(features["in_sum"], features["out_sum"]) + 1.0
    )
    features["out_amount_to_balance"] = features["out_max"] / (features["avg_monthly_balance"] + 1.0)
    features["fan_out_ratio"] = features["unique_dst"] / (features["out_count"] + 1.0)
    features["fan_in_ratio"] = features["unique_src"] / (features["in_count"] + 1.0)

    # Log-transform skewed monetary & temporal columns
    skewed_cols = [
        "account_age_days", "avg_monthly_balance", "out_sum", "out_mean", "out_max",
        "in_sum", "in_mean", "in_max", "min_inter_tx_sec", "median_inter_tx_sec"
    ]
    for col in skewed_cols:
        features[col] = np.log1p(np.maximum(features[col], 0.0))

    feature_names = list(features.columns)
    X_raw = features.sort_index().values.astype(np.float32)
    y = accounts_df.sort_values("account_id")["is_fraud_node"].values.astype(np.int64)

    # Stratified Train / Val / Test split
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=y, random_state=seed
    )
    relative_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val, stratify=y[temp_idx], random_state=seed
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Fit scaler strictly on training nodes to prevent leakage
    scaler = StandardScaler()
    X_scaled = X_raw.copy()
    X_scaled[train_idx] = scaler.fit_transform(X_raw[train_idx])
    X_scaled[val_idx] = scaler.transform(X_raw[val_idx])
    X_scaled[test_idx] = scaler.transform(X_raw[test_idx])

    # Build bidirectional transaction edges + aggregated edge attributes
    edge_agg = tx_df.groupby(["src", "dst"]).agg(
        tx_count=("amount", "count"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
    ).reset_index()

    src = edge_agg["src"].values
    dst = edge_agg["dst"].values

    edge_feats = np.column_stack([
        np.log1p(edge_agg["tx_count"].values),
        np.log1p(edge_agg["total_amount"].values),
        np.log1p(edge_agg["mean_amount"].values),
    ]).astype(np.float32)

    # Bidirectional edges for message passing in both money flow directions
    edge_index = torch.tensor(
        np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(np.vstack([edge_feats, edge_feats]), dtype=torch.float32)

    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.feature_names = feature_names
    return data


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    acc_df = pd.read_csv("data/raw/accounts.csv")
    tx_df = pd.read_csv("data/raw/transactions.csv")
    graph_data = build_graph_data(acc_df, tx_df)
    torch.save(graph_data, "artifacts/graph_data.pt")
    print(f"Saved graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, {graph_data.num_features} features.")