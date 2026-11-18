import json
import os
import yaml
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from src.models import GCNFraudDetector, GraphSAGEFraudDetector, GATv2FraudDetector
from src.train import train_gnn_model


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "PR_AUC": round(float(average_precision_score(y_true, y_probs)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, y_probs)), 4),
        "F1": round(float(f1_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Threshold": round(threshold, 4),
    }


def run_benchmark():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    data = torch.load(cfg["paths"]["processed_data"], weights_only=False)
    X = data.x.numpy()
    y = data.y.numpy()
    train_m = data.train_mask.numpy()
    val_m = data.val_mask.numpy()
    test_m = data.test_mask.numpy()

    results = {}

    # 1. Tabular Baseline: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=cfg["seed"])
    lr.fit(X[train_m], y[train_m])
    lr_thresh = find_best_threshold(y[val_m], lr.predict_proba(X[val_m])[:, 1])
    results["Logistic_Regression"] = compute_metrics(
        y[test_m], lr.predict_proba(X[test_m])[:, 1], lr_thresh
    )

    # 2. Tabular Baseline: XGBoost
    scale_pos = (y[train_m] == 0).sum() / max((y[train_m] == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=cfg["seed"],
    )
    xgb.fit(X[train_m], y[train_m])
    xgb_thresh = find_best_threshold(y[val_m], xgb.predict_proba(X[val_m])[:, 1])
    results["XGBoost"] = compute_metrics(
        y[test_m], xgb.predict_proba(X[test_m])[:, 1], xgb_thresh
    )

    # 3. GNN Models: GCN, GraphSAGE, GATv2
    in_dim = data.num_features
    edge_dim = data.edge_attr.size(1)
    gnn_candidates = {
        "GCN": GCNFraudDetector(in_dim, cfg["model"]["hidden_dim"], cfg["model"]["dropout"]),
        "GraphSAGE": GraphSAGEFraudDetector(in_dim, cfg["model"]["hidden_dim"], cfg["model"]["dropout"]),
        "GATv2": GATv2FraudDetector(
            in_dim,
            cfg["model"]["hidden_dim"],
            cfg["model"]["heads"],
            edge_dim,
            cfg["model"]["dropout"],
        ),
    }

    for name, model in gnn_candidates.items():
        save_path = cfg["paths"]["best_model"] if name == "GATv2" else None
        trained_model, _ = train_gnn_model(
            model,
            data,
            lr=cfg["model"]["lr"],
            weight_decay=cfg["model"]["weight_decay"],
            epochs=cfg["model"]["epochs"],
            patience=cfg["model"]["patience"],
            save_path=save_path,
        )
        trained_model.eval()
        with torch.no_grad():
            all_probs = torch.sigmoid(
                trained_model(data.x, data.edge_index, data.edge_attr)
            ).numpy()

        thresh = find_best_threshold(y[val_m], all_probs[val_m])
        results[name] = compute_metrics(y[test_m], all_probs[test_m], thresh)

    os.makedirs("artifacts", exist_ok=True)
    with open(cfg["paths"]["metrics_report"], "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== GraphShield Test Set Benchmark ===")
    header = f"{'Model':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'F1':<8} | {'Precision':<9} | {'Recall':<8}"
    print(header)
    print("-" * len(header))
    for model_name, m in results.items():
        print(
            f"{model_name:<22} | {m['PR_AUC']:<8.4f} | {m['ROC_AUC']:<8.4f} | "
            f"{m['F1']:<8.4f} | {m['Precision']:<9.4f} | {m['Recall']:<8.4f}"
        )


if __name__ == "__main__":
    run_benchmark()