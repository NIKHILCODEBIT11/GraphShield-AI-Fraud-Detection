import os
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from src.models import GCNFraudDetector, GraphSAGEFraudDetector, GATv2FraudDetector


def train_gnn_model(
    model: nn.Module,
    data,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    epochs: int = 120,
    patience: int = 20,
    save_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # Compute positive class weight for imbalanced fraud labels
    train_y = data.y[data.train_mask].float()
    num_pos = train_y.sum().item()
    num_neg = len(train_y) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_prauc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index, data.edge_attr)[data.val_mask]
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()

            val_prauc = average_precision_score(val_true, val_probs)
            val_rocauc = roc_auc_score(val_true, val_probs)

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)

    return model.cpu(), best_val_prauc


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data = torch.load(cfg["paths"]["processed_data"], weights_only=False)
    in_dim = data.num_features
    edge_dim = data.edge_attr.size(1)

    gat_model = GATv2FraudDetector(
        in_channels=in_dim,
        hidden_channels=cfg["model"]["hidden_dim"],
        heads=cfg["model"]["heads"],
        edge_dim=edge_dim,
        dropout=cfg["model"]["dropout"],
    )

    _, val_score = train_gnn_model(
        gat_model,
        data,
        lr=cfg["model"]["lr"],
        weight_decay=cfg["model"]["weight_decay"],
        epochs=cfg["model"]["epochs"],
        patience=cfg["model"]["patience"],
        save_path=cfg["paths"]["best_model"],
    )
    print(f"Training complete. Best Validation PR-AUC (GATv2): {val_score:.4f}")