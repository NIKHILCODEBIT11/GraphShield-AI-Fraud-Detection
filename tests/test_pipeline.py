import torch
from src.data_generator import generate_transaction_network
from src.graph_builder import build_graph_data
from src.models import GATv2FraudDetector


def test_end_to_end_graph_and_forward_pass():
    acc_df, tx_df = generate_transaction_network(
        num_accounts=300, num_transactions=1500, fraud_ring_count=5, seed=7
    )
    data = build_graph_data(acc_df, tx_df, seed=7)

    # Verify disjoint train/val/test splits (no data leakage)
    assert not torch.any(data.train_mask & data.val_mask)
    assert not torch.any(data.train_mask & data.test_mask)
    assert not torch.any(data.val_mask & data.test_mask)

    model = GATv2FraudDetector(
        in_channels=data.num_features,
        hidden_channels=32,
        heads=4,
        edge_dim=data.edge_attr.size(1),
    )
    logits, (edge_idx, attn) = model(
        data.x, data.edge_index, data.edge_attr, return_attention=True
    )

    assert logits.shape == (data.num_nodes,)
    assert not torch.isnan(logits).any()
    assert attn.shape[0] == edge_idx.shape[1]