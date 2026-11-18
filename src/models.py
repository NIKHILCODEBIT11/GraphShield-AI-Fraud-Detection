import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv


class GCNFraudDetector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)


class GraphSAGEFraudDetector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)


class GATv2FraudDetector(nn.Module):
    """
    2-layer GATv2 architecture incorporating edge attributes (tx count, volume)
    with residual skip-connection to prevent over-smoothing on fraud rings.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 4,
        edge_dim: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATv2Conv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.gat2 = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        return_attention: bool = False,
    ):
        x_skip = self.input_proj(x)

        h1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        h1 = self.bn1(h1 + x_skip)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        if return_attention:
            h2, attn_weights = self.gat2(
                h1, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
            attn_weights = None

        h2 = self.bn2(h2 + h1)
        h2 = F.elu(h2)

        # Concatenate initial projected features with 2-hop structural embedding
        out = torch.cat([x_skip, h2], dim=-1)
        logits = self.classifier(out).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits