"""The graph convolutional network used by train.py (see README, Architecture)."""
import torch
import torch.nn as nn


class GCN(nn.Module):
    """Graph layers on the hex graph -> mean and max pooling over all hexes -> fully connected layers -> one logit.

    Graph layer i, for every hex v and channel k:
        h_v[k] = ReLU(a_k · x_v + b_k · sum_{u neighbour of v} x_u + c_k)
    with a_k = graph_self[i].weight[k], c_k = graph_self[i].bias[k], b_k = graph_nb[i].weight[k].
    The hex graph (adjacency matrix) is passed to forward(), so the same model works for any map shape.
    """

    def __init__(self, in_features=8, width=16, depth=1, hidden=(8,)):
        super().__init__()
        self.graph_self = nn.ModuleList()
        self.graph_nb = nn.ModuleList()
        n = in_features
        for _ in range(depth):
            self.graph_self.append(nn.Linear(n, width))
            self.graph_nb.append(nn.Linear(n, width, bias=False))
            n = width
        self.dense = nn.ModuleList()
        n = 2 * width
        for h in hidden:
            self.dense.append(nn.Linear(n, h))
            n = h
        self.out = nn.Linear(n, 1)

    def forward(self, x, adj):
        """x: (batch, hexes, 8) one-hot features, adj: (hexes, hexes) adjacency matrix -> (batch,) logits."""
        for lin_self, lin_nb in zip(self.graph_self, self.graph_nb):
            x = torch.relu(lin_self(x) + lin_nb(adj @ x))
        x = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        for layer in self.dense:
            x = torch.relu(layer(x))
        return self.out(x).squeeze(1)
