import torch
import torch.nn as nn
import torch.nn.functional as F


class HexConv2d(nn.Conv2d):
    """3x3 conv on an axial grid restricted to the hex itself and its 6 neighbours."""

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)
        mask = torch.ones(1, 1, 3, 3)
        mask[..., 0, 0] = 0  # offset (-1, -1): not a neighbour in axial coordinates
        mask[..., 2, 2] = 0  # offset (+1, +1)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=1)


class MLP(nn.Module):
    def __init__(self, in_shape, width=128, depth=2, dropout=0.0, **_):
        super().__init__()
        layers, n = [nn.Flatten()], in_shape[0] * in_shape[1] * in_shape[2]
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.ReLU(), nn.Dropout(dropout)]
            n = width
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    """Stack of 3x3 convs (hex-masked if hex=True), then either
    head='pool': mean+max over all hexes (position invariant) or
    head='flatten': a linear layer over all positions (can use where on the map a pattern is)."""

    def __init__(self, in_shape, hex_mask, width=32, depth=3, dropout=0.0, hex=False, head='pool', **_):
        super().__init__()
        c = in_shape[0]
        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(HexConv2d(c, width) if hex else nn.Conv2d(c, width, 3, padding=1))
            c = width
        self.register_buffer('hex_mask', torch.as_tensor(hex_mask, dtype=torch.float32)[None, None])
        self.head = head
        n = 2 * width if head == 'pool' else width * in_shape[1] * in_shape[2]
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n, 1)

    def forward(self, x):
        for conv in self.convs:
            # zero cells that are not hexes, so padding never carries information
            x = F.relu(conv(x)) * self.hex_mask
        if self.head == 'pool':
            n_hex = self.hex_mask.sum()
            mean = x.sum(dim=(2, 3)) / n_hex
            mx = (x - 1e4 * (1 - self.hex_mask)).amax(dim=(2, 3))
            x = torch.cat([mean, mx], dim=1)
        else:
            x = x.flatten(1)
        return self.out(self.dropout(x)).squeeze(1)


class Square(nn.Module):
    """Quadratic activation x^2 (Gromov 2023, modular arithmetic): (a*x_v + b*sum x_u)^2 contains products
    of a hex's features with its neighbours' features, i.e. colour-pair detectors."""

    def forward(self, x):
        return x * x


ACTIVATIONS = {'relu': nn.ReLU, 'leaky': lambda: nn.LeakyReLU(0.2), 'square': Square, 'sigmoid': nn.Sigmoid}


class GraphHead(nn.Module):
    """(B, items, F) -> logit. head='flatten': dense layers over all items (position-specific),
    head='pool': mean+max over items (position invariant, works for any river layout).
    Then one fully connected ReLU layer per entry of `hidden` (e.g. [113, 8]) -> 1."""

    def __init__(self, n_items, width, hidden, head, dropout, act='relu'):
        super().__init__()
        self.head = head
        n = n_items * width if head == 'flatten' else 2 * width
        layers = [nn.Dropout(dropout)]
        for h in ([hidden] if isinstance(hidden, int) else hidden):
            layers += [nn.Linear(n, h), ACTIVATIONS[act]()]
            n = h
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        h = h.flatten(1) if self.head == 'flatten' else torch.cat([h.mean(1), h.amax(1)], dim=1)
        return self.net(h).squeeze(1)


class GCN(nn.Module):
    """Graph convolution on the hex graph. Per layer and hex v:
        h'_v = act(W_self h_v + W_nb * sum_{u neighbour of v} h_u + b)      (act: ReLU by default)
    Weights are shared by all hexes; the neighbour sum does not know the direction (isotropic),
    unlike HexConv2d, which has one weight per neighbour direction."""

    def __init__(self, n_nodes, edges, in_features=8, width=8, depth=1, hidden=8, head='flatten', dropout=0.0,
                 act_graph='relu', act_dense='relu', **_):
        super().__init__()
        self.act = ACTIVATIONS[act_graph]()
        adj = torch.zeros(n_nodes, n_nodes)
        e = torch.as_tensor(edges)
        adj[e[:, 0], e[:, 1]] = 1
        adj[e[:, 1], e[:, 0]] = 1
        self.register_buffer('adj', adj)
        self.self_lin = nn.ModuleList()
        self.nb_lin = nn.ModuleList()
        f = in_features
        for _ in range(depth):
            self.self_lin.append(nn.Linear(f, width))
            self.nb_lin.append(nn.Linear(f, width, bias=False))
            f = width
        self.head = GraphHead(n_nodes, width, hidden, head, dropout, act_dense)

    def forward(self, x, adj=None):  # x: (B, nodes, features); adj overrides the stored hex graph
        adj = self.adj if adj is None else adj
        for ls, ln in zip(self.self_lin, self.nb_lin):
            x = self.act(ls(x) + ln(adj @ x))
        return self.head(x)


class EdgeGCN(nn.Module):
    """One neuron group per edge between two hexes, fed only by these two hexes:
        e_uv = ReLU(W (x_u + x_v) + b)
    With one-hot input, x_u + x_v encodes the unordered colour pair of the two neighbours."""

    def __init__(self, n_nodes, edges, in_features=8, width=8, hidden=8, head='flatten', dropout=0.0, **_):
        super().__init__()
        self.register_buffer('edges', torch.as_tensor(edges))
        self.lin = nn.Linear(in_features, width)
        self.head = GraphHead(len(edges), width, hidden, head, dropout)

    def forward(self, x):
        e = F.relu(self.lin(x[:, self.edges[:, 0]] + x[:, self.edges[:, 1]]))
        return self.head(e)


def build_model(name, in_shape, hex_mask, **kw):
    if name == 'gcn':
        return GCN(**kw)
    if name == 'edgegcn':
        return EdgeGCN(**kw)
    if name == 'mlp':
        return MLP(in_shape, **kw)
    if name == 'cnn':
        return CNN(in_shape, hex_mask, hex=False, **kw)
    if name == 'hexcnn':
        return CNN(in_shape, hex_mask, hex=True, **kw)
    raise ValueError(name)
