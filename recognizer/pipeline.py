"""Training pipeline behind train.py: maps from two generator files -> GCN -> hard example mining -> model.py.

A generator file provides  generate_map(seed) -> str  (map text as read by map_is_smart.py).
Maps of different shapes are allowed; batches are formed per shape (hex graph).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from recognizer.genload import generate
from recognizer.model_template import features, hex_graph, parse_map
from recognizer.models import GCN

# disjoint seed ranges for the different map sets
SEEDS = {'train': 0, 'val': 10_000_000, 'test': 20_000_000, 'final': 900_000_000}
MINING_SEED_BASE, MINING_STRIDE = 100_000_000, 1_000_000

class MapSet:
    """Maps with labels (1 = smart, 0 = dumb), one-hot node features and a hex graph per map shape."""
    graphs = {}  # shape signature -> adjacency (torch)

    def __init__(self, texts=(), labels=()):
        self.texts, self.y, self.sig, self.x = list(texts), np.array(labels, dtype=np.float32), [], []
        for t in self.texts:
            rows, indented = parse_map(t)
            sig = (tuple(len(r) for r in rows), tuple(indented))
            if sig not in MapSet.graphs:
                MapSet.graphs[sig] = torch.from_numpy(hex_graph(rows, indented)[1])
            cells = [(r, c) for r in range(len(rows)) for c in range(len(rows[r]))]
            self.sig.append(sig)
            self.x.append(features(rows, cells))

    @classmethod
    def from_generators(cls, smart, dumb, seed_base, n, workers):
        seeds = range(seed_base, seed_base + n)
        texts = generate(smart, seeds, workers) + generate(dumb, seeds, workers)
        return cls(texts, [1] * n + [0] * n)

    def subset(self, idx):
        s = MapSet()
        s.texts = [self.texts[i] for i in idx]
        s.y = self.y[idx]
        s.sig = [self.sig[i] for i in idx]
        s.x = [self.x[i] for i in idx]
        return s

    def __len__(self):
        return len(self.y)

    def batches(self, batch_size, shuffle):
        """Yields (indices, x, adj, y); each batch contains maps of one shape only."""
        groups = {}
        for i, s in enumerate(self.sig):
            groups.setdefault(s, []).append(i)
        chunks = []
        for sig, idx in groups.items():
            idx = np.random.permutation(idx) if shuffle else np.array(idx)
            chunks += [(sig, idx[k:k + batch_size]) for k in range(0, len(idx), batch_size)]
        if shuffle:
            chunks = [chunks[k] for k in np.random.permutation(len(chunks))]
        for sig, idx in chunks:
            x = torch.from_numpy(np.stack([self.x[i] for i in idx]))
            yield idx, x, MapSet.graphs[sig], torch.from_numpy(self.y[idx])


def new_model(width, depth, hidden):
    # n_nodes/edges only initialise the stored graph; the graph of each batch is passed to forward()
    return GCN(1, np.zeros((0, 2), dtype=np.int64), width=width, depth=depth, hidden=hidden, head='pool')


@torch.no_grad()
def predict(model, data, batch_size=1024):
    model.eval()
    prob = np.zeros(len(data), dtype=np.float32)
    for idx, x, adj, _ in data.batches(batch_size, shuffle=False):
        prob[idx] = torch.sigmoid(model(x, adj)).numpy()
    return prob


def loss_of(model, data):
    p = np.clip(predict(model, data), 1e-7, 1 - 1e-7)
    return float(-np.mean(data.y * np.log(p) + (1 - data.y) * np.log(1 - p)))


def train(model, data, val, lr, epochs, patience, batch_size, log=print):
    """Adam + early stopping on the validation loss; returns the model of the best epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, epoch = math.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for _, x, adj, y in data.batches(batch_size, shuffle=True):
            loss = F.binary_cross_entropy_with_logits(model(x, adj), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = loss_of(model, val)
        if vl < best:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log(f'  trained {epoch} epochs on {len(data)} maps, best validation loss {best:.4f}')
    return model


def errors(model, data):
    pred = predict(model, data) > 0.5
    wrong = np.flatnonzero(pred != (data.y == 1))
    return wrong, int(((data.y == 0) & pred).sum()), int(((data.y == 1) & ~pred).sum())


def fine_tune_set(data, wrong, hard_frac, rng):
    """All wrongly judged maps + correctly judged maps of the same set (half per class), so that the
    wrongly judged ones make up hard_frac."""
    n_normal = int(round(len(wrong) * (1 - hard_frac) / hard_frac))
    correct = np.setdiff1d(np.arange(len(data)), wrong)
    picks = [rng.permutation(correct[data.y[correct] == c])[:n_normal // 2 + (n_normal % 2) * c] for c in (0, 1)]
    return data.subset(np.concatenate([wrong] + picks).astype(int))


def mcnemar_p(wrong_a, wrong_b, n):
    a, b = np.zeros(n, bool), np.zeros(n, bool)
    a[wrong_a], b[wrong_b] = True, True
    k1, k2 = int((a & ~b).sum()), int((~a & b).sum())
    if k1 + k2 == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(k1 + k2, i) for i in range(min(k1, k2) + 1)) / 2 ** (k1 + k2))


def mining(model, cfg, smart, dumb, val, args, log=print):
    """Hard example mining with fresh maps from the generators (see README)."""
    rng = np.random.default_rng(args.seed)
    start = new_model(**cfg)
    start.load_state_dict(model.state_dict())
    best = model
    cur = MapSet.from_generators(smart, dumb, MINING_SEED_BASE, args.eval_n, args.workers)
    cur_wrong, fp, fn = errors(best, cur)
    log(f'mining: evaluation set 0: {len(cur_wrong)} errors (FP {fp}, FN {fn}) of {len(cur)}')
    failed, accepted = 0, []
    for k in range(1, args.rounds + 1):
        if len(cur_wrong) == 0:
            break
        cand = new_model(**cfg)
        cand.load_state_dict(best.state_dict())
        ft = fine_tune_set(cur, cur_wrong, args.hard_frac, rng)
        log(f'round {k}: further training on {len(cur_wrong)} wrongly judged + {len(ft) - len(cur_wrong)} other fresh maps')
        cand = train(cand, ft, val, args.ft_lr, args.ft_epochs, args.ft_patience, args.batch_size, log)
        nxt = MapSet.from_generators(smart, dumb, MINING_SEED_BASE + k * MINING_STRIDE, args.eval_n, args.workers)
        w_best, fp_b, fn_b = errors(best, nxt)
        w_cand, fp_c, fn_c = errors(cand, nxt)
        log(f'  new evaluation set: saved model {len(w_best)} errors (FP {fp_b}, FN {fn_b}) | '
            f'further-trained model {len(w_cand)} errors (FP {fp_c}, FN {fn_c}) | McNemar p = {mcnemar_p(w_best, w_cand, len(nxt)):.3f}')
        if len(w_cand) < len(w_best):
            best, failed, cur_wrong = cand, 0, w_cand
            accepted.append(k)
            log('  -> further-trained model accepted')
        else:
            failed += 1
            cur_wrong = w_best
            log(f'  -> saved model kept ({failed}/{args.mining_patience} failed attempts)')
            if failed >= args.mining_patience:
                break
        cur = nxt
    result = {'accepted_rounds': accepted}
    if accepted and args.final_n:
        final = MapSet.from_generators(smart, dumb, SEEDS['final'], args.final_n, args.workers)
        w_s, _, _ = errors(start, final)
        w_e, _, _ = errors(best, final)
        result.update(final_n=len(final), start_errors=len(w_s), end_errors=len(w_e),
                      mcnemar_p=mcnemar_p(w_s, w_e, len(final)))
        log(f'mining effect on {len(final)} fresh maps: {len(w_s)} errors before -> {len(w_e)} after mining '
            f'(McNemar p = {result["mcnemar_p"]:.3f})')
    return best, result
