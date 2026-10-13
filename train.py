"""Train a model that tells maps of two generators apart, and write it as a standalone script.

    python train.py generators/smart_generator.py generators/dumb_generator.py --out model.py
    python model.py map.txt          # used exactly like map_is_smart.py

Each generator file must provide  generate_map(seed) -> str  (one map row per line, e.g. "U,S,G,B,...").
Steps: generate training/validation maps -> train the GCN -> hard example mining with fresh maps ->
evaluate once on a fresh test set -> write model.py (numpy only). Needs torch (see requirements.txt).
With the defaults a run takes about 1-2 hours on a laptop CPU; use smaller --n/--eval-n to try it out.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from recognizer.export import export
from recognizer.pipeline import SEEDS, MapSet, errors, mining, new_model, train


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('smart', help='generator file of the "smart" class (label 1)')
    p.add_argument('dumb', help='generator file of the "dumb" class (label 0)')
    p.add_argument('--out', default='model.py', help='output script (default: model.py)')
    p.add_argument('--n', type=int, default=5000, help='training maps per generator')
    p.add_argument('--val-n', type=int, default=1000, help='validation maps per generator (early stopping)')
    p.add_argument('--test-n', type=int, default=2000, help='test maps per generator (final evaluation)')
    p.add_argument('--width', type=int, default=16, help='channels per hex in the graph layer')
    p.add_argument('--depth', type=int, default=1, help='number of graph layers')
    p.add_argument('--hidden', type=int, nargs='+', default=[8], help='fully connected layers after pooling')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--patience', type=int, default=30, help='early stopping: epochs without improvement')
    p.add_argument('--no-mining', action='store_true', help='skip hard example mining')
    p.add_argument('--eval-n', type=int, default=5000, help='mining: maps per generator in each fresh evaluation set')
    p.add_argument('--rounds', type=int, default=10, help='mining: maximum number of rounds')
    p.add_argument('--mining-patience', type=int, default=2, help='mining: failed attempts in a row before stopping')
    p.add_argument('--hard-frac', type=float, default=0.05, help='mining: share of wrongly judged maps in further training')
    p.add_argument('--ft-lr', type=float, default=3e-4)
    p.add_argument('--ft-epochs', type=int, default=50)
    p.add_argument('--ft-patience', type=int, default=8)
    p.add_argument('--final-n', type=int, default=10000, help='mining: maps per generator to measure the mining effect (0 = skip)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    log = lambda msg: print(f'[{(time.time() - t0) / 60:5.1f} min] {msg}', flush=True)

    log(f'generating {args.n} + {args.n} training and {args.val_n} + {args.val_n} validation maps')
    data = MapSet.from_generators(args.smart, args.dumb, SEEDS['train'], args.n, args.workers)
    val = MapSet.from_generators(args.smart, args.dumb, SEEDS['val'], args.val_n, args.workers)
    shapes = {s[0] for s in data.sig}
    log(f'{len(shapes)} map shape(s), {sum(next(iter(shapes)))} hexes'
        + ('' if len(shapes) == 1 else ' (first shape)'))

    cfg = dict(width=args.width, depth=args.depth, hidden=args.hidden)
    log(f'training GCN {cfg}')
    model = train(new_model(**cfg), data, val, args.lr, args.epochs, args.patience, args.batch_size, log)

    report = {'config': cfg, 'args': vars(args)}
    if not args.no_mining:
        model, report['mining'] = mining(
            model, cfg, args.smart, args.dumb, val, eval_n=args.eval_n, rounds=args.rounds,
            patience=args.mining_patience, hard_frac=args.hard_frac, ft_lr=args.ft_lr, ft_epochs=args.ft_epochs,
            ft_patience=args.ft_patience, batch_size=args.batch_size, final_n=args.final_n, workers=args.workers,
            seed=args.seed, log=log)

    log(f'final evaluation on {args.test_n} + {args.test_n} fresh test maps')
    test = MapSet.from_generators(args.smart, args.dumb, SEEDS['test'], args.test_n, args.workers)
    wrong, fp, fn = errors(model, test)
    acc = 1 - len(wrong) / len(test)
    report['test'] = {'n': len(test), 'accuracy': acc, 'fp': fp, 'fn': fn,
                      'fp_rate': fp / args.test_n, 'fn_rate': fn / args.test_n}
    log(f'test accuracy {acc:.4f} | FP rate {fp / args.test_n:.4f} | FN rate {fn / args.test_n:.4f}')

    n_params = sum(p.numel() for p in model.parameters())
    info = (f'GCN {args.depth} graph layer(s) x {args.width} channels, dense {args.hidden}, {n_params} parameters; '
            f'smart = {os.path.basename(args.smart)}, dumb = {os.path.basename(args.dumb)}; '
            f'test accuracy {acc:.4f} (FP rate {fp / args.test_n:.4f}, FN rate {fn / args.test_n:.4f})')
    export(model.state_dict(), args.out, info)
    with open(os.path.splitext(args.out)[0] + '_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    log(f'wrote {args.out} (use: python {args.out} map.txt) and {os.path.splitext(args.out)[0]}_report.json')


if __name__ == '__main__':
    main()
