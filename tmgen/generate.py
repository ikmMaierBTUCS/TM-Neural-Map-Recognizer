"""Generate labelled maps from the smart and the dumb generator.

Usage:
    python -m tmgen.generate --split train --n 2000
    python -m tmgen.generate --split train --n 2000 --start 2000   # extend later
    python -m tmgen.generate --split val   --n 500
    python -m tmgen.generate --split test  --n 1000

By default the final tests (number/river_access/luck) are skipped: every create_map()
result is used. --tests re-enables the original retry loop (much slower).
Shards go to data/notests/ or data/tests/ respectively.

Writes the shard <out>/<layout>_<split>_<start>-<end>.npz with
    grids  int8 (N, H, W): 0..6 = red,yel,bro,bla,blu,grn,gry (colour wheel order),
                           7 = river, -1 = no hex (end of a short row)
    labels int8 (N,):      1 = smart generator, 0 = dumb generator
    attempts int16 (N,):   number of create_map() runs until all tests passed
"""
import argparse
import contextlib
import io
import multiprocessing as mp
import os
import random
import time

import numpy as np

from tmgen.fast import FastDumbHexagonalMap, FastHexagonalMap

# (height, width, form, rivers, start) as in the smart generator's example section
LAYOUTS = {
    # original Terra Mystica base map river layout ("original" in the generator's example section)
    'original': (9, 13, 0, [(1,1),(2,1),(5,1),(6,1),(9,1),(10,1),(0,2),(1,2),(3,2),(5,2),(7,2),(9,2),(11,2),(12,2),(3,3),(4,3),(7,3),(9,3),(8,4),(9,4),(2,5),(3,5),(6,5),(7,5),(8,5),(0,6),(1,6),(2,6),(4,6),(6,6),(8,6),(3,7),(4,7),(5,7),(8,7),(9,8)], (5, 4)),
    'fi': (9, 13, 1, [(1,0),(5,0),(2,1),(6,1),(7,1),(8,1),(2,2),(3,2),(4,2),(8,2),(9,2),(10,2),(11,2),(12,2),(5,3),(9,3),(0,4),(1,4),(3,4),(4,4),(9,4),(2,5),(3,5),(5,5),(6,5),(7,5),(10,5),(1,6),(7,6),(10,6),(2,7),(7,7),(10,7),(2,8),(7,8),(10,8)], (7, 5)),
    'loon': (9, 13, 1, [(8,0),(9,0),(3,1),(4,1),(7,1),(10,1),(1,2),(2,2),(6,2),(10,2),(3,3),(7,3),(8,3),(10,3),(2,4),(5,4),(6,4),(8,4),(1,5),(4,5),(6,5),(8,5),(1,6),(2,6),(3,6),(9,6),(10,6),(3,7),(7,7),(8,7),(11,7),(2,8),(4,8),(5,8),(6,8)], (5, 3)),
    'fjords': (9, 13, 0, [(2,0),(2,1),(6,1),(7,1),(8,1),(9,1),(10,1),(3,2),(4,2),(6,2),(11,2),(0,3),(1,3),(2,3),(4,3),(5,3),(11,3),(12,3),(3,4),(6,4),(11,4),(2,5),(6,5),(10,5),(2,6),(7,6),(10,6),(1,7),(7,7),(8,7),(9,7),(1,8),(2,8),(7,8)], (4, 3)),
}

CODES = {'red': 0, 'yel': 1, 'bro': 2, 'bla': 3, 'blu': 4, 'grn': 5, 'gry': 6, '~~~': 7}
RIVER = 7
NO_HEX = -1

# disjoint seed ranges so that train/val/test never share a random stream
SPLIT_SEED_OFFSET = {'train': 0, 'val': 100_000_000, 'test': 200_000_000, 'ana': 400_000_000}  # ana: error analysis
KIND_SEED_OFFSET = {'smart': 0, 'dumb': 50_000_000}


def passes_tests(hex_map, kind):
    if not hex_map.number_test() or not hex_map.river_access_test():
        return False
    # the luck test depends on the work distances, which the dumb generator does not use
    if kind == 'smart' and not hex_map.luck_test():
        return False
    return True


def generate_map(kind, layout, seed, tests=False):
    """Run the generator until a map is complete (and, with tests=True, passes the final tests
    like in the original retry loop)."""
    random.seed(seed)
    height, width, form, rivers, start = LAYOUTS[layout]
    # fast drop-in versions: identical maps for the same seed (see tmgen.fast_check)
    cls = FastHexagonalMap if kind == 'smart' else FastDumbHexagonalMap
    attempts = 0
    while True:
        attempts += 1
        hex_map = cls(height, width, form, rivers, start)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hex_map.create_map()
        except Exception:
            continue
        if not tests or passes_tests(hex_map, kind):
            return hex_map, attempts


def to_grid(hex_map):
    grid = np.full((hex_map.height, hex_map.width), NO_HEX, dtype=np.int8)
    for (col, row), color in hex_map.map.items():
        grid[row, col] = CODES[color]
    return grid


def _worker(args):
    kind, layout, seed, tests = args
    hex_map, attempts = generate_map(kind, layout, seed, tests)
    return to_grid(hex_map), attempts


def load_split(layout, split, data_dir='data/notests'):
    """Concatenate all shards of a split."""
    files = sorted(f for f in os.listdir(data_dir) if f.startswith(f'{layout}_{split}_') and f.endswith('.npz'))
    if not files:
        raise FileNotFoundError(f'no shards for {layout}/{split} in {data_dir}')
    parts = [np.load(os.path.join(data_dir, f)) for f in files]
    return (np.concatenate([p['grids'] for p in parts]),
            np.concatenate([p['labels'] for p in parts]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='fjords', choices=LAYOUTS)
    parser.add_argument('--split', required=True, choices=SPLIT_SEED_OFFSET)
    parser.add_argument('--n', type=int, required=True, help='maps per generator')
    parser.add_argument('--start', type=int, default=0, help='first map index (to extend a split by a new shard)')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--tests', action='store_true', help='apply the final tests (rejection sampling)')
    parser.add_argument('--out', default=None, help='default: data/tests or data/notests')
    args = parser.parse_args()

    jobs = []
    for kind in ('smart', 'dumb'):
        base = SPLIT_SEED_OFFSET[args.split] + KIND_SEED_OFFSET[kind]
        jobs += [(kind, args.layout, base + i, args.tests) for i in range(args.start, args.start + args.n)]
    labels = np.array([1 if job[0] == 'smart' else 0 for job in jobs], dtype=np.int8)
    if args.out is None:
        args.out = os.path.join('data', 'tests' if args.tests else 'notests')

    t0 = time.time()
    grids, attempts = [], []
    with mp.Pool(args.workers) as pool:
        for i, (grid, att) in enumerate(pool.imap(_worker, jobs, chunksize=8)):
            grids.append(grid)
            attempts.append(att)
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(jobs)} maps, {time.time() - t0:.0f}s', flush=True)

    attempts = np.array(attempts, dtype=np.int16)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{args.layout}_{args.split}_{args.start}-{args.start + args.n}.npz')
    np.savez_compressed(path, grids=np.stack(grids), labels=labels, attempts=attempts)
    print(f'wrote {path} in {time.time() - t0:.0f}s; '
          f'mean attempts smart={attempts[labels == 1].mean():.1f} dumb={attempts[labels == 0].mean():.1f}')


if __name__ == '__main__':
    main()
