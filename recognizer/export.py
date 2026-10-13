"""Write a trained GCN (recognizer/gcn.py) as a standalone, numpy-only script (same usage as map_is_smart.py).

    python -m recognizer.export model.pt map_is_smart.py --info "..."     (model.pt: {'state_dict': ...})
"""
import argparse
import json
import os

import torch

TEMPLATE = os.path.join(os.path.dirname(__file__), 'model_template.py')
PLACEHOLDER = 'WEIGHTS = None  # filled in by recognizer/export.py'


def weights_from_state(state, info=''):
    """State dict of recognizer.gcn.GCN -> plain nested lists (graph layers, dense layers, output)."""
    to = lambda t: t.detach().cpu().numpy().round(7).tolist()
    count = lambda prefix: len([k for k in state if k.startswith(prefix) and k.endswith('.weight')])
    graph = [{'self_w': to(state[f'graph_self.{i}.weight']), 'self_b': to(state[f'graph_self.{i}.bias']),
              'nb_w': to(state[f'graph_nb.{i}.weight'])} for i in range(count('graph_self.'))]
    dense = [{'w': to(state[f'dense.{i}.weight']), 'b': to(state[f'dense.{i}.bias'])} for i in range(count('dense.'))]
    out = {'w': to(state['out.weight'])[0], 'b': to(state['out.bias'])}
    return {'info': info, 'graph': graph, 'dense': dense, 'out': out}


def export(state, out_path, info=''):
    with open(TEMPLATE, encoding='utf-8') as f:
        code = f.read()
    assert code.count(PLACEHOLDER) == 1
    w = json.dumps(weights_from_state(state, info), separators=(',', ':'))
    code = code.replace(PLACEHOLDER, f"WEIGHTS = json.loads('{w}')")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('out')
    p.add_argument('--info', default='')
    args = p.parse_args()
    export(torch.load(args.checkpoint)['state_dict'], args.out, args.info)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
