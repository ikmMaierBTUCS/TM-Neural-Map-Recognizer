"""Dumb generator (no same-coloured neighbours, colours weighted by contingent^0.5) for train.py / generate_maps.py.

A generator file only has to provide  generate_map(seed) -> str  returning a map in the format of
map_is_smart.py (one row per line, e.g. "U,S,G,B,..."). Change LAYOUT to use another river layout
(see tmgen/layouts.py: original, fjords, fi, loon).
"""
import contextlib
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmgen.fast import FastDumbHexagonalMap  # noqa: E402
from tmgen.layouts import LAYOUTS  # noqa: E402

LAYOUT = 'original'


def generate_map(seed):
    random.seed(seed)
    height, width, form, rivers, start = LAYOUTS[LAYOUT]
    while True:
        hex_map = FastDumbHexagonalMap(height, width, form, rivers, start)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hex_map.create_map()
                return hex_map.bga_format()
        except Exception:  # no colour allowed for a hex; retry
            continue
