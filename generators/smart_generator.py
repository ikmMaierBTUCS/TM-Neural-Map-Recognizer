"""Smart generator (Terra Mystica Map Generator, branch 3) for train.py / generate_maps.py.

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
from tmgen.fast import FastHexagonalMap  # noqa: E402  (identical maps to the original code, 5x faster)
from tmgen.layouts import LAYOUTS  # noqa: E402

LAYOUT = 'original'


def generate_map(seed):
    random.seed(seed)
    height, width, form, rivers, start = LAYOUTS[LAYOUT]
    while True:
        hex_map = FastHexagonalMap(height, width, form, rivers, start)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hex_map.create_map()
                return hex_map.bga_format()
        except Exception:  # the generator occasionally gets stuck; retry like the original code
            continue
