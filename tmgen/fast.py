"""Faster drop-in versions of the smart and dumb generator.

Only helper methods are replaced; create_map() and all scoring formulas stay the original code,
so for the same random seed the maps are identical to those of tmgen/smart.py and tmgen/dumb.py.

What is cached / tracked:
  - river_access(hex): depends only on the river layout -> computed once per layout
  - river_access_score(color), absolute_stock(color), stock(color): kept up to date on every
    assignment to self.map instead of scanning all hexes on every call
  - out_of_bounds: set lookup instead of range checks
  - next_hex: precomputed table for in-bounds hexes
  - number_of_landhexes: constant per layout
work_distance is deliberately left unchanged.
"""
from tmgen.dumb import DumbHexagonalMap
from tmgen.smart import HexagonalMap

LAND = ('red', 'yel', 'bro', 'bla', 'blu', 'grn', 'gry')
_layout_cache = {}


class TrackedMap(dict):
    """dict hex -> colour that keeps per-colour totals of count, hex value and river access."""

    def __init__(self, items, river_access, hex_value):
        super().__init__(items)
        self.ra = river_access
        self.hv = hex_value
        self.count = dict.fromkeys(LAND, 0)
        self.value = dict.fromkeys(LAND, 0.0)
        self.access = dict.fromkeys(LAND, 0)
        for h, c in self.items():
            self._add(h, c, 1)

    def _add(self, h, c, sign):
        if c in self.count:
            self.count[c] += sign
            self.value[c] += sign * self.hv[h]
            self.access[c] += sign * self.ra[h]

    def __setitem__(self, h, c):
        old = self.get(h)
        if old is not None:
            self._add(h, old, -1)
        super().__setitem__(h, c)
        self._add(h, c, 1)


class FastMixin:

    def _setup_fast(self):
        key = (self.height, self.width, self.form, tuple(self.rivers))
        if key not in _layout_cache:
            valid = frozenset(self.map)
            nxt = {(h, d): HexagonalMap.next_hex(self, h, d) for h in valid for d in range(6)}
            # river_access/value_of_hex of the original code, evaluated once on the empty layout
            self._valid, self._next = valid, nxt
            ra = {h: HexagonalMap.river_access(self, h) for h in valid}
            hv = {h: HexagonalMap.value_of_hex(self, h) for h in valid}
            n_land = HexagonalMap.number_of_landhexes(self)
            _layout_cache[key] = (valid, nxt, ra, hv, n_land)
        self._valid, self._next, ra, hv, self._n_land = _layout_cache[key]
        self.map = TrackedMap(self.map, ra, hv)

    def out_of_bounds(self, hexf):
        return hexf not in self._valid

    def next_hex(self, hexf, direction):
        r = self._next.get((hexf, direction))
        return r if r is not None else HexagonalMap.next_hex(self, hexf, direction)

    def mapfunction(self, hexf):
        return self.map.get(hexf, '')

    def number_of_landhexes(self):
        return self._n_land

    def absolute_stock(self, color):
        return self.map.count[color] if color in self.map.count else list(self.map.values()).count(color)

    def stock(self, color):
        # only used as "is there any hex of this colour" (value_of_hex > 0 on every hex)
        return self.map.value[color] if color in self.map.value else HexagonalMap.stock(self, color)

    def river_access(self, hexf):
        if hexf in self.map.ra and self.map[hexf] not in ('~~~', ' ~ '):
            return self.map.ra[hexf]
        return HexagonalMap.river_access(self, hexf)

    def river_access_score(self, color):
        return self.map.access[color] if color in self.map.access else HexagonalMap.river_access_score(self, color)


class FastHexagonalMap(FastMixin, HexagonalMap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_fast()

    def reset(self):
        return FastHexagonalMap(self.height, self.width, self.form, self.rivers, self.start)


class FastDumbHexagonalMap(FastMixin, DumbHexagonalMap):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_fast()

    def reset(self):
        return FastDumbHexagonalMap(self.height, self.width, self.form, self.rivers, self.start)
