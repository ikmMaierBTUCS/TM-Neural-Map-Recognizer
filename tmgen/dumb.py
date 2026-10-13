# Dumb generator: same map traversal (spiral walk from the start hex) as the smart
# generator, but the colour of each hex is chosen only by
#   - adjacency: colours already adjacent to the hex are forbidden
#   - contingent: the remaining colours are weighted by contingent(color)**0.5,
#     the same contingent factor the smart generator uses in share_score.
# No work distances, no river access saldo, no luck.
import random

from tmgen.smart import HexagonalMap

LAND_COLORS = ['red', 'yel', 'bro', 'bla', 'blu', 'grn', 'gry']


class DumbHexagonalMap(HexagonalMap):

    def reset(self):
        return DumbHexagonalMap(self.height, self.width, self.form, self.rivers, self.start)

    def is_adjacent(self, hexf, color):
        for direction in range(6):
            if self.mapfunction(self.next_hex(hexf, direction)) == color:
                return True
        return False

    def dumb_score(self, hexf, color):
        if self.is_adjacent(hexf, color):
            return 0
        return round(self.contingent(color)**0.5 * 10)

    def paint(self, hexf):
        if self.out_of_bounds(hexf):
            return
        if self.map[hexf] == ' ~ ':
            self.map[hexf] = '~~~'
            return
        scores = [self.dumb_score(hexf, color) for color in LAND_COLORS]
        # raises ValueError if every colour scores 0; the caller retries, like the smart generator
        randint = random.randint(0, sum(scores) - 1)
        for color, score in zip(LAND_COLORS, scores):
            if randint < score:
                self.map[hexf] = color
                return
            randint -= score

    def create_map(self):
        # randomize start field (as in smart generator)
        current_hex = self.start
        if self.map[current_hex] != ' ~ ':
            self.map[current_hex] = self.colors[random.randint(0, 6)]
        else:
            self.map[current_hex] = '~~~'

        # randomize first step
        current_direction = random.randint(0, 5)
        current_hex = self.next_hex(current_hex, current_direction)
        self.paint(current_hex)
        current_direction = (current_direction + 2) % 6

        # further steps: identical spiral walk as in the smart generator
        while '???' in self.map.values() or ' ~ ' in self.map.values():
            while not self.out_of_bounds(self.next_hex(current_hex, current_direction)) and not self.map[self.next_hex(current_hex, current_direction)] in ['???', ' ~ ']:
                current_direction = (current_direction - 1) % 6
            current_hex = self.next_hex(current_hex, current_direction)
            self.paint(current_hex)
            current_direction = (current_direction + 1) % 6
        return ''
