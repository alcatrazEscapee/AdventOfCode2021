# Day 9: Smoke Basin
# Leaderboard Rank: 60 / 20

from utils import FiniteGrid, get_input_lines
from collections import Counter


def main():
    g = FiniteGrid.of_iter([[int(c) for c in line] for line in get_input_lines()], default=10)

    risk = 0
    for x, y in g.locations():
        p = g[x, y]
        if p < g[x + 1, y] and p < g[x - 1, y] and p < g[x, y + 1] and p < g[x, y - 1]:
            risk += 1 + p

    print('Part 1:', risk)

    basins = Counter()  # collect the number of points that terminate in any one basin
    for x, y in g.locations():
        if g[x, y] != 9:  # height 9 do not belong to any basin
            while True:  # step from this point until we reach a sink
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if g[x + dx, y + dy] < g[x, y]:
                        x += dx
                        y += dy
                        break
                else:  # Reached the bottom of a basin
                    basins[x, y] += 1
                    break

    a, b, c = basins.most_common(3)
    print('Part 2:', a[1] * b[1] * c[1])


if __name__ == '__main__':
    main()
