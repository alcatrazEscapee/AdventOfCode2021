# Day 9: Smoke Basin
# Leaderboard Rank: 60 / 20

from utils import FiniteGrid, get_input
from collections import Counter


def main(text: str):
    grid: FiniteGrid[int] = FiniteGrid.of_str(text, default='9').map_values(int)

    risk = 0
    for pos in grid.locations():
        height = grid[pos]
        if all(height < grid[adj] for adj in pos.neighbors()):
            risk += 1 + height

    print('Part 1:', risk)

    basins = Counter()  # collect the number of points that terminate in any one basin
    for pos in grid.locations():
        if grid[pos] != 9:  # height 9 do not belong to any basin
            while any(grid[low := adj] < grid[pos] for adj in pos.neighbors()):
                pos = low  # Step downwards until we reach a low point
            basins[pos] += 1

    a, b, c = basins.most_common(3)
    print('Part 2:', a[1] * b[1] * c[1])


if __name__ == '__main__':
    main(get_input())
