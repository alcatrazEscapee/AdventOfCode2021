# Day 15: Chiton
# Leaderboard Rank: 60 / 147

from utils import FiniteGrid, Point2, get_input

import heapq


def main(text: str):
    grid = FiniteGrid.of_str(text).map_values(int)
    grid5x5 = FiniteGrid.of_empty(grid.width * 5, grid.height * 5, 0)
    for x, y in grid5x5.locations():
        grid5x5[x, y] = ((grid[x % grid.width, y % grid.height] + x // grid.width + y // grid.height - 1) % 9) + 1

    print('Part 1:', dijkstra(grid))
    print('Part 2:', dijkstra(grid5x5))


def dijkstra(grid: FiniteGrid) -> int:
    queue = [(0, Point2(0, 0))]
    shortest = {(0, 0): 0}
    target = grid.width - 1, grid.height - 1
    while queue:
        risk, pos = heapq.heappop(queue)
        if pos == target:
            return risk
        for adj in pos.neighbors():
            if adj in grid:
                adj_risk = risk + grid[adj]
                if adj not in shortest or adj_risk < shortest[adj]:
                    shortest[adj] = adj_risk
                    heapq.heappush(queue, (adj_risk, adj))
    raise ValueError('Dijkstra\'s terminated without finding a shortest path length to the target')


if __name__ == '__main__':
    main(get_input())
