# Day 11: Dumbo Octopus
# Leaderboard Rank: 63 / 55

from utils import get_input, FiniteGrid
from itertools import count


def main(text: str):
    grid = FiniteGrid.of_str(text).map_values(int)
    total_flashes = 0

    for step in count(1):
        # All octopuses' energy increases by one
        # Then each octopus with energy > 9 flashes, and all adjacent octopuses increment by one and flash, recursively.
        # Finally, each octopus that flashed has its energy reset to zero
        queue = []
        for pos in grid.locations():
            grid[pos] += 1
            if grid[pos] > 9:
                queue.append(pos)

        flashes = set(queue)
        while queue:
            pos = queue.pop()
            for adj in pos.moore_neighbors():
                if adj in grid:
                    grid[adj] += 1
                    if grid[adj] > 9 and adj not in flashes:
                        queue.append(adj)
                        flashes.add(adj)

        total_flashes += len(flashes)
        for pos in flashes:
            grid[pos] = 0

        if step == 100:
            print('Part 1:', total_flashes)  # The total number of flashes after 100 steps

        if len(flashes) == len(grid):
            print('Part 2:', step)  # The first step where all octopuses flash
            break


if __name__ == '__main__':
    main(get_input())
