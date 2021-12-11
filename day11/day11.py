# Day 11: Dumbo Octopus
# Leaderboard Rank: 63 / 55

from utils import get_input, FiniteGrid


def main(text: str):
    grid = FiniteGrid.of_str(text).map_values(int)
    total_octopuses = grid.width * grid.height
    total_flashes = 0

    step = 0
    while True:
        step += 1

        # All octopuses energy increases by one
        # Then each octopus with energy > 9 flashes, and all adjacent octopuses increment by one and flash, recursively.
        # Finally, each octopus that flashed has it's energy reset to zero
        for pos in grid.locations():
            grid[pos] += 1

        # Simple recursive flashing - compute until we compute no flashes in a single iteration
        flashes = set()
        modified = True
        while modified:
            modified = False
            for pos in grid.locations():
                if grid[pos] > 9 and pos not in flashes:
                    flashes.add(pos)
                    modified = True
                    for adj in pos.moore_neighbors():
                        if adj in grid:
                            grid[adj] += 1

        total_flashes += len(flashes)
        for pos in flashes:
            grid[pos] = 0

        if step == 100:
            print('Part 1:', total_flashes)  # The total number of flashes after 100 steps

        if len(flashes) == total_octopuses:
            print('Part 2:', step)  # The first step where all octopuses flash
            break


if __name__ == '__main__':
    main(get_input())
