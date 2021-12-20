# Day 20: Trench Map
# Leaderboard Rank: 58 / 52

from utils import FiniteGrid, get_input


def main(text: str):
    algorithm, image = text.split('\n\n')
    grid = FiniteGrid.of_str(image, default='.')

    for step in range(50):

        if step == 2:
            print('Part 1:', sum(grid[p] == '#' for p in grid.locations()))

        # Key realization is that the infinite surrounding region's points also needs to be tracked
        # In this case, we do so via the default parameter of FiniteGrid
        # Other than that, we can expand the tracked area by one in all directions each iteration
        default = algorithm[0b111111111 if grid.default == '#' else 0]
        next_grid = FiniteGrid.of_empty(grid.width + 2, grid.height + 2, default=default)
        for x, y in next_grid.locations():
            parity = 0
            for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):  # Order is important
                parity = (parity << 1) | (grid[x + dx - 1, y + dy - 1] == '#')
            next_grid[x, y] = algorithm[parity]
        grid = next_grid

    print('Part 2:', sum(grid[p] == '#' for p in grid.locations()))


if __name__ == '__main__':
    main(get_input())
