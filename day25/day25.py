# Day 25: Sea Cucumber
from utils import FiniteGrid, get_input


def main(text: str):
    grid = FiniteGrid.of_str(text)

    step = 0
    movement = True
    while movement:
        next_grid = FiniteGrid.of_empty(grid.width, grid.height, '.')
        movement = False

        for x, y in grid.locations():
            if grid[x, y] == '>':
                x0 = (x + 1) % grid.width
                if grid[x0, y] == '.':
                    movement = True
                    next_grid[x0, y] = '>'
                else:
                    next_grid[x, y] = '>'
            elif grid[x, y] == 'v':
                next_grid[x, y] = 'v'

        grid = next_grid
        next_grid = FiniteGrid.of_empty(grid.width, grid.height, '.')

        for x, y in grid.locations():
            if grid[x, y] == 'v':
                y0 = (y + 1) % grid.height
                if grid[x, y0] == '.':
                    next_grid[x, y0] = 'v'
                    movement = True
                else:
                    next_grid[x, y] = 'v'
            elif grid[x, y] == '>':
                next_grid[x, y] = '>'

        grid = next_grid
        step += 1

    print('Part 1:', step)


if __name__ == '__main__':
    main(get_input())
