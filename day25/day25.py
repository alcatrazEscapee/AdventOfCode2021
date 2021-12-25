# Day 25: Sea Cucumber

from utils import FiniteGrid, get_input


def main(text: str):
    grid = FiniteGrid.of_str(text)

    step = 0
    movement = True
    while movement:
        next_grid = FiniteGrid.of_empty(grid.width, grid.height, '.')
        movement = False

        # Move the east herd first. Check against the existing open spaces
        for x, y in grid.locations():
            if grid[x, y] == '>':
                x0 = (x + 1) % grid.width
                if grid[x0, y] == '.':
                    movement = True
                    next_grid[x0, y] = '>'
                else:
                    next_grid[x, y] = '>'

        # Move the south herd second. The herd locations are in grid, but the east locations are in next_grid
        # An 'empty' spot, has grid[x,y] != 'v', and also that next_grid[x, y] != '>'
        for x, y in grid.locations():
            if grid[x, y] == 'v':
                y0 = (y + 1) % grid.height
                if grid[x, y0] != 'v' and next_grid[x, y0] != '>':
                    next_grid[x, y0] = 'v'
                    movement = True
                else:
                    next_grid[x, y] = 'v'

        grid = next_grid
        step += 1

    print('Part 1:', step)


if __name__ == '__main__':
    main(get_input())
