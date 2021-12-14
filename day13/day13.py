from utils import InfiniteGrid, get_input, ints, map_to_callable
from aoc_font import recognize


def main(text: str):
    dots, folds = text.split('\n\n')
    dots = set(ints(line) for line in dots.split('\n'))
    folds = [((s := line.split('='))[0][-1], int(s[1])) for line in folds.split('\n')]

    part1 = False
    for axis, value in folds:
        if axis == 'x':
            dots = {(mirror(x, value), y) for x, y in dots}
        else:
            dots = {(x, mirror(y, value)) for x, y in dots}

        if not part1:
            part1 = True
            print('Part 1:', len(dots))

    grid = InfiniteGrid.of_points(dots)
    print('Part 2:', recognize(grid))
    print(grid.map_values(map_to_callable({'.': ' ', '#': '\u2588'})))


def mirror(v: int, k: int) -> int:
    return k - abs(v - k)


if __name__ == '__main__':
    main(get_input())
