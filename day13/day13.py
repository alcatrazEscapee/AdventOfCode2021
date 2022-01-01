from utils import get_input, ints
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

    min_x, max_x, min_y, max_y = min(d[0] for d in dots), max(d[0] for d in dots), min(d[1] for d in dots), max(d[1] for d in dots)
    print('\n'.join(''.join(
        '\u2588' if (x, y) in dots else ' '
        for x in range(min_x, 1 + max_x))
        for y in range(min_y, 1 + max_y)))
    print('Part 2:', recognize(dots, min_x, max_x, min_y, max_y))


def mirror(v: int, k: int) -> int:
    return k - abs(v - k)


if __name__ == '__main__':
    main(get_input())
