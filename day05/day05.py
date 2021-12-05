# Day 5: Hydrothermal Venture

from utils import get_input_lines, ints, sign
from typing import List, Tuple, Iterable


def main():
    lines: List[Tuple[int, ...]] = [ints(line) for line in get_input_lines()]
    print('Part 1:', run(lines, False))
    print('Part 2:', run(lines, True))


def run(lines: List[Tuple[int, ...]], diagonals: bool) -> int:
    points = set()
    duplicates = set()

    for x0, y0, x1, y1 in lines:
        if x0 == x1 or y0 == y1 or diagonals:  # Either straight line, or we allow diagonals
            for p in project(x0, y0, x1, y1):
                if p in points:
                    duplicates.add(p)
                else:
                    points.add(p)

    return len(duplicates)

def project(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """ Projects points along the line (x0, y0) -> (x1, y1). Works for horizontal, vertical, or exactly diagonal lines. """
    dx = x1 - x0
    dy = y1 - y0
    sx = sign(dx)
    sy = sign(dy)
    for d in range(1 + max(abs(dx), abs(dy))):
        yield x0 + d * sx, y0 + d * sy


if __name__ == '__main__':
    main()
