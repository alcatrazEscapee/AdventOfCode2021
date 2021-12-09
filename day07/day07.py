# Day 7: The Treachery of Whales

from utils import get_input, ints
from typing import Tuple, Callable


def main(text: str):
    values = ints(text)

    # Analytical improvements:
    # For part 1, we can show the target location will lie in an interval where there are equal number of points on either side (so +1/-1 of the median)
    # For part 2, we can show the target location will lie within +/- 1 of the mean.
    # This improves the overall runtime by about 10x as opposed to checking every location

    ordered = sorted(values)
    midpoint = len(values) // 2
    mean = sum(values) // len(values)
    print('Part 1:', fuel(values, lambda x: x, ordered[midpoint - 1], ordered[midpoint + 1]))
    print('Part 2:', fuel(values, lambda x: (x + 1) * x // 2, mean - 1, mean + 1))

def fuel(values: Tuple[int], cost: Callable[[int], int], lo: int, hi: int) -> int:
    return min(sum(cost(abs(p - c)) for c in values) for p in range(lo, 1 + hi))


if __name__ == '__main__':
    main(get_input())
