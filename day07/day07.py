from utils import get_input, ints
from typing import Tuple, Callable


def main():
    values = ints(get_input())
    print('Part 1:',fuel(values, lambda x: x))
    print('Part 2:', fuel(values, lambda x: (x + 1) * x // 2))

def fuel(values: Tuple[int], cost: Callable[[int], int]) -> int:
    return min(sum(cost(abs(p - c)) for c in values) for p in values)

if __name__ == '__main__':
    main()
