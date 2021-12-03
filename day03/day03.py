# Day 3: Binary Diagnostic

from typing import List, Set
from collections import Counter

from utils import get_input_lines


def main():
    rows: List[str] = get_input_lines()

    gamma = epsilon = ''
    o2 = set(rows)
    co2 = set(rows)

    for i in range(len(rows[0])):
        c0, c1 = Counter(r[i] for r in rows).most_common(2)  # returns (most common char, count), (least common char, count)
        gamma += c0[0]
        epsilon += c1[0]
        o2 = reduce(o2, i, True)
        co2 = reduce(co2, i, False)

    print('Part 1:', int(gamma, 2) * int(epsilon, 2))
    print('Part 2:', int(o2.pop(), 2) * int(co2.pop(), 2))

def reduce(values: Set[str], i: int, hi: bool) -> Set[str]:
    if len(values) > 1:
        c0, c1 = Counter(r[i] for r in values).most_common(2)
        bit = ('1' if hi else '0') if c0[1] == c1[1] else (c0 if hi else c1)[0]
        return {x for x in values if x[i] == bit}
    return values


if __name__ == '__main__':
    main()
