# Day 3: Binary Diagnostic

from typing import List
from collections import Counter

from utils import get_input


def main(text: str):
    rows: List[str] = text.split('\n')

    gamma = epsilon = ''
    o2 = list(rows)
    co2 = list(rows)

    for i in range(len(rows[0])):
        c = Counter(r[i] for r in rows)
        if c['1'] > c['0']:
            gamma += '1'
            epsilon += '0'
        else:
            gamma += '0'
            epsilon += '1'

        o2 = reduce(o2, i, True)
        co2 = reduce(co2, i, False)

    print('Part 1:', int(gamma, 2) * int(epsilon, 2))
    print('Part 2:', int(o2[0], 2) * int(co2[0], 2))

def reduce(values: List[str], i: int, hi: bool) -> List[str]:
    if len(values) > 1:
        c = Counter(r[i] for r in values)
        bit = '1' if hi == (c['1'] >= c['0']) else '0'
        return [x for x in values if x[i] == bit]
    return values


if __name__ == '__main__':
    main(get_input())
