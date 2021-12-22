# Day 14: Extended Polymerization
# Leaderboard Rank: 16 / 191

from utils import get_input, windows
from collections import Counter


def main(text: str):
    start, rules = text.split('\n\n')
    rules = dict(r.split(' -> ') for r in rules.split('\n'))

    pairs = Counter(windows(start, 2))
    elements = Counter(start)
    for i in range(40):

        if i == 10:
            print('Part 1:', checksum(elements))

        next_pairs = Counter()
        next_elements = Counter(elements)

        for pair, count in pairs.items():
            lhs, rhs = pair
            mid = rules[lhs + rhs]

            # Generate two independent pairs
            next_pairs[lhs, mid] += count
            next_pairs[mid, rhs] += count

            # And increment the count for the new elements added
            next_elements[mid] += count
        pairs = next_pairs
        elements = next_elements

    print('Part 2:', checksum(elements))

def checksum(elements: Counter):
    most, *_, least = elements.most_common()
    return most[1] - least[1]


if __name__ == '__main__':
    main(get_input())
