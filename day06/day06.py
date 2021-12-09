# Day 6: Lanternfish
# Leaderboard Rank: 31 / 85

from utils import get_input, ints

from typing import Tuple
from collections import Counter


def main(text: str):
    values = ints(text)
    print('Part 1:', run(values, 80))
    print('Part 2:', run(values, 256))


def run(values: Tuple[int, ...], iterations: int) -> int:
    counts = Counter(values)
    for _ in range(iterations):
        next_counts = Counter()

        for days, count in counts.items():  # Handle day > 0 fish
            if days != 0:
                next_counts[days - 1] = count

        # Handle day 0 fish
        next_counts[6] += counts[0]
        next_counts[8] = counts[0]

        counts = next_counts
    return sum(counts.values())


if __name__ == '__main__':
    main(get_input())
