# Day 01: Sonar Sweep

from utils import get_input, ints, windows


def main(text: str):
    values = ints(text)

    print('Part 1:', sum(a < b for a, b in windows(values, 2)))
    print('Part 2:', sum(a < b for a, _, _, b in windows(values, 4)))


if __name__ == '__main__':
    main(get_input())
