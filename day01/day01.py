# Day 01: Sonar Sweep

from utils import get_input, ints


def main(text: str):
    values = ints(text)

    print('Part 1:', sum(values[i] < values[i + 1] for i in range(len(values) - 1)))
    print('Part 2:', sum(values[i] < values[i + 3] for i in range(len(values) - 3)))


if __name__ == '__main__':
    main(get_input())
