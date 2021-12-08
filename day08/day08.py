# Day 8: Seven Segment Search

from utils import get_input_lines


def main():
    letters = 'abcdefg'
    numbers = {'abcefg': 0, 'cf': 1, 'acdeg': 2, 'acdfg': 3, 'bcdf': 4, 'abdfg': 5, 'abdefg': 6, 'acf': 7, 'abcdefg': 8, 'abcdfg': 9}

    part1 = part2 = 0
    for line in get_input_lines():
        digits, values = map(lambda x: x.split(' '), line.split(' | '))

        # Part 1: count the number of ones, fours, sevens, and eights in the values
        # They are the only numbers identifiable by number of lit segments
        part1 += sum(len(v) in {2, 3, 4, 7} for v in values)

        # Part 2: reverse engineer the digit mapping and use that to compute the actual displayed value
        def of_length(n: int) -> str:
            return next(v for v in digits if len(v) == n)

        one = of_length(2)
        seven = of_length(3)
        four = of_length(4)

        six = next(d for d in digits if len(d) == 6 and any(v not in d for v in one))  # 6 is the only length 6 digit that has a segment that isn't in 1

        a = next(v for v in seven if v not in one)  # 'a', present in 7 but not 1
        c = next(v for v in letters if v not in six)  # 'c', only one missing from 6
        f = next(v for v in one if v != c)  # 'f', the only other segment in 1 now that 'c' is known

        two_three_five = [v for v in digits if len(v) == 5]  # 2, 3, and 5 are the only length 5 digits

        five = next(v for v in two_three_five if f in v and c not in v)  # 5 has 'f' but not 'c'
        two = next(v for v in two_three_five if c in v and f not in v)  # 2 has 'c' but not 'f'

        b = next(v for v in five if v not in two and v not in one)  # 'b' is unique in 5, given it's not in 2 or 1
        e = next(v for v in two if v not in five and v not in one)  # 'e' is unique in 5, given it's not in 2 or 1
        d = next(v for v in four if v not in {b, c, f})  # 'd', only missing letter from 4
        g = next(v for v in letters if v not in {a, b, c, d, e, f})  # 'g', only missing letter

        segments = {a: 'a', b: 'b', c: 'c', d: 'd', e: 'e', f: 'f', g: 'g'}
        part2 += sum(
            d * numbers[''.join(sorted(segments[c] for c in v))]
            for d, v in zip((1000, 100, 10, 1), values)
        )

    print('Part 1:', part1)
    print('Part 2:', part2)


if __name__ == '__main__':
    main()
