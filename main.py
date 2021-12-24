# Entrypoint for running all, or individual puzzles
# Times each puzzle's execution

from typing import Tuple, Optional
from time import perf_counter_ns
from contextlib import redirect_stdout
from io import StringIO

from utils import get_input

try:
    from day01 import day01
    from day02 import day02
    from day03 import day03
    from day04 import day04
    from day05 import day05
    from day06 import day06
    from day07 import day07
    from day08 import day08
    from day09 import day09
    from day10 import day10
    from day11 import day11
    from day12 import day12
    from day13 import day13
    from day14 import day14
    from day15 import day15
    from day16 import day16
    from day17 import day17
    from day18 import day18
    from day19 import day19
    from day20 import day20
    from day21 import day21
    from day22 import day22
    from day23 import day23
    from day24 import day24
    from day25 import day25
except:
    pass

PuzzleOutput = Tuple[Optional[str], Optional[str]]

def main():
    for day in range(1, 1 + 25):
        try:
            tick = perf_counter_ns()
            run_day(day)
            tock = perf_counter_ns()
            print('Day %02d:' % day, format_ns(tock - tick))
        except Exception as e:
            print('Day %02d:' % day, e)

def get_example(day: int, example: int) -> str:
    return get_input('./day%02d/examples.txt' % day).split('\n\n=====\n\n')[example - 1]

def get_day_input(day: int) -> str:
    return get_input('./day%02d/input.txt' % day)

def run_day_with_example(day: int, example: int) -> PuzzleOutput:
    return run_day(day, get_example(day, example))

def run_day(day: int, inp: Optional[str] = None) -> PuzzleOutput:
    if inp is None:
        try:
            inp = get_day_input(day)
        except:
            inp = None

    with StringIO() as out, redirect_stdout(out):
        match day:
            case 1:  day01.main(inp)
            case 2:  day02.main(inp)
            case 3:  day03.main(inp)
            case 4:  day04.main(inp)
            case 5:  day05.main(inp)
            case 6:  day06.main(inp)
            case 7:  day07.main(inp)
            case 8:  day08.main(inp)
            case 9:  day09.main(inp)
            case 10: day10.main(inp)
            case 11: day11.main(inp)
            case 12: day12.main(inp)
            case 13: day13.main(inp)
            case 14: day14.main(inp)
            case 15: day15.main(inp)
            case 16: day16.main(inp)
            case 17: day17.main(inp)
            case 18: day18.main(inp)
            case 19: day19.main(inp)
            case 20: day20.main(inp)
            case 21: day21.main(inp)
            case 22: day22.main(inp)
            case 23: day23.main(inp)
            case 24: day24.main(inp)
            case 25: day25.main(inp)
            case _: raise ValueError('Incomplete')

        result = out.getvalue()

    # Decode the output
    part1 = part2 = None
    for line in result.split('\n'):
        if line.startswith('Part 1: '):
            part1 = line[8:]
        elif line.startswith('Part 2: '):
            part2 = line[8:]
    return part1, part2

def format_ns(delta: int) -> str:
    if delta < 10_000:
        return '%5d ns' % delta
    elif delta < 10_000_000:
        return '%5d us' % int(delta * 0.001)
    elif delta < 10_000_000_000:
        return '%5d ms' % int(delta * 0.000_001)
    else:
        return '%5d  s' % int(delta * 0.000_000_001)


if __name__ == '__main__':
    main()
