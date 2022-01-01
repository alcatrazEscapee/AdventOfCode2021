# Entrypoint for running all, or individual puzzles
# Times each puzzle's execution

from typing import NamedTuple, Tuple, Optional, Any
from contextlib import redirect_stdout
from time import perf_counter_ns
from io import StringIO

from utils import get_input

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


PuzzleOutput = Tuple[Optional[str], Optional[str]]
Days = day01, day02, day03, day04, day05, day06, day07, day08, day09, day10, day11, day12, day13, day14, day15, day16, day17, day18, day19, day20, day21, day22, day23, day24, day25

class ExampleInput(NamedTuple):
    example: int


def main():
    benchmark_all()

def benchmark_all():
    for day in range(1, 1 + 25):
        benchmark_day(day)

def benchmark_day(day: int):
    try:
        inp = resolve_input(day)
        tick = perf_counter_ns()
        exec_day(day, inp)
        tock = perf_counter_ns()
        print('Day %02d:' % day, format_ns(tock - tick))
    except Exception as e:
        print('Day %02d:' % day, e)

def run_day(day: int, inp: Any = None) -> PuzzleOutput:
    inp = resolve_input(day, inp)
    out = exec_day(day, inp)
    return decode_output(out)

def exec_day(day: int, inp: str) -> str:
    with StringIO() as out, redirect_stdout(out):
        Days[day - 1].main(inp)
        return out.getvalue()

# Inputs

def resolve_input(day: int, inp: Any = None) -> str:
    if inp is None:
        return get_input('./day%02d/input.txt' % day)
    elif isinstance(inp, ExampleInput):
        return get_input('./day%02d/examples.txt' % day).split('\n\n=====\n\n')[inp.example - 1]
    return inp

def example(index: int) -> ExampleInput:
    return ExampleInput(index)

def decode_output(output: str) -> PuzzleOutput:
    part1 = part2 = None
    for line in output.split('\n'):
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
