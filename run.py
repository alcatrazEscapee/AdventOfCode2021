
import os
import time
import subprocess

from typing import Optional, Tuple, DefaultDict
from collections import defaultdict


def main():
    expected: DefaultDict[int, Tuple[Optional[str], Optional[str]]] = defaultdict(lambda: ('', ''), {
        1: ('1482', '1518')
    })

    for i in range(1, 1 + 25):
        result = run_day(i)
        if result:
            p1, p2, delta = result
            result = 'Day %2d' % i + format_part(1, p1, expected[i][0]) + format_part(2, p2, expected[i][1]) + ' | Time: ' + format_time(delta)
        else:
            result = 'Day %2d | No solution' % i
        print(result)


def run_day(day: int, cmd: str = 'python') -> Optional[Tuple[Optional[str], Optional[str], float]]:
    try:
        os.chdir('day%02d' % day)
    except:
        return None
    try:
        return run_process('%s day%02d.py' % (cmd, day))
    except:
        return None
    finally:
        os.chdir('../')


def run_process(cmd: str) -> Tuple[str, str, float]:
    part1 = part2 = None
    now = time.time()
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        output = proc.stdout.readline().decode('utf-8').replace('\r', '').replace('\n', '')
        if output.startswith('Part 1: '):
            part1 = output[8:]
        elif output.startswith('Part 2: '):
            part2 = output[8:]
    proc.wait()
    then = time.time()
    return part1, part2, then - now


def format_time(delta: float) -> str:
    if delta > 1:
        return '%3d  s' % int(delta)
    elif delta > 0.0001:
        return '%3d ms' % int(delta * 1000)
    elif delta > 0.000_0001:
        return '%3d us' % int(delta * 1000_000)
    else:
        return '%3d ns' % int(delta * 1000_000_000)


def format_part(part: int, result: Optional[str], expected: Optional[str]) -> str:
    desc = 'n/a   ' if not expected else ('passed' if result == expected else 'failed')
    return f' | Part {part}: {desc}'


if __name__ == '__main__':
    main()
