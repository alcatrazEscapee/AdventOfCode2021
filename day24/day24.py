# Day 24: Arithmetic Logic Unit
# Leaderboard Rank: 170 / 159

from utils import get_input, ints
from typing import List, Tuple


def main(text: str):
    python, rules = decode_input(text)

    # The python code found in solve_manually(), without comments
    print('\n'.join(python))

    print('Part 1:', solve(rules, max))
    print('Part 2:', solve(rules, min))


def decode_input(text: str) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """
    Given a set of assembly input, produces two things:
    1. An equivalent form of the assembly into python
    2. The set of digit rules the assembly is implementing (in the form (A, B, c), representing dA = dB + c
    """
    lines = text.split('\n')

    decoded = [
        'x = 0',
        'z = 0',
        *['d%d = 0' % d for d in range(14)]
    ]

    # The instructions implement a stack, and then use that to translate digit parity.
    # The stack is empty if all digit parity rules have been met, aka 0
    # z = z * 26 + x => 'push x'
    # z = z // 26 => 'pop'
    # x = (z % 26) => 'peek'
    rules = []
    stack = []
    last = None

    for digit in range(14):  # 14 digits of a serial number
        sub_program = lines[digit * 18:digit * 18 + 18]  # 18 lines per sub program
        _, _, divisor, addend, *_, tail_addend = ints(' '.join(sub_program))

        # Build the python code
        decoded.append('x = (z %% 26) + %d' % addend)
        if divisor == 26:
            decoded.append('z //= 26')
        decoded.append('if d%d != x:' % digit)
        decoded.append('    z = (26 * z) + d%d + %d' % (digit, tail_addend))

        if divisor == 26:
            last = stack.pop()
        if addend > 0:  # In practice, these are all > 10 to guarantee the if statement is hit
            stack.append((digit, tail_addend))
        else:
            # restriction is discovered
            last_digit, last_addend = last
            rules.append((digit, last_digit, last_addend + addend))

    return decoded, rules


def solve(rules: List[Tuple[int, int, int]], optimizer) -> int:
    digits = [0] * 14
    for a, b, delta in rules:
        if a > b:  # WLOG, take a < b
            a, b, delta = b, a, -delta
        digits[b] = optimizer([d for d in range(1, 10) if 1 <= d + delta <= 9])
        digits[a] = digits[b] + delta

    return int(''.join(map(str, digits)))


def solve_manually():
    # Decoded rules
    # d4 = d3 + 3
    # d5 = d2 - 4
    # d7 = d6 - 6
    # d9 = d8 + 5
    # d11 = d10 + 2
    # d12 = d1 + 7
    # d13 = d0 - 7

    x = 0
    z = 0
    d0 = 0
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    d6 = 0
    d7 = 0
    d8 = 0
    d9 = 0
    d10 = 0
    d11 = 0
    d12 = 0
    d13 = 0

    # There are 7 statements that translate to stack pop operations (z //= 26)
    # There are 14 if statements that guard stack push operations (z = z * 26 + x)
    # 7 of these if statements are always true
    # Thus, find the 7 if statements that are *not* always true, and determine what they must entail

    x = (z % 26) + 11  # x = 11
    z //= 1
    assert d0 != x
    if d0 != x:  # true
        z = (26 * z) + d0 + 1  # z = d0 + 1
    x = (z % 26) + 11  # x = d0 + 12
    z //= 1  # z = d0 + 1
    assert d1 != x
    if d1 != x:  # true
        z = (26 * z) + d1 + 11  # z = 26 * (d0 + 1) + d1 + 11
    x = (z % 26) + 14  # x = d1 + 25
    z //= 1  # z = 26 * (d0 + 12) + d1 + 11
    assert d2 != x
    if d2 != x:  # true
        z = (26 * z) + d2 + 1  # z = 26 * (26 * (d0 + 1) + d1 + 11) + d2 + 1
    x = (z % 26) + 11  # x = d2 + 12
    z //= 1
    assert d3 != x
    if d3 != x:  # true
        z = (26 * z) + d3 + 11  # z = 26 * (26 * (26 * (d0 + 1) + d1 + 11) + d2 + 1) + d3 + 11
    x = (z % 26) + -8  # x = d3 + 3
    z //= 26  # z = 26 * (26 * (d0 + 1) + d1 + 11) + d2 + 1
    assert d4 == x
    if d4 != x:  # false (1): d4 == d3 + 3
        z = (26 * z) + d4 + 2
    x = (z % 26) + -5  # x = d2 - 4
    z //= 26  # z = 26 * (d0 + 1) + d1 + 11
    assert d5 == x
    if d5 != x:  # false (2): d5 = d2 - 4
        z = (26 * z) + d5 + 9  # noop
    x = (z % 26) + 11  # x = d1 + 22
    z //= 1
    assert d6 != x
    if d6 != x:  # true
        z = (26 * z) + d6 + 7  # z = 26 * (26 * (d0 + 1) + d1 + 11) + d6 + 7
    x = (z % 26) + -13  # x = d6 - 6
    z //= 26  # z = 26 * (d0 + 12) + d1 + 11
    assert d7 == x, 'x = ' + str(x)
    if d7 != x:  # false (3): d7 = d6 - 6
        z = (26 * z) + d7 + 11
    x = (z % 26) + 12  # d1 + 22
    z //= 1
    assert d8 != x
    if d8 != x:  # true
        z = (26 * z) + d8 + 6  # z = 26 * (26 * (d0 + 1) + d1 + 11) + d8 + 6
    x = (z % 26) + -1  # x = d8 + 5
    z //= 26  # z = 26 * (d0 + 12) + d1 + 11
    assert d9 == x
    if d9 != x:  # false (4): d9 = d8 + 5
        z = (26 * z) + d9 + 15
    x = (z % 26) + 14  # x = d11 + 25
    z //= 1
    assert d10 != x
    if d10 != x:  # true
        z = (26 * z) + d10 + 7  # z = 26 * (26 * (d0 + 1) + d1 + 11) + d10 + 7
    x = (z % 26) + -5  # x = d10 + 2
    z //= 26  # z = 26 * (d0 + 12) + d1 + 11
    assert d11 == x
    if d11 != x:  # false (5): d11 = d10 + 2
        z = (26 * z) + d11 + 1
    x = (z % 26) + -4  # x = d1 + 7
    z //= 26  # z = d0 + 12
    assert d12 == x
    if d12 != x:  # false (6): d12 = d1 + 7
        z = (26 * z) + d12 + 8
    x = (z % 26) + -8  # x = d0 - 7
    z //= 26  # z = 0
    assert d13 == x, 'x = ' + str(x)
    if d13 != x:  # false (7): d13 == d0 - 7
        z = (26 * z) + d13 + 6

    assert z == 0
    print(''.join(map(str, [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13])))


def run(text: str, value: int) -> int:
    """ Runs the assembly code directly with the given digits as input, and returns the value in z """
    memory = [0, 0, 0, 0]  # w x y z
    indexes = {'w': 0, 'x': 1, 'y': 2, 'z': 3}

    def ref(p):
        if p in indexes:
            return memory[indexes[p]]
        return int(p)

    digits = list(map(int, str(value)))  # Handy trick to convert base 10 int to msb -> lsb digits
    for instruction in text.split('\n'):
        parts = instruction.split(' ')
        if parts[0] == 'inp':
            memory[indexes[parts[1]]] = digits.pop(0)
        elif parts[0] == 'add':
            memory[indexes[parts[1]]] += ref(parts[2])
        elif parts[0] == 'mul':
            memory[indexes[parts[1]]] *= ref(parts[2])
        elif parts[0] == 'div':
            memory[indexes[parts[1]]] //= ref(parts[2])
        elif parts[0] == 'mod':
            memory[indexes[parts[1]]] %= ref(parts[2])
        elif parts[0] == 'eql':
            memory[indexes[parts[1]]] = 1 if ref(parts[2]) == ref(parts[1]) else 0

    return memory[3]


if __name__ == '__main__':
    main(get_input())
