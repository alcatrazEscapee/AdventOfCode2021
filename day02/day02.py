from utils import get_input_lines


def main():
    aim = pos = part1_depth = part2_depth = 0
    for line in get_input_lines():
        op, x = line.split(' ')  # op X
        x = int(x)
        if op == 'forward':
            pos += x
            part2_depth += aim * x
        elif op == 'up':
            part1_depth -= x
            aim -= x
        elif op == 'down':
            part1_depth += x
            aim += x

    print('Part 1:', part1_depth * pos)
    print('Part 2:', part2_depth * pos)


if __name__ == '__main__':
    main()
