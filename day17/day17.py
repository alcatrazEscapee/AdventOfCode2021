from utils import get_input, ints
from math import sqrt, floor, ceil


def main(text: str):
    min_x, max_x, min_y, max_y = ints(text)

    velocities = set()
    for yt in range(min_y, 1 + max_y):
        for n in range(1, 1 + 2 * abs(yt)):
            if (2 * yt) % n == 0 and (v0 := 2 * yt - n + n * n) % (2 * n) == 0:
                v = v0 // (2 * n)
                for xt in range(min_x, 1 + max_x):
                    # Case: u <= n
                    k = sqrt(2 * xt)
                    for u in range(floor(k) - 1, 1 + ceil(k)):
                        if u <= n and u * (u + 1) == 2 * xt:
                            velocities.add((u, v))
                    # Case: u > n
                    if (2 * xt) % n == 0 and (u0 := 2 * xt - n + n * n) % (2 * n) == 0:
                        u = u0 // (2 * n)
                        if u > n:
                            velocities.add((u, v))

    print('Part 1:', max(v * (v + 1) // 2 for _, v in velocities))
    print('Part 2:', len(velocities))


if __name__ == '__main__':
    main(get_input())
