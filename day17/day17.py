from utils import get_input, ints

from math import sqrt, floor, ceil
from collections import defaultdict


def main(text: str):
    min_x, max_x, min_y, max_y = ints(text)

    n2v = defaultdict(set)

    for yt in range(min_y, 1 + max_y):
        for n in range(1, 1 + 2 * abs(yt)):
            if (2 * yt) % n == 0 and (v0 := 2 * yt - n + n * n) % (2 * n) == 0:
                v = v0 // (2 * n)
                n2v[n].add(v)

    trajectories = set()
    for xt in range(min_x, 1 + max_x):
        k = sqrt(2 * xt)
        for u in range(floor(k) - 1, 1 + ceil(k)):
            if u * (u + 1) == 2 * xt:
                for n, vs in n2v.items():
                    if u <= n:
                        for v in vs:
                            trajectories.add((u, v))
        for n, vs in n2v.items():
            if (2 * xt) % n == 0 and (u0 := 2 * xt - n + n * n) % (2 * n) == 0:
                u = u0 // (2 * n)
                if u > n:
                    for v in vs:
                        trajectories.add((u, v))

    print('Part 1:',  max(v * (v + 1) // 2 for _, v in trajectories if v > 0))
    print('Part 2:', len(trajectories))


if __name__ == '__main__':
    main(get_input())
