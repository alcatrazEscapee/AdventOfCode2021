from utils import get_input, ints, sign


def main(text: str):
    global MINX, MAXX, MINY, MAXY
    MINX, MAXX, MINY, MAXY = ints(text)

    # target area: x=70..96, y=-179..-124
    my = 0
    count = 0
    for vx in range(1, 1000):
        for vy in range(-1000, 1000):

            win, max_y = shoot(vx, vy)
            if win:
                my = max(max_y, my)
                count += 1
    print('Part 1:', my)
    print('Part 2:', count)


def shoot(vx: int, vy: int):
    x = 0
    y = 0
    y_max = 0
    while True:
        x += vx
        y += vy

        vx -= sign(vx)
        vy -= 1

        y_max = max(y, y_max)

        if MINX <= x <= MAXX and MINY <= y <= MAXY:
            return True, y_max

        if y < MINY:
            return False, 0  # no hit

        if vx == 0:
            if x < MINX or x > MAXX:
                return False, 0

        if vx > 0:
            if x > MAXX:
                return False, 0


if __name__ == '__main__':
    main(get_input())
