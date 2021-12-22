# Day 22: Reactor Reboot
# Leaderboard Rank: 18 / 580

from utils import get_input, ints
from typing import NamedTuple, Tuple, List, Optional, Generator


class Box(NamedTuple):
    """ A 3 dimensional box defined by [x0, x1] x [y0, y1] x [z0, z1] """
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def intersect(self, other: 'Box') -> Optional['Box']:
        """ Compute the intersection of this box with another, if it exists. """
        if not (self.x1 < other.x0 or self.x0 > other.x1 or
                self.y1 < other.y0 or self.y0 > other.y1 or
                self.z1 < other.z0 or self.z0 > other.z1):
            return Box(
                max(self.x0, other.x0), min(self.x1, other.x1),
                max(self.y0, other.y0), min(self.y1, other.y1),
                max(self.z0, other.z0), min(self.z1, other.z1)
            )

    def decompose(self, other: 'Box') -> Generator['Box', None, None]:
        """ Where other is a strict subset of this box, return a set of disjoint boxes which partition a - b """
        a, b = self, other
        for box in (
            Box(a.x0, b.x0 - 1, a.y0, a.y1, a.z0, a.z1),
            Box(b.x1 + 1, a.x1, a.y0, a.y1, a.z0, a.z1),
            Box(b.x0, b.x1, a.y0, b.y0 - 1, a.z0, a.z1),
            Box(b.x0, b.x1, b.y1 + 1, a.y1, a.z0, a.z1),
            Box(b.x0, b.x1, b.y0, b.y1, a.z0, b.z0 - 1),
            Box(b.x0, b.x1, b.y0, b.y1, b.z1 + 1, a.z1)
        ):
            if box.area() > 0:
                yield box

    def area(self) -> int:
        return max(0, 1 + self.x1 - self.x0) * max(0, 1 + self.y1 - self.y0) * max(0, 1 + self.z1 - self.z0)


def main(text: str):
    boxes = parse(text)
    print('Part 1:', solve_part1(boxes))
    print('Part 2:', solve_part2(boxes))


def parse(text: str) -> List[Tuple[str, Box]]:
    return [(line.split(' ')[0], Box(*ints(line))) for line in text.split('\n')]


def solve_part1(boxes: List[Tuple[str, Box]]) -> int:
    """ This solves part 1, in a naive (but simple) solution: simply enumerate all the points in each box, since we're bounded by a 100x100x100 region. """
    on = set()
    for step, box in boxes:
        points = {
            (x, y, z)
            for x in range(max(-50, box.x0), 1 + min(50, box.x1))
            for y in range(max(-50, box.y0), 1 + min(50, box.y1))
            for z in range(max(-50, box.z0), 1 + min(50, box.z1))
        }
        if step == 'on':
            on |= points
        else:
            on -= points
    return len(on)


def solve_part2(boxes: List[Tuple[str, Box]]) -> int:
    """ This is required in order to solve part 2. It uses an iterative cube-splitting setup to create at all times a list of disjoint boxes that are currently on """

    on = []
    for step, box in boxes:
        if step == 'on':
            # To add a new box, we first set up a queue of new boxes to add
            # Then, we repeatedly take a box from this queue, and compare it against all other boxes
            # If it is mutually disjoint with all of them, it gets added to the 'on' list
            # If we find an intersection with any existing box, the queued box gets decomposed and re-added to the queue
            adding = [box]  # queued boxes to add, not necessarily disjoint
            while adding:
                to_add = adding.pop()
                for box_on in on:
                    if (intersect := to_add.intersect(box_on)) is not None:
                        for c in to_add.decompose(intersect):
                            adding.append(c)
                        break
                else:
                    on.append(to_add)  # No intersections were found with boxes in 'on', we can add this and maintain the disjoint invariant
        else:
            # When removing a box, we follow a similar procedure to adding, but with a different disjoint check
            # We create a similar queue of boxes to add, except this time we pretend we are re-adding every box
            # On the addition of a singular box, we intersect it with the to_remove box, and if any intersection is found, we add the decomposition of the to_add box instead.
            # Unlike in the addition case, we don't need to re-queue the decomposition as we know they are disjoint with to_remove and the rest of 'on'
            adding = on
            on = []
            to_remove = box
            for to_add in adding:
                if (intersect := to_add.intersect(to_remove)) is not None:
                    for c in to_add.decompose(intersect):
                        on.append(c)
                else:
                    on.append(to_add)

    return sum(map(Box.area, on))


def solve_part2_coordinate_subdivision(boxes: List[Tuple[str, Box]]) -> int:
    """ An alternative method to solve part 2 which uses coordinate subdivisions to make a new grid.
    On the puzzle input, this is roughly a 800x800x800 grid, which actually takes some time to compute through (~3 min)
    It runs all the examples however, in under 3 seconds.
    """
    # The boxes are in [a, b] form. Replace them with coordinate divisions that are [a, b)
    x_divisions = sorted({b.x0 for _, b in boxes} | {b.x1 + 1 for _, b in boxes})
    y_divisions = sorted({b.y0 for _, b in boxes} | {b.y1 + 1 for _, b in boxes})
    z_divisions = sorted({b.z0 for _, b in boxes} | {b.z1 + 1 for _, b in boxes})

    # Map of lower corner coordinates to index into the divisions
    x_index = {x: i for i, x in enumerate(x_divisions)}
    y_index = {y: i for i, y in enumerate(y_divisions)}
    z_index = {z: i for i, z in enumerate(z_divisions)}

    on = set()
    for step, box in boxes:
        points = {
            (x, y, z)
            for x in range(x_index[box.x0], x_index[box.x1 + 1])
            for y in range(y_index[box.y0], y_index[box.y1 + 1])
            for z in range(z_index[box.z0], z_index[box.z1 + 1])
        }
        if step == 'on':
            on |= points
        else:
            on -= points

    # Calculate the actual area held by all boxes
    def area(pos: Tuple[int, int, int]) -> int:
        x, y, z = pos
        return ((x_divisions[x + 1] - x_divisions[x]) *
                (y_divisions[y + 1] - y_divisions[y]) *
                (z_divisions[z + 1] - z_divisions[z]))

    return sum(map(area, on))


if __name__ == '__main__':
    main(get_input())
