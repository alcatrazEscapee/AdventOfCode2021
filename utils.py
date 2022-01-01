
from typing import Tuple, List, Optional, Sequence, Callable, Generic, TypeVar, NamedTuple, Generator, Iterable

import re


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def get_input(path: str = './input.txt') -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def ints(text: str, sign_prefixes: bool = True) -> Tuple[int, ...]:
    regex = r'([\-+]?\d+)' if sign_prefixes else r'(\d+)'
    return tuple(map(int, re.findall(regex, text)))

def windows(x: Sequence[T], n: int) -> Generator[Tuple[T, ...], None, None]:
    """ Returns sliding windows of n length across the sequence x """
    if n <= len(x):
        for i in range(1 + len(x) - n):
            yield tuple(x[i:i + n])

def sign(a: int) -> int:
    """ Returns the sign of a """
    return 0 if a == 0 else (-1 if a < 0 else 1)

def cyclic_mod(value: int, min_inclusive: int, max_inclusive: int) -> int:
    """ Given a value x and a range [a, b], wraps x so that it lies within the range [a, b] """
    return ((value - min_inclusive) % (max_inclusive - min_inclusive + 1)) + min_inclusive

def cross(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, int]:
    """ Returns the cross product of the three-dimensional vectors a and b """
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx

def norm1(x: Iterable[int]) -> int:
    """ Returns the manhattan distance (1-norm) of x """
    return sum(map(abs, x))


class Point2(NamedTuple):
    x: int
    y: int

    def __add__(self, other: Tuple[int, int]) -> 'Point2':
        return Point2(self.x + other[0], self.y + other[1])

    def __sub__(self, other: Tuple[int, int]) -> 'Point2':
        return Point2(self.x - other[0], self.y - other[1])

    def norm1(self):
        return norm1(self)

    def neighbors(self) -> Generator['Point2', None, None]:
        yield Point2(self.x + 1, self.y)
        yield Point2(self.x - 1, self.y)
        yield Point2(self.x, self.y + 1)
        yield Point2(self.x, self.y - 1)

    def moore_neighbors(self) -> Generator['Point2', None, None]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield Point2(self.x + dx, self.y + dy)


class Point3(NamedTuple):
    x: int
    y: int
    z: int

    def __add__(self, other: Tuple[int, int, int]) -> 'Point3':
        return Point3(self.x + other[0], self.y + other[1], self.z + other[2])

    def __sub__(self, other: Tuple[int, int, int]) -> 'Point3':
        return Point3(self.x - other[0], self.y - other[1], self.z - other[2])

    def norm1(self):
        return norm1(self)


class FiniteGrid(Generic[T]):

    @staticmethod
    def of_empty(width: int, height: int, default: Optional[T] = None, wrap: bool = False) -> 'FiniteGrid[T]':
        return FiniteGrid(width, height, [default] * width * height, default, wrap)

    @staticmethod
    def of_str(s: str, default: Optional[T] = None, wrap: bool = False) -> 'FiniteGrid[T]':
        return FiniteGrid.of_iter(s.strip().split('\n'), default, wrap)

    @staticmethod
    def of_iter(seq: Sequence[Sequence[T]], default: Optional[T] = None, wrap: bool = False) -> 'FiniteGrid[T]':
        height, width = len(seq), len(seq[0])
        assert all(len(sub) == width for sub in seq), 'Different widths of sub sequences'
        return FiniteGrid(width, height, [e for r in seq for e in r], default, wrap)

    def __init__(self, width: int, height: int, array: List[T], default: Optional[T], wrap: bool):
        self.width: int = width
        self.height: int = height
        self.default: T = default
        self.wrap: bool = wrap
        self.array: List[T] = array

    def __getitem__(self, key: Tuple[int, int]) -> T:
        if self.wrap:
            x, y = key[0] % self.width, key[1] % self.height
        else:
            x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.array[x + self.width * y]
        elif self.default is not None:
            return self.default
        raise ValueError('Tried to get the value at invalid location (%d, %d)' % (x, y))

    def __setitem__(self, key: Tuple[int, int], value: T):
        if self.wrap:
            x, y = key[0] % self.width, key[1] % self.height
        else:
            x, y = key
        if 0 <= x < self.width and 0 <= y < self.height:
            self.array[x + self.width * y] = value
        elif self.default is None:
            raise ValueError('Tried to set the value at invalid location (%d, %d)' % (x, y))

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return 0 <= key[0] < self.width and 0 <= key[1] < self.height

    def __len__(self) -> int:
        return len(self.array)

    def __eq__(self, other):
        return other is not None and isinstance(other, FiniteGrid) and self.array == other.array

    def __repr__(self):
        return '[%d x %d Grid]\n%s' % (self.width, self.height, str(self))

    def __str__(self) -> str:
        return '\n'.join(''.join(str(self.array[x + self.width * y]) for x in range(self.width)) for y in range(self.height))

    def map(self, f: Callable[[Point2], T]) -> 'FiniteGrid[T]':
        return FiniteGrid(self.width, self.height, [f(pos) for pos in self.locations()], self.default, self.wrap)

    def map_values(self, f: Callable[[T], V]) -> 'FiniteGrid[V]':
        return FiniteGrid(self.width, self.height, [f(v) for v in self.array], f(self.default) if self.default is not None else None, self.wrap)

    def row(self, row: int) -> Generator[T, None, None]:
        for x in range(self.width):
            yield self.array[x + self.width * row]

    def col(self, col: int) -> Generator[T, None, None]:
        for y in range(self.height):
            yield self.array[col + self.width * y]

    def copy(self) -> 'FiniteGrid[T]': return self.map(lambda p: self[p])
    def rotate_cw(self) -> 'FiniteGrid[T]': return self.map(lambda p: self[p.y, self.width - 1 - p.x])
    def rotate_ccw(self) -> 'FiniteGrid[T]': return self.map(lambda p: self[self.height - 1 - p.y, p.x])
    def mirror_y(self) -> 'FiniteGrid[T]': return self.map(lambda p: self[self.width - 1 - p.x, p.y])
    def mirror_x(self) -> 'FiniteGrid[T]': return self.map(lambda p: self[p.x, self.height - 1 - p.y])

    def permutations(self) -> Generator['FiniteGrid[T]', None, None]:
        """ Iterates through all permutations (rotations and mirrors) of this grid """
        grid = self
        for _ in range(4):
            yield grid
            yield grid.mirror_y()
            grid = grid.rotate_cw()

    def locations(self) -> Generator[Point2, None, None]:
        """ An iterator over all coordinate positions within the grid """
        for y in range(self.height):
            for x in range(self.width):
                yield Point2(x, y)
