
from typing import Tuple, List, Optional, Sequence, Callable, Generic, TypeVar, Iterator

import re


T = TypeVar('T')
V = TypeVar('V')


def get_input(path: str = './input.txt') -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_input_lines(path: str = './input.txt') -> List[str]:
    return get_input(path).split('\n')

def ints(text: str, sign_prefixes: bool = True) -> Tuple[int, ...]:
    regex = r'([\-+]?\d+)' if sign_prefixes else r'(\d+)'
    return tuple(map(int, re.findall(regex, text)))

def sign(a: int) -> int:
    """ Returns the sign of a """
    return 0 if a == 0 else (-1 if a < 0 else 1)


class FiniteGrid(Generic[T]):

    @staticmethod
    def of_iter(seq: Sequence[Sequence[T]], default: Optional[T] = None, wrap: bool = False) -> 'FiniteGrid[T]':
        height, width = len(seq), len(seq[0])
        assert set(len(sub) for sub in seq) == {width}, 'Different widths of sub sequences'
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

    def __eq__(self, other):
        return other is not None and isinstance(other, FiniteGrid) and self.array == other.array

    def __repr__(self):
        return '[%d x %d Grid]\n%s' % (self.width, self.height, str(self))

    def __str__(self) -> str:
        return '\n'.join(''.join(str(self.array[x + self.width * y]) for x in range(self.width)) for y in range(self.height))

    def map(self, f: Callable[[int, int], V]) -> 'FiniteGrid[V]':
        return FiniteGrid(self.width, self.height, [f(x, y) for x, y in self.locations()], self.default, self.wrap)

    def row(self, row: int) -> Iterator[T]:
        for x in range(self.width):
            yield self.array[x + self.width * row]

    def col(self, col: int) -> Iterator[T]:
        for y in range(self.height):
            yield self.array[col + self.width * y]

    def copy(self) -> 'FiniteGrid[T]': return self.map(lambda x, y: self[x, y])
    def rotate_cw(self) -> 'FiniteGrid[T]': return self.map(lambda x, y: self[y, self.width - 1 - x])
    def rotate_ccw(self) -> 'FiniteGrid[T]': return self.map(lambda x, y: self[self.height - 1 - y, x])
    def mirror_y(self) -> 'FiniteGrid[T]': return self.map(lambda x, y: self[self.width - 1 - x, y])
    def mirror_x(self) -> 'FiniteGrid[T]': return self.map(lambda x, y: self[x, self.height - 1 - y])

    def permutations(self) -> Iterator['FiniteGrid[T]']:
        """ Iterates through all permutations (rotations and mirrors) of this grid """
        grid = self
        for _ in range(4):
            yield grid
            yield grid.mirror_y()
            grid = grid.rotate_cw()

    def locations(self) -> Iterator[Tuple[int, int]]:
        """ An iterator over all coordinate positions within the grid """
        for y in range(self.height):
            for x in range(self.width):
                yield x, y

    def count(self, e: T) -> int:
        return sum(v == e for v in self.array)
