# Day 18: Snailfish

from utils import get_input
from typing import List, Union, Optional

import functools
import operator
import math


# Recursive Types
SnailfishNumber = List[Union['SnailfishNumber', int]]


class Node:
    def __init__(self, value: Optional[int] = None):
        self.parent: Optional['Node'] = None
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.value: Optional[int] = value

    def add_left(self, child: 'Node'):
        self.left = child
        child.parent = self

    def add_right(self, child: 'Node'):
        self.right = child
        child.parent = self

    def __str__(self):
        return str(self.value) if self.value is not None else '[' + str(self.left) + ',' + str(self.right) + ']'

    def __add__(self, other):
        node = Node()
        node.add_left(self)
        node.add_right(other)
        reduce(node)
        return node

    def magnitude(self) -> int:
        return self.value if self.value is not None else 3 * self.left.magnitude() + 2 * self.right.magnitude()

    def listify(self) -> SnailfishNumber:
        """ Converts this tree representation of a snailfish number back into a list form """
        return self.value if self.value is not None else [self.left.listify(), self.right.listify()]


def main(text: str):
    lists = [eval(line) for line in text.split('\n')]
    print('Part 1:', functools.reduce(operator.add, map(convert, lists)).magnitude())
    print('Part 2:', max(
        (convert(a) + convert(b)).magnitude()
        for a in lists
        for b in lists
    ))


def convert(number: SnailfishNumber) -> Node:
    """ Converts a snail number list-like specification into a node tree structure """
    if isinstance(number, list):
        node = Node()
        lhs, rhs = number
        node.add_left(convert(lhs))
        node.add_right(convert(rhs))
        return node
    elif isinstance(number, int):
        return Node(value=number)
    else:
        raise ValueError('Not a snail number: %s' % str(number))

def reduce(node: Node):
    """ Performs all reductions after a single addition. """
    while try_explode(node) or try_split(node):
        pass

def try_split(node: Optional[Node]) -> bool:
    """ Performs a single split operation if possible. Returns True if a split was done. Mutates the input. """
    if node is None:
        return False
    if node.value is not None and node.value >= 10:
        # Split the current node
        node.add_left(Node(value=math.floor(node.value / 2)))
        node.add_right(Node(value=math.ceil(node.value / 2)))
        node.value = None
        return True
    return try_split(node.left) or try_split(node.right)

def try_explode(node: Optional[Node], depth: int = 0) -> bool:
    """ Performs a single explode operation if possible. Returns True if an explode was done. Mutates the input. """
    if node is None:
        return False
    if node.value is None and depth == 4:
        # Explode the current node
        left_value, right_value = node.left.value, node.right.value
        node.left = node.right = None
        node.value = 0
        explode_cascade(node, left_value, False)
        explode_cascade(node, right_value, True)
        return True
    return try_explode(node.left, depth + 1) or try_explode(node.right, depth + 1)

def explode_cascade(node: Node, value: int, reverse_direction: bool):
    """
    Cascades a single value, and adds is to the leftmost adjacent value
    Since the tree is a balanced tree, we know that the process to find the adjacent value will be to go up -> left once -> right
    The reverse_direction param simply switches the order of the left and right operations
    """
    left, right = lambda n: n.left, lambda n: n.right
    if reverse_direction:
        left, right = right, left

    while node.parent is not None:
        if left(node.parent) != node:
            node = left(node.parent)
            while right(node) is not None:
                node = right(node)
            node.value += value
            return
        node = node.parent


if __name__ == '__main__':
    main(get_input())
