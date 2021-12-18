import functools

from utils_all import *

INPUT = [
[[[4,[8,0]],[[6,9],[0,3]]],[[[2,9],[6,3]],[1,[9,9]]]],
[[[2,[5,0]],[[5,6],4]],[[[2,0],[4,8]],[7,8]]],
[[[5,[0,6]],[9,6]],4],
[[9,[5,3]],[[4,9],[8,2]]],
[[5,[6,2]],[[9,3],[4,[3,2]]]],
[[[9,[9,8]],[2,9]],[[4,[6,5]],[[6,2],9]]],
[[[9,8],7],[[0,[2,1]],[[9,5],[4,5]]]],
[[4,[0,1]],[[[3,6],5],[4,9]]],
[[0,9],[[3,1],3]],
[[[3,[7,4]],[9,8]],5],
[[6,[2,8]],[0,[[6,9],[8,1]]]],
[[1,[[1,1],[9,4]]],[[3,[2,9]],[7,[5,4]]]],
[[[4,[9,3]],[4,[3,8]]],[[[1,3],[7,1]],5]],
[[[[2,9],8],7],[4,[5,4]]],
[[8,[[1,3],5]],[[6,5],6]],
[[[[5,5],[3,1]],[[6,8],2]],[[8,5],3]],
[[[7,[7,3]],7],0],
[[7,[[1,0],0]],[2,[9,2]]],
[[3,[1,7]],7],
[[2,[6,3]],[[[1,6],7],[9,3]]],
[5,5],
[[[[5,1],[1,8]],[[3,0],2]],[[8,2],[9,0]]],
[[[[4,1],[0,8]],[9,[9,7]]],[[[2,4],3],[8,9]]],
[[3,[8,2]],9],
[[7,1],[[[5,8],[9,9]],2]],
[[[[2,3],[7,7]],[[6,3],7]],[8,[[1,8],[1,5]]]],
[[0,[0,5]],[[[1,5],6],0]],
[8,[[[7,9],[2,9]],6]],
[[5,[3,[8,7]]],[3,4]],
[[[8,2],[3,[5,2]]],4],
[[[[9,6],[3,3]],[3,[1,8]]],[[[6,1],4],[[1,3],2]]],
[[[[5,7],[3,6]],[0,[6,4]]],[[[0,2],8],3]],
[[[2,4],[3,[9,1]]],6],
[[[[9,6],[2,0]],[4,0]],[[5,[0,9]],[[5,3],[6,6]]]],
[[[3,5],9],[[7,[8,1]],[[2,6],[0,6]]]],
[[[9,2],[[3,2],8]],[4,4]],
[[[4,[5,6]],5],[[7,[8,7]],2]],
[[[4,8],[3,[7,1]]],1],
[8,[[1,[9,4]],7]],
[[[[2,3],5],[7,[4,9]]],[[4,8],[[8,1],[3,1]]]],
[[[[4,2],4],[1,[0,7]]],[[1,[4,5]],[9,[3,6]]]],
[[[[7,2],[4,9]],[6,2]],[[6,7],[2,[0,2]]]],
[[4,5],[[[4,1],3],5]],
[[[9,[2,2]],[[0,1],[3,2]]],2],
[[2,[[7,5],3]],[[[1,0],[7,4]],0]],
[6,[[3,[7,2]],[[6,5],[4,7]]]],
[[4,[[5,3],[1,8]]],[[[6,0],3],[2,7]]],
[[[3,[4,3]],[8,1]],[8,[3,[0,7]]]],
[[[[8,5],[0,5]],[[8,0],9]],[[[7,7],[3,0]],[4,[2,7]]]],
[[4,[[2,0],[5,7]]],[8,2]],
[[[[6,3],[8,9]],[[7,5],[4,3]]],[9,[2,4]]],
[9,[[[2,1],[9,7]],[5,[3,8]]]],
[[0,[[6,3],[7,8]]],[[7,[8,2]],3]],
[[7,[[2,4],4]],6],
[[3,[2,8]],[[[1,8],1],0]],
[[1,[[5,0],1]],6],
[[5,[2,[5,4]]],[[[8,5],9],[8,[7,2]]]],
[[[[5,0],[6,1]],0],[[7,9],[2,3]]],
[[[[9,6],[0,0]],[0,[5,2]]],5],
[[[[6,6],6],[0,[9,4]]],[[[0,7],[3,8]],[8,5]]],
[6,8],
[[6,[7,1]],5],
[[[[2,9],1],[[7,6],5]],[[9,2],[[9,5],2]]],
[[2,[6,5]],[2,[0,8]]],
[[[4,9],8],0],
[[3,[[8,4],[3,5]]],[[0,3],[2,8]]],
[[[3,[1,6]],1],[3,[[7,4],4]]],
[[[1,[0,8]],6],[[2,5],[6,[1,2]]]],
[[7,[[5,9],5]],[[7,9],7]],
[[[3,[9,9]],[[5,0],2]],[[8,[6,6]],9]],
[[9,4],[[2,[6,1]],6]],
[[2,[[3,2],5]],[9,8]],
[[[1,[5,7]],4],[9,[[7,2],3]]],
[[[[4,0],[3,9]],[[2,4],[9,4]]],0],
[[[6,5],8],[[[1,7],3],7]],
[[[[5,9],4],6],[[[3,3],[0,4]],[3,[2,2]]]],
[[[[3,5],[7,4]],[[7,2],[3,2]]],1],
[[[0,9],[1,[4,6]]],[3,[[6,9],9]]],
[[[[3,8],4],[8,[5,6]]],[6,[[0,1],8]]],
[[5,[[4,3],5]],[[2,[2,8]],[5,[5,7]]]],
[[[4,[2,7]],0],[[7,6],[[5,8],[4,4]]]],
[[[2,[3,3]],[6,[1,7]]],[[[2,8],[9,1]],[[2,7],[9,2]]]],
[[9,[3,5]],[[[9,4],[1,8]],[[7,2],[9,6]]]],
[[5,[4,[4,0]]],[[5,5],[[8,0],7]]],
[0,[[[1,9],9],[7,[0,3]]]],
[[[[5,3],8],1],[[[7,3],[5,4]],9]],
[[[[4,0],4],[9,[1,9]]],[[[8,9],7],[[5,9],[0,3]]]],
[[[0,8],[[7,2],7]],[[1,[8,4]],[[8,3],2]]],
[[[[2,9],[0,0]],[0,[2,2]]],[6,9]],
[2,[6,2]],
[[[[9,9],[8,1]],5],6],
[[4,1],[[[5,9],[3,2]],[0,[4,0]]]],
[[[[8,9],3],[8,0]],[[[4,6],[2,3]],1]],
[[[0,5],[3,[8,2]]],[3,2]],
[[[[3,3],[1,8]],[[5,8],[2,7]]],[[[1,5],9],[4,2]]],
[[[[7,0],5],2],[1,[[5,1],6]]],
[3,[[9,[9,3]],[1,[2,8]]]],
[[5,[[7,4],[0,3]]],[4,[[4,4],[6,8]]]],
[[[8,7],[5,1]],[[4,5],[7,[3,8]]]],
[[[[4,5],[5,5]],[[2,7],[0,5]]],[[5,[7,0]],[[9,6],5]]]
]


class Node:
    def __init__(self, left = None, right = None, value = None):
        self.left_parent = None
        self.right_parent = None
        self.left = left
        self.right = right
        self.value = value

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        else:
            return '[' + str(self.left) + ',' + str(self.right) + ']'

    def parent(self):
        return self.left_parent if self.left_parent is not None else self.right_parent


def main(text: str):

    test_explode([[[[[9,8],1],2],3],4], [[[[0,9],2],3],4])
    test_explode([7,[6,[5,[4,[3,2]]]]], [7,[6,[5,[7,0]]]])
    test_explode([[6,[5,[4,[3,2]]]],1], [[6,[5,[7,0]]],3])
    test_explode([[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]], [[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]])
    test_explode([[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]], [[3,[2,[8,0]]],[9,[5,[7,0]]]])
    test_explode([[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]], [[[[0,7],4],[7,[[8,4],9]]],[1,1]])
    test_explode([[[[0,7],4],[7,[[8,4],9]]],[1,1]], [[[[0,7],4],[15,[0,13]]],[1,1]])

    test_split([[[[0,7],4],[15,[0,13]]],[1,1]], [[[[0,7],4],[[7,8],[0,13]]],[1,1]])
    test_split([[[[0,7],4],[[7,8],[0,13]]],[1,1]], [[[[0,7],4],[[7,8],[0,[6,7]]]],[1,1]])

    test_add_list(([1,1], [2,2], [3,3], [4,4]), [[[[1,1],[2,2]],[3,3]],[4,4]])
    test_add_list(([1,1], [2,2], [3,3], [4,4], [5,5]), [[[[3,0],[5,3]],[4,4]],[5,5]])
    test_add_list(([1,1], [2,2], [3,3], [4,4], [5,5], [6,6]), [[[[5,0],[7,4]],[5,5]],[6,6]])

    test_add_list((
        [[[0, [4, 5]], [0, 0]], [[[4, 5], [2, 6]], [9, 5]]],
        [7, [[[3, 7], [4, 3]], [[6, 3], [8, 8]]]],
        [[2, [[0, 8], [3, 4]]], [[[6, 7], 1], [7, [1, 6]]]],
        [[[[2, 4], 7], [6, [0, 5]]], [[[6, 8], [2, 8]], [[2, 1], [4, 5]]]],
        [7, [5, [[3, 8], [1, 4]]]],
        [[2, [2, 2]], [8, [8, 1]]],
        [2, 9],
        [1, [[[9, 3], 9], [[9, 0], [0, 7]]]],
        [[[5, [7, 4]], 7], 1],
        [[[[4, 2], 2], 6], [8, 7]],
    ), [[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]])

    test_magnitude([[1,2],[[3,4],5]], 143)
    test_magnitude([[[[0,7],4],[[7,8],[6,0]]],[8,1]], 1384)
    test_magnitude([[[[1,1],[2,2]],[3,3]],[4,4]], 445)
    test_magnitude([[[[3,0],[5,3]],[4,4]],[5,5]], 791)
    test_magnitude([[[[5,0],[7,4]],[5,5]],[6,6]], 1137)
    test_magnitude([[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]], 3488)

    print('Part 1:', magnitude(add_list([convert(ls) for ls in INPUT])))
    print('Part 2:', max(
        magnitude(add(convert(a), convert(b)))
        for a in INPUT
        for b in INPUT
    ))

def test_explode(lhs, rhs):
    lhs = convert(lhs)
    try_explode(lhs)
    lhs = str(lhs)
    rhs = str(convert(rhs))
    assert lhs == rhs, 'Actual: ' + lhs

def test_split(lhs, rhs):
    lhs = convert(lhs)
    try_split(lhs)
    lhs = str(lhs)
    rhs = str(convert(rhs))
    assert lhs == rhs, 'Actual: ' + lhs

def test_add_list(lhs, rhs):
    lhs = add_list((convert(ls) for ls in lhs))
    lhs = str(lhs)
    rhs = str(convert(rhs))
    assert lhs == rhs, 'Actual: ' + lhs

def test_magnitude(node, value):
    lhs = magnitude(convert(node))
    assert lhs == value, 'Actual: ' + str(value)


def magnitude(node):
    if node.value is not None:
        return node.value
    else:
        return 3 * magnitude(node.left) + 2 * magnitude(node.right)


def add_list(nodes):
    return functools.reduce(add, nodes)


def add(lhs, rhs):
    node = Node(lhs, rhs)
    lhs.left_parent = node
    rhs.right_parent = node
    reduce(node)
    return node


def reduce(node):
    while True:
        if try_explode(node):
            continue
        if try_split(node):
            continue
        break
    return node

def try_split(node) -> bool:
    if node.value is not None and node.value >= 10:
        node.left = Node(value=math.floor(node.value / 2))
        node.left.left_parent = node
        node.right = Node(value=math.ceil(node.value / 2))
        node.right.right_parent = node
        node.value = None
        return True
    if node.left is not None:
        if try_split(node.left):
            return True
    if node.right is not None:
        if try_split(node.right):
            return True
    return False

def try_explode(node, depth = 0):
    if node.value is None:
        if depth == 4:
            explode(node)
            return True
        if node.left is not None:
            if try_explode(node.left, depth + 1):
                return True
        if node.right is not None:
            if try_explode(node.right, depth + 1):
                return True
    return False

def explode(node):
    lv, rv = node.left.value, node.right.value
    assert lv is not None and rv is not None
    if node.left_parent:
        assert node.left_parent.left == node
        parent = node.left_parent
        parent.left = Node(value=0)
        parent.left.left_parent = parent
    elif node.right_parent:
        assert node.right_parent.right == node
        parent = node.right_parent
        parent.right = Node(value=0)
        parent.right.right_parent = parent
    else:
        assert False, 'no parent'

    explode_left(node, lv)
    explode_right(node, rv)


def explode_left(node, value):
    curr = node
    while True:
        if curr.right_parent is not None:
            curr = curr.right_parent.left
            break
        elif curr.left_parent is not None:
            curr = curr.left_parent
        else:
            return
    while curr.right is not None:
        curr = curr.right
    assert curr.value is not None
    curr.value += value


def explode_right(node, value):
    curr = node
    while True:
        if curr.left_parent is not None:
            curr = curr.left_parent.right
            break
        elif curr.right_parent is not None:
            curr = curr.right_parent
        else:
            return
    while curr.left is not None:
        curr = curr.left
    assert curr.value is not None
    curr.value += value


def convert(lists):
    if isinstance(lists, list):
        lhs, rhs = lists
        lhs = convert(lhs)
        rhs = convert(rhs)
        par = Node(lhs, rhs)
        lhs.left_parent = par
        rhs.right_parent = par
        return par
    elif isinstance(lists, int):
        return Node(value=lists)
    assert False


if __name__ == '__main__':
    main(get_input())
