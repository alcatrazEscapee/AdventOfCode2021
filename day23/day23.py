from utils import get_input

from typing import Tuple, List, Set, Dict, Union, TypeVar
from heapq import heappop, heappush

T = TypeVar('T')

BURROWS: Dict[str, int] = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
HALLWAY_STOPS: Tuple[int, ...] = 0, 1, 3, 5, 7, 9, 10  # Valid x positions to stop in
HALLWAY_POSITIONS = dict(
    (
        (start_x, end_x),
        sorted((x for x in HALLWAY_STOPS if start_x < x <= end_x or end_x <= x < start_x), key=lambda k: k if start_x < end_x else -k)
    )
    for start_x in range(0, 11)
    for end_x in range(0, 11)
)

BURROW_X: Dict[str, int] = {'A': 2, 'B': 4, 'C': 6, 'D': 8}  # Target burrow x position for each amphipod
MOVEMENT_COST: Dict[str, int] = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}  # Cost for one step of movement for each amphipod

Hallway = Tuple[str, ...]
Burrow = Tuple[Union[str, int], ...]
State = Tuple[Hallway, Tuple[Burrow, ...]]

def parse_text(text: str) -> Tuple[str, ...]:
    _, _, *rows, _ = text.split('\n')
    return tuple(
        ''.join(c for c in r if c in 'ABCD')
        for r in rows
    )

def parse_rows(*rows: str) -> State:
    hallway = tuple('...........')
    return hallway, tuple(
        (len(burrow) - len(rem := ''.join(burrow).rstrip(target)), *rem)
        for target, *burrow in zip('ABCD', *rows)
    )

def pretty_print(state: State, depth: int):
    hallway, burrows = state
    burrow_contents = (('.' * (depth - len(b) - b[0] + 1) + ''.join(b[1:]) + ('ABCD'[i] * b[0])) for i, b in enumerate(burrows))
    print('\n'.join((
        '#############',
        '#' + ''.join(hallway) + '#',
        *('###' + '#'.join(bc) + '###' for bc in zip(*burrow_contents)),
        '#############',
    )))


def main(text: str):
    # We are only interested in lines 2 and 3, which show the positions of the letters in the top and bottom rooms
    # We infer the rest of the structure of the hallway from the puzzle description
    top_row, bottom_row = parse_text(text)

    # The state, we represent as a sequence of positions, either holding '.' (for empty), or a letter (for a amphipod)
    # The state is ordered (free spaces 0, ... 10, hallways at 2, .. 8 row 1, 2... )
    start: State = parse_rows(top_row, bottom_row)
    end: State = parse_rows('ABCD', 'ABCD')

    print('Part 1:', solve(start, end, 2))

    # Part 2, add two more rows in the middle
    start = parse_rows(top_row, 'DCBA', 'DBAC', bottom_row)
    end = parse_rows('ABCD', 'ABCD', 'ABCD', 'ABCD')

    print('Part 2:', solve(start, end, 4))


def solve(start: State, end: State, depth: int) -> int:
    queue: List[Tuple[int, State]] = [(0, start)]
    seen: Set[State] = set()
    while queue:
        cost, state = heappop(queue)
        if state == end:
            return cost

        if state in seen:
            continue
        seen.add(state)

        hallway, burrows = state

        # Skip any states where we cannot empty any hallway - if this is true, we will never be able to move amphipods into another burrow
        if all(not can_empty(hallway, i, b) for i, b in enumerate(burrows)):
            continue

        # Movements from the hallway into a burrow
        for x in HALLWAY_STOPS:
            pod = hallway[x]
            if pod != '.':
                # Only allowable movement is to move into the target burrow
                target_x = BURROW_X[pod]
                target_burrow_index = BURROWS[pod]
                target_burrow = burrows[target_burrow_index]
                if is_hallway_clear(hallway, x, target_x) and len(target_burrow) == 1:  # the hallway movement is clear, and the target burrow is empty
                    target_y = depth - target_burrow[0]
                    next_state = (
                        replace(hallway, x, '.'),
                        replace(burrows, target_burrow_index, (target_burrow[0] + 1,))
                    )
                    enqueue(queue, cost, abs(x - target_x) + target_y, pod, next_state)

        for burrow_index, burrow in enumerate(burrows):
            if len(burrow) > 1:  # This burrow has occupants that can be moved. Try and move the most open one

                pod = burrow[1]
                x = (burrow_index + 1) * 2
                y = depth - len(burrow) + 2 - burrow[0]
                target_x = BURROW_X[pod]

                # Currently in a room
                # There are two allowable moves: into a hallway, and into another room through a hallway
                # If we can move directly to the target room, we only do that, and don't try moving into a hallway first
                target_burrow_index = BURROWS[pod]
                target_burrow = burrows[target_burrow_index]
                if is_hallway_clear(hallway, x, target_x) and len(target_burrow) == 1:
                    target_y = depth - target_burrow[0]
                    next_state = (
                        hallway,
                        replace2(burrows, burrow_index, (burrow[0], *burrow[2:]), target_burrow_index, (target_burrow[0] + 1,))
                    )
                    enqueue(queue, cost, abs(x - target_x) + y + target_y, pod, next_state)

                else:
                    for stop_x in HALLWAY_STOPS:
                        if is_hallway_clear(hallway, x, stop_x):
                            # A valid move into the hallway is possible
                            next_state = (
                                replace(hallway, stop_x, pod),
                                replace(burrows, burrow_index, (burrow[0], *burrow[2:]))
                            )
                            enqueue(queue, cost, abs(x - stop_x) + y, pod, next_state)

    raise ValueError('Termination without reaching the end')

def enqueue(queue: List[Tuple[int, State]], cost: int, movement: int, pod: str, next_state: State):
    heappush(queue, (cost + movement * MOVEMENT_COST[pod], next_state))


def is_hallway_clear(hallway: Hallway, start_x: int, end_x: int) -> bool:
    """ Checks if the path from start_x -> end_x is clear of other amphipods. """
    return all(hallway[x] == '.' for x in HALLWAY_POSITIONS[start_x, end_x])


def replace(values: Tuple[T, ...], index: int, value: T) -> Tuple[T, ...]:
    return tuple(
        value if i == index else
        v
        for i, v in enumerate(values))

def replace2(values: Tuple[T, ...], first_index: int, first_value: T, second_index: int, second_value: T) -> Tuple[T, ...]:
    return tuple(
        first_value if i == first_index else
        second_value if i == second_index else
        v
        for i, v in enumerate(values))

def can_empty(hallway: Hallway, burrow_index: int, burrow: Burrow) -> bool:
    """ If a burrow at a given index can be emptied completely - there must be enough hallway space to do so """
    count = len(burrow) - 1
    if count == 0:
        return True
    x = (burrow_index + 1) * 2
    for j in HALLWAY_POSITIONS[x, 0]:
        if hallway[j] != '.':
            break
        count -= 1
        if count == 0:
            return True
    for j in HALLWAY_POSITIONS[x, 10]:
        if hallway[j] != '.':
            break
        count -= 1
        if count == 0:
            return True
    return False


if __name__ == '__main__':
    main(get_input())
