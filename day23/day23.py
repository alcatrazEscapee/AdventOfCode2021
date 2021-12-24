from utils import get_input

from typing import Tuple, List, Set, Dict
from heapq import heappop, heappush

BURROWS: Tuple[int, ...] = 2, 4, 6, 8  # Burrow x positions
HALLWAY_STOPS: Tuple[int, ...] = 0, 1, 3, 5, 7, 9, 10  # Valid x positions to stop in
ROOM_TARGET: Dict[str, int] = {'A': 2, 'B': 4, 'C': 6, 'D': 8}  # Target burrow x position for each amphipod
MOVEMENT_COST: Dict[str, int] = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}  # Cost for one step of movement for each amphipod

State = Tuple[str, ...]


def main(text: str):
    # We are only interested in lines 2 and 3, which show the positions of the letters in the top and bottom rooms
    # We infer the rest of the structure of the hallway from the puzzle description
    _, _, top_row, bottom_row, _ = text.split('\n')

    # The state, we represent as a sequence of positions, either holding '.' (for empty), or a letter (for a amphipod)
    # The state is ordered (free spaces 0, ... 10, hallways at 2, .. 8 row 1, 2... )
    start: State = tuple('...........') + row_of(top_row) + row_of(bottom_row)
    end: State = tuple('...........ABCDABCD')

    print('Part 1:', solve(start, end))

    # Part 2, add two more rows in the middle
    start = tuple('...........') + row_of(top_row) + tuple('DCBADBAC') + row_of(bottom_row)
    end = end + tuple('ABCDABCD')

    print('Part 2:', solve(start, end))


def solve(start: State, end: State) -> int:
    queue: List[Tuple[int, State]] = [(0, start)]
    seen: Set[State] = set()
    depth: int = get_depth(start)
    while queue:
        cost, state = heappop(queue)

        if state == end:
            return cost

        if state in seen:
            continue
        seen.add(state)

        # Iterate through each location, and if an amphipod is at that location, attempt possible moves
        for index, pod in enumerate(state):
            if pod != '.':  # Amphipod at the current location
                x, y = to_position(index)
                target_x = ROOM_TARGET[pod]

                if y == 0:
                    # Currently in a hallway
                    # Only allowable movement is to move into the target burrow
                    # Additionally, no other amphipods of a different letter must be present
                    if is_hallway_clear(state, x, target_x) and (target_y := find_clear_burrow(state, pod, target_x, depth)) != -1:
                        enqueue(queue, cost, abs(x - target_x) + target_y, state, pod, index, target_x, target_y)
                else:
                    # Currently in a room
                    # There are two allowable moves: into a hallway, and into another room through a hallway
                    # Before computing both, we observe two optimizations:
                    # 1. If we are already in our destination room (with no others below us), we don't move as this is optimal already
                    if x == target_x and all(state[to_index(x, dy)] == pod for dy in range(1 + y, 1 + depth)):
                        continue

                    # 2. If we can move directly to the target room, we only do that, and don't try moving into a hallway first
                    if is_burrow_clear(state, x, y):
                        # Now, compute valid moves from a burrow -> hallway -> burrow
                        # We are still restricted as before, in only moving to the target burrow
                        if is_hallway_clear(state, x, target_x) and (target_y := find_clear_burrow(state, pod, target_x, depth)) != -1:
                            enqueue(queue, cost, abs(x - target_x) + y + target_y, state, pod, index, target_x, target_y)
                            continue

                        for stop_x in HALLWAY_STOPS:
                            if is_hallway_clear(state, x, stop_x, False):
                                # A valid move into the hallway is possible
                                enqueue(queue, cost, abs(x - stop_x) + y, state, pod, index, stop_x, 0)

    raise ValueError('Termination without reaching the end')


def enqueue(queue: List[Tuple[int, State]], cost: int, movement: int, state: State, pod: str, index: int, x: int, y: int):
    destination = to_index(x, y)
    next_state = tuple(
        pod if i == destination else
        ('.' if i == index else s)
        for i, s in enumerate(state))
    heappush(queue, (cost + movement * MOVEMENT_COST[pod], next_state))


def row_of(row: str) -> State:
    return tuple(c for c in row if c in 'ABCD')

def to_index(x: int, y: int) -> int:
    """ Where x represents the horizontal movement, y is the hallway depth """
    return x if y == 0 else 6 + (x // 2) + 4 * y

def to_position(index: int) -> Tuple[int, int]:
    """ Inverse of to_index """
    if index <= 10:
        return index, 0
    k = index - 11
    return BURROWS[k % 4], 1 + k // 4

def get_depth(state: State) -> int:
    return (len(state) - 11) // 4

def is_burrow_clear(state: State, x: int, y: int) -> bool:
    """ If the route from (x, y) is clear to move into the hallway """
    return all(state[to_index(x, dy)] == '.' for dy in range(y))

def is_hallway_clear(state: State, start_x: int, end_x: int, exclude_start_x: bool = True) -> bool:
    """ Checks if the path from start_x -> end_x is clear of other amphipods """
    min_x = min(start_x, end_x)
    max_x = max(start_x, end_x)
    return all(state[to_index(x, 0)] == '.' or (exclude_start_x and x == start_x) for x in range(min_x, 1 + max_x))

def find_clear_burrow(state: State, pod: str, target_x: int, depth: int) -> int:
    """ Finds a clear position in the burrow for the given pod, if one exists """
    y = 0
    while y < depth and state[to_index(target_x, y + 1)] == '.':
        y += 1
    if y > 0:  # Must have found at least one empty space
        if all(state[to_index(target_x, dy)] == pod for dy in range(1 + y, 1 + depth)):  # All deeper must be of the same type
            return y
    return -1  # Invalid burrow


if __name__ == '__main__':
    main(get_input())
