# Day 4: Giant Squid
# Leaderboard Rank: 27 / 12

from utils import get_input, ints
from typing import Set, Tuple

BingoBoard = Tuple[Tuple[int, ...], ...]

def main(text: str):
    # Parse input
    order, *boards = text.split('\n\n')
    order = ints(order)
    boards = [tuple(ints(row) for row in board.split('\n')) for board in boards]

    first = last = None
    called = set()
    for call in order:
        called.add(call)

        remaining = []
        for board in boards:
            if victory(board, called):
                if first is None:
                    first = score(board, called, call)
            else:
                remaining.append(board)

        # Remove any boards that have already won, and record the last board
        if not remaining:
            last = score(boards[0], called, call)  # Assumes a single unique last winner as per the problem definition
            break

        boards = remaining

    print('Part 1:', first)
    print('Part 2:', last)


def victory(board: BingoBoard, called: Set[int]) -> bool:
    return any(
        all(board[i][j] in called for j in range(5)) or
        all(board[j][i] in called for j in range(5))
        for i in range(5)
    )

def score(board: BingoBoard, called: Set[int], last: int) -> int:
    # Sum of all not-called numbers on the board multiplied by the last called value
    return sum(x for row in board for x in row if x not in called) * last


if __name__ == '__main__':
    main(get_input())
