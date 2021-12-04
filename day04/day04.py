# Day 4: Giant Squid
# Leaderboard Rank: 27 / 12

from utils import get_input, ints
from typing import Set, List


def main():
    # Parse input
    text = get_input()
    order = ints(text.split('\n')[0])
    boards = [[list(ints(row)) for row in board.split('\n')] for board in text.split('\n\n')[1:]]

    last_score = None
    called = set()
    for call in order:
        called.add(call)

        remaining = []
        for idx, board in enumerate(boards):
            if victory(board, called):
                if last_score is None:
                    # Part 1 is the first board to victory
                    print('Part 1:', score(board, called, call))
                last_score = score(board, called, call)
            else:
                remaining.append(board)

        # Remove any boards that have already won
        boards = remaining

    # Part 2 is the final board to win
    print('Part 2:', last_score)


def victory(board: List[List[int]], called: Set[int]) -> bool:
    for i in range(5):  # Check every row and column for a bingo
        if all(board[i][j] in called for j in range(5)) or all(board[j][i] in called for j in range(5)):
            return True
    return False  # Otherwise, this board has not won yet

def score(board: List[List[int]], called: Set[int], last: int) -> int:
    # Sum of all not-called numbers on the board multiplied by the last called value
    return sum(x for row in board for x in row if x not in called) * last


if __name__ == '__main__':
    main()