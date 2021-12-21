from utils import get_input, ints, cyclic_mod

from collections import Counter
from functools import lru_cache
from typing import Tuple


def main(text: str):
    p1, p2 = (ints(line)[-1] for line in text.split('\n'))

    print('Part 1:', play_deterministic_dice(p1, p2))
    print('Part 2:', max(play_dirac_dice(p1, p2)))


def play_deterministic_dice(p1: int, p2: int, dice_state: int = 1, total_dice: int = 0, p1s: int = 0, p2s: int = 0) -> int:
    """ p1, p2 are the player positions, p1s, p2s are the player's scores. The current turn to be taken is by player one.
    Returns the product of total dice rolled * losing player's score
    """
    p1_dice, dice_state = deterministic_dice(dice_state)
    total_dice += 3
    p1 = cyclic_mod(p1 + p1_dice, 1, 10)
    p1s += p1

    if p1s >= 1000:
        return total_dice * p2s
    return play_deterministic_dice(p2, p1, dice_state, total_dice, p2s, p1s)


def deterministic_dice(state: int) -> Tuple[int, int]:
    """ Rolls a deterministic dice three times, returning the dice sum and the state value of the dice """
    new_state = cyclic_mod(state + 3, 1, 100)
    return 3 * state + 3, new_state


@lru_cache(None)
def play_dirac_dice(p1: int, p2: int, p1s: int = 0, p2s: int = 0) -> Tuple[int, int]:
    """ p1, p2 are the player positions. p1s, p2s are the player's scores. The current turn to be taken is by player one .
    Returns the pair of counts of universes in which p1 and p2 win, respectively.
    """
    p1_wins = p2_wins = 0  # Counts of the universes in which each player wins
    for roll, count in dirac_dice().items():
        p1_next = cyclic_mod(p1 + roll, 1, 10)
        p1s_next = p1s + p1_next
        if p1s_next < 21:
            # player 1 has not yet won. We swap the players and calculate wins, then swap the sums
            a, b = play_dirac_dice(p2, p1_next, p2s, p1s_next)
            p1_wins += b * count
            p2_wins += a * count
        else:
            # player 1 has won in these universes, so we just increment their wins
            p1_wins += count
    return p1_wins, p2_wins

@lru_cache(1)
def dirac_dice() -> Counter:
    """ The map of roll value -> universe count of a set of three dirac dice """
    counts = Counter()
    for a in range(1, 1 + 3):
        for b in range(1, 1 + 3):
            for c in range(1, 1 + 3):
                counts[a + b + c] += 1
    return counts


if __name__ == '__main__':
    main(get_input())
