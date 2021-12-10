from utils import get_input
from statistics import median


def main(text: str):

    syntax_score = {')': 3, ']': 57, '}': 1197, '>': 25137}
    autocomplete_score = {')': 1, ']': 2, '}': 3, '>': 4}
    inverse = {'(': ')', '[': ']', '{': '}', '<': '>'}

    part1 = 0  # Total syntax score
    part2 = []  # List of scores for incomplete lines
    for line in text.split('\n'):
        stack = []
        for c in line:
            if c in '([{<':
                stack.append(c)
            elif c != inverse[stack.pop()]:
                part1 += syntax_score[c]
                break
        else:
            if stack:  # Only consider score from incomplete lines
                score = 0  # The score is the autocompletion, as a 5-digit binary number with
                for c in reversed(stack):
                    score = score * 5 + autocomplete_score[inverse[c]]
                part2.append(score)

    print('Part 1:', part1)
    print('Part 2:', median(part2))


if __name__ == '__main__':
    main(get_input())
