from utils import get_input
from collections import defaultdict
from typing import Set, List, Tuple, DefaultDict


def main(text: str):

    # Assemble the graph
    graph: DefaultDict[str, Set[str]] = defaultdict(set)
    for line in text.split('\n'):
        s, e = line.split('-')
        graph[s].add(e)
        graph[e].add(s)

    # Remove edges from 'end' -> X and X -> 'start', to give nice termination conditions
    graph['end'] = set()
    for k, v in graph.items():
        v -= {'start'}

    part1 = set()
    part2 = set()
    queue: List[Tuple[Tuple[str, ...], Set[str], bool]] = [(('start',), set(), True)]

    while queue:
        path, seen, repeat = queue.pop()
        node = path[-1]

        if node == 'end':
            part2.add(path)
            if repeat:  # If the repeated node is still available, this path qualifies for part1 counting
                part1.add(path)

        for step in graph[node]:
            if step.isupper():  # Allowed to traverse large nodes any number of times
                queue.append(((*path, step), seen, repeat))
            elif step not in seen:  # Small nodes must be encountered once (in general)
                queue.append(((*path, step), seen | {step}, repeat))
            elif repeat:  # Except for a single small node to this path, which can be encountered twice
                queue.append(((*path, step), seen, False))

    print('Part 1:', len(part1))
    print('Part 2:', len(part2))


if __name__ == '__main__':
    main(get_input())
