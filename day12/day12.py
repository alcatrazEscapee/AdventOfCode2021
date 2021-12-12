from utils import get_input
from collections import defaultdict
from typing import Set, List, Tuple, Optional, DefaultDict


def main(text: str):

    # Assemble the graph
    edges: DefaultDict[str, Set[str]] = defaultdict(set)
    for line in text.split('\n'):
        s, e = line.split('-')
        edges[s].add(e)
        edges[e].add(s)

    # Remove edges from 'end' -> X and X -> 'start', to give nice termination conditions
    edges['end'] = set()
    for k, v in edges.items():
        v -= {'start'}

    print('Part 1:', total_paths(edges, False))
    print('Part 2:', total_paths(edges, True))


def total_paths(graph: DefaultDict[str, Set[str]], allow_double_small_node_traversal: bool) -> int:
    """ Number of paths given that:
    - Large nodes can be encountered any number of times
    - Small nodes can be encountered once (or if allow_double_small_node_traversal, then a single node may be encountered twice)
    In order to track this, we actually track the sum of all paths that have a *specific* small node that may be encountered twice
    """
    paths = set()
    queue: List[Tuple[Tuple[str, ...], Set[str], Optional[str]]]

    if allow_double_small_node_traversal:
        queue = [(('start',), set(), n) for n in graph.keys() if n.islower()]
    else:
        queue = [(('start',), set(), None)]  # This path artificially has no small nodes which it can traverse twice

    while queue:
        path, seen, small = queue.pop()
        node = path[-1]

        if node == 'end':
            paths.add(path)

        for step in graph[node]:
            if step.isupper():  # Allowed to traverse large nodes any number of times
                queue.append(((*path, step), set(seen), small))
            elif step not in seen:  # Small nodes must be encountered once (in general)
                queue.append(((*path, step), seen | {step}, small))
            elif step == small:  # Except for a *specific* small node to this path, which can be encountered twice
                queue.append(((*path, step), set(seen), None))
    return len(paths)


if __name__ == '__main__':
    main(get_input())
