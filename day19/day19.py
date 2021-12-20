from utils import Point3, get_input, ints, cross
from typing import List, Tuple, Set, FrozenSet, Dict, Callable, Optional
from functools import lru_cache


def main(text: str):
    scanners: Dict[int, FrozenSet[Point3]] = {
        scanner_id: frozenset(Point3(*ints(beacon)) for beacon in scanner.split('\n')[1:])
        for scanner_id, scanner in enumerate(text.split('\n\n'))
    }

    positions, beacons = find_scanner_positions(scanners, 12)

    print('Part 1:', len(beacons))
    print('Part 2:', max(
        abs(px - qx) + abs(py - qy) + abs(pz - qz)
        for px, py, pz in positions.values()
        for qx, qy, qz in positions.values()
    ))


def find_scanner_positions(scanners: Dict[int, FrozenSet[Point3]], threshold: int) -> Tuple[Dict[int, Point3], Set[Point3]]:

    # This is a heuristic that vastly reduces the runtime, by estimating the most likely pairings of scanners.
    # We use the set of pairwise manhattan distances between each scanner, and order them by the largest intersection of such distances
    fingerprints = {key: fingerprint(scanner) for key, scanner in scanners.items()}
    fingerprint_pairs = sorted([
        (len(fingerprints[a] & fingerprints[b]), a, b)
        for a in scanners.keys()
        for b in scanners.keys()
        if a < b  # WLOG take a < b, exclude cases where a = b
    ], key=lambda k: k[0])  # Last = highest fingerprint match

    positions: Dict[int, Point3] = {0: Point3(0, 0, 0)}  # The positions of each scanner
    oriented: Dict[int, FrozenSet[Point3]] = {0: scanners[0]}  # The oriented beacons of each individual scanner, in order to use the correct target orientation when finding an overlap
    beacons: Set[Point3] = set(scanners[0])  # The positioned beacons in the same orientation to the zero scanner

    remaining = {key for key in scanners.keys() if key != 0}  # Remaining scanner ids that have not been used
    cursor = len(fingerprint_pairs) - 1  # The current index into the fingerprint_pairs sequence
    while remaining:
        _, a, b = fingerprint_pairs[cursor]
        if a in remaining and b in remaining:
            # Both are unknown, we need to skip for now
            # We cannot compute the relative distance as we don't know the orientation of either of them relative to the known scanners
            cursor -= 1
            continue
        elif (a_unknown := a in remaining) or b in remaining:
            # Only one of a or b is unknown
            # Take the target to be known, the candidate to be unknown
            # Attempt to match, checking all orientations. If a valid one is found, we reset to the top of the pairs sequence and continue iterating
            # Otherwise, we can remove this pairing and continue iterating
            if a_unknown:
                target_id, candidate_id = b, a
            else:
                target_id, candidate_id = a, b
            target, candidate = oriented[target_id], scanners[candidate_id]
            if overlap := detect_overlap_any_orientation(target, candidate, threshold):
                pos, oriented_candidate = overlap
                oriented[candidate_id] = oriented_candidate  # Record the oriented match
                real_pos = positions[candidate_id] = pos + positions[target_id]  # Calculate the absolute position of the candidate scanner
                beacons |= {real_pos + beacon for beacon in oriented_candidate}  # Record absolute positions of all beacons

                remaining.remove(candidate_id)
                fingerprint_pairs.pop(cursor)
                cursor = len(fingerprint_pairs) - 1
                continue

        # If we've reached here, there is no possible gain from this pair, and we can safely remove it.
        fingerprint_pairs.pop(cursor)
        cursor -= 1

    return positions, beacons


def fingerprint(beacons: FrozenSet[Point3]) -> FrozenSet[int]:
    return frozenset(
        (a - b).norm1()
        for a in beacons
        for b in beacons
        if a != b
    )


def detect_overlap_any_orientation(beacons: FrozenSet[Point3], candidate: FrozenSet[Point3], threshold: int) -> Optional[Tuple[Point3, FrozenSet[Point3]]]:
    for t in build_transformations():
        oriented_candidate = frozenset(t(pos) for pos in candidate)
        if pos := detect_overlap(beacons, oriented_candidate, threshold):
            return pos, oriented_candidate
    return None

def detect_overlap(beacons: FrozenSet[Point3], candidate: FrozenSet[Point3], threshold: int) -> Optional[Point3]:
    """ Return the vector of scanner positions candidate - beacons """
    for p in beacons:  # absolute positions
        for c in candidate:  # relative positions to unknown scanner pos b
            # Choose b s.t. b + c = p
            b = p - c

            # Count overlapping points, (c0 + b) = p0 : c0 is a candidate and p0 is an arbitrary existing beacon
            overlap = 0
            for c0 in candidate:
                c0 = b + c0
                if c0 in beacons:
                    overlap += 1
                    if overlap >= threshold:
                        return b
    return None

@lru_cache(1)
def build_transformations() -> Tuple[Callable[[Point3], Point3], ...]:
    def build(ps: List[Tuple[int, int]]) -> Callable[[Point3], Point3]:
        # ps is pairs of (index into original coordinates, sign of new coordinates)
        px, py, pz = ps

        def apply(p: Point3) -> Point3:  # def is necessary here to bind the value of ps
            return Point3(p[px[0]] * px[1], p[py[0]] * py[1], p[pz[0]] * pz[1])
        return apply

    basis_vectors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    zero_vector = (0, 0, 0)
    transformations = []

    for i in basis_vectors:
        for j in basis_vectors:
            if (k := cross(i, j)) != zero_vector:
                transformations.append(build([next((n, s) for n, s in enumerate(v) if s != 0) for v in (i, j, k)]))

    assert len(transformations) == 24, 'Expected 24 total transformations'
    return tuple(transformations)


if __name__ == '__main__':
    main(get_input())
