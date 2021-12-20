from utils_all import *


def main(text: str):
    scanners = [{ints(beacon) for beacon in scanner.split('\n')[1:]} for scanner in text.split('\n\n')]

    ps, bs = find_scanner_positions(scanners, 12)
    for k, v in ps.items():
        print(k, v)

    print('Part 1:', len(bs))
    print('Part 2:', max(
        abs(px - qx) + abs(py - qy) + abs(pz - qz)
        for px, py, pz in ps.values()
        for qx, qy, qz in ps.values()
    ))


def find_scanner_positions(scanners, threshold: int):
    root, *remaining = enumerate(scanners)
    targets = [root]  # (id, beacons) pairs
    positions = {root[0]: (0, 0, 0)}  # id -> Position
    beacons_relative_to_root = set(root[1])
    while remaining:  # While there are still scanners to place
        new_targets = []
        if not targets:
            raise ValueError('No targets left?')
        print('remaining', [r for r, _ in remaining], 'targets', [t for t, _ in targets])
        for target_id, target in targets:  # Loop through target scanners - ones that we have not checked as adjacent to existing scanners
            still_remaining = []
            for candidate_id, candidate in remaining:
                key = target_id, candidate_id

                print('checking', target_id, '->', candidate_id)
                if overlap := detect_overlap_any_orientation(target, candidate, threshold):
                    # An overlap was found, with candidate and target. pos is the difference between candidate and target's position
                    pos, oriented_candidate = overlap
                    print('overlap between', target_id, '->', candidate_id, 'at', pos)
                    positions[candidate_id] = sum_iter(pos, positions[target_id])
                    new_targets.append((candidate_id, oriented_candidate))  # And since this one was positioned, we can now consider it a new target

                    assert detect_overlap(target, oriented_candidate, threshold) == pos

                    root_pos = positions[candidate_id]
                    i = 0
                    for beacon in oriented_candidate:
                        b_pos = sum_iter(root_pos, beacon)
                        if b_pos in beacons_relative_to_root:
                            i += 1
                        beacons_relative_to_root.add(b_pos)
                else:
                    still_remaining.append((candidate_id, candidate))  # An overlapping position was not found, so we keep this one as remaining

            remaining = still_remaining
        targets = new_targets
    return positions, beacons_relative_to_root


def detect_overlap_any_orientation(beacons, candidate, threshold: int):
    for orientation in ORIENTATIONS:
        oriented_candidate = {orientation(pos) for pos in candidate}
        if pos := detect_overlap(beacons, oriented_candidate, threshold):
            return pos, oriented_candidate
    return None

def detect_overlap(beacons, candidate, threshold: int):
    """ Return the vector of scanner positions candidate - beacons """
    for p in beacons:  # absolute positions
        for c in candidate:  # relative positions to unknown scanner pos b
            # Choose b s.t. b + c = p
            b = sum_iter(p, c, -1)

            # Count overlapping points, (c0 + b) = p0 : c0 is a candidate and p0 is an arbitrary existing beacon
            overlap = 0
            for c0 in candidate:
                c0 = sum_iter(b, c0)
                if c0 in beacons:
                    overlap += 1
                    if overlap >= threshold:
                        return b
    return None

def build_orientations():
    basis_vectors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    zero_vector = (0, 0, 0)

    basis_orientations = set()
    for i in basis_vectors:
        for j in basis_vectors:
            if (k := cross(i, j)) != zero_vector:
                basis_orientations.add((i, j, k))

    assert len(basis_orientations) == 24

    def coordinate_transform(b_vec):
        b_index, b_sign = next((n, s) for n, s in enumerate(b_vec) if s != 0)

        def apply(pos) -> int:
            return b_sign * pos[b_index]
        return apply

    def position_transform(b_vec):
        ci, cj, ck = map(coordinate_transform, b_vec)

        def apply(pos):
            return ci(pos), cj(pos), ck(pos)
        return apply

    return [position_transform(b) for b in basis_orientations]


def cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


ORIENTATIONS = build_orientations()

if __name__ == '__main__':
    main(get_input())
