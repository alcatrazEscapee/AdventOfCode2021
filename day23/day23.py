from utils_all import *


def main(text: str):
    lines = text.split('\n')

    hallways = 2, 4, 6, 8
    # points 0, ... 10 inclusive, and hallways with (x, 1), and (x, 2)

    input_setup = ((6, 1), (8, 2), (2, 1), (2, 2), (4, 1), (4, 2), (6, 2), (8, 1))

    example_1 = ((2, 2), (8, 2), (2, 1), (6, 1), (4, 1), (6, 2), (4, 2), (8, 1))

    # interpretation
    # state: (A, A, B, B, C, C, D, D, cost)
    letters = 'AABBCCDD'
    rooms = (2, 2, 4, 4, 6, 6, 8, 8)
    costs = [1, 1, 10, 10, 100, 100, 1000, 1000]
    valid_stopping_spaces = (0, 1, 3, 5, 7, 9, 10)
    queue: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = [(0, input_setup)]
    seen_positions = set(queue[0][1])
    while queue:
        last_cost, positions = heapq.heappop(queue)
        occupied = {a: letters[i] for i, a in enumerate(positions)}

        if positions in seen_positions:
            continue
        seen_positions.add(positions)


        if positions == ((2, 2), (8, 2), (2, 1), (3, 0), (4, 1), (6, 2), (4, 2), (8, 1)):
            print('step 1')
        if positions == ((2, 2), (8, 2), (2, 1), (3, 0), (6, 1), (6, 2), (4, 2), (8, 1)):
            print('step 2')
        if positions == ((2, 2), (8, 2), (2, 1), (3, 0), (6, 1), (6, 2), (5, 0), (8, 1)):
            print('step 3')
        if positions == ((2, 2), (8, 2), (2, 1), (4, 2), (6, 1), (6, 2), (5, 0), (8, 1)):
            print('step 4')
        if positions == ((2, 2), (8, 2), (4, 1), (4, 2), (6, 1), (6, 2), (5, 0), (8, 1)):
            print('step 5')

        if positions == ((2, 2), (8, 2), (4, 1), (4, 2), (6, 1), (6, 2), (5, 0), (8, 1)):
            print('step 5')
        if positions == ((2, 2), (8, 2), (4, 1), (4, 2), (6, 1), (6, 2), (5, 0), (7, 0)):
            print('step 6')

        x_positions = tuple(p[0] for p in positions)
        if x_positions == rooms:
            print('finished!')
            print(last_cost)
            break

        # try and move each one
        for i, an in enumerate(positions):
            letter = letters[i]
            room_target = rooms[i]
            room1 = room_target, 1
            room2 = room_target, 2
            if an[1] > 0:  # in a room
                if an == room2:  # already in target room at bottom position - should never move
                    continue
                if an == room1 and room2 in occupied and occupied[room2] == letter:  # both are already in target room - should never move
                    continue
                step_cost = 0
                if an[1] == 1:
                    # next to hallway
                    step_cost = 1
                elif an[1] == 2 and (an[0], 1) not in occupied:
                    # one away from hallway, and the space is not occupied, so they can move into the hallway
                    step_cost = 2
                if step_cost == 0:
                    continue  # cannot exit this room

                for stop in valid_stopping_spaces:  # each stopping space from this hallway
                    min_x = min(stop, an[0])
                    max_x = max(stop, an[0])
                    # need all the movement space to be free
                    if not all(
                        (x, 0) not in occupied
                        for x in range(min_x, 1 + max_x)
                    ):
                        continue

                    # then we can move into the stopping spot
                    move_cost = (max_x - min_x + step_cost) * costs[i]
                    next_states = tuple(s if j != i else (stop, 0) for j, s in enumerate(positions))
                    heapq.heappush(queue, (last_cost + move_cost, next_states))

                # or, we can see if we can move directly into a target room
                min_x = min(an[0], room_target)
                max_x = max(an[0], room_target)
                if not all(
                        (x, 0) not in occupied
                        for x in range(min_x, 1 + max_x)
                ):
                    continue  # cannot traverse to the target room
                step_into_cost = 0
                if room1 not in occupied and room2 not in occupied:
                    # move into room 2
                    step_into_cost = 2
                    end_room = room2
                elif room1 not in occupied and room2 in occupied and occupied[room2] == letter:
                    # move into room 1
                    step_into_cost = 1
                    end_room = room1
                else:
                    continue  # cannot traverse into target room

                move_cost = (max_x - min_x + step_cost + step_into_cost) * costs[i]
                next_states = tuple(s if j != i else end_room for j, s in enumerate(positions))
                heapq.heappush(queue, (last_cost + move_cost, next_states))

            else:  # in a hallway
                # can only move into a room
                # can only move into destination room
                # can only move into destination if unoccupied, or occupied by the correct person
                room1 = room_target, 1
                room2 = room_target, 2
                if room1 not in occupied and room2 not in occupied:
                    # move into room2
                    # need to assert space is available
                    min_x = min(an[0], room_target)
                    max_x = max(an[0], room_target)
                    if not all(
                        (x, 0) not in occupied or x == an[0]
                        for x in range(min_x, 1 + max_x)
                    ):
                        continue
                    # space is available. so move to the right x and two into the space
                    move_cost = (max_x - min_x + 2) * costs[i]
                    next_states = tuple(s if j != i else room2 for j, s in enumerate(positions))
                    heapq.heappush(queue, (last_cost + move_cost, next_states))

                elif room1 not in occupied and room2 in occupied and occupied[room2] == letter:
                    # move into (room_target, 1)
                    min_x = min(an[0], room_target)
                    max_x = max(an[0], room_target)
                    if not all(
                            (x, 0) not in occupied or x == an[0]
                            for x in range(min_x, 1 + max_x)
                    ):
                        continue
                    # space is available. so move to the right x and one into the space
                    move_cost = (max_x - min_x + 1) * costs[i]
                    next_states = tuple(s if j != i else room1 for j, s in enumerate(positions))
                    heapq.heappush(queue, (last_cost + move_cost, next_states))
                else:
                    continue



if __name__ == '__main__':
    main(get_input())
